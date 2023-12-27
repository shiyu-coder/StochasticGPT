from io import StringIO

import numpy as np
import streamlit as st
import time

from content_analysis import section_analysis_async
from paper_scoring import paper_scoring
from util import *
from paper_class import *
from structure_extraction import extract_paper_structure, extract_title

# 设置页面配置
st.set_page_config(
    page_title="StochasticGPT",
    page_icon="🧊",
    layout="wide",
)

# 初始化状态变量
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# 初始化会话状态中的 paper 对象
if 'paper' not in st.session_state:
    st.session_state['paper'] = Paper()


def set_page_state_to_uploaded():
    st.session_state.file_uploaded = True


def show_file_progress_and_result(placeholder, uploaded_file):
    with placeholder:
        with st.spinner('文件处理中，请稍候...'):
            progress_bar = st.progress(0)
            paper = st.session_state['paper']
            # 检查是否存在论文缓存
            file_name = uploaded_file.name
            paper.file_name = file_name
            paper_cache = load_from_cache(file_name)
            if paper_cache is None:
                # 将latex文件转成字符串
                paper_content = process_uploaded_paper_data(uploaded_file)
                paper.paper_content = paper_content
                progress_bar.progress(5)
                # 抽取论文题目
                paper_title = extract_title(paper_content)
                if paper_title is None:
                    paper_title = "Unknown Title"
                paper.title = paper_title
                # 抽取论文结构
                dt_paper = extract_paper_structure(paper_content)
                progress_bar.progress(50)
                paper.overall_structure = dt_paper['overall_stu']
                paper.dt_section_structure = dt_paper['section_stu']
                paper.dt_section_content = dt_paper['section_con']
                # 全文内容检查
                ls_analysis_result = section_analysis_async(paper.dt_section_content, paper.dt_section_structure, paper.overall_structure)
                dt_analysis_result = dict(zip(list(paper.dt_section_content.keys()), ls_analysis_result))
                paper.dt_analysis_result = dt_analysis_result
                progress_bar.progress(90)
                # 全文打分
                dt_score = paper_scoring(paper.paper_content)
                paper.dt_score = dt_score
                paper.paper_score = np.mean(list(paper.dt_score.values()))
                progress_bar.progress(95)
                # 初始化润色结果
                paper.initial_polishing_result()
                # 存储论文
                save_cache(paper)
            else:
                paper = paper_cache
                progress_bar.progress(100)
            st.session_state['paper'] = paper
            progress_bar.empty()
            set_page_state_to_uploaded()
            st.success(f'Paper {uploaded_file.name} processing completed!')
        time.sleep(1)  # 为用户提供简短的成功提示时间
        placeholder.empty()  # 清空占位符内容，用于在同一位置展示文件评审页面
        review_system_page(placeholder)  # 在相同的占位符中显示新的页面内容


def upload_page(placeholder):
    with placeholder:
        st.title('Upload you .tex paper')
        uploaded_file = st.file_uploader("", type=['tex'])
        if uploaded_file is not None:
            show_file_progress_and_result(placeholder, uploaded_file)


# 使用一个函数来切换按钮状态并在需要时显示表单
def toggle_custom_input(box_title, button_type):
    key = f'show_{button_type}_{box_title}'
    if key not in st.session_state:
        st.session_state[key] = False
    st.session_state[key] = not st.session_state[key]


def slot_rescoring():
    paper = st.session_state['paper']
    polishing_paper = get_polishing_paper(paper.paper_content, paper.dt_polishing_result)
    dt_score = paper_scoring(polishing_paper)

    st.session_state['paper'].dt_score = dt_score
    st.session_state['paper'].paper_score = np.mean(list(paper.dt_score.values()))


def review_system_page(placeholder):
    with placeholder.container():
        paper = st.session_state['paper']
        st.title(paper.title)

        # 展示评分结果
        st.subheader("Total Score")
        # 获取最新的总分数并展示
        total_score = paper.paper_score
        st.write(f"Total score: {total_score} / 10")

        st.subheader("Each Subitem Score")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Consistency: {paper.dt_score['Consistency']} / 10")
            st.write(f"Coherence: {paper.dt_score['Coherence']} / 10")
        with col2:
            st.write(f"Conciseness: {paper.dt_score['Conciseness']} / 10")
            st.write(f"Substantiveness: {paper.dt_score['Substantiveness']} / 10")

        # "重新评分" 按钮
        st.button("Rescoring", on_click=slot_rescoring)

        # 循环创建框和按钮，假设创建两个框，可以按照需要进行修改
        ls_section_label = list(paper.dt_section_content.keys())
        for i in range(len(paper.dt_section_content)):
            section_label = ls_section_label[i]
            box_title = f'Box {i}'
            with st.container():  # 使用容器包裹整个框的内容
                left_col, right_col = st.columns([3, 2])  # 分配3:2的比例给左右两列

                with left_col:
                    st.subheader(section_label)
                    # 假设的Markdown内容，可以根据实际需求调整
                    content = paper.dt_analysis_result[section_label]
                    st.markdown(content, unsafe_allow_html=True)

                    # 按钮横向排列
                    btn_cols = st.columns(5)  # 5个按钮分布于5列，自适应宽度
                    with btn_cols[0]:
                        st.button('Language polishing', key=f'lang_btn_{i}')
                    with btn_cols[1]:
                        st.button('Structural polishing', key=f'struct_btn_{i}')
                    with btn_cols[2]:
                        st.button('Section polishing', key=f'chapter_btn_{i}')
                    with btn_cols[3]:
                        if st.button('Custom polishing', key=f'custom_btn_{i}'):
                            toggle_custom_input(box_title, 'custom_input')
                    with btn_cols[4]:
                        if st.button('Polishing scheme', key=f'design_btn_{i}'):
                            toggle_custom_input(box_title, 'design_input')

                    # 如果点击了“自定义润色”，显示自定义输入框和确认按钮
                    if st.session_state.get(f'show_custom_input_{box_title}', False):
                        with st.form(key=f'custom_form_{i}'):
                            custom_input = st.text_area('Please enter your custom polishing requests:')
                            submit_button = st.form_submit_button(label='Confirm')
                            if submit_button:
                                right_col.markdown(f'#### {box_title} 自定义润色要求')
                                right_col.write(custom_input)
                                toggle_custom_input(box_title, 'custom_input')

                    # 如果点击了“设计润色方案”，显示设计方案输入框和确认按钮
                    if st.session_state.get(f'show_design_input_{box_title}', False):
                        with st.form(key=f'design_form_{i}'):
                            design_input = st.text_area('How would you like to design the polish plan?')
                            submit_button = st.form_submit_button(label='Confirm')
                            if submit_button:
                                right_col.markdown(f'#### {box_title} 设计润色方案')
                                right_col.write(design_input)
                                toggle_custom_input(box_title, 'design_input')

                with right_col:
                    st.subheader(f'{section_label} - Polishing results')
                    st.markdown("", unsafe_allow_html=True)


if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False

placeholder = st.empty()  # 创建一个初始的占位符

if not st.session_state.file_uploaded:
    upload_page(placeholder)  # 上传文件的逻辑
else:
    review_system_page(placeholder)  # 文件已上传时，显示评审系统的逻辑
