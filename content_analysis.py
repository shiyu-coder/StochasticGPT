import re

import json
import traceback

from llm_api import GPT, parse_json, llm_request
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import LatexTextSplitter
from util import multiprocess, get_cpu_count


def replace_at_sentences(text):
    # 定义正则表达式模式，匹配@句子@(句子)格式
    pattern = re.compile(r'@\s*([^@]+)\s*@\s*\(([^)]+)\)')

    # 替换函数
    def replace(match):
        sentence1 = match.group(1)  # 获取第一个句子
        sentence2 = match.group(2)  # 获取第二个句子
        # 返回替换后格式的字符串
        return f'<span style="color:red;">{sentence1}</span>(<span style="color:orange;">{sentence2}</span>)'

    # 使用正则表达式的sub()方法进行查找和替换
    new_text = pattern.sub(replace, text)

    return new_text


_content_analysis_prompt = """
请你完成一个英文论文片段（latex格式）的语言表达和行文逻辑的检查任务，具体要求如下：
1. 我会提供给你一篇论文的一个section，需要你帮忙检查这个section的语言表达和行文逻辑
<1> 语言表达方面：检查内容是否有语法错误、口语化表达、不够好的表达等问题
<2> 行文逻辑方面：检查内容的行文逻辑是否合理，上下文衔接是否顺畅，是否存在逻辑不通顺等问题
2. 无论是整篇文章的行文逻辑，还是单个section的行文逻辑，都是使用JSON-encoded directed acyclic graph (DAG)的结构表示
3. 整篇文章的逻辑结构为：{overall_structure}
4. 这个section的逻辑结构为：{section_structure}
5. 这个section的内容为：{section_content}
6. 输出格式要求：
<1> 输出的内容应该是markdown格式，请将latex格式的section内容转成markdown格式，删去图像，表格等不包含长文本的内容。内容总共分"section内容和标注"，以及"行文逻辑点评"两个部分。
<2> "section内容和标注"：对于你认为语言表达问题的句子，请用@符号包裹，同时在后面插入一个括号，括号中注明该句子的语言表达问题是什么
<3> "行文逻辑点评"：在输出整个标注过的section后，在最后另起一段（这段用#符号包括），用1句话简单的评价一下该段落的行文逻辑，写明是否存在问题，如果存在，问题是什么。
注意，这里我们只关注文章语言表达方面的行文逻辑问题，不要关注文章讲述的内容是否正确。
7. 下面是一个供你参考输出格式的例子：
{content_analysis_example}
"""

content_analysis_prompt = """
Your task is to review a section of an academic paper (in LaTeX format) and check it for language proficiency and logical flow. Follow these steps:

I will provide you with a section (titled "{section_label}") of the paper. You need to check the section for:

Language: Look for grammatical errors, colloquial expressions, or any subpar phrases and suggest improvements.
Logical Flow: Ascertain whether the content follows a reasonable sequence, whether transitions between ideas are smooth, and pinpoint any incoherence in the logic.
Both the whole paper and individual sections' logical flow are depicted using JSON-encoded directed acyclic graph (DAG) structures.

The logical structure of the entire paper is: {paper_structure}

The logical structure of the specific section is: {section_structure}

The content of the section is: {section_content}

Formatting Requirements:

Present your output in Markdown format (You need to output in markdown format, not this latex text should be converted to markdown format). Convert LaTeX section content to Markdown, omitting images, tables, and other non-textual elements. Divide your output into "Section Content with Annotations" and "Logical Flow Commentary."
"Section Content with Annotations": Wrap any sentences with language issues in "@" symbols and immediately follow them with a parenthesis stating what the issue is.
"Logical Flow Commentary": After presenting the annotated section, add a separate part summarizing the logical flow of the paragraph. State whether it's logical or if any issues are present, and describe them briefly.
Please focus solely on the language-related logical flow issues for this task, without considering the factual accuracy of the content.

Here's an example for reference:

{content_analysis_example}

Ensure your response is clear and it enables gpt to understand and fulfill the task effectively.
"""

content_analysis_example = """
### Section Content with Annotations
Planning a trip to the moon differs in no essential respect from planning a trip to the beach. \
@You hve to decide what to take along, what to leave behind.@(There is a typographical error: "hve" should be "have.". It is generally more common to see "take with you" instead of "take along.") \
Should the thermos jug go? The child's rubber horse? The dill pickles? \
@These are the sometimes fateful decisions on which the success or failure of the whole outing turns.@\
(The first issue with this sentence is the subject-verb disagreement. "These" is plural, but "is" is singular. \
The correct verb form should be "are" to match the plural subject.The last part of the sentence, "on which the success or failure of the whole outing turns," \
is a bit awkward. A smoother way to write this might be: "These decisions can sometimes be crucial, determining the success or failure of the entire outing.")
Among the items they sent along, of course, was the little jointed flagpoles and the flag that could be stiffened to the breeze that did not blow. \
(It is traditional among explorers to plant the flag.) Yet the two men who stepped out on the surface of the moon were in a class by themselves and should \
have been equipped accordingly: they were of the new breed of men, those who had seen the earth whole.
### Logical Flow Commentary
The passage is mostly coherent and logically consistent, but the phrase "should have been equipped accordingly" is a bit abrupt without further explanation \
of what specific equipment would have been appropriate for their unique status. This leaves the reader with a question of what the author means by \
"equipped accordingly," which slightly disrupts the flow.
"""

_modify_scheme_design_prompt = """
请你完成一个英文论文（latex格式）的一个section（标题为{section_label}）的语言润色任务，具体的要求如下：
1. 我有一个大概的润色要求供你参考：{user_instruction}
2. 整篇论文的逻辑结构为：{overall_structure}
3. 这个section的逻辑结构为：{section_structure}
4. 这个section的内容为：{section_content}
请你参考我的润色要求和文章片段设计一个更加具体的润色方案，并分点给出。
注意：只关注语言表达方面，不要关注section讲述的具体内容是否合理和正确。
Please only provide suggestions for improvement, not the results of the improvement.
"""

modify_scheme_design_prompt = """
I need assistance with an English-language editing task for a specific section of a LaTeX-formatted academic paper. The details are as follows:

Editing instructions provided for reference: {user_instruction}
The logical flow of the entire paper: {overall_structure}
The logical flow within the specific section: {section_structure}
The content of the section needing language polishing: {section_content}
Please devise a language polishing plan based on these provided elements, and outline the plan in bullet points. Keep in mind that the focus should be solely on linguistic expression; there's no need to evaluate the actual subject matter for its validity or accuracy.
Please only provide suggestions for improvement, not the results of the improvement.
"""


def section_analysis(section_label, section_content, section_structure, overall_structure):
    if len(section_content) < 100:
        return section_content
    request = content_analysis_prompt.format(section_label=section_label, paper_structure=overall_structure, section_structure=section_structure,
                                             section_content=section_content,
                                             content_analysis_example=content_analysis_example)
    max_try = 3
    try_count = 0
    while try_count < max_try:
        try:
            reply = llm_request(request)
            reply = replace_at_sentences(reply)
            return reply
        except KeyboardInterrupt:
            return
        except:
            traceback.print_exc()
            try_count += 1


def section_analysis_async(dt_section, dt_section_structure, overall_structure):
    print("Analysis section asynchronously")

    def fun(section_pair, overall_structure):
        section_label, section_content, section_structure = section_pair
        return section_analysis(section_label, section_content, section_structure, overall_structure)

    ls_section_pair = []
    for section_label in dt_section.keys():
        ls_section_pair.append((section_label, dt_section[section_label], dt_section_structure[section_label]))
    ls_analysis_result = multiprocess(
        func=fun,
        paras=ls_section_pair,
        overall_structure=overall_structure,
        n_processes=min(len(ls_section_pair), get_cpu_count())
    )
    return ls_analysis_result


def modify_scheme_design(user_instruction, section_label, section_content, section_structure, overall_structure):
    request = modify_scheme_design_prompt.format(user_instruction=user_instruction, section_label=section_label, overall_structure=overall_structure,
                                                 section_structure=section_structure,
                                                 section_content=section_content)
    max_try = 3
    try_count = 0
    while try_count < max_try:
        try:
            reply = llm_request(request)
            reply = replace_at_sentences(reply)
            return reply
        except KeyboardInterrupt:
            return
        except:
            traceback.print_exc()
            try_count += 1


if __name__ == '__main__':
    paper_structure = {'nodes': [{'name': 'Title', 'type': 'title', 'parents': []}, {'name': 'Abstract', 'type': 'abstract', 'parents': ['Title']},
                                 {'name': 'Introduction', 'type': 'section', 'parents': ['Abstract']},
                                 {'name': 'Methodology', 'type': 'section', 'parents': ['Introduction']},
                                 {'name': 'Data Analysis', 'type': 'subsection', 'parents': ['Methodology']},
                                 {'name': 'Image Processing', 'type': 'subsection', 'parents': ['Methodology']},
                                 {'name': 'Gap Filling', 'type': 'subsection', 'parents': ['Methodology']},
                                 {'name': 'Experiments', 'type': 'section', 'parents': ['Methodology']},
                                 {'name': 'Experimental Approach', 'type': 'subsection', 'parents': ['Experiments']},
                                 {'name': 'Results', 'type': 'subsection', 'parents': ['Experiments']},
                                 {'name': 'Conclusions and Lessons Learned', 'type': 'section', 'parents': ['Results']},
                                 {'name': 'Acknowledgements', 'type': 'section', 'parents': ['Conclusions and Lessons Learned']}],
                       'edges': [{'from': 'Title', 'to': 'Abstract'}, {'from': 'Abstract', 'to': 'Introduction'},
                                 {'from': 'Introduction', 'to': 'Methodology'},
                                 {'from': 'Methodology', 'to': 'Data Analysis'}, {'from': 'Methodology', 'to': 'Image Processing'},
                                 {'from': 'Methodology', 'to': 'Gap Filling'}, {'from': 'Methodology', 'to': 'Experiments'},
                                 {'from': 'Experiments', 'to': 'Experimental Approach'}, {'from': 'Experiments', 'to': 'Results'},
                                 {'from': 'Results', 'to': 'Conclusions and Lessons Learned'},
                                 {'from': 'Conclusions and Lessons Learned', 'to': 'Acknowledgements'}]}

    section_structure = {'nodes': [
        {'name': 'Challenge Overview',
         'content': 'The Urban Life and Air Pollution task at MediaEval 2022 is introduced, which required participants to predict the air quality index (AQI) value at future intervals using a variety of data sources.',
         'parents': []}, {'name': 'Data Gaps Issue',
                          'content': 'The paper acknowledges the common issue of gaps in air quality datasets, which is particularly problematic in poorer or developing countries.',
                          'parents': ['Challenge Overview']}, {'name': 'Research Contribution',
                                                               'content': 'The paper outlines its contribution by describing the approach taken to address the large gaps in the air quality data encountered.',
                                                               'parents': ['Data Gaps Issue']}], 'edges': [
        {'from': 'Challenge Overview', 'to': 'Data Gaps Issue'}, {'from': 'Data Gaps Issue', 'to': 'Research Contribution'}]}

    section_content = '\\label{sec:intro}\n\n\\begin{comment}\n% Just in case we want some form of Intro\nAccording to the World Health Organisation (WHO), 91\\% of the world\'s population reside in conditions where WHO\'s air quality guidelines levels were not met \\cite{organizacion2021global}. This report on 2016 also showed that ambient (outdoor) air pollution in both cities and rural areas was estimated to cause 4.2 million premature deaths worldwide. The research concluded that policies and investments supporting cleaner transport, energy-efficient homes, power generation, industry and better municipal waste management would would be crucial to the reduction of outdoor air pollution. In a separate report, it is estimated that air pollution globally accounts for roughly seven million premature deaths a year \\cite{Gar21}, where it was again stated that the majority of those deaths are caused by outdoor air pollution with the rest generally attributed to poor air quality from indoor cooking. While the majority of these deaths occur in developing countries, with China and India accounting for roughly 50\\%, developed countries also have a problem with deaths resulting from air pollution.\nIn this research, the focus will mostly be on the modelling of concentrations in particulate matter - tiny particles in the air generated both by natural processes and human activity. These particles are generally \n12  categorised (in the public health domain) by their diameter; fine particles with diameter less than 2.5 $\\mu$m are referred to as "PM2.5" and coarse particles with diameter between 2.5 and 10 $\\mu$m are referred to as "PM10".\\\\\n\\end{comment}\n\nThe Urban Life and Air Pollution task at MediaEval 2022 required participants to predict the air quality index (AQI) value at +1, +5 and +7 days using an archive of air quality, weather and images from 16 CCTV cameras, one image taken every 60 seconds  \\cite{UA22}. Participating groups were required to download the data from online sources for local processing.\nGaps in air quality datasets are common with the problem exacerbated for data gathered in poorer or developing countries \\cite{PINDER2019116794, Falge2001, Hui2004, Moffat2007, Kim2020}. In this paper we describe how we addressed the very large gaps in data that we encountered in the data we downloaded.\n\n\n%\\section{Related Work}\n%\\label{sec:rr}\n%.. a snapshot of how our graph? technique has been used for solving other problems.\n\n% Mark: I restructured this to stop duplication of text'
    section_label = "Introduction"

    # print(section_analysis(section_label, section_content, section_structure, paper_structure))

    user_input = "Please help to make the language expression more beautiful"
    print(modify_scheme_design(user_input, section_label, section_content, section_structure, paper_structure))
