import copy
import os
import re

import json
import traceback

import joblib

from llm_api import GPT, parse_json, llm_request
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import LatexTextSplitter
from util import multiprocess, get_cpu_count

root_cache_path = "../StochasticGPT_data/cache"


def load_from_cache(file_name):
    if f"{file_name}.pkl" in os.listdir(root_cache_path):
        ls_cache = joblib.load(f"{root_cache_path}/{file_name}.pkl")
        paper = Paper()
        paper.load_cache(ls_cache)
        paper.file_name = file_name
        return paper
    return None


def save_cache(paper):
    ls_cache = paper.get_cache()
    joblib.dump(ls_cache, f"{root_cache_path}/{paper.file_name}.pkl")


def replace_section_content(latex_content, section_title, new_content):
    # 转义为正则表达式的特殊字符
    escaped_title = re.escape(section_title)

    # 构建查找section或subsection的正则表达式
    section_regex = re.compile(
        r"(\\section\{" + escaped_title + r"\}|\\subsection\{" + escaped_title + r"\})"
                                                                                 ".*?"
                                                                                 r"(?=\\section|\\subsection|$)",
        re.DOTALL
    )

    # 在文档中查找对应的section或subsection
    match = section_regex.search(latex_content)
    if match:
        start_index = match.start()
        end_index = match.end()

        # 替换section内容
        replaced_content = (
                latex_content[:start_index] +
                match.group(1) + "\n" + new_content + "\n" +
                latex_content[end_index:]
        )

        return replaced_content
    else:
        # 没有找到对应的section或subsection
        print("Section or Subsection not found.")
        return latex_content


def get_polishing_paper(paper_content, dt_polishing_result):
    polishing_paper = copy.deepcopy(paper_content)
    for section_label in dt_polishing_result.keys():
        section_polishing_result = dt_polishing_result[section_label]
        if len(section_polishing_result) > 0:
            polishing_paper = replace_section_content(polishing_paper, section_label, section_polishing_result)
    return polishing_paper


class Paper:

    def __init__(self):
        self.file_name = None
        self.title = None
        self.paper_content = None
        self.overall_structure = None
        self.dt_section_structure = None
        self.dt_section_content = None
        self.dt_analysis_result = None
        self.paper_score = None
        self.dt_score = None
        self.dt_polishing_result = None

    def initial_polishing_result(self):
        self.dt_polishing_result = {}
        for section_label in self.dt_section_content.keys():
            self.dt_polishing_result[section_label] = ""

    def get_cache(self):
        return [self.title, self.paper_content, self.overall_structure, self.dt_section_structure, self.dt_section_content,
                self.dt_analysis_result, self.paper_score, self.dt_score, self.dt_polishing_result]

    def load_cache(self, ls_cache):
        self.title, self.paper_content, self.overall_structure, self.dt_section_structure, self.dt_section_content, self.dt_analysis_result, self.paper_score, self.dt_score, self.dt_polishing_result = ls_cache




if __name__ == '__main__':
    file_name = 'DCU-AQ.tex'
    # ls_cache = joblib.load(f"{root_cache_path}/{file_name}")
    paper = load_from_cache(file_name)
    section_name = 'Data Analysis'
    print(paper.dt_section_content[section_name])
    print(paper.dt_section_structure[section_name])

    # title = "Managing Large Dataset Gaps in Urban Air Quality Prediction: DCU-Insight-AQ at MediaEval 2022"
    # dt_polishing_result = {}
    # for section_label in ls_cache[4].keys():
    #     dt_polishing_result[section_label] = ""
    # joblib.dump([*ls_cache, dt_polishing_result], f"{root_cache_path}/{file_name}")













