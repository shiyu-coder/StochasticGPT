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


# Make the paper worse

_rewrite_worse_prompt = """
我现在有一个任务需要你完成，具体要求如下：
1. 我会提供给你一篇latex格式的学术论文：{paper_content}
2. 我感觉这篇论文在语言表达方面做的很好，现在我想将这篇论文修改的更差，具体包括以下几个方面：
<1> 错误：论文在英文表达上出现了很多的单词拼写错误，语法错误，时态错误（但是跟latex相关的代码不能出现错误）
<2> 表达问题：论文在英文表达上出现了过于口语化，歧义，模糊不清楚等问题
<3> 上下文衔接问题：论文在上下文衔接方面出现了问题，行文逻辑混乱
<4> 语言问题：论文在英文语言的使用上不够好，重复性表达多，使用的句式和单次都很平庸
3. 假设将这篇论文改的更差有三种不同的程度：有点差/比较差/非常差
请你按照{rewrite_degree}的程度，参考我上面列出的方面将这篇论文改的更差，只给出修改后的论文内容。
"""

rewrite_worse_prompt = """
I have a task for you that involves editing an academic paper written in LaTeX format. Below are the specific requirements:

I will provide you with the content of the paper in LaTeX format: 

"{paper_content}"

The language in the paper is currently well-expressed, but I want to intentionally deteriorate the quality of writing in several specific areas: a. Introduce errors such as misspellings, grammatical mistakes, and incorrect verb tenses. b. Make the language excessively colloquial, introduce ambiguity, and ensure that expressions are vague. c. Disrupt the logical flow and cohesion between sections of the paper, leading to disorganized content structure. d. Degrade the language by using repetitive expressions, mundane sentence structures, and unremarkable vocabulary.

Assume there are three levels of quality deterioration you can apply: slightly_poor, moderately_poor, and extremely_poor. Based on the degree of rewriting specified as '{rewrite_degree}', please revise the paper following the above guidelines to make it worse. Provide only the modified content of the paper.

Please meaningfully integrate these changes while keeping the LaTeX structure intact.
"""

root_path = f"../StochasticGPT_data/paper/2212.10273/DCU-AQ.tex"
save_path = "../StochasticGPT_data/paper/exp2"
paper_text = TextLoader(root_path).load()[0].page_content

for rewrite_degree in ["slightly_poor", "moderately_poor", "extremely_poor"]:
    request = rewrite_worse_prompt.format(paper_content=paper_text, rewrite_degree=rewrite_degree)
    result = llm_request(request)
    print(result)
    print('------------------------------------------------------------------------')
    with open(f"{save_path}/{rewrite_degree}.tex", 'w') as f:
        f.write(result)























