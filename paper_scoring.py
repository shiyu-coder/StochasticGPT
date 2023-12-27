import re
import json
import traceback

from langchain.document_loaders import TextLoader

from llm_api import GPT, parse_json, llm_request
from util import multiprocess, get_cpu_count


_paper_scoring_prompt = """
我有一篇latex格式的学术论文，请你帮忙对这篇论文的语言表达层面进行打分，具体要求如下：
1. 请分析这篇论文的语言表达、上下文逻辑衔接，结构布局等方面，不要关注论文讲述的具体内容的正确合理与否
2. 请从以下几个方面分别给出你的打分（0-10分）
<1> Consistency: Also known as the principle of integrity, totality, or unity, consistency refers to the idea that when conceiving or writing an English academic paper, everything from a single sentence to the entire piece should closely revolve around the main theme of the paper, maintaining semantic and structural integrity and unity.
<2> Coherence: The principle of coherence refers to creating a natural, smooth, and logical connection between words, sentences, and paragraphs through a reasonable layout and appropriate word choice. Coherence is mainly reflected in the organization of the paper's discourse and sensible transitions, to ensure the paper is logical and does not give the reader a sense of confusion.
<3> Conciseness: Conciseness in English academic writing means there are no superfluous words or sentences; word choice and phrasing should be succinct and forceful.
<4> Substantiveness: The principle of substantiveness requires enriching the content and readability of an English paper by using a variety of words and sentence structures that are diverse and dynamic.

论文内容：
{paper_content}

请用JSON格式返回你的打分结果，key是Consistency/Coherence/Conciseness/Substantiveness，value是对应的分数。
"""

paper_scoring_prompt = """
I have an academic paper in LaTeX format, and I need an evaluation of its linguistic expression. Please focus on the language use, rather than the accuracy or validity of the content itself. Consider the following aspects and score each on a scale from 0 to 10, where 0 is the worst and 10 is the best:

Consistency: Evaluate if the paper maintains a clear and unified theme throughout, with all components—from individual sentences to full sections—contributing to the overall argument or purpose of the work.

Coherence: Assess the logical flow and smooth connections between words, sentences, and paragraphs. Does the paper facilitate an easy and logical read, without leaving the reader confused?

Conciseness: Determine if the paper is free from unnecessary words or sentences, with precise and effective use of language.

Substantiveness: Consider the richness and readability of the paper's content, taking into account the diversity and vitality of the vocabulary and sentence structure.

The content of the paper is as follows: {paper_content}

Please provide your ratings in JSON format, with keys for "Consistency", "Coherence", "Conciseness" and "Substantiveness" and their corresponding scores as the values.
"""


def paper_scoring(paper_content):
    print("Paper scoring")
    request = paper_scoring_prompt.format(paper_content=paper_content)
    max_try = 3
    try_count = 0
    while try_count < max_try:
        try:
            reply = llm_request(request)
            dt_score = parse_json(reply)
            return dt_score
        except KeyboardInterrupt:
            return
        except:
            traceback.print_exc()
            try_count += 1


if __name__ == '__main__':
    root_path = f"../StochasticGPT_data/paper/2212.10273/DCU-AQ.tex"
    paper_text = TextLoader(root_path).load()[0].page_content
    result = paper_scoring(paper_text)
    print(result)






