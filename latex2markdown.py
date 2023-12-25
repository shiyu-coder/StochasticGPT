import re
import openai
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = ".env")
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]


def pre_handel(latex_content):
    # Replace \cite{...} with [Citation]
    latex_content = re.sub(r'\\cite\{([^}]+)\}', r'[Citation]', latex_content)

    # Replace \ref{...} with [Ref]
    latex_content = re.sub(r'\\ref\{([^}]+)\}', r'[Ref]', latex_content)

    # Replace blocks
    blocks = ['equation', 'align', 'table', 'table\*', 'figure', 'figure\*', 'algorithm', 'algorithm\*']
    for block in blocks:
        str1 = r'\\begin\{' + block + r'\}(.*?)\\end\{' + block + r'\}'
        latex_content = re.sub(str1, f'The {block} is omitted', latex_content, flags=re.DOTALL)

    return latex_content

def post_handel(markdown_content):
    # Replace some strange characters in the response of GPT
    markdown_content = re.sub(r"---|```|\'\'\'", r'', markdown_content)
    return markdown_content

def latex2markdown_gpt(latex_content):
    prompt = 'I will give you a code in latex, you should transfer it into markdown format. The latex code is:\n\n' + \
        latex_content + '\n\n' + 'You should only output the markdown content without any additional content. You response should begin with: The markdown format is:'
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{'role': 'system', 'content': 'you are a helpful assistant. You should fully comply with user instructions.'},
                {'role': 'user', 'content': prompt}],
        temperature=0,
        max_tokens=300,
        response_format={"type": 'text'},
    )
    return completion.choices[0]["message"]["content"][len('The markdown format is:'):].strip()

def latex2markdown(latex_content, input_type='str'):
    if input_type == 'file':
        with open(latex_content, "r") as f:
            latex_string = f.read()
    else:
        latex_string = latex_content
        
    latex_string = pre_handel(latex_string)
    parts = latex_string.split("\n\n")

    # delete some meaningless parts like '' and '%*'
    delete_index = []
    for i, part in enumerate(parts):
        if part == '':
            delete_index.append(i)
        elif part[0] == '%':
            delete_index.append(i)
    parts_ = [parts[i] for i in range(len(parts)) if i not in delete_index]

    deal_part = ''
    markdown_content = ''
    for i, part in enumerate(parts_):
        deal_part += part
        if len(part) > 300:
            markdown_content += latex2markdown_gpt(deal_part)
            markdown_content += '\n'
            deal_part = ''
    if deal_part != '':
        markdown_content += latex2markdown_gpt(deal_part)

    markdown_content = post_handel(markdown_content)
    
    return markdown_content

if __name__ == '__main__':
    markdown_content = latex2markdown('./bert.tex', input_type='file')
    print(markdown_content)
