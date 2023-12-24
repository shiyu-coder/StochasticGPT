from datetime import datetime, timedelta
from joblib import Parallel, delayed
import logging
import time
import traceback
from hashlib import md5
import json
import numpy as np
import openai
import re

import os
import tiktoken
from langchain.embeddings import OpenAIEmbeddings


openai_api_base = "http://27.102.66.157:8000/v1"
with open("../openai_key", "r") as f:
    openai_api_key = f.read().strip()
openai.api_base = openai_api_base
openai.api_key = openai_api_key


def get_date(date0, days):
    stamp1 = datetime.strptime(date0, "%Y-%m-%d") + timedelta(days=days)
    date1 = stamp1.strftime("%Y-%m-%d")
    return date1


def hashcode(plainText):
    """
    将字符串转换为hash值
    """
    plainTextBytes = plainText.encode("utf-8")
    encryptor = md5()
    encryptor.update(plainTextBytes)
    hashCode = encryptor.hexdigest()
    return hashCode


def info_to_text(info):
    """
    将信息转换为可读的文本
    """
    if isinstance(info, dict):
        text = "\n".join([f"{str(key)}: {str(value)}" for key, value in info.items() if key not in ["embedding", "hash"]])
    elif isinstance(info, list):
        text = "\n".join([str(item) for item in info])
    else:
        text = str(info)
        # raise AssertionError(f"不支持的数据类型 {type(info)}")

    return text


def dict_to_tuple(d: dict):
    return tuple(sorted(d.items()))


def load_json(json_path):
    with open(json_path, "r") as f:
        json_obj = json.load(f)
    return json_obj


def save_json(json_obj, save_path):
    with open(save_path, "w") as f:
        json.dump(json_obj, f, indent=4)
    return


def split_list(items, split_num=None, division=None):
    """
    将list分割成split_num份，或者每份division个元素
    """
    if division is not None:
        return [items[i : i + division] for i in range(0, len(items), division)]

    elif split_num is not None:
        division = len(items) / float(split_num)
        return [items[int(round(division * i)) : int(round(division * (i + 1)))] for i in range(split_num)]
    else:
        raise ValueError("split_num和division必须有一个不为None")
    return


def list_paths(folder_dir: str, sort: str = "time", reverse_order: bool = False) -> list:
    """
    列出指定文件夹下的所有文件，并按指定方式排序。

    参数:
        folder_path (str): 要检查的文件夹的路径。
        sort (str): 排序方式，可以是'time'或'name'。
        reverse_order (bool): 如果为True，将按照从新到旧或从Z到A的顺序返回文件；否则，按照从旧到新或从A到Z的顺序返回。

    返回:
        list: 按指定方式排序的文件名列表。
    """
    if not os.path.exists(folder_dir):
        return []

    # 获取文件夹下的所有文件名
    all_files = [f for f in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, f))]

    # 根据指定的排序方式进行排序
    if sort == "time":
        sorted_files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(folder_dir, x)), reverse=reverse_order)
    elif sort == "name":
        sorted_files = sorted(all_files, reverse=reverse_order)
    else:
        raise ValueError(f"Invalid sort type: {sort}. Valid options are 'time' or 'name'.")

    sorted_paths = [os.path.join(folder_dir, f) for f in sorted_files]

    return sorted_paths


def parse_numbers(text):
    try:
        matches = re.findall(r"#(\d+)\$", text)
        numbers_list = [float(match) for match in matches]
        return numbers_list
    except BaseException as error:
        raise AssertionError(f"数字解析失败 {error}")


def parse_json(text, key=None):
    """
    直接利用json.loads解析字符串
    """
    try:
        json_obj = json.loads(text)
        if key is not None:
            json_obj = json_obj[key]
        return json_obj

    except:
        try:
            pattern = re.compile(r"```json\n([\s\S]*?)\n```")
            match = pattern.search(text)
            json_text = match.group(1)
            json_obj = json.loads(json_text)
            if key is not None:
                json_obj = json_obj[key]
            return json_obj

        except BaseException as error:
            raise AssertionError(f"JSON解析失败 {error}\n{text}")


def parse_func(fun_str):
    matchObj = re.search(r"def (.*)\(.*\)", fun_str, re.M | re.I)
    factor_name = matchObj.group(1)
    return factor_name


def parse_code(text):
    """
    解析代码块，返回代码字符串
    """
    try:
        if "NO INDICATOR FUNCTION" in text:
            return None
        matchObj = re.search(r"`{3}([Pp]ython)?\n?([\s\S]*)\n?`{3}", text, re.M | re.I)
        code = matchObj.group(2)
        # functions = re.findall(r'def .*?return .*?\n', code_text, re.DOTALL)

        return code
    except BaseException as error:
        logging.error(f"解析失败\n {text}")
        raise AssertionError(f"代码解析失败 {error}\n{text}")


def parse_args(text):
    try:
        # matches = re.findall(r"\(([\d.,\s]+)\)", text)
        # args_list = [tuple(map(eval, m.split(','))) for m in matches]
        # matches = re.findall(r"\(([\d.,\s]+)\)", text)
        # args_list = [tuple(map(eval, filter(None, m.split(",")))) for m in matches]

        args_dict_list = parse_json(text, key="arguments")
        args_list = [dict_to_tuple(args_dict) for args_dict in args_dict_list]

        # 检查list，保证tuple的长度一致
        if len(set([len(args) for args in args_list])) != 1:
            raise AssertionError(f"参数长度不一致: {args_list}")
        # 去除list中的重复tuple
        args_list = sorted(list(set(args_list)))
        return args_list

    except BaseException as error:
        raise AssertionError(f"参数解析失败 {error}")


def parse_error(traceback_text, code_text):
    """
    解析错误信息，定位错误行，返回可读的错误信息
    """

    line_msg = ""
    for line in traceback_text.splitlines():
        if """File "<string>", line""" in line:
            line_msg = line
            break

    error_text_list = ["Error: ", traceback_text.splitlines()[0], line_msg]
    line_match = re.search(r"""File "<string>", line (\d+)""", line_msg)
    if line_match:
        line_number = line_match.group(1)
        code_line = code_text.splitlines()[int(line_number) - 1]
        error_text_list.append(code_line)

    error_text_list.append(traceback_text.splitlines()[-1])
    error_text = "\n".join(error_text_list)

    return error_text


def calc_tokens_num_from_text(text):
    """
    统计文本中的token数量
    """
    num_tokens = len(tiktoken.encoding_for_model("gpt-4").encode(text))
    return num_tokens


class GPT:
    def __init__(self):
        self.fee_path = "./record/fee.json"
        self.max_try_num = 3
        self.token_fee_dict = {
            "gpt-4": (0.03, 0.06),
            "gpt-4-1106-preview": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0015, 0.002),
            "gpt-3.5-turbo-1106": (0.001, 0.002),
            "text-embedding-ada-002": (0.0001, 0.0),
        }
        self.llm_system_content = "Assuming you are an expert in English paper polishing."
        self.reset_conversation()

    def _record_fee(self, prompt_tokens, completion_tokens, llm_model):
        """
        记录大模型费用
        """

        request_fee_per_token = self.token_fee_dict[llm_model][0]
        reply_fee_per_token = self.token_fee_dict[llm_model][1]

        prompt_fee = prompt_tokens * request_fee_per_token / 1000
        completion_fee = completion_tokens * reply_fee_per_token / 1000

        fee = round(prompt_fee + completion_fee, 3)
        try:
            info = load_json(self.fee_path)
        except:
            info = {"fee": 0.0}
        info["fee"] += fee
        info["fee"] = round(info["fee"], 3)
        save_json(info, self.fee_path)
        return

    def _request(self, messages, llm_model, temperature, response_type, max_tokens=None):
        """
        使用大模型生成回复
        """

        for _ in range(self.max_try_num):
            try:
                completion = openai.ChatCompletion.create(
                    model=llm_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": response_type},
                )
                reply = completion.choices[0]["message"]["content"]
                token_num = completion.usage.to_dict()
                self._record_fee(token_num["prompt_tokens"], token_num["completion_tokens"], llm_model)
                return reply

            except openai.error.APIConnectionError:
                time.sleep(3)
            except openai.error.RateLimitError:
                time.sleep(60)
            except BaseException as error:
                logging.error(f"大模型调用失败: \n{error}")
                time.sleep(10)
        return None

    def embedding(self, text):
        """
        使用大模型计算文本的嵌入
        """
        text = info_to_text(text)
        embedding_model = "text-embedding-ada-002"
        for _ in range(self.max_try_num):
            try:
                reply = openai.Embedding.create(input=text, model=embedding_model)
                tokens_num = reply["usage"]["total_tokens"]
                embedding = np.array(reply["data"][0]["embedding"])
                self._record_fee(tokens_num, 0, llm_model=embedding_model)
                return embedding
            except BaseException as error:
                time.sleep(5)
        logging.warning(f"生成embedding失败")

        # 生成全是nan的embedding
        embedding = np.full(1536, np.nan)
        # embedding = np.random.randn(1536)
        # embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def fee(self):
        """
        输出当前累计费用
        """
        try:
            info = load_json(self.fee_path)
        except:
            info = {"fee": 0.0}
        return info["fee"]

    def reset_conversation(self, conversation=None):
        """
        重置对话历史
        """
        if conversation is None:
            conversation = [{"role": "system", "content": self.llm_system_content}]
        self.conversation = conversation
        return

    def replace_content(self, old_content, new_content):
        """
        将对话历史中所有包含old_content的内容替换为new_content
        """
        for item in self.conversation:
            if old_content in item["content"]:
                item["content"] = item["content"].replace(old_content, new_content)
        return

    def conversation_tokens_num(self):
        """
        计算对话历史中的token数量
        """
        text = "\n".join([item["content"] for item in self.conversation])
        num_tokens = calc_tokens_num_from_text(text)
        return num_tokens

    def chat(self, request, llm_model='gpt-4-1106-preview', temperature=0.4, response_type="text", max_tokens=None, parse_func=None):
        """
        与大模型对话
        若回复成功，则返回解析后的reply，并将request和reply添加到对话历史中
        若回复失败，则返回None，不将request和reply添加到对话历史中
        """
        # logging.info(f"开始请求 \n{request}")

        request = info_to_text(request)
        messages = self.conversation + [{"role": "user", "content": request}]
        reply = self._request(
            messages, llm_model=llm_model, temperature=temperature, response_type=response_type, max_tokens=max_tokens
        )

        # 若reply为None，则表示请求失败
        if reply is None:
            return None

        if parse_func is not None:
            parse_reply = parse_func(reply)
        else:
            parse_reply = reply

        # 若对话完成，则将记录添加到对话历史中，注意这里记录的是request和原始reply
        self.conversation.append({"role": "user", "content": request})
        self.conversation.append({"role": "assistant", "content": reply})
        # logging.info(f"请求成功 \n{reply}")

        # 返回解析后的reply
        return parse_reply

    def robust_chat(self, request, llm_model, temperatures: list, response_type="text", max_tokens=None, parse_func=None):
        """
        鲁棒请求，给定一组温度，当请求失败时，自动改变温度，重新请求
        """

        for temperature in temperatures:
            try:
                reply = self.chat(
                    request,
                    llm_model=llm_model,
                    temperature=temperature,
                    response_type=response_type,
                    max_tokens=max_tokens,
                    parse_func=parse_func,
                )
                return reply
            except AssertionError as error:
                logging.error(f"解析失败：{error}")
            except:
                logging.error(f"chat报错：\n{traceback.format_exc()}")
                time.sleep(3)
        return None


def llm_request(request, system_content=None, temperature=0.0, max_tokens=None, llm_model="gpt-4-1106-preview", response_type="text"):
    messages = [{'role': 'user', 'content': request}]
    if not system_content:
        system_content = "Assuming you are an expert in English paper polishing."
    messages = [{'role': 'system', 'content': system_content}] + messages
    max_try = 10
    cur_try = 0
    while cur_try < max_try:
        try:
            completion = openai.ChatCompletion.create(
                model=llm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                response_format={"type": response_type}
            )
            reply = completion.choices[0]['message']['content']
            return reply
        except KeyboardInterrupt:
            return
        except openai.error.APIConnectionError:
            print("APIConnectionError")
            time.sleep(3)
            cur_try += 1
        except openai.error.RateLimitError:
            traceback.print_exc()
            time.sleep(60)
            cur_try += 1
        except:
            traceback.print_exc()
            time.sleep(3)
            cur_try += 1
    return None


if __name__ == '__main__':
    model = GPT()
    print(model.chat("给我讲个笑话"))






