import json
import os
import erniebot
import time  # 添加时间模块的导入

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 列出支持的模型
models = erniebot.Model.list()
print(models)

# 设置鉴权参数
erniebot.api_type = "aistudio"
erniebot.access_token = os.environ["BAIDU_API_KEY"]

def handle_rate_limit_and_timeout(func):
    """
    装饰器：处理 API 调用中的速率限制和超时异常，自动重试。
    """
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Executing function: {func.__name__}")  # 打印执行的函数名
                time.sleep(1)  # 使用 time.sleep() 实现延迟重试
                print(f"handle_rate_limit_and_timeout Error: {e}. Retrying...")
    return wrapper
@handle_rate_limit_and_timeout
def get_response_from_llm(
    msg,  # 用户输入的消息
    model,  # 指定使用的模型
    system_message,  # 系统消息，用于设定对话上下文
    print_debug=False,  # 是否打印调试信息
    msg_history=None,  # 对话的历史记录
    temperature=0.75,  # 生成的文本的多样性
):
    """
    从 LLM 获取响应。

    参数:
    msg (str): 用户输入的消息。
    model (str): 指定使用的模型。
    system_message (str): 系统消息，用于设定对话上下文。
    print_debug (bool): 是否打印调试信息。
    msg_history (list): 对话的历史记录。
    temperature (float): 生成的文本的多样性。

    返回:
    tuple: 生成的内容和更新后的消息历史记录。
    """
    if msg_history is None:
        msg_history = []  # 如果没有提供历史记录，则初始化为空列表

    new_msg_history = msg_history + [{"role": "user", "content": msg}]  # 将用户消息添加到历史记录中
    content, new_msg_history = call_llm_api(
        model, system_message, new_msg_history, temperature, n_responses=1
    )

    # 如果设置了打印调试信息
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)  # 打印调试分隔符
        for j, msg in enumerate(new_msg_history):  # 遍历打印消息历史记录
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)  # 打印生成的内容
        print("*" * 21 + " LLM END " + "*" * 21)  # 打印调试分隔符
        print()

    return content, new_msg_history  # 返回生成的内容和更新后的消息历史记录

@handle_rate_limit_and_timeout
def get_batch_responses_from_llm(
    msg,  # 用户输入的消息
    model,  # 指定使用的模型
    system_message,  # 系统消息，用于设定对话上下文
    print_debug=False,  # 是否打印调试信息
    msg_history=None,  # 对话的历史记录
    temperature=0.75,  # 生成的文本的多样性
    n_responses=1,  # 需要生成的响应数量
):
    """
    从 LLM 获取批量响应。

    参数:
    msg (str): 用户输入的消息。
    model (str): 指定使用的模型。
    system_message (str): 系统消息，用于设定对话上下文。
    print_debug (bool): 是否打印调试信息。
    msg_history (list): 对话的历史记录。
    temperature (float): 生成的文本的多样性。
    n_responses (int): 需要生成的响应数量。

    返回:
    tuple: 生成的内容和更新后的消息历史记录。
    """
    if msg_history is None:
        msg_history = []  # 如果没有提供历史记录，则初始化为空列表

    new_msg_history = msg_history + [{"role": "user", "content": msg}]  # 将用户消息添加到历史记录中
    content, new_msg_history = call_llm_api(
        model, system_message, new_msg_history, temperature, n_responses
    )

    # 如果设置了打印调试信息
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)  # 打印调试分隔符
        for j, msg in enumerate(new_msg_history[0]):  # 遍历打印第一条消息历史记录
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)  # 打印生成的内容
        print("*" * 21 + " LLM END " + "*" * 21)  # 打印调试分隔符
        print()

    return content, new_msg_history  # 返回生成的内容和更新后的消息历史记录

def call_llm_api(model, system_message, msg_history, temperature, n_responses):
    """
    调用 LLM API 获取响应。

    参数:
    model (str): 使用的模型名称。
    system_message (str): 系统消息。
    msg_history (list): 历史消息记录。
    temperature (float): 生成文本的多样性。
    n_responses (int): 请求生成的响应数量。

    返回:
    tuple: 生成的内容和更新后的消息历史记录。
    """
    response = erniebot.ChatCompletion.create(
        model=model,  # 使用的模型名称
        messages=[
            {"role": "user", "content": system_message},  # 系统消息
            *msg_history,  # 历史消息记录
        ],
        temperature=temperature,  # 生成文本的多样性
    )
    print(response)
    if n_responses == 1:
        content = response.choices[0].message.content  # 从响应中提取生成的文本内容
        new_msg_history = msg_history + [{"role": "assistant", "content": content}]  # 更新历史记录
        return content, new_msg_history
    else:
        content = [r.message.content for r in response.choices]  # 从响应中提取生成的文本内容
        new_msg_history = [
            msg_history + [{"role": "assistant", "content": c}] for c in content  # 将每个响应加入新的历史记录
        ]
        return content, new_msg_history

def extract_json_between_markers(llm_output):
    """
    从 LLM 输出中提取 JSON 数据。

    参数:
    llm_output (str): LLM 的输出。

    返回:
    dict: 解析后的 JSON 对象，如果解析失败则返回 None。
    """
    # 定义 JSON 开始和结束的标记
    json_start_marker = "```json"
    json_end_marker = "```"

    # 找到 JSON 字符串的开始和结束索引
    start_index = llm_output.find(json_start_marker)
    if (start_index != -1):
        start_index += len(json_start_marker)  # 将起始索引移动到标记之后的位置
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None  # 如果没有找到开始标记，则返回 None

    if end_index == -1:
        return None  # 如果没有找到结束标记，则返回 None

    # 提取 JSON 字符串
    json_string = llm_output[start_index:end_index].strip()  # 去除前后空格
    try:
        parsed_json = json.loads(json_string)  # 尝试解析 JSON 字符串
        return parsed_json  # 如果成功，返回解析后的 JSON 对象
    except json.JSONDecodeError:
        return None  # 如果解析失败（无效的 JSON 格式），返回 None