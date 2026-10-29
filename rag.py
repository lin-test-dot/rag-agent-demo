from openai import OpenAI
from config import API_KEY, BASE_URL, CHAT_MODEL

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 1. 定义工具清单（告诉大模型你有什么工具）
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": "当用户询问星海大学图书馆的规定、惩罚、人物信息时，调用此工具检索本地知识库。",
            "parameters": {
                "type": "object",#在 JSON 标准里，object 就对应 Python 的 dict（字典）
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "从用户问题中提取的搜索关键词，例如'螺蛳粉'、'王大妈'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def ask_model(query, context_text=""):
    # 把参考资料和问题拼起来
    prompt = f"""
    你是一个严谨的图书馆助手。请优先根据以下【参考资料】回答问题。
    如果资料为空，或者资料没有提及，你可以调用 search_library 工具去查找。
    如果工具也查不到，请直接回答“抱歉，资料库中没有相关信息”。

    【参考资料】：
    {context_text}

    【问题】：
    {query}
    """

    messages = [
        {"role": "system", "content": "你是一个智能图书馆客服。"},
        {"role": "user", "content": prompt}
    ]

    # 呼叫大模型，注意传入了 tools 参数
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=tools,  # 把工具菜单交给模型
        tool_choice="auto"  # 让模型自己决定用不用
    )

    return response.choices[0].message

def stream_answer(messages):
    response=client.chat.completions.create(
    model=CHAT_MODEL,
    messages=messages,
    stream=True
  )

    for chunk in response:
        delta=chunk.choices[0].delta.content
        if delta:
            yield delta