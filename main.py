from config import CHUNK_SIZE, CHUNK_OVERLAP
from chunker import split_text
from retriever import search
from rag import ask_model
import json

# 1. 读取文件并切块
with open("test.txt", 'r', encoding='utf-8') as file:
    content = file.read()
chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"✅ 资料加载完毕，共 {len(chunks)} 块。")

print("\n🤖 Agent 已上线！输入 '退出' 结束。")

while True:
    query = input("\n你: ").strip()
    if query == "退出":
        break
    if not query:
        continue

    # --- 第一轮对话：让大模型自己决定要不要用工具 ---
    message = ask_model(query)

    # 如果大模型决定调用工具 (tool_calls 不为空)
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        print(f"🔍 [Agent 思考] 决定调用工具！参数：{tool_call.function.arguments}")

        # 解析模型传过来的参数（把 JSON 字符串转成字典）
        args = json.loads(tool_call.function.arguments)
        search_query = args.get("query", query)

        # 执行你的本地代码：去向量库里查
        matched_chunks = search(search_query, chunks)

        # --- 第二轮对话：把查到的结果交给大模型 ---
        print("🤖 [Agent 思考] 检索完成，正在组织语言...")
        final_message = ask_model(query, "\n".join(matched_chunks))
        print(f"🤖: {final_message.content}")

    else:
        # 大模型觉得不需要用工具，直接回答
        print(f"🤖: {message.content}")