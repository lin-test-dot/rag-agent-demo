import streamlit as st
from openai import embeddings
from streamlit import divider, spinner

from config import CHUNK_SIZE, CHUNK_OVERLAP
from chunker import split_text
from main import content
from retriever import search,get_embedding,get_collection
from rag import ask_model,stream_answer
import json

# ==========================================
# 1. 网页全局设置
# ==========================================
st.set_page_config(page_title="AI 图书馆助手", page_icon="📚", layout="centered")
st.title("📚 星海大学 AI 图书馆助手")
st.caption("基于 RAG 与 Function Calling 智能体架构")

# ==========================================
# 2. 侧边栏设置（控制面板）
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统设置")
    st.write("在这里调整参数和清空对话。")

    # 演示：一个温度滑动条（目前用来展示，没有真正传给 rag.py，因为咱们的 ask_model 还没写接收逻辑）
    user_temp = st.slider("模型发散度 (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    st.divider()  # 画一条分割线

    # 清空对话记录的按钮
    if st.button("🗑️ 清空当前对话", use_container_width=True):
        # 重置历史记录，只保留系统人设
        st.session_state.chat_history = [
            {"role": "system",
             "content": "你是一个严谨的图书馆助手。请优先根据提供的资料回答。如果资料没有，请调用工具查询。"}
        ]
        st.rerun()  # 强制刷新网页，让主界面立刻清空

    st,divider()
    st.subheader("知识库管理")#小标题
    uploaded_file = st.file_uploader("上传 TXT 文档", type=["txt"])
    if uploaded_file  is not None:
        if st.button("开始解析入库"):
            with st.spinner("正在解析....."):
                text_content=uploaded_file.read().decode("utf-8")
                new_chunks=split_text(text_content,CHUNK_SIZE,CHUNK_OVERLAP)
                st.success("解析成功，共切为"+str(len(new_chunks))+"块")
                collection=get_collection()
                current_count=collection.count()
                for i,chunk in enumerate(new_chunks):
                    emb=get_embedding(chunk)
                    if emb:
                        collection.add(
                            embeddings=[emb],
                            documents=[chunk],
                            ids=[str(current_count)]
                        )
                st.success(f"解析成功,当前总共有{collection.count()}条数据")





# ==========================================
# 3. 加载知识库（带缓存）
# ==========================================
@st.cache_resource
def load_knowledge_base():
    with open("test.txt", 'r', encoding='utf-8') as file:
        content = file.read()
    chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


chunks = load_knowledge_base()

# ==========================================
# 4. 初始化网页记忆云盘
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "你是一个严谨的图书馆助手。请优先根据提供的资料回答。如果资料没有，请调用工具查询。"}
    ]

# ==========================================
# 5. 渲染历史聊天记录

# ==========================================
# 遍历云盘里的记录，画出气泡
for msg in st.session_state.chat_history:
    # 兼容处理：如果是字典，就用 [] 取；如果是对象（大模型返回的），就用 . 取
    role = msg["role"] if isinstance(msg, dict) else msg.role
    content = msg["content"] if isinstance(msg, dict) else msg.content

    # 过滤掉 system 和 tool 的底层通信，只展示用户和 AI 的对话
    if role in ["user", "assistant"]:
        # 如果 content 是空的（比如调工具那轮没说话），就跳过不画
        if content:
            with st.chat_message(role):
                 st.markdown(content)

# ==========================================
# 6. 接收用户输入并处理 (Agent 核心逻辑)
# ==========================================
if prompt := st.chat_input("请输入你的问题，例如：吃螺蛳粉怎么罚？"):

    # 6.1 把用户的问题画出来
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6.2 把用户问题存入记忆
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 6.3 准备 AI 气泡
    with st.chat_message("assistant"):

        # --- 第一轮对话：让大模型判断要不要调工具（这里会等一小会儿）---
        with st.spinner("🤖 正在分析意图..."):
            message = ask_model(st.session_state.chat_history)

        if message.tool_calls:
            tool_call = message.tool_calls[0]

            # 把大模型的指令对象存入记忆
            st.session_state.chat_history.append(message)

            # 解析参数并执行本地查库
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", prompt)
            matched_chunks = search(search_query, chunks)

            # 把工具查到的结果存入记忆（带上取件码 ID）
            st.session_state.chat_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "\n".join(matched_chunks)
            })

            # --- 第二轮对话：流式输出（打字机）---
            reply = st.write_stream(stream_answer(st.session_state.chat_history))
            st.session_state.chat_history.append({"role": "assistant", "content": reply})


        else:
            # --- 大模型觉得不需要工具，直接流式回答 ---
            # 也是用流式输出，保证体验一致
            reply = st.write_stream(stream_answer(st.session_state.chat_history))
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # 注意：这里绝对不能再写 st.markdown(reply) 了！因为上面已经画完了。