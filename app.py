import streamlit as st
from config import CHUNK_SIZE, CHUNK_OVERLAP
from chunker import split_text
from retriever import search, get_embedding, get_collection
from rag import ask_model, stream_answer
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

    user_temp = st.slider("模型发散度 (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    st.divider()

    # 清空对话记录的按钮
    if st.button("🗑️ 清空当前对话", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "system",
             "content": "你是一个严谨的图书馆助手。请优先根据提供的资料回答。如果资料没有，请调用工具查询。"}
        ]
        st.rerun()

    st.divider()
    st.subheader("📁 知识库管理")

    uploaded_file = st.file_uploader("上传 TXT 文档", type=["txt"])

    if uploaded_file is not None:
        if st.button("开始解析入库"):
            with st.spinner("正在解析并存入数据库..."):
                text_content = uploaded_file.read().decode("utf-8")
                new_chunks = split_text(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
                st.success(f"解析成功，共切分为 {len(new_chunks)} 块")

                collection = get_collection()
                current_count = collection.count()

                # 👇 修复了 ID 冲突的 Bug（加了 + i）
                for i, chunk in enumerate(new_chunks):
                    emb = get_embedding(chunk)
                    if emb:
                        collection.add(
                            embeddings=[emb],
                            documents=[chunk],
                            ids=[str(current_count + i)]
                        )
                st.success(f"🎉 入库成功！当前总共有 {collection.count()} 条数据")
                st.rerun()  # 刷新网页


# ==========================================
# 3. 加载知识库（带缓存）
# ==========================================
@st.cache_resource
def load_knowledge_base():
    with open("test.txt", 'r', encoding='utf-8') as file:
        content = file.read()
    return split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)


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
for msg in st.session_state.chat_history:
    role = msg["role"] if isinstance(msg, dict) else msg.role
    content = msg["content"] if isinstance(msg, dict) else msg.content

    if role in ["user", "assistant"] and content:
        with st.chat_message(role):
            st.markdown(content)

# ==========================================
# 6. 接收用户输入并处理 (Agent 核心逻辑)
# ==========================================
if prompt := st.chat_input("请输入你的问题，例如：吃螺蛳粉怎么罚？"):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🤖 正在分析意图..."):
            message = ask_model(st.session_state.chat_history)

        if message.tool_calls:
            tool_call = message.tool_calls[0]

            # 👇 极其重要的修复：把对象转成纯字典存入历史，防止序列化报错！
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })

            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", prompt)
            matched_chunks = search(search_query, chunks)

            st.session_state.chat_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "\n".join(matched_chunks)
            })

            reply = st.write_stream(stream_answer(st.session_state.chat_history))
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        else:
            reply = st.write_stream(stream_answer(st.session_state.chat_history))
            st.session_state.chat_history.append({"role": "assistant", "content": reply})