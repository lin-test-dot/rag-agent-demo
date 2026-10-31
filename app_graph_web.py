import streamlit as st
from agent_graph import app as agent_app
from langchain_core.messages import HumanMessage

# ==========================================
# 1. 网页全局设置
# ==========================================
st.set_page_config(page_title="AI 图书馆助手 (LangGraph版)", page_icon="🚀", layout="centered")
st.title("🚀 星海大学 AI 图书馆助手")
st.caption("基于 LangGraph 状态图架构的智能问答 Agent")

# ==========================================
# 2. 侧边栏设置
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统设置")
    st.write("当前版本：LangGraph 架构")

    if st.button("🗑️ 清空当前对话", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ==========================================
# 3. 初始化网页记忆云盘
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 4. 渲染历史聊天记录（极致精简版）
# ==========================================
for msg in st.session_state.chat_history:
    # 因为我们自己存的全是字典，直接取就行
    if msg["role"] in ["user", "assistant"] and msg["content"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ==========================================
# 5. 接收用户输入，驱动 LangGraph 流转
# ==========================================
if prompt := st.chat_input("请输入你的问题，例如：吃螺蛳粉怎么罚？"):

    # 5.1 渲染并保存用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 5.2 驱动 LangGraph 图运行
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent 正在思考并查库中..."):

            # 包装初始状态
            inputs = {"messages": [HumanMessage(content=prompt)]}
            final_answer = ""

            # 启动图！recursion_limit=5 防止死循环
            for event in agent_app.stream(inputs, {"recursion_limit": 5}):
                for node_name, node_output in event.items():
                    # 我们只关心大模型节点产生的最终文本回答
                    if node_name == "agent":
                        last_msg = node_output["messages"][-1]
                        # 如果大模型直接说话了（不是发 tool_calls）
                        if last_msg.content:
                            final_answer = last_msg.content

            # 5.3 把结果画在网页上
            st.markdown(final_answer)

            # 5.4 存入网页记忆
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})