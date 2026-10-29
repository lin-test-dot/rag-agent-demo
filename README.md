# 📚 基于 RAG 与 Function Calling 的智能问答 Agent

这是一个基于通义千问大模型的智能体项目，支持本地知识库检索和工具调用。

## 🛠️ 技术栈
* **大模型**：通义千问 (qwen-plus)
* **向量库**：ChromaDB
* **前端**：Streamlit

## 🚀 怎么运行
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env` 文件
3. 运行：`streamlit run app.py`

## 🚧 踩坑记录
缺失 `tool_call_id` 导致 API 报错 400
*   **现象**：在第一轮工具执行完后，将结果作为 `role: tool` 发给大模型，接口返回 `400 Bad Request`。
*   **解决**：大模型对上下文有严格校验。必须将大模型返回的 `assistant message`（含 `tool_calls`）原封不动存入历史列表，并在回传工具结果时，附带对应的 `tool_call_id` 形成闭环，大模型才能识别。
网页刷新导致对话记忆丢失
*   **现象**：终端版记忆正常，但在 Streamlit 网页中，每次发送新消息页面都会重跑，导致大模型“失忆”。
*   **解决**：理解 Streamlit 的重新执行机制。引入 `st.session_state` 维护全局历史列表，并在渲染时过滤 `system` 与 `tool` 消息，只渲染 `user` 与 `assistant` 气泡。
