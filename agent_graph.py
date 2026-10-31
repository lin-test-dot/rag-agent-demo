# ==========================================
# 1. 导入模块（清理了所有垃圾导入）
# ==========================================
import operator
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# 导入你自己的配置和工具
from config import API_KEY, BASE_URL, CHAT_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from chunker import split_text
from retriever import search

# ==========================================
# 2. 定义黑板
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

# ==========================================
# 3. 定义工具
# ==========================================
@tool
def search_library(query: str) -> str:
    """当用户询问星海大学图书馆的规定、惩罚、人物信息时，调用此工具检索本地知识库。"""
    with open("test.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
    matched_chunks = search(query, chunks)
    return "\n".join(matched_chunks)

tools = [search_library]
# 👇 修复了拼写错误：tolls 改成了 tools
tools_by_name = {"search_library": search_library}

# ==========================================
# 4. 初始化大模型
# ==========================================
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=CHAT_MODEL,
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 5. 包装节点
# ==========================================
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_outputs = []
    for tool_call in last_message.tool_calls:
        # 👇 这里也同步改成了 tools_by_name
        tool_func = tools_by_name[tool_call["name"]]
        observation = tool_func.invoke(tool_call["args"])
        tool_outputs.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )
    return {"messages": tool_outputs}

# ==========================================
# 6. 组装图并编译
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()