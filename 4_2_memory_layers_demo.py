"""
========================================================
LangChain Phase 4-2
短期记忆 vs 长期记忆 vs 工作记忆 教材级 Demo
========================================================

📚 学习目标：
1. 理解三种记忆层次及其适用场景
2. 短期记忆：会话内 InMemoryChatMessageHistory / LangGraph Checkpointer
3. 长期记忆：VectorStore 存储用户偏好，RAG 检索注入
4. 工作记忆：LangGraph GraphState 保存单次推理中间态

🔗 记忆分层架构：
┌─────────────────────────────────────────────────────────────┐
│  短期记忆（会话内）                                           │
│    InMemoryChatMessageHistory / Checkpointer + thread_id      │
│    → 记住「刚才说了什么」                                     │
├─────────────────────────────────────────────────────────────┤
│  长期记忆（跨会话）                                           │
│    VectorStore 存用户偏好/历史结论 → RAG 检索                  │
│    → 记住「用户是谁、偏好什么」                               │
├─────────────────────────────────────────────────────────────┤
│  工作记忆（单次推理）                                         │
│    LangGraph State 字段（如 retrieved_docs, tool_results）    │
│    → 记住「这一步的中间结果」                                 │
└─────────────────────────────────────────────────────────────┘
"""
import os
from numbers import Number
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from model_provider import check_api_key, get_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()
_user_profile_store: dict[str, list[Document]] = {}

def get_embeddings():
    from langchain_community.embeddings import ZhipuAIEmbeddings
    return ZhipuAIEmbeddings(api_key=get_api_key())

def demo_1_short_term_session():
    """同一会话 ID 下，历史消息被保留并在后续轮次使用"""
    print('\n' + '=' * 65)
    print('Demo 1: 短期记忆 — InMemoryChatMessageHistory')
    print('=' * 65)
    sessions: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in sessions:
            sessions[session_id] = InMemoryChatMessageHistory()
        return sessions[session_id]
    hist_a = get_history('user_A')
    hist_a.add_user_message('我喜欢喝咖啡')
    hist_a.add_ai_message('好的，已记住您喜欢咖啡。')
    hist_a.add_user_message('我刚才说我喜欢什么？')
    print(f'user_A 历史消息数: {len(hist_a.messages)}')
    print('最近一轮用户问: 我刚才说我喜欢什么？')
    model = get_model()
    response = model.invoke([*hist_a.messages])
    print(f'🤖 回答: {response.content}')
    hist_b = get_history('user_B')
    print(f'\nuser_B 历史消息数: {len(hist_b.messages)}（独立会话，不受 user_A 影响）')

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def demo_2_checkpointer():
    """
    LangGraph Checkpointer + thread_id 是 Agent 场景的短期记忆标准方案。
    你在 langgraph_demo_examples.py 中已学过 InMemorySaver。
    """
    print('\n' + '=' * 65)
    print('Demo 2: 短期记忆 — LangGraph Checkpointer')
    print('=' * 65)

    def chat_node(state: ChatState):
        response = get_model().invoke(state['messages'])
        return {'messages': [response]}
    graph = StateGraph(ChatState)
    graph.add_node('chat', chat_node)
    graph.add_edge(START, 'chat')
    graph.add_edge('chat', END)
    checkpointer = InMemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    config = {'configurable': {'thread_id': 'demo-thread-001'}}
    app.invoke({'messages': [HumanMessage(content='记住：我的宠物叫小白')]}, config)
    result = app.invoke({'messages': [HumanMessage(content='我的宠物叫什么？')]}, config)
    print(f"🤖 回答: {result['messages'][-1].content}")
    print('💡 同一 thread_id 下，Checkpointer 自动保存/恢复 messages 状态')

def save_user_profile(user_id: str, facts: list[str]):
    """将用户事实写入「长期记忆库」（Demo 用 dict + 简单检索）"""
    docs = [Document(page_content=f, metadata={'user_id': user_id}) for f in facts]
    _user_profile_store[user_id] = docs

def retrieve_user_profile(user_id: str, query: str, k: int=2) -> str:
    """从长期记忆库检索与用户问题相关的偏好/事实"""
    docs = _user_profile_store.get(user_id, [])
    if not docs:
        return '（无长期记忆）'
    hits = [d.page_content for d in docs if any((w in d.page_content for w in query.split()))]
    if not hits:
        hits = [d.page_content for d in docs[:k]]
    return '\n'.join((f'- {h}' for h in hits[:k]))

def demo_3_long_term_vector():
    """跨会话长期记忆：即使短期 history 清空，VectorStore 中的偏好仍在"""
    print('\n' + '=' * 65)
    print('Demo 3: 长期记忆 — VectorStore 用户偏好')
    print('=' * 65)
    user_id = 'user_001'
    save_user_profile(user_id, ['用户偏好：回答时使用中文', '用户职业：后端工程师，熟悉 Python 和 Go', '用户禁忌：不要推荐 Java 框架'])
    query = '帮我推荐技术栈'
    profile_context = retrieve_user_profile(user_id, query)
    print(f'检索到的用户画像:\n{profile_context}')
    prompt = f'根据以下用户长期偏好回答问题：\n{profile_context}\n\n用户问题：{query}'
    response = get_model().invoke([HumanMessage(content=prompt)])
    print(f'\n🤖 结合长期记忆的回答: {response.content[:200]}...')
    print('\n💡 长期记忆跨 session 持久化；短期 history 清空后仍可 RAG 检索用户画像')

class AgentWorkState(TypedDict):
    """工作记忆：单次推理过程中的中间变量"""
    question: str
    retrieved_docs: list[str]
    draft_answer: str
    final_answer: str

def demo_4_working_memory():
    """
    LangGraph State 的自定义字段就是「工作记忆」：
    只在当前这次 invoke 的生命周期内有意义。
    """
    print('\n' + '=' * 65)
    print('Demo 4: 工作记忆 — GraphState 中间态')
    print('=' * 65)

    def retrieve_node(state: AgentWorkState):
        docs = ['LangChain 支持 LCEL 管道组合', 'LangGraph 用于有状态流程编排']
        return {'retrieved_docs': docs}

    def generate_node(state: AgentWorkState):
        context = '\n'.join(state['retrieved_docs'])
        draft = f'基于资料 [{context[:50]}...] 的草稿回答'
        final = get_model().invoke([HumanMessage(content=f"问题：{state['question']}\n资料：{context}\n请简洁回答")]).content
        return {'draft_answer': draft, 'final_answer': final}
    graph = StateGraph(AgentWorkState)
    graph.add_node('retrieve', retrieve_node)
    graph.add_node('generate', generate_node)
    graph.add_edge(START, 'retrieve')
    graph.add_edge('retrieve', 'generate')
    graph.add_edge('generate', END)
    app = graph.compile()
    result = app.invoke({'question': 'LangChain 和 LangGraph 有什么区别？', 'retrieved_docs': [], 'draft_answer': '', 'final_answer': ''})
    print(f"工作记忆 retrieved_docs: {result['retrieved_docs']}")
    print(f"工作记忆 draft_answer: {result['draft_answer'][:60]}...")
    print(f"最终输出 final_answer: {result['final_answer'][:150]}...")

def demo_5_three_layers_summary():
    print('\n' + '=' * 65)
    print('Demo 5: 三层记忆协同总结')
    print('=' * 65)
    print('\n    生产级聊天机器人典型组合：\n\n    1. 短期：Checkpointer(thread_id) 保存最近 N 轮 messages\n    2. 长期：VectorStore 存 user_id → 偏好/事实，每轮 RAG 检索注入 system prompt\n    3. 工作：State 字段存 retrieved_docs / tool_outputs，节点间传递\n\n    数据流：\n      用户输入\n        → [长期] RAG 检索用户画像 → 注入 system\n        → [短期] 加载 thread 历史 messages\n        → [工作] retrieve 节点写入 retrieved_docs\n        → LLM 生成 → 更新短期记忆\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 4-2: 记忆分层架构                     ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('短期 session', '当前窗口内的消息'),
        ('Checkpointer', '跨 invoke 接续'),
        ('长期向量', '跨会话检索旧记忆'),
        ('工作记忆', '当前任务相关的临时变量'),
        ('三层对照表', '该用哪一层'),
    ])
    check_api_key()
    try:
        demo_1_short_term_session()
        demo_2_checkpointer()
        demo_3_long_term_vector()
        demo_4_working_memory()
        demo_5_three_layers_summary()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 4-2 完成！下一步: 4_3_memory_in_chain_demo.py\n')