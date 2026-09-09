"""
LangGraph Phase 4-2: Store API 跨线程长期记忆 Demo

学习目标:
  · Checkpointer = 单次对话快照 (thread_id)
  · Store = 跨 thread 的长期记忆 (namespace + key)
  · 节点内通过 Runtime.store 访问

运行: python lg_4_2_store_api_demo.py
"""
import os
from numbers import Number
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()

class ChatState(TypedDict):
    user_id: str
    user_message: str
    reply: str

def save_profile_node(state: ChatState, runtime: Runtime) -> dict:
    """首次对话：写入 Store"""
    if runtime.store is None:
        return {}
    namespace = ('users', state['user_id'])
    if runtime.store.get(namespace, 'profile') is None:
        runtime.store.put(namespace, 'profile', {'name': '张三', 'lang': '中文', 'preference': '回答简洁'})
        print('  💾 Store 写入用户档案')
    return {}

def chat_node(state: ChatState, runtime: Runtime) -> dict:
    """从 Store 读取档案并回复"""
    profile = {'name': '未知', 'lang': '中文'}
    if runtime.store:
        item = runtime.store.get(('users', state['user_id']), 'profile')
        if item:
            profile = item.value
            print(f'  📖 Store 读取: {profile}')
    llm = get_model()
    prompt = f"用户档案: {profile}\n用户说: {state['user_message']}\n请用用户偏好的语言简短回复。"
    reply = llm.invoke([HumanMessage(content=prompt)]).content
    return {'reply': reply}
if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 4-2: Store API 跨线程长期记忆')
    print('=' * 65)
    store = InMemoryStore()
    g = StateGraph(ChatState)
    g.add_node('save_profile', save_profile_node)
    g.add_node('chat', chat_node)
    g.add_edge(START, 'save_profile')
    g.add_edge('save_profile', 'chat')
    g.add_edge('chat', END)
    app = g.compile(checkpointer=InMemorySaver(), store=store)
    cfg_a = {'configurable': {'thread_id': 'thread-A'}}
    print_step(1, "thread-A 写入 Store", "按 user_id=u001 存档案，和 thread_id 无关")
    r1 = app.invoke({'user_id': 'u001', 'user_message': '你好', 'reply': ''}, cfg_a)
    print(f"  thread-A: {r1['reply'][:80]}")
    print_step(2, "thread-B 读取同一档案", "换了会话仍记得用户偏好")
    cfg_b = {'configurable': {'thread_id': 'thread-B'}}
    r2 = app.invoke({'user_id': 'u001', 'user_message': '你还记得我吗？', 'reply': ''}, cfg_b)
    print(f"  thread-B: {r2['reply'][:80]}")
    print('\n💡 Store 按 user_id 共享；Checkpointer 按 thread_id 隔离短期对话')