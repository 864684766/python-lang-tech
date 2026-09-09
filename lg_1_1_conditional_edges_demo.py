"""
LangGraph Phase 1-1: 条件边 add_conditional_edges 教材级 Demo

学习目标:
  · 跑完节点 A 后，根据 state 动态选择下一跳
  · 掌握 route_function + 映射表 的标准写法

运行: python lg_1_1_conditional_edges_demo.py
"""
import os
from numbers import Number
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()

class IntentState(TypedDict):
    question: str
    intent: str
    result: str

def classify_node(state: IntentState) -> dict:
    """用 LLM 判断用户意图: translate / summarize / unknown"""
    llm = get_model()
    prompt = f"判断用户意图，只回复一个词: translate(翻译) / summarize(总结) / unknown\n用户输入: {state['question']}"
    intent = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    if 'translate' in intent or '翻译' in intent:
        intent = 'translate'
    elif 'summarize' in intent or '总结' in intent:
        intent = 'summarize'
    else:
        intent = 'unknown'
    print(f'  🎯 意图识别: {intent}')
    return {'intent': intent}

def translate_node(state: IntentState) -> dict:
    llm = get_model()
    text = llm.invoke([HumanMessage(content=f"将以下内容翻译为英文:\n{state['question']}")]).content
    return {'result': f'[翻译] {text}'}

def summarize_node(state: IntentState) -> dict:
    llm = get_model()
    text = llm.invoke([HumanMessage(content=f"用一句话总结:\n{state['question']}")]).content
    return {'result': f'[总结] {text}'}

def unknown_node(state: IntentState) -> dict:
    return {'result': '[错误] 无法识别意图，请说明要「翻译」还是「总结」'}

def route_by_intent(state: IntentState) -> Literal['translate', 'summarize', 'unknown']:
    """条件边的路由函数：返回值必须是下面 mapping 的 key。"""
    return state['intent']

def build_graph():
    g = StateGraph(IntentState)
    g.add_node('classify', classify_node)
    g.add_node('translate', translate_node)
    g.add_node('summarize', summarize_node)
    g.add_node('unknown', unknown_node)
    g.add_edge(START, 'classify')
    # 和 add_edge 的差别：下一跳由 route_by_intent(state) 动态决定
    g.add_conditional_edges('classify', route_by_intent, {'translate': 'translate', 'summarize': 'summarize', 'unknown': 'unknown'})
    g.add_edge('translate', END)
    g.add_edge('summarize', END)
    g.add_edge('unknown', END)
    return g.compile()
if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 1-1: 条件边 add_conditional_edges')
    print('=' * 65)
    app = build_graph()
    print_step(1, "翻译意图", "classify 应路由到 translate 节点")
    for q in ['请翻译：你好世界', '请总结：LangGraph 是有状态图编排框架']:
        print(f'\n问题: {q}')
        r = app.invoke({'question': q, 'intent': '', 'result': ''})
        print(f"结果: {r['result'][:120]}...")
    print_tip("把 mapping 里的 key 写错，运行时会找不到下一跳。")