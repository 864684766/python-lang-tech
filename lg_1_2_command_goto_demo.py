"""
LangGraph Phase 1-2: Command(goto=..., update=...) 教材级 Demo

学习目标:
  · 节点内直接决定下一跳 + 局部更新 state
  · 无需外部 route_function

运行: python lg_1_2_command_goto_demo.py
"""
import os
from numbers import Number
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()

class QualityState(TypedDict):
    topic: str
    draft: str
    score: int
    attempts: int
    final: str

def generate_node(state: QualityState):
    llm = get_model(temperature=0.7)
    draft = llm.invoke([HumanMessage(content=f"写一段关于「{state['topic']}」的 80 字介绍")]).content
    return {'draft': draft, 'attempts': state['attempts'] + 1}

def score_node(state: QualityState):
    """质量打分过低 → Command(goto='generate') 原地重试"""
    llm = get_model()
    prompt = f"给以下文本质量打分 0-10（只回复数字）:\n{state['draft']}"
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    score = int(''.join((c for c in raw if c.isdigit()))[:2] or '5')
    print(f"  📊 第 {state['attempts']} 次尝试，质量分: {score}")
    if score < 7 and state['attempts'] < 3:
        # 节点自己决定下一跳 + 把 score 写进 state，无需外部 route 函数
        return Command(goto='generate', update={'score': score})
    return Command(goto='finalize', update={'score': score})

def finalize_node(state: QualityState) -> dict:
    return {'final': f"[score={state['score']}] {state['draft']}"}

def build_graph():
    g = StateGraph(QualityState)
    g.add_node('generate', generate_node)
    g.add_node('score', score_node)
    g.add_node('finalize', finalize_node)
    g.add_edge(START, 'generate')
    g.add_edge('generate', 'score')
    g.add_edge('finalize', END)
    return g.compile()
if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 1-2: Command(goto, update)')
    print('=' * 65)
    app = build_graph()
    print_step(1, "生成并打分", "低分会 Command(goto='generate') 循环，最多 3 次")
    r = app.invoke({'topic': 'LangGraph', 'draft': '', 'score': 0, 'attempts': 0, 'final': ''})
    print(f"\n✅ 最终输出:\n{r['final'][:200]}")
    print_tip("和 lg_1_1 对比：1-1 用外部 route 函数；1-2 把路由写在节点返回值里。")