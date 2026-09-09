"""
LangGraph Phase 5-1: 图中 Structured Output Demo

学习目标:
  · 在 Graph 节点内用 with_structured_output 约束 LLM 输出
  · 下游节点直接使用 Pydantic 字段做条件路由

运行: python lg_5_1_structured_output_demo.py
"""
import os
from numbers import Number
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()

class FeedbackAnalysis(BaseModel):
    """LLM 必须按这个 schema 填字段，下游才能 if/else。"""
    summary: str = Field(description='反馈摘要')
    keywords: list[str] = Field(description='关键词列表')
    # Literal = 三选一，不是列表；模型不能输出「开心」这种第四种值
    sentiment: Literal['positive', 'neutral', 'negative'] = Field(description='情感极性')

class FlowState(TypedDict):
    feedback: str
    analysis: dict
    action: str

def analyze_node(state: FlowState) -> dict:
    # 把 Pydantic schema 绑到模型上：输出不再是自由文本
    structured = get_model().with_structured_output(FeedbackAnalysis)
    result = structured.invoke([HumanMessage(content=f"分析用户反馈:\n{state['feedback']}")])
    # 有的版本返回 Pydantic 对象（有 model_dump），有的已是 dict
    data = result.model_dump() if hasattr(result, 'model_dump') else result
    print(f"  📊 结构化结果: sentiment={data['sentiment']}, keywords={data['keywords']}")
    return {'analysis': data}

def route_node(state: FlowState) -> dict:
    """本节点不再调 LLM，只读结构化字段做路由 —— 这就是 schema 的价值。"""
    sentiment = state['analysis']['sentiment']
    if sentiment == 'negative':
        action = 'escalate_to_human'
    elif sentiment == 'positive':
        action = 'send_thanks'
    else:
        action = 'auto_reply'
    print(f"  🚦 因 sentiment={sentiment!r} → action={action}")
    return {'action': action}

def build_graph():
    g = StateGraph(FlowState)
    g.add_node('analyze', analyze_node)
    g.add_node('route', route_node)
    g.add_edge(START, 'analyze')
    g.add_edge('analyze', 'route')
    g.add_edge('route', END)
    return g.compile()
if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 5-1: 图中 Structured Output')
    print('=' * 65)
    app = build_graph()
    print_step(1, "正面反馈", "期望 sentiment=positive → send_thanks")
    for i, fb in enumerate(['产品太棒了，非常满意！', '发货慢，我要投诉'], 1):
        print_step(i, "跑图", f"用户反馈: {fb}")
        r = app.invoke({'feedback': fb, 'analysis': {}, 'action': ''})
        print(f"  → 最终 action: {r['action']}")
    print_tip("若模型返回自由文本，route_node 无法稳定分支；Literal 把情感锁成三个值。")