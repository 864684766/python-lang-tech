"""
LangGraph Phase 3-3: Send API 并行分发 Demo

学习目标:
  · Command(send=[Send(node, payload), ...]) 动态并行触发 worker
  · Reducer 汇聚多个 worker 的结果

运行: python lg_3_3_send_api_demo.py
"""
import os
from numbers import Number
from typing import Annotated, TypedDict
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()

class MapReduceState(TypedDict):
    headlines: list[str]
    summaries: Annotated[list[str], operator.add]

class WorkerState(TypedDict):
    headline: str
    summary: str

def dispatch_workers(state: MapReduceState):
    """
    返回 list[Send]，不是直接调用 summarize_one()。
    运行时会并行调度 N 个 worker；完成顺序可能和 headlines 列表不一致。
    """
    print(f"  📤 按列表顺序下发 {len(state['headlines'])} 张 Send 工单（并行执行）")
    return [Send('summarize_one', {'headline': h, 'summary': ''}) for h in state['headlines']]

def summarize_one(state: WorkerState) -> dict:
    llm = get_model()
    text = llm.invoke([HumanMessage(content=f"用一句话总结新闻标题: {state['headline']}")]).content
    print(f"  · Worker 完成: {state['headline'][:20]}...")
    return {'summaries': [f"[{state['headline']}] {text}"]}

def build_graph():
    g = StateGraph(MapReduceState)
    g.add_node('summarize_one', summarize_one)
    g.add_conditional_edges(START, dispatch_workers, ['summarize_one'])
    g.add_edge('summarize_one', END)
    return g.compile()
if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 3-3: Send API 并行 Map-Reduce')
    print('=' * 65)
    app = build_graph()
    headlines = ['LangGraph 发布新版本', 'AI Agent 应用爆发', 'Python 3.13 性能提升']
    print_step(1, "Map", f"为 {len(headlines)} 条标题各 Send 一个 summarize_one")
    r = app.invoke({'headlines': headlines, 'summaries': []})
    print_step(2, "Reduce", "summaries 使用 operator.add，多路结果拼成一个 list")
    print('\n汇总结果（完成顺序可能与下发顺序不同）:')
    for s in r['summaries']:
        print(f'  {s[:80]}...')
    print_tip("Send 是调度指令；要稳定顺序请在 payload 里带 index，汇总后再排序。")