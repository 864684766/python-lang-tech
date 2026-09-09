# -*- coding: utf-8 -*-
"""
LangGraph Phase 5-4: LangSmith Tracing 可视化调试 Demo

学习目标:
  · LangGraph 与 LCEL 共用 LangSmith tracing
  · 在 Dashboard 查看 DAG 回放、节点耗时

运行: python lg_5_4_langsmith_tracing_demo.py

可选 .env:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=lsv2_...
  LANGCHAIN_PROJECT=langgraph-demo
"""

import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from demo_explain import print_lesson, print_step, print_tip


class State(TypedDict):
    value: int


def double_node(state: State) -> dict:
    """故意用两个纯计算节点：没有 LLM 也能在 LangSmith 里看到 DAG。"""
    return {"value": state["value"] * 2}


def add_ten_node(state: State) -> dict:
    return {"value": state["value"] + 10}


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("LangGraph 5-4: LangSmith Tracing")
    print("=" * 65)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "")
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    print_step(1, "看环境变量", "没配 KEY 也能本地跑图；配上才会上报 Dashboard")
    print(f"  LANGCHAIN_TRACING_V2 = {tracing or '(未设置)'}")
    print(f"  LANGCHAIN_API_KEY    = {'已设置' if api_key else '(未设置)'}")

    g = StateGraph(State)
    g.add_node("double", double_node)
    g.add_node("add_ten", add_ten_node)
    g.add_edge(START, "double")
    g.add_edge("double", "add_ten")
    g.add_edge("add_ten", END)
    app = g.compile()

    print_step(2, "invoke 并打 tags", "config.tags / metadata 会出现在 LangSmith 的这次 trace 上")
    r = app.invoke(
        {"value": 5},
        config={"tags": ["lg-5-4"], "metadata": {"demo": "langgraph-tracing"}},
    )
    print(f"\n结果: {r}  （5×2+10=20）")
    if tracing.lower() == "true" and api_key:
        print("✅ Trace 已上报 → https://smith.langchain.com")
    else:
        print_tip("配置 LANGCHAIN_TRACING_V2=true 后可在 LangSmith 回放 double → add_ten")
