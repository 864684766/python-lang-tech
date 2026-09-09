# -*- coding: utf-8 -*-
"""
LangGraph Phase 5-2: 错误处理与降级 Demo

学习目标:
  · 节点内 try/except + Command(goto="fallback") 软着陆
  · 避免 LLM/API 异常导致整图崩溃

运行: python lg_5_2_error_fallback_demo.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from demo_explain import print_lesson, print_step, print_tip


class State(TypedDict):
    question: str
    answer: str
    used_fallback: bool


def llm_node(state: State):
    """模拟 LLM 调用失败"""
    if "崩溃" in state["question"]:
        raise ConnectionError("模拟 API 超时")
    return {"answer": f"正常回答: {state['question']}", "used_fallback": False}


def llm_node_safe(state: State):
    try:
        return llm_node(state)
    except Exception as exc:
        print(f"  ⚠️  捕获异常: {exc} → 跳转 fallback")
        return Command(goto="fallback", update={"used_fallback": True})


def fallback_node(state: State) -> dict:
    return {
        "answer": "抱歉，服务暂时不可用，请稍后重试。（来自 fallback 节点）",
        "used_fallback": True,
    }


def build_graph():
    g = StateGraph(State)
    g.add_node("llm", llm_node_safe)
    g.add_node("fallback", fallback_node)
    g.add_edge(START, "llm")
    g.add_edge("fallback", END)
    return g.compile()


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("LangGraph 5-2: 错误处理与降级")
    print("=" * 65)
    app = build_graph()
    print_step(1, "正常路径", "问题不含「崩溃」，llm_node 直接返回")
    r1 = app.invoke({"question": "你好", "answer": "", "used_fallback": False})
    print(f"正常: {r1['answer']}, fallback={r1['used_fallback']}")
    print_step(2, "降级路径", "模拟超时 → Command(goto='fallback')，整图不崩溃")
    r2 = app.invoke({"question": "请崩溃", "answer": "", "used_fallback": False})
    print(f"降级: {r2['answer']}, fallback={r2['used_fallback']}")
    print_tip("生产环境应对 LLM/HTTP 包一层 try/except，用 Command 或 fallback 链软着陆。")
