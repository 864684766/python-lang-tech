"""
专门演示：LangGraph Studio 可视化调试 +（可选）LangSmith Trace。

Studio（本地，不需要 LangSmith Key）:
  1. 本文件顶层已导出 graph
  2. langgraph.json 指向本文件
  3. 启动:  $env:PYTHONUTF8="1"; langgraph dev --port 2025 --allow-blocking
  4. 打开:  https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2025
  5. Input 里 question 填一句话，点 Submit
  6. 点节点看每一步的 state；含「详细/为什么/怎么」会走 detail，否则走 short

LangSmith Trace（云端面板，需要 Key）:
  Key 不要发给任何人。在网页点 Generate API Key，写进本目录 .env:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_你的key
    LANGSMITH_PROJECT=langgraph-test
  然后再跑一次图，到 https://smith.langchain.com 的 Tracing 项目里看。

也可直接: python lg_studio_debug_demo.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from demo_explain import print_lesson, print_step, print_tip

load_dotenv()


@dataclass
class StudioState:
    question: str = "爱情是什么"
    route: str = ""
    answer: str = ""


def _want_detail(question: str) -> bool:
    return any(word in question for word in ("详细", "为什么", "怎么", "如何"))


def route_node(state: StudioState) -> dict:
    """只做分流，方便在 Studio 里看条件边。"""
    route = "detail" if _want_detail(state.question) else "short"
    print(f"  [route] question={state.question!r} → {route}")
    return {"route": route}


def short_node(state: StudioState) -> dict:
    answer = f"[简短] {state.question}：一句话就能说清。"
    print(f"  [short] {answer}")
    return {"answer": answer}


def detail_node(state: StudioState) -> dict:
    answer = (
        f"[详细] 关于「{state.question}」：\n"
        "1. 先定义问题\n"
        "2. 再拆成可执行的步骤\n"
        "3. 最后给一个可检查的结论"
    )
    print(f"  [detail]\n{answer}")
    return {"answer": answer}


def pick_branch(state: StudioState) -> str:
    return state.route


def build_graph():
    g = StateGraph(StudioState)
    g.add_node("route", route_node)
    g.add_node("short", short_node)
    g.add_node("detail", detail_node)
    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        pick_branch,
        {"short": "short", "detail": "detail"},
    )
    g.add_edge("short", END)
    g.add_edge("detail", END)
    return g.compile(name="studio_debug")


graph = build_graph()


def _tracing_status() -> tuple[bool, str]:
    tracing = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or ""
    ).lower() == "true"
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "(未设置)"
    )
    return tracing and bool(key), project


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("Studio 调试案例：route → short / detail")
    print("=" * 65)

    enabled, project = _tracing_status()
    print_step(1, "LangSmith Key", "Studio 本地调试不需要；云端 Trace 才需要")
    if enabled:
        print(f"  Trace 已开启，项目: {project}")
    else:
        print("  未配置 LANGSMITH_API_KEY，只在本地跑，不会上报云端。")
        print_tip("Key 在 LangSmith 页面点 Generate API Key，写进 .env，不要贴到聊天里。")

    print_step(2, "跑两条路径", "Studio 里改 question 再 Submit，效果相同")
    for q in ["爱情是什么", "详细说明爱情是什么"]:
        result = graph.invoke({"question": q, "route": "", "answer": ""})
        print(f"  Q: {q}")
        print(f"  route={result['route']}")
        print(f"  {result['answer'].splitlines()[0]}")
        print()
