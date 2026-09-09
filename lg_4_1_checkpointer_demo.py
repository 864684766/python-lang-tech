# -*- coding: utf-8 -*-
"""
LangGraph Phase 4-1: 生产级 Checkpointer Demo

学习目标:
  · InMemorySaver vs SqliteSaver 对比
  · 跨进程/重启后会话接续

运行: python lg_4_1_checkpointer_demo.py

可选安装 SQLite 后端:
  pip install langgraph-checkpoint-sqlite
"""

import os
import tempfile
from typing import Annotated, TypedDict

import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from demo_explain import print_lesson, print_step, print_tip


class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    turn_count: Annotated[int, operator.add]


def chat_node(state: ChatState) -> dict:
    last = state["messages"][-1] if state["messages"] else {"content": ""}
    reply = f"收到: {last.get('content', '')}"
    return {
        "messages": [{"role": "assistant", "content": reply}],
        "turn_count": 1,
    }


def demo_in_memory():
    print("\n--- InMemorySaver ---")
    g = StateGraph(ChatState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    app = g.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "mem-session-1"}}
    print_step("内存", "同一 thread_id 两次 invoke，messages 会累加")
    app.invoke({"messages": [{"role": "user", "content": "你好"}], "turn_count": 0}, cfg)
    r = app.invoke({"messages": [{"role": "user", "content": "再见"}], "turn_count": 0}, cfg)
    print(f"  消息数: {len(r['messages'])}, 轮次: {r['turn_count']}")
    print("  ⚠️  进程重启后 InMemorySaver 数据丢失")


def demo_sqlite():
    print("\n--- SqliteSaver（持久化）---")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print("  未安装 langgraph-checkpoint-sqlite")
        print("  运行: pip install langgraph-checkpoint-sqlite")
        print("  代码示例:")
        print("""
    with SqliteSaver.from_conn_string("./chat.db") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        app.invoke(..., config={"configurable": {"thread_id": "user-1"}})
    # 重启进程后同 thread_id 仍可接续
        """)
        return

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "chat.db")
        with SqliteSaver.from_conn_string(db) as checkpointer:
            g = StateGraph(ChatState)
            g.add_node("chat", chat_node)
            g.add_edge(START, "chat")
            g.add_edge("chat", END)
            app = g.compile(checkpointer=checkpointer)
            cfg = {"configurable": {"thread_id": "sqlite-session-1"}}

            app.invoke({"messages": [{"role": "user", "content": "第一轮"}], "turn_count": 0}, cfg)
            r = app.invoke({"messages": [{"role": "user", "content": "第二轮"}], "turn_count": 0}, cfg)
            print(f"  消息数: {len(r['messages'])}, DB: {db}")
            print("  ✅ 数据写入 SQLite，重启进程后可恢复")


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("LangGraph 4-1: Checkpointer 持久化")
    print("=" * 65)
    demo_in_memory()
    demo_sqlite()
