# -*- coding: utf-8 -*-
"""
LangGraph Phase 2-3: Time Travel / State Editing 教材级 Demo

学习目标:
  · interrupt 暂停后，用 get_state / update_state 篡改历史快照
  · Command(resume=...) 从修改后的状态继续执行

运行: python lg_2_3_time_travel_demo.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from demo_explain import print_lesson, print_step, print_tip


class ApprovalState(TypedDict):
    draft: str
    approved: bool
    feedback: str
    final: str


def write_draft(_: ApprovalState) -> dict:
    return {"draft": "【AI 初稿】本周项目进展良好", "approved": False}


def review_node(state: ApprovalState):
    print(f"  ⏸  等待审批，当前草稿: {state['draft']}")
    # 括号里的 dict = 暂停时给人看的说明书（不是 interrupt 的返回值）
    # 恢复后 decision = Command(resume={...}) 里那个 dict
    decision = interrupt({"action": "请审批草稿", "draft": state["draft"]})
    return {
        "approved": decision.get("approve", False),  # 键名必须对上 resume
        "feedback": decision.get("feedback", ""),
    }


def finalize(state: ApprovalState) -> dict:
    if state["approved"]:
        return {"final": f"✅ 已发布: {state['draft']}"}
    return {"final": f"❌ 驳回: {state['feedback']}"}


def build_graph():
    g = StateGraph(ApprovalState)
    g.add_node("write", write_draft)
    g.add_node("review", review_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "write")
    g.add_edge("write", "review")
    g.add_edge("review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("LangGraph 2-3: Time Travel / State Editing")
    print("=" * 65)

    app = build_graph()
    # thread_id = 这一条审批单；invoke / get_state / resume 必须用同一把钥匙
    config = {"configurable": {"thread_id": "approval-tt-001"}}

    print_step(1, "跑到 interrupt", "图暂停；说明书在 snapshot.interrupts[0].value，State 在 snapshot.values")
    result = app.invoke(
        {"draft": "", "approved": False, "feedback": "", "final": ""},
        config=config,
    )
    print("  检测到中断，准备 Time Travel 编辑状态...")
    snapshot = app.get_state(config)
    if snapshot.interrupts:
        print(f"  📤 interrupt 给人看的载荷: {snapshot.interrupts[0].value}")
    print(f"  编辑前 draft (State): {snapshot.values.get('draft')}")

    print_step(2, "update_state", "改的是图 State 里的 draft，不是 resume 的键")
    app.update_state(config, {"draft": "【老板修改版】本周项目超额完成，重点突出 Q3 成果"})
    snapshot2 = app.get_state(config)
    print(f"  编辑后 draft: {snapshot2.values.get('draft')}")

    print_step(3, "Command(resume=...)", "resume 的键对准 decision.get('approve'/'feedback')")
    final = app.invoke(
        Command(resume={"approve": True, "feedback": "修改后通过"}),
        config=config,
    )
    print(f"\n✅ {final['final']}")
    print_tip("update_state 改草稿；resume 交审批意见。config.thread_id 把三次调用钉在同一条线上。")
