# -*- coding: utf-8 -*-
"""
LangGraph Phase 2-1: Tool-Calling Agent 手动 ReAct 教材级 Demo

学习目标:
  · 在 StateGraph 节点中 bind_tools + 手动执行 tool_calls
  · 理解 create_agent 底层的 Graph 结构
  · invoke_with_logged_retry —— with_retry 等价封装，异常重试时打印日志
  · model_provider.get_model —— Zhipu 失败自动切换 OpenAI

运行: python lg_2_1_tool_calling_agent_demo.py
"""

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from model_provider import check_api_key, get_model
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
)
from demo_explain import print_lesson, print_step, print_tip


@tool
def get_weather(city: str) -> str:
    """查询城市天气（模拟）"""
    data = {"北京": "25°C 晴", "上海": "22°C 多云"}
    return data.get(city, f"暂无 {city} 数据")


@tool
def calculator(expression: str) -> str:
    """数学计算"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误: {e}"


TOOLS = [get_weather, calculator]
TOOL_MAP = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_wait_strategy(
    *,
    mode: str = "exponential_jitter",
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exp_base: float = 2.0,
    fixed_delay: float = 2.0,
):
    """
    重试等待策略（tenacity wait）:

      fixed              → 每次固定等 fixed_delay 秒
      exponential        → initial × exp_base^n，上限 max_delay（无随机抖动）
      exponential_jitter → 同上 + 随机 jitter（默认，与 with_retry 行为接近）
    """
    if mode == "fixed":
        return wait_fixed(fixed_delay)
    if mode == "exponential":
        return wait_exponential(
            multiplier=initial_delay,
            min=initial_delay,
            max=max_delay,
            exp_base=exp_base,
        )
    if mode == "exponential_jitter":
        return wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            exp_base=exp_base,
        )
    raise ValueError(f"未知 wait_mode: {mode!r}，可选 fixed / exponential / exponential_jitter")


def invoke_with_logged_retry(
    runnable: Any,
    input: Any,
    *,
    max_attempts: int = 3,
    wait_mode: str = "exponential_jitter",
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exp_base: float = 2.0,
    fixed_delay: float = 2.0,
) -> Any:
    """
    等价于 runnable.with_retry(stop_after_attempt=max_attempts)，
    但每次因异常触发重试前会 print 日志（含等待秒数）。
    """

    def _log_before_retry(retry_state) -> None:
        exc = retry_state.outcome.exception()
        attempt = retry_state.attempt_number
        wait_sec = (
            retry_state.next_action.sleep
            if retry_state.next_action is not None
            else retry_state.upcoming_sleep
        )
        print(f"  ⚠️  model 调用失败（第 {attempt}/{max_attempts} 次）: {type(exc).__name__}: {exc}")
        if attempt < max_attempts:
            print(f"  ⏳ 等待 {wait_sec:.2f}s 后执行第 {attempt + 1} 次重试...")

    def _log_retry_start(retry_state) -> None:
        if retry_state.attempt_number > 1:
            print(f"  🔄 等待结束，开始第 {retry_state.attempt_number} 次重试")

    result = None
    for attempt in Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=_build_wait_strategy(
            mode=wait_mode,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exp_base=exp_base,
            fixed_delay=fixed_delay,
        ),
        retry=retry_if_exception_type(Exception),
        before=_log_retry_start,
        before_sleep=_log_before_retry,
        reraise=True,
    ):
        with attempt:
            result = runnable.invoke(input)
    return result


def call_model(state: AgentState) -> dict:
    """agent 节点：把工具 schema 绑到模型上，让它决定调不调、调哪个。

    get_model() 已含主备切换；invoke_with_logged_retry 再给超时打等待日志。
    """
    model = get_model().bind_tools(TOOLS)
    response = invoke_with_logged_retry(
        model,
        state["messages"],
        max_attempts=3,
        wait_mode="exponential_jitter",
        initial_delay=1.0,
        max_delay=8.0,
        exp_base=2.0,
    )
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """tools 节点：执行上一轮 AIMessage.tool_calls，结果写成 ToolMessage 回灌。"""
    last: AIMessage = state["messages"][-1]
    outputs = []
    for tc in last.tool_calls:
        result = TOOL_MAP[tc["name"]].invoke(tc["args"])
        print(f"  🔧 {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)}) → {result}")
        outputs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    """条件边：有 tool_calls 就去 tools，没有就结束（模型已经能直接回答）。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph():
    # 这就是 create_agent 内部的骨架：agent ↔ tools 循环
    g = StateGraph(AgentState)
    g.add_node("agent", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")  # 工具结果必须回到 agent，让模型看观察再决定
    return g.compile()


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("LangGraph 2-1: 手动 Tool-Calling ReAct 循环")
    print("=" * 65)
    check_api_key()

    app = build_graph()
    q = "北京今天多少度？温度乘以 2 是多少？"
    print_step(1, "一次提问走完 ReAct", "期望先 get_weather，再 calculator，最后自然语言终答")
    print(f"问题: {q}\n")
    r = app.invoke({"messages": [HumanMessage(content=q)]})
    print(f"\n🤖 最终: {r['messages'][-1].content}")
    print_tip("对照 lg_2_2：同样能力，create_agent 只要三行；本文件是把循环拆开给你看。")
