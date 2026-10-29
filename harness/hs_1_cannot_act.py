"""
Harness 教材 ① 模型会说不会做

上一节用概念图分清了模型与 Harness。本课首次实际运行工具调用。

这一节把问题往前推一步。如果工具列表里根本没有 add 呢？

先看一个现象。问「3+5 等于几」，模型可以直接回文字「8」。记录里只有这两条：

    [
        {"kind": "user", "text": "3+5 等于几？只答数字"},
        {"kind": "model", "text": "8"},
    ]
    called_tools(...) == []     # 没有名为 add 的那一条

数字是对的，add(3, 5) 却没执行过。

这里的问题是：终答里有 "8"，只说明它嘴上会说。kind 为 "tool_result" 的记录一条都没有。

以前的做法是系统提示写「请仔细算」或「你是计算器」。算错了就再求一次。这句话管的是措辞，管不了有没有调用函数。

这一节用同一道题跑两遍：

    Runner(model, [unused]).run("3+5 等于几？只答数字")          # 工具表里没有 add
    Runner(model, [add], system=MUST_USE_ADD).run("3+5 等于几？") # 工具表里有 add

第一遍 called_tools 是 []。第二遍要检查的是 "add" 在不在 called_tools 里，不是终答里有没有 8。

反面教材：测试只写 assert "8" in 终答。例如不挂 add，问「3+5 等于几？只答数字」，模型回「8」，测试通过。记录里却没有 kind 为 "tool_result" 的那一条，called_tools 是 []。换一道它不会心算的题，同样的测试就会把嘴算当成工具已经执行过。正确的检查是 "add" in called_tools。

影响：能力边界由 Harness 画。列表里没有的函数，模型自己变不出来。

本课也是工具调用接口的唯一入门课。Runner 内部的 LangChain 接法是：

    @tool / bind_tools       提供工具名称、说明和参数 schema，不上传或执行函数
    AIMessage.tool_calls     模型提出 name / args / id，可能有零个或多个调用
    本地工具.invoke(args)    Harness 检查工具名，工具包装器校验参数，再执行函数
    ToolMessage              工具返回值作为 content，用 tool_call_id 对应调用的 id
    再次 invoke              保留原始 AIMessage 和配对的工具结果，让模型继续处理

不需要给工具结果额外拼接一段说明文字。这些类和方法属于 LangChain；其他 SDK
也能实现同样的机制。工具声明不代替权限校验，也不保证模型一定调用工具。
没有 tool_calls 时，Runner 结束本轮，返回 Turn(status, items, final_text 等)。
items 由 Harness 维护，是完整记录；实际发给模型的上下文可以只是其中一部分。
实现对应 mini.py 的 _invoke / _run_tools / items_to_messages / _loop。
后续 hs_2 只研究前一步结果如何成为下一步工具参数，不再重新介绍单次调用。

运行: python harness/hs_1_cannot_act.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import MUST_USE_ADD, add, live_model, print_harness_job
from harness.mini import Runner, called_tools
from langchain_core.tools import tool


@tool
def unused() -> str:
    """本课无工具对照用的空工具。"""
    return "unused"


def demo() -> None:
    print("\n" + "=" * 65)
    print("① 会说，不等于做过")
    print("=" * 65)

    print_harness_job(
        "给不给工具，是 Harness 的能力边界。模型自己变不出 add()",
        prop="同一道 3+5",
    )

    print_idea(
        phenomenon="GLM 回复 8。看起来对，但可能根本没调用任何计算器。",
        gap="「说」和「做」要分开。数字必须经过 add。",
        before="System 里写「你是计算器」。错了只能再求它一次。",
        impact="有工具时，Item 里会出现 tool_result。这才算 Harness 在工作。",
        counter="只看终答是不是 8。嘴算也能碰巧对。",
    )

    model = live_model()

    print_section("从前：不给加法工具，只能嘴算")
    mouth = Runner(model, [unused]).run("3+5 等于几？只答数字，不要调用工具")
    print_ok(f"无 add 时调用了 {called_tools(mouth)}，终答 {mouth.final_text!r}")

    print_section("现在：给 add，并要求必须用工具")
    real = Runner(model, [add], system=MUST_USE_ADD).run("3+5 等于几？")
    tools = called_tools(real)
    assert "add" in tools, f"GLM 没有调用 add，调用了 {tools}，终答 {real.final_text!r}"
    print_ok(f"调用了 {tools}，终答 {real.final_text!r}")

    print_section("同一轮的调用、结果和终答分别在哪里")
    for item in real.items:
        if item.kind == "model" and item.raw is not None and item.raw.tool_calls:
            print_ok(f"AIMessage.tool_calls = {item.raw.tool_calls}")
        elif item.kind == "tool_result":
            print_ok(f"ToolMessage: content={item.text!r}, tool_call_id={item.tool_call_id!r}")
    print_ok(f"Runner 返回 Turn：status={real.status!r}，完整 items 共 {len(real.items)} 条")
    print_ok("模型返回的是本次 AIMessage；完整 items 和 final_text 由 Harness 包装返回")

    print_trap(
        "终答对 ≠ Harness 有效",
        "上一格即使答对也没有 add。这一格要看 tool_result。",
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
