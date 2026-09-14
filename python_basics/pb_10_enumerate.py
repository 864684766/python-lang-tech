"""
Python 类型教材 ⑩ enumerate
给一排数据配上下标的内置函数，不是 list / dict 那种容器。
写 Agent 时：给工具编号、按序执行 tool_calls、调试消息列表。
运行: python python_basics/pb_10_enumerate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_counter,
    print_lesson,
    print_ok,
    print_section,
    print_tip,
    print_trap,
)


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑩ enumerate：给已有序列编号（内置函数，不是容器类型）")
    print("=" * 65)

    print_section("它是什么")
    tools = ["search", "read_file", "write_file"]
    numbered = enumerate(tools)
    print_ok(f"enumerate(tools) → {numbered}")
    print_ok(f"type={type(numbered)}    要用完才看清内容: {list(enumerate(tools))}")
    print_trap(
        "把它当成「一种基本类型」和 list 并列",
        "enumerate 是函数，返回一次性迭代器。容器仍是左边那个 list。",
    )

    print_section("和 range 的分工")
    print("    range：自己造整数    for i in range(3)")
    print("    enumerate：给已有的 list/tuple 配下标")
    print_ok("Agent 里遍历工具、消息、tool_calls，用 enumerate，不要 range(len(...))")

    print_section("Agent：给工具清单编号（给模型看）")
    for i, name in enumerate(tools, start=2):
        print(f"    {i}. {name}")
    print_ok("start=1 时编号从 1 起，下标仍是 list 的 0、1、2")

    print_section("Agent：按顺序执行多条 tool_calls")
    tool_calls = [
        {"name": "search", "args": {"q": "LangGraph"}},
        {"name": "read_file", "args": {"path": "README.md"}},
    ]
    for i, call in enumerate(tool_calls):
        print_ok(f"第 {i} 次调用 {call['name']}  参数={call['args']}")

    print_section("Agent：调试对话消息")
    messages = [
        {"role": "user", "content": "查天气"},
        {"role": "assistant", "content": "我去调工具"},
        {"role": "tool", "content": "晴，22°C"},
    ]
    for i, msg in enumerate(messages):
        print(f"    [{i}] {msg['role']}: {msg['content']}")
    print_ok("compaction / 截断上下文时，常用 enumerate 定位「从第几条砍」")

    print_section("常见报错")
    print_counter(
        "enumerate 结果不能按下标取",
        lambda: enumerate(tools)[0],
        TypeError,
    )
    print_ok(f"要下标先转 list: {list(enumerate(tools))[0]}")
    print_trap(
        "for i in range(len(tools)): 再用 tools[i]",
        "能跑，但啰嗦，也容易写错边界。直接 for i, name in enumerate(tools)。",
    )

    print_tip(
        "enumerate 用完即空。要反复看，list(enumerate(xs))。"
        " JS 对照: arr.entries()。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
