"""
Python 类型教材 ⑪ array，以及：内置 / 标准库 / 第三方
array 在标准库 array 模块里，不是 list，也不是 JS 的 Array。
Agent 日常存工具名、消息、参数用 list；本节用来认清「要不要 import」。
运行: python python_basics/pb_11_array.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from array import array

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
    print("⑪ array + Python 自带分三层")
    print("=" * 65)

    print_section("三层分别是什么（「Python 自带」常把前两层混着说）")
    print("    1. 内置 builtins     不用 import：list、dict、enumerate、len、print")
    print("    2. 标准库 stdlib     安装 Python 就有，但要 import：array、json、pathlib")
    print("    3. 第三方            pip 才有：langchain、fastapi、numpy")
    print_ok("「自带」= 装上解释器就有（1+2）。「不用写 import」= 只有 1。")
    print_ok("json 是自带的标准库；langchain 不是自带，要 pip。")

    print_section("JS Array 对应的是 list，不是 array")
    tools = ["search", "read_file"]
    print_ok(f"Agent 工具列表用 list → {tools}    type={type(tools)}")
    nums = array("i", [1, 2, 3])  # "i" = 有符号整数
    print_ok(f"array('i', [1,2,3]) → {nums}    type={type(nums)}")
    print_trap(
        "从 JS 过来写成 array = [1, 2] 或 new Array()",
        "Python 里一排任意值永远用 list。array 只能装同一类数字，还要先 import。",
    )

    print_section("array 能干什么、不能干什么")
    nums.append(4)
    print_ok(f"可以 append 同类数字: {nums}    nums[0]={nums[0]}")
    print_ok(f"转成 list: {nums.tolist()}")
    print_counter(
        "不能往 int 数组里塞字符串",
        lambda: array("i", [1, "search"]),
        TypeError,
    )
    print_counter(
        "不能像 list 那样混放 dict（消息/工具参数）",
        lambda: array("i", [{"name": "search"}]),
        TypeError,
    )

    print_section("和 Agent 的关系（实话）")
    print("    写 Agent：工具名、messages、tool_calls、状态 全部用 list / dict。")
    print("    array.array 几乎不会出现在 LangChain / LangGraph 代码里。")
    print("    向量 embedding：接口先给你 list[float]，算相似度用 numpy.ndarray（⑫）。")
    print_ok("认 array 是为了：别和 list 搞混，并第一次碰到「标准库要 import」。")

    print_section("Agent 更常 import 的标准库（本节不展开，知道名字即可）")
    print("    json       工具参数、模型返回的结构化文本")
    print("    pathlib    拼 SKILL.md、本地文件路径")
    print("    copy       复制 state，避免改到同一份")
    print("    os / sys   读环境变量、改模块搜索路径")
    print("    collections  deque、defaultdict（图状态里偶见）")

    print_tip(
        "内置不用 import；标准库要 import 但不用 pip；第三方要 pip。"
        " Agent 容器选 list/dict。向量计算见 ⑫ NumPy。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
