"""
Python 类型教材 ⑤ list（列表）
一排可以改的值，方括号。对照 JS：Array。
运行: python python_basics/pb_5_list.py
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
    print("⑤ list：列表（可变）")
    print("=" * 65)

    print_section("创建、下标、切片")
    tools = ["add", "get_me", "list_issues"]
    print_ok(f"{tools}    type={type(tools)}  len={len(tools)}")
    print_ok(f"第 0 个 {tools[0]}    最后 1 个 {tools[-1]}")
    print_ok(f"切片 [0:2]（含头不含尾）→ {tools[0:2]}")

    print_section("改原地")
    tools.append("search")
    print_ok(f"append 后 {tools}")
    tools[0] = "subtract"
    print_ok(f"改下标 0 后 {tools}")
    tools.remove("search")  # 按值删掉第一次出现的那个
    print_ok(f"remove('search') 后 {tools}")
    print_ok(f"'get_me' in tools → {'get_me' in tools}")

    print_section("遍历（JS 的 for...of）")
    for i, name in enumerate(tools):
        print(f"    {i}: {name}")

    print_section("赋值不是复制")
    a = [1, 2]
    b = a
    b.append(3)
    print_trap(
        "b = a 之后改 b，a 也变了",
        f"a={a}  b={b}。两个名字指向同一份列表。",
    )
    c = a.copy()  # 或 a[:]
    c.append(99)
    print_ok(f"a.copy() 再改: a={a}  c={c}")

    print_section("常见报错")
    print_counter("下标超出", lambda: tools[99], IndexError)
    print_counter("列表没有 JS 的 .find()", lambda: tools.find("get_me"), AttributeError)
    print_counter("remove 一个不存在的值", lambda: tools.remove("不存在"), ValueError)

    print_tip("要两份列表用 .copy()。找元素用 for+break，没有 .find()。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
