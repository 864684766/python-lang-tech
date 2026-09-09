"""
Python 类型教材 ⓪ 总览：Python 到底有哪些常用类型

先建立地图，再按节打开 ①～⑨ 动手。
每个值都有类型，用 type(x) 查看。
运行: python python_basics/pb_0_types_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_counter, print_lesson, print_ok, print_section, print_tip


def demo() -> None:
    print("\n" + "=" * 65)
    print("⓪ Python 常用内置类型（新手先认这些）")
    print("=" * 65)

    # 一行一个：名字、例子、可变、JS 对照
    rows = [
        ("None", None, "不可变", "null / undefined（Python 合成一个）"),
        ("bool", True, "不可变", "boolean"),
        ("int", 42, "不可变", "number 的整数部分"),
        ("float", 3.14, "不可变", "number 的小数部分"),
        ("str", "hello", "不可变", "string"),
        ("list", [1, 2], "可变", "Array"),
        ("tuple", (1, 2), "不可变", "没有直系亲戚，像写死的一组值"),
        ("dict", {"a": 1, }, "可变", "普通对象 / Map"),
        ("set", {1, 2}, "可变", "Set"),
        ("range", range(3), "不可变", "没有；用来数次数"),
    ]
    print(f"  {'类型':<8}{'例子':<16}{'type()':<20}可变?     JS 对照")
    print("  " + "-" * 72)
    for name, value, mutable, js in rows:
        print(f"  {name:<8}{value!s:<16}{str(type(value)):<20}{mutable:<8} {js}")

    print_section("怎么认类型")
    x = "mcp"
    print_ok(f"x = {x!r}")
    print(f"    type(x) → {type(x)}")
    print(f"    isinstance(x, str) → {isinstance(x, str)}")

    print_section("可变 vs 不可变")
    print("    可变 = 改了原地还是原来那个对象（list / dict / set）。")
    print("    不可变 = 看起来像改，其实是造出新值（str / int / tuple）。")

    print_section("新手先不用管")
    print("    complex（复数）、bytes（二进制）、frozenset（不可变集合）")
    print("    函数、类、模块 在 Python 里也是对象，后面再学。")

    print_section("类型对不上会怎样")
    # Python 不会像 JS 那样把 '3'+1 悄悄变成 '31'
    print_counter("字符串不能直接加数字（JS 会变成 '31'）", lambda: "3" + 1, TypeError)

    print_tip("先跑本文件认地图，再按 ① None → ⑨ range 一节一节跑。每节都有【反例】。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
