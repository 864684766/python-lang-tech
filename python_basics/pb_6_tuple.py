"""
Python 类型教材 ⑥ tuple（元组）
一排不能改的值，圆括号。没有 JS 直系亲戚，用来表示「固定几个槽位」。
运行: python python_basics/pb_6_tuple.py
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
    print("⑥ tuple：元组（不可变）")
    print("=" * 65)

    print_section("创建和读取")
    args = ("-y", "some-package")
    print_ok(f"{args}    type={type(args)}  len={len(args)}")
    print_ok(f"args[0] → {args[0]}")

    def mutate() -> None:
        args[0] = "no"  # type: ignore[index]

    print_counter("不能改下标、也不能 append", mutate, TypeError)

    print_section("解包")
    flag, package = args
    print_ok(f"flag={flag}  package={package}")

    def pair():
        return "read", "write"  # 多返回值其实是一个 tuple

    r, w = pair()
    print_ok(f"多返回值: r={r} w={w}  实际类型={type(pair())}")

    def bad_unpack() -> None:
        x, y, z = args

    print_counter("左边 3 个变量，右边只有 2 个值", bad_unpack, ValueError)

    print_section("单元素必须有逗号")
    not_tuple = ("hello")
    one_tuple = ("hello",)
    print_trap(
        "('hello') 只是括号，不是元组",
        f"{not_tuple!r} type={type(not_tuple)}。正确: {one_tuple!r} type={type(one_tuple)}",
    )

    print_ok(f"list ↔ tuple: {tuple([1, 2])}    {list((1, 2))}")
    print_tip("要改内容用 list；固定几个槽位用 tuple。单元素写 ('x',)。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
