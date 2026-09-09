"""
Python 类型教材 ③ int 与 float
整数 int、小数 float。Python 把它们分成两种类型（JS 几乎都叫 number）。
运行: python python_basics/pb_3_int_float.py
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
    print("③ int（整数）和 float（小数）")
    print("=" * 65)

    print_section("两种数字")
    print_ok(f"42 type={type(42)}    3.14 type={type(3.14)}")

    print_section("四则运算")
    a, b = 7, 2
    print_ok(f"{a} + {b} = {a + b}    {a} - {b} = {a - b}    {a} * {b} = {a * b}")
    print_ok(f"{a} / {b} = {a / b}     ← 除法永远得到 float")
    print_ok(f"{a} // {b} = {a // b}    ← 整除，丢掉小数（不是四舍五入）")
    print_ok(f"{a} % {b} = {a % b}     ← 余数")
    print_ok(f"{a} ** {b} = {a ** b}    ← 乘方")

    print_section("类型转换")
    print_ok(f"int(3.9) → {int(3.9)}     ← 直接截断")
    print_ok(f"float(3) → {float(3)}")
    print_ok(f"int('10') → {int('10')}")
    print_counter("int('3.9') 不行，字符串必须是整数字面量", lambda: int("3.9"), ValueError)
    print_ok(f"正确：先 float 再 int → {int(float('3.9'))}")
    print_counter("除以 0", lambda: 1 / 0, ZeroDivisionError)

    print_section("比较")
    print_ok(f"1 == 1.0 → {1 == 1.0}    值相等，类型不同")
    print_trap(
        "浮点会有精度误差",
        f"0.1 + 0.2 == 0.3 → {0.1 + 0.2 == 0.3}（实际是 {0.1 + 0.2}）。钱不要用 float 硬比。",
    )

    print_tip("日常除法用 / ；要整数下标用 //。字符串转整数用 int()，不能直接 int('3.9')。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
