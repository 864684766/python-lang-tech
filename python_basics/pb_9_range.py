"""
Python 类型教材 ⑨ range
「从哪数到哪」的整数序列，常用在 for 循环里。本身不是 list。
运行: python python_basics/pb_9_range.py
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
    print("⑨ range：数次数")
    print("=" * 65)

    print_section("含头不含尾")
    r = range(3)  # 0, 1, 2
    print_ok(f"range(3) → {r}    type={type(r)}")
    print_ok(f"变成 list 才能一眼看清: {list(r)}")
    print_trap(
        "以为 range(3) 会包含 3",
        f"实际是 {list(range(3))}，没有 3。和切片 [0:3] 一样，右边是开区间。",
    )

    print_section("for 循环")
    print("    for i in range(3):")
    for i in range(3):
        print(f"      i = {i}")
    print_ok(f"range(2, 6) → {list(range(2, 6))}     从 2 到 5")
    print_ok(f"range(0, 10, 3) → {list(range(0, 10, 3))}  步长 3")
    print_ok(f"3 in range(10) → {3 in range(10)}")

    print_section("常见报错")
    print_counter("range 不能和 list 直接相加", lambda: range(3) + [3, 4], TypeError)
    print_ok(f"要拼接先变成 list: {list(range(3)) + [3, 4]}")

    print()
    print("  —— 十种类型走完了 ——")
    print("  必会：None / bool / int / float / str / list / tuple / dict / set / range")
    print("  接着：⑩ enumerate  ⑪ array/标准库  ⑫ NumPy ndarray")
    print("  或：python python_basics/pb_run_all.py 一次跑完")

    print_tip("range(n) 是 0 到 n-1。要看里面有谁，包一层 list()。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
