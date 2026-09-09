"""
Python 类型教材 ② bool（真假）
只有 True 和 False 两个值。注意首字母大写。
对照 JS：true / false（JS 全小写）。
运行: python python_basics/pb_2_bool.py
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
    print("② bool：True / False")
    print("=" * 65)

    print_section("关键字必须大写")
    print_ok(f"True type={type(True)}    False type={type(False)}")
    print_counter("写成 JS 的 true（未定义的名字）", lambda: true, NameError)  # noqa: F821

    print_section("比较会得到 bool")
    print_ok(f"3 > 1 → {3 > 1}")
    print_ok(f"'a' == 'A' → {'a' == 'A'}    ← 区分大小写")

    print_section("逻辑运算（JS 是 && || !）")
    print_ok(f"True and False → {True and False}")
    print_ok(f"True or False → {True or False}")
    print_ok(f"not True → {not True}")
    # and/or 返回的是「其中一个操作数」，不一定是 True/False
    print_trap(
        "1 and 2 得到 2，不是 True",
        f"1 and 2 → {1 and 2}；'' or '缺省' → {'' or '缺省'}。"
        "需要真正的布尔值时包一层 bool()。",
    )

    print_section("if 里什么算假")
    print("    假：None、False、0、0.0、''、[]、()、{}、set()")
    for value in (None, False, 0, "", [], {}, set()):
        print(f"    bool({value!r}) → {bool(value)}")
    print_ok(f"非空都是真：bool('0')={bool('0')}  bool([0])={bool([0])}")
    print_trap(
        "套 JS 习惯：空数组 [] 在 JS 是真",
        "Python 里 if [] 进不去。空列表要用 if not items: 判断。",
    )

    print_tip("True/False 首字母大写。空容器都算假。and/or 可能返回原值而不是布尔。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
