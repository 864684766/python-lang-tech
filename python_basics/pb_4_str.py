"""
Python 类型教材 ④ str（字符串）
文字。单引号双引号都行。字符串不可变：方法会返回新串，不改原来那份。
对照 JS：string。
运行: python python_basics/pb_4_str.py
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
    print("④ str：字符串")
    print("=" * 65)

    print_section("怎么写")
    a = "hello"
    b = "mcp"  # 单引号双引号一样
    print_ok(f"{a!r}  type={type(a)}")

    print_section("拼接、重复、f-string")
    print_ok(f"拼接: {a + ' ' + b}")
    print_ok(f"重复: {'na' * 2}")
    name, n = "get_me", 3
    print_ok(f"f-string: 工具 {name}，共 {n} 个    ← 等于 JS 的 `工具 ${{name}}`")

    print_section("常用方法（都返回新串）")
    raw = "  Get_Me  \n"
    print_ok(f"原串 {raw!r}")
    print_ok(f"strip 去两端空白 → {raw.strip()!r}")
    print_ok(f"lower → {raw.strip().lower()!r}")
    print_trap(
        "只调用 strip、不接回来",
        f"raw.strip() 之后 raw 仍是 {raw!r}。必须 raw = raw.strip()。",
    )

    print_section("切和拼、下标")
    parts = "owner,repo".split(",")
    print_ok(f"split(',') → {parts}    join → {'/'.join(parts)}")
    s = "python"
    print_ok(f"s[0]={s[0]}  s[-1]={s[-1]}  s[0:3]={s[0:3]!r}  len={len(s)}")
    print_ok(f"'Me' in 'Get_Me' → {'Me' in 'Get_Me'}    ← JS 的 includes")
    print_counter("下标超出长度", lambda: s[100], IndexError)

    def mutate_str() -> None:
        s[0] = "P"  # type: ignore[index]

    print_counter("字符串不能改某一个字符（不可变）", mutate_str, TypeError)

    print_tip("方法不改原串。要留下结果必须赋值：s = s.strip()")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
