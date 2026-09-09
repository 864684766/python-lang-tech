"""
Python 类型教材 ⑧ set（集合）
一堆不重复、没有顺序的值。对照 JS：Set。
运行: python python_basics/pb_8_set.py
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
    print("⑧ set：集合（可变，去重，无顺序）")
    print("=" * 65)

    print_section("去重、增删、成员")
    tags = {"http", "stdio", "http"}  # 重复的只会留一份
    print_ok(f"{tags}    type={type(tags)}  len={len(tags)}")
    tags.add("sse")
    print_ok(f"add 后 {tags}")
    tags.discard("stdio")  # 没有也不报错；.remove 没有会报错
    print_ok(f"discard 后 {tags}")
    print_ok(f"'http' in tags → {'http' in tags}")

    print_section("并 / 交 / 差")
    a, b = {1, 2, 3}, {3, 4}
    print_ok(f"并集 a|b → {a | b}")
    print_ok(f"交集 a&b → {a & b}")
    print_ok(f"差集 a-b → {a - b}    （在 a 不在 b）")

    print_section("空集合和去重")
    print_trap(
        "{} 是空字典，不是空集合",
        f"type({{}})={type({})}。空集合必须写 set() → {set()!r}",
    )
    names = ["a", "b", "a"]
    print_ok(f"列表去重 list(set(...)) → {list(set(names))}    顺序不保证")

    print_section("常见报错")
    print_counter("set 没有下标，不能 tags[0]", lambda: tags[0], TypeError)
    print_counter("list 不能放进 set（不可哈希）", lambda: {1, [2, 3]}, TypeError)
    print_counter("remove 一个不存在的元素", lambda: tags.remove("没有"), KeyError)

    print_tip("空集合写 set()。要按下标取用 list。set 里只能放不可变的东西。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
