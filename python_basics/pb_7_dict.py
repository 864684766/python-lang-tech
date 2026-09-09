"""
Python 类型教材 ⑦ dict（字典）
键 → 值。花括号。对照 JS：普通对象 / Map。key 通常是字符串。
运行: python python_basics/pb_7_dict.py
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
    print("⑦ dict：字典（可变）")
    print("=" * 65)

    print_section("创建和读取")
    user = {"name": "alice", "n_tools": 3}
    print_ok(f"{user}    type={type(user)}")
    print_ok(f"键一定在时用 [] → {user['name']}")
    print_ok(f"键可能不在用 .get → {user.get('url')}    ← 没有就是 None，不报错")
    empty: dict = {}
    print_ok(f"带默认值 .get('url', empty) → {user.get('url', empty)}")

    print_section("in 查的是键，不是值")
    print_ok(f"'name' in user → {'name' in user}")
    print_trap(
        "用 in 去查值",
        f"'alice' in user → {'alice' in user}（False，那是值）。"
        f"'alice' in user.values() → {'alice' in user.values()}",
    )

    print_section("写入、删除、遍历")
    user["url"] = "https://example.com"
    print_ok(f"写入后 keys: {list(user.keys())}")
    del user["url"]
    print_ok(f"del 之后 {user}")
    print("    遍历 items：")
    for k, v in user.items():
        print(f"      {k} = {v}")

    print_section("合并（JS 的 ...展开）")
    merged = {**{"transport": "http"}, **{"url": "https://api.example.com"}}
    print_ok(f"**展开合并 → {merged}")

    print_section("常见报错")
    print_counter("键不存在还用 []", lambda: user["missing"], KeyError)
    print_counter("del 一个没有的键", lambda: user.__delitem__("missing"), KeyError)

    print_tip("不确定键在不在，用 .get，不要 user['x']。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
