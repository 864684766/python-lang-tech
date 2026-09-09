"""
Python 类型教材 ① None
「没有值」只有这一种写法：None。类型叫 NoneType。
对照 JS：null 和 undefined 在 Python 里都对应 None。
运行: python python_basics/pb_1_none.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_lesson, print_ok, print_section, print_tip, print_trap


def demo() -> None:
    print("\n" + "=" * 65)
    print("① None：没有值")
    print("=" * 65)

    print_section("长什么样")
    x = None
    print_ok(f"x = {x}    type={type(x)}")

    print_section("怎么判断「是不是没有」")
    # is 问的是：是不是同一个对象。None 全项目只有一份，所以用 is。
    print_ok(f"x is None → {x is None}")
    print_ok(f"x is not None → {x is not None}")

    print_section("哪些地方会冒出 None")

    def do_nothing():
        pass  # 没写 return，Python 自动 return None

    print_ok(f"函数没写 return → {do_nothing()}")
    data = {"name": "mcp"}
    print_ok(f"dict.get 找不到键 → {data.get('url')}")

    print_section("反例：空字符串不是 None")
    name = ""
    print_trap(
        "if not name 会把 ''、0、[] 都当成「没有」",
        f"name={name!r}  is None={name is None}  bool(name)={bool(name)}。"
        "「没拿到值」用 is None；「是不是空串」用 == ''。",
    )
    print_trap(
        "写成 == None 能跑，但不地道",
        "PEP 8 要求用 is None。新手统一写 is / is not 即可。",
    )

    print_tip("问「有没有这个东西」用 is None；问「是不是空串/空列表」要另写判断。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
