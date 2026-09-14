"""
Python 标准库教材 ③ copy（= 不是备份；浅拷贝不管嵌套）

是什么:
  三层: = 两个名字同一份；.copy / copy.copy 只复制外层；deepcopy 嵌套也新造。
  对照: C# MemberwiseClone 是浅的；JS 的 {...obj} 也是浅的。

用在:
  1) 图 state / messages 要改副本、不能动原对象 → deepcopy
  2) 没有嵌套的 list/dict → .copy() 就够

不用:
  str / int / tuple 不可变，赋值即可。不要用 = 当「备份」

运行: python python_stdlib/ps_3_copy.py
"""
from __future__ import annotations

import copy  # 模块名 copy。里面两个函数：copy.copy / copy.deepcopy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_lesson,
    print_ok,
    print_overview,
    print_section,
    print_tip,
    print_trap,
)


def demo() -> None:
    print("\n" + "=" * 65)
    print("③ copy：三个层次，别混")
    print("=" * 65)

    print_overview(
        "复制有三层：= 是两个名字同一份；.copy 只复制外层；deepcopy 嵌套也新造。",
        [
            "图 state / messages 要改副本、不能动原对象 → deepcopy",
            "没有嵌套的 list/dict → .copy() 就够",
        ],
        "str / int / tuple 不可变，赋值即可。不要用 = 当「备份」。",
    )

    print_section("b = a：两个名字，同一份对象")
    print("    = 不复制。改 b 就是改 a。类型课 pb_5 的 list 已经见过。")
    a = {"messages": [{"role": "user", "content": "hi"}]}
    b = a
    b["messages"].append({"role": "assistant", "content": "ok"})
    print_trap("b = a 再改 b", f"a 也变了: a={a}")

    print_section("浅拷贝：copy.copy(x) 或 x.copy()")
    print("    shallow = 浅。只新建最外层那个 dict/list。")
    print("    里面的嵌套 list、dict 仍是同一份。")
    print("    dict.copy() / list.copy() 和 copy.copy(x) 效果一样，都是浅的。")
    state = {"messages": [{"role": "user", "content": "hi"}], "step": "start"}
    shallow = state.copy()
    shallow["step"] = "next"  # 外层键，两份分开了
    print_ok(f"改外层 step: 原={state['step']!r}  拷贝={shallow['step']!r}")
    shallow["messages"].append({"role": "tool", "content": "晴"})
    print_trap(
        "浅拷贝后改嵌套 list",
        f"原 state['messages'] 也多了一条: {state['messages']}",
    )

    print_section("深拷贝：copy.deepcopy(x)")
    print("    deep = 深。嵌套多少层都新造一份。")
    print("    参数 = 任意对象；返回值 = 结构相同、互不影响的新对象。")
    origin = {"messages": [{"role": "user", "content": "hi"}], "step": "start"}
    deep = copy.deepcopy(origin)
    deep["messages"].append({"role": "assistant", "content": "ok"})
    print_ok(f"改 deep 后 origin 仍是 {origin['messages']}")
    print_ok(f"deep = {deep['messages']}")

    print_section("怎么选")
    print("    没有嵌套的 list/dict     x.copy() 够了")
    print("    messages / 图 state      copy.deepcopy")
    print("    str / int / tuple        不可变，赋值即可，不用拷")
    print_ok("图状态被原地改坏，十次有九次是浅拷贝了嵌套 messages")

    print_tip("记住: = 同对象；.copy 浅；deepcopy 深。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
