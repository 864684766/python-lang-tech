"""
Python 语言教材 ① 类型标注和 typing 库

是什么:
  给参数/返回值写说明书。语法像 TypeScript。
  普通函数运行时不拦；@tool / FastAPI / LangGraph 会在运行时读标注。

用在:
  1) 给自己和编辑器看：def lookup(term: str) -> str
  2) 给框架读：@tool 的参数、FastAPI 的 422、LangGraph 的 Annotated
  3) 给 dict 规定键：TypedDict（造值仍写花括号，不能点属性）

不用:
  别指望 : str 能挡住 echo(123)
  要运行时拦脏数据，用 Pydantic（FastAPI fb_1）或自己 isinstance

运行: python python_lang/pl_1_typing.py
"""
from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict, get_args, get_origin, get_type_hints

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_counter,
    print_lesson,
    print_ok,
    print_overview,
    print_section,
    print_tip,
    print_trap,
)


def lookup(term: str) -> str:
    """① 最像 TS：参数 str，返回 str。"""
    return term.lower()


def echo(x: str) -> str:
    """原样返回。不用 .lower()，避免「数字没有该方法」掩盖「标注不校验」。"""
    return x


def take_stop(stop: Optional[list[str]] = None) -> int:
    """② Optional[list[str]] = list[str] | None。"""
    return 0 if stop is None else len(stop)


class CounterState(TypedDict):
    """给 dict 规定键。运行时仍是普通 dict，不是 class 实例。"""

    visited_nodes: Annotated[list, operator.add]
    current_step: str


def merge_visited(old: list, new: list) -> list:
    """LangGraph 对 Annotated[..., operator.add] 做的事：list 用 + 拼。"""
    return operator.add(old, new)


def demo() -> None:
    print("\n" + "=" * 65)
    print("① 类型标注 / typing：像 TS，但默认不校验")
    print("=" * 65)

    print_overview(
        "给参数/返回值写说明书。语法像 TS。普通函数运行时不拦。",
        [
            "给自己和编辑器看：def lookup(term: str) -> str",
            "给框架读：@tool / FastAPI / LangGraph 的 Annotated",
            "给 dict 规定键：TypedDict（造值仍写花括号）",
        ],
        "别指望 : str 能挡住 echo(123)。要运行时拦脏数据，用 Pydantic 或 isinstance。",
    )

    print_section("标注写在哪")
    print_ok(f"lookup('ReAct') → {lookup('ReAct')!r}")
    print_ok("写法: def lookup(term: str) -> str    对照 TS: (term: string) => string")

    print_section("Python 默认不按标注拦")
    print_ok(f"echo 写了 x: str，仍能 echo(123) → {echo(123)!r}    没有 TypeError")
    print_ok("lookup(123) 会炸，是因为 123 没有 .lower()，不是 : str 在拦")
    print_trap(
        "当成 C# / 严格 TS：写了 : str 就会编译失败",
        "解释器不管。要拦脏数据用 Pydantic（FastAPI fb_1）或自己 isinstance。",
    )

    print_section("Optional / 联合 / 列表")
    print_ok(f"stop=None → {take_stop()}")
    print_ok(f"stop=['\\n'] → {take_stop(['\\n'])}")
    print_ok("新写法 list[str] | None 和 Optional[list[str]] 一个意思")
    box: Any = 123
    print_ok(f"Any 放弃检查，什么都能塞: box={box!r}")
    print_ok("源码里占位用 Any；自己能写清楚就写清楚")

    print_section("TypedDict：dict 的接口，不是能 new 的类")
    state: CounterState = {"visited_nodes": ["A"], "current_step": "a"}
    print_ok(f"正确：花括号造 dict → {state}    type={type(state)}")
    print_ok(f"取值用中括号: state['current_step']={state['current_step']!r}")
    print_counter(
        "当成 class / dataclass 去点属性",
        lambda: state.current_step,  # type: ignore[attr-defined]
        AttributeError,
    )
    print_trap(
        "写成 CounterState('A')，像 C# 的 new CounterState(\"A\")",
        "TypedDict 没有构造函数。造值永远写 {'visited_nodes': [...], 'current_step': '...'}。",
    )

    print_section("Annotated：类型后面夹带私货")
    hint = get_type_hints(CounterState, include_extras=True)["visited_nodes"]
    print_ok(f"get_origin={get_origin(hint)}  get_args={get_args(hint)}")
    print_ok("有 from __future__ import annotations 时，要用 get_type_hints 才能拿到真正的 Annotated")
    print_ok(f"合并 ['A'] + ['B'] → {merge_visited(['A'], ['B'])}")
    print_trap(
        "把 Annotated 只当成注释",
        "LangGraph 会读第二段 operator.add：节点 return 列表时是追加，不是覆盖。",
    )

    def apply(fn: Callable[[str], str], text: str) -> str:
        return fn(text)

    print_section("Callable：函数也有类型")
    print_ok(f"Callable[[str], str]  ≈  TS (s: string) => string  → {apply(str.upper, 'mcp')}")

    print_section("和框架怎么接")
    print("    @tool 的参数标注 → 运行时生成 JSON schema（模型填参）")
    print("    FastAPI 的参数标注 → 运行时校验，错了 422")
    print("    普通函数的标注 → 给人看；mypy/Pylance 可选检查")
    print_ok("同一套语法，三档用途。看到 Annotated 先问：谁在运行时读它？")

    print_tip(
        "from __future__ import annotations 让标注推迟求值，方便写 str | None。"
        " 本仓库几乎每个文件第一行都有。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
