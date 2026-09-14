"""
Python 语言教材 ② with 和 contextlib

是什么:
  with：进入时打开资源，离开时一定关掉，中途报错也关。
  contextlib 是制造 with 的工具箱（@contextmanager / suppress / ExitStack）。

用在:
  1) 读文件：with path.open() as f
  2) MCP / 数据库会话：用完必须断开
  3) 个数不定的多把资源：ExitStack

不用:
  没有「打开/关闭」这对动作（纯计算、拼字符串）不必套 with
  手动 close 碰到 return / 异常会漏关

运行: python python_lang/pl_2_contextlib.py
"""
from __future__ import annotations

import sys
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

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


@contextmanager
def speak(name: str):
    """yield 之前 = 进入 with；之后 = 离开（包括报错）。"""
    print(f"    进入: 打开 {name}")
    try:
        yield name
    finally:
        print(f"    离开: 关闭 {name}")


def demo() -> None:
    print("\n" + "=" * 65)
    print("② with / contextlib：进入时开，离开时关")
    print("=" * 65)

    print_overview(
        "进入时打开资源，离开时一定关掉，中途报错也关。",
        [
            "读文件：with path.open() as f",
            "MCP / 数据库会话：用完必须断开",
            "个数不定的多把资源：ExitStack",
        ],
        "没有「打开/关闭」这对动作不必套 with。手动 close 碰到异常会漏关。",
    )

    print_section("内置：with 打开文件")
    path = Path(__file__).with_name("_demo_skill.md")
    path.write_text("# demo\n", encoding="utf-8")
    with path.open(encoding="utf-8") as f:
        text = f.read()
    print_ok(f"读到 {text!r}    出了 with，文件已经关")
    path.unlink()

    print_section("自己造 with：@contextmanager")
    with speak("session") as n:
        print_ok(f"with 里面拿到 yield 的值: {n}")

    print_section("报错也会走 finally")
    try:
        with speak("broken"):
            raise RuntimeError("模拟工具失败")
    except RuntimeError as e:
        print_ok(f"里面炸了: {e!s}    但上面已经打印了「离开」")

    print_section("suppress：某种错直接吞掉")
    with suppress(FileNotFoundError):
        Path("_no_such_file_langtest.txt").unlink()
    print_ok("文件不存在也不报错。少写一层 try/except")

    print_trap(
        "打开资源不 with，只靠手动 close",
        "中途 return / 异常会漏关。文件、锁、MCP session 都用 with / async with。",
    )

    def after_with():
        p = Path(__file__).with_name("_demo_skill.md")
        p.write_text("x", encoding="utf-8")
        with p.open(encoding="utf-8") as f:
            pass
        p.unlink()
        return f.read()

    print_counter("出了 with 再读同一个文件对象", after_with, ValueError)

    print_section("几个资源：嵌套 with")
    with speak("math"):
        with speak("github"):
            print_ok("先开 math，再开 github；离开时反过来：先关 github，再关 math")

    print_section("个数不一定时：ExitStack")
    print("    嵌套 with 要事先知道有几层。要连 1 个还是 3 个 MCP，写不成固定缩进。")
    with ExitStack() as stack:
        a = stack.enter_context(speak("math"))
        b = stack.enter_context(speak("github"))
        print_ok(f"进栈两把: {a}, {b}    出这一个 with 时，后进的先关")
    print_ok("AsyncExitStack 就是异步版，写法一样，换成 async with。MCP 连多个 Server 时会看到")
    print("    @asynccontextmanager        造「能 async with 的东西」，见 ③ asyncio")
    print_ok("contextlib 不是一种数据，是制造 with 的工具箱")

    print_tip(
        "看到 with / async with 就想：进入准备，离开清理。"
        " FastAPI lifespan、MCP ClientSession 都是这套。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
