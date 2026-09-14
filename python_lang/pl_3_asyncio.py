"""
Python 语言教材 ③ asyncio

是什么:
  单线程里交错等 I/O。写 async def / await，不是开很多系统线程。
  入口：asyncio.run(协程)。只写 tick() 不会跑。

用在:
  1) 同时等两个慢请求：asyncio.gather（看时间戳：快的 B 会先结束）
  2) FastAPI 中间件 / MCP ClientSession / LangChain ainvoke
  3) 对照 JS/C#：一调用 async 函数就跑到第一个 await；Python 要对齐用 create_task

不用:
  纯 CPU 计算不会因为 async 变快
  普通脚本里 time.sleep 就够，不必上事件循环
  async 函数里不要 time.sleep（会卡住整条循环）

运行: python python_lang/pl_3_asyncio.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
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


async def tick(name: str, seconds: float, t0: float) -> str:
    print(f"    {time.perf_counter() - t0:.2f}s  {name} 开始等 {seconds:.2f}s")
    await asyncio.sleep(seconds)
    print(f"    {time.perf_counter() - t0:.2f}s  {name} 结束")
    return name


@asynccontextmanager
async def session(name: str):
    print(f"    async 进入: {name}")
    try:
        yield name
    finally:
        print(f"    async 离开: {name}")


async def demo_async() -> None:
    print_section("async def + await：写在后面的，要等前面的醒")
    t0 = time.perf_counter()
    got = await tick("lookup", 0.05, t0)
    print_ok(f"await 的返回值 → {got!r}    这一行在 lookup 结束之后才执行")

    print_section("顺序 await：先 a 整段结束，才轮到 b")
    t0 = time.perf_counter()
    await tick("慢的A", 0.20, t0)
    await tick("快的B", 0.05, t0)
    sequential = time.perf_counter() - t0
    print_ok(f"总耗时 ≈ {sequential:.2f}s    结束顺序一定是 A 然后 B（B 再快也得等 A）")

    print_section("gather：两个一起等，谁先醒谁先打印结束")
    t0 = time.perf_counter()
    names = await asyncio.gather(
        tick("慢的A", 0.20, t0),
        tick("快的B", 0.05, t0),
    )
    parallel = time.perf_counter() - t0
    print_ok(f"总耗时 ≈ {parallel:.2f}s    看上面：B 在 0.05s 结束，A 在 0.20s 结束")
    print_ok(f"gather 的返回值仍按参数顺序 {names}    不是按谁先结束")

    print_section("和 JS / C#（三句）")
    print("    1. await A; await B  在 Python / JS / C# 都是 A 整段完才开始 B。")
    print("    2. JS/C# 一调用 async 函数就跑到第一个 await；Python 只写 tick() 不会跑。")
    print("    3. Python 要「先点火再交错」用 create_task，对应 JS 里不写 await 直接 tick()。")
    t0 = time.perf_counter()
    ta = asyncio.create_task(tick("A", 0.20, t0))
    tb = asyncio.create_task(tick("B", 0.05, t0))
    print_ok("第 3 句的例子：这行在 A/B「开始」之前打印")
    await ta
    await tb

    print_section("async with = 异步的 with")
    async with session("mcp") as n:
        print_ok(f"会话里: {n}")

    print_section("不能在普通 def 里直接 await")
    print_ok("入口用 asyncio.run(协程)。本文件最底下就是")


def demo() -> None:
    print("\n" + "=" * 65)
    print("③ asyncio：async / await / gather / async with")
    print("=" * 65)

    print_overview(
        "单线程里交错等 I/O。写 async def / await，不是开很多线程。",
        [
            "同时等两个慢请求：asyncio.gather",
            "FastAPI 中间件 / MCP ClientSession / LangChain ainvoke",
            "入口：asyncio.run(协程)。只写 tick() 不会跑",
        ],
        "纯 CPU 计算不会因为 async 变快。async 函数里不要 time.sleep。",
    )

    print_ok("asyncio.run 开始：会把 demo_async 整段跑完（含所有 await）才回来")
    asyncio.run(demo_async())
    print_ok("asyncio.run 结束：现在才到 print_trap。50-51 行早就跑完了")

    print_trap(
        "在 async 函数里调用 time.sleep 做等待",
        "time.sleep 会卡住整条事件循环。等 I/O 用 await asyncio.sleep。",
    )
    def add_coro():
        c = tick("x", 0, time.perf_counter())
        try:
            return c + "!"
        finally:
            c.close()

    print_counter("把没 await 的协程当字符串用", add_coro, TypeError)
    pending = tick("x", 0, time.perf_counter())
    print_ok(f"只写 tick(...) 得到 {pending!r}    要 await 或 asyncio.run 才会执行")
    pending.close()

    print_section("仓库里已经出现、但没有当课讲的")
    print("    FastAPI fb_2 中间件写成 async def —— 框架要求，那节没讲事件循环")
    print("    LangChain 5_3 是 ainvoke / abatch，不是 asyncio 语法课")
    print("    MCP 7_2 才大量 async with + asyncio.run")
    print_ok("语法在本节；接到 HTTP / 模型 / MCP 时再对号入座")

    print_tip(
        "async def 声明协程；await 等 I/O；asyncio.run 开事件循环。"
        " CPU 死循环不会因为 async 变快。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
