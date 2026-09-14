"""
Python 语言教材总入口

⓪ 面向对象 → ① typing 标注 → ② with/contextlib → ③ asyncio

单节: python python_lang/pl_0_oop.py
全部: python python_lang/pl_run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_lesson, print_run_plan, print_step

DEMOS = [
    ("pl_0_oop.py", "封装 / 继承 / 多态；和 C# 的差别；dataclass vs Enum"),
    ("pl_1_typing.py", "类型标注 + typing：Optional、TypedDict、Annotated"),
    ("pl_2_contextlib.py", "with 保证开关；@contextmanager / suppress"),
    ("pl_3_asyncio.py", "async / await / gather / async with"),
]


def main() -> None:
    print_lesson(__file__)
    print_run_plan([(name, why) for name, why in DEMOS])
    here = Path(__file__).parent
    for filename, why in DEMOS:
        print_step(filename, why)
        runpy.run_path(str(here / filename), run_name="__main__")
    print("\n语言四节走完。下一步：python python_stdlib/ps_run_all.py（json / 路径 / 时间 / 日志）。\n")


if __name__ == "__main__":
    main()
