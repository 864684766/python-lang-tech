"""
FastAPI 教材总入口（给会一点 Koa 的人）

⓪ 最小接口 → ① 校验 → ② 中间件 / Depends

单节: python fastapi_basics/fb_0_hello.py
全部: python fastapi_basics/fb_run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_lesson, print_run_plan, print_step

DEMOS = [
    ("fb_0_hello.py", "最小 GET：没有 ctx，return dict 就是 JSON"),
    ("fb_1_validation.py", "类型注解和 Pydantic，错了 422"),
    ("fb_2_middleware.py", "洋葱中间件 + Depends 鉴权"),
]


def main() -> None:
    print_lesson(__file__)
    print_run_plan([(name, why) for name, why in DEMOS])
    here = Path(__file__).parent
    for filename, why in DEMOS:
        print_step(filename, why)
        runpy.run_path(str(here / filename), run_name="__main__")
    print("\nFastAPI 三节走完。校验看签名，横切看中间件/Depends。\n")


if __name__ == "__main__":
    main()
