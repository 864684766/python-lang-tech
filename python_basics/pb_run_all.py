"""
Python 类型教材总入口（给完全新手）

先 ⓪ 认地图，再 ①～⑨ 每种类型单独跑一遍。

单节: python python_basics/pb_0_types_overview.py
全部: python python_basics/pb_run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_lesson, print_run_plan, print_step

DEMOS = [
    ("pb_0_types_overview.py", "先看 Python 有哪些常用类型"),
    ("pb_1_none.py", "None：没有值"),
    ("pb_2_bool.py", "bool：True / False，以及什么算假"),
    ("pb_3_int_float.py", "int / float：整数和小数"),
    ("pb_4_str.py", "str：字符串"),
    ("pb_5_list.py", "list：可变列表"),
    ("pb_6_tuple.py", "tuple：不可变的一组值"),
    ("pb_7_dict.py", "dict：键值对"),
    ("pb_8_set.py", "set：去重集合"),
    ("pb_9_range.py", "range：数次数"),
]


def main() -> None:
    print_lesson(__file__)
    print_run_plan([(name, why) for name, why in DEMOS])
    here = Path(__file__).parent
    for filename, why in DEMOS:
        print_step(filename, why)
        runpy.run_path(str(here / filename), run_name="__main__")
    print("\n类型走完了。下一步：python fastapi_basics/fb_run_all.py（校验和中间件）。\n")


if __name__ == "__main__":
    main()
