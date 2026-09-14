"""
Python 标准库教材总入口

json → pathlib → glob → copy → os/sys → datetime → logging

单节: python python_stdlib/ps_0_json.py
全部: python python_stdlib/ps_run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_lesson, print_run_plan, print_step

DEMOS = [
    ("ps_0_json.py", "dumps/loads；工具参数和配置"),
    ("ps_1_pathlib.py", "Path 拼路径、读 SKILL.md"),
    ("ps_2_glob.py", "不知道文件名时按样子列出；7_1 扫 SKILL.md"),
    ("ps_3_copy.py", "赋值 vs 浅拷贝 vs deepcopy"),
    ("ps_4_os_sys.py", "环境变量、sys.path、argv"),
    ("ps_5_datetime.py", "UTC / 上海时区、timedelta、和 time 怎么分工"),
    ("ps_6_logging.py", "级别和 logger 名字"),
]


def main() -> None:
    print_lesson(__file__)
    print_run_plan([(name, why) for name, why in DEMOS])
    here = Path(__file__).parent
    for filename, why in DEMOS:
        print_step(filename, why)
        runpy.run_path(str(here / filename), run_name="__main__")
    print("\n标准库七节走完。下一步：python fastapi_basics/fb_run_all.py。\n")


if __name__ == "__main__":
    main()
