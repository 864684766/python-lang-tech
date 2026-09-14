"""
Python 标准库教材 ④ os / sys（环境变量、import 路径、命令行）

是什么:
  os 管操作系统留给进程的东西；sys 管解释器自己。
  每天碰三个名字: os.environ、sys.path、sys.argv。

用在:
  1) 读 API Key：os.environ.get('ZHIPUAI_API_KEY')（不要写进源码）
  2) 让 import 找到仓库根：sys.path.insert（本仓库每课都在做）
  3) 读命令行：python xx.py --foo → sys.argv[1]

不用:
  拼文件路径优先 pathlib，少用 os.path.join
  密钥不要写进源码

运行: python python_stdlib/ps_4_os_sys.py
"""
from __future__ import annotations

import os
import sys
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


def demo() -> None:
    print("\n" + "=" * 65)
    print("④ os / sys：三个每天会碰到的名字")
    print("=" * 65)

    print_overview(
        "os 管操作系统留给进程的东西；sys 管解释器自己。每天碰 environ / path / argv。",
        [
            "读 API Key：os.environ.get('ZHIPUAI_API_KEY')",
            "让 import 找到仓库根：sys.path.insert",
            "读命令行：python xx.py --foo → sys.argv",
        ],
        "拼文件路径优先 pathlib。密钥不要写进源码。",
    )

    print_section("os.environ：环境变量（一块 dict 一样的表）")
    print("    环境变量 = 操作系统留给进程的键值，不写进源码。API Key 放这里。")
    print("    os.environ['KEY'] = 'x'     写入（本进程有效）")
    print("    os.environ.get('KEY')       读取；没有返回 None，不报错")
    print("    os.environ['KEY']           没有就 KeyError（和 dict 一样）")
    print("    del os.environ['KEY']       删掉")
    print("    对照: JS 的 process.env.KEY；C# 的 Environment.GetEnvironmentVariable")
    os.environ["LANGTEST_DEMO"] = "from-lesson"
    print_ok(f".get('LANGTEST_DEMO') → {os.environ.get('LANGTEST_DEMO')!r}")
    print_ok(f"没有的键 .get → {os.environ.get('NO_SUCH_LANGTEST_KEY')!r}")
    print_counter(
        "没设的键还用 os.environ['KEY']",
        lambda: os.environ["NO_SUCH_LANGTEST_KEY"],
        KeyError,
    )
    print_trap(
        "把 API Key 写进源码",
        "只从环境或 .env 读。load_dotenv() 做的事就是把 .env 填进 os.environ。",
    )
    del os.environ["LANGTEST_DEMO"]

    print_section("sys.path：import 时去哪些目录找 .py")
    print("    sys.path 是 list[str]。import demo_explain 时按这个列表从头搜。")
    print("    sys.path.insert(0, 仓库根)  插到最前面，本课才能 from demo_explain import ...")
    print("    sys.executable            当前这个 python.exe 的路径（别用错 venv）")
    print_ok(f"sys.path[0] = {sys.path[0]!r}")
    print_ok(f"sys.executable = {sys.executable}")

    print_section("sys.argv：命令行参数")
    print("    argv = argument vector。list[str]。")
    print("    argv[0] 永远是脚本路径；你在命令行多写的从 argv[1] 开始。")
    print("    例: python ps_4_os_sys.py --foo  → argv[1] == '--foo'")
    print_ok(f"argv[0] = {sys.argv[0]!r}")
    print_ok(f"argv[1:] = {sys.argv[1:]}")

    print_section("和 pathlib 怎么分工")
    print("    os        环境、工作目录 os.getcwd()、进程")
    print("    sys       解释器：path / argv / exit / version")
    print("    pathlib   拼文件路径、读文件（优先用它，少用 os.path.join）")
    print_ok("读 Key → environ.get；拼 SKILL.md → Path；改 import → sys.path")

    print_tip("sys.exit(1) 会结束整个进程。教材 run_all 里不要用，否则后面的课跑不到。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
