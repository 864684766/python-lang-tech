"""
Python 标准库教材 ① pathlib（打开「已经知道名字」的那一份）

是什么:
  Path 是路径对象，不是字符串。用 / 拼接，再 exists / read_text / write_text。
  对照: JS 的 path.join / C# 的 Path.Combine。

用在:
  1) 已经知道完整路径：skills/translate/SKILL.md
  2) 本仓库每课开头用 Path(__file__) 定位仓库根
  3) 判断在不在、读文本、写文本、删临时文件

不用:
  不知道文件叫什么、要按样子找一批 —— 下一节 glob
  不要用字符串 + 拼路径（Windows 分隔符会 silently 写错）

运行: python python_stdlib/ps_1_pathlib.py
"""
from __future__ import annotations

import sys
from pathlib import Path  # Path = 路径对象，不是字符串

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
    print("① pathlib：Path 是什么、常用方法叫什么")
    print("=" * 65)

    print_overview(
        "用对象表示磁盘上的某个位置，再拼、判断、读、写。打开「已经知道名字」的那一份。",
        [
            "打开已知路径：skills/translate/SKILL.md",
            "本仓库每课用 Path(__file__) 定位仓库根",
            "对照 JS path.join / C# Path.Combine，Python 用 / 拼",
        ],
        "不知道文件叫什么、要按样子找一批 —— 下一节 glob。不要用字符串 + 拼路径。",
    )

    print_section("__file__ / resolve / parent 是什么")
    print("    __file__     本脚本的路径（字符串）。几乎每个教材文件第一段都用它。")
    print("    Path(...)    把字符串变成路径对象，才能 .parent / read_text。")
    print("    .resolve()   展开成绝对路径（去掉 . 和 ..）。")
    print("    .parent      上一级目录。  .parents[1] = 上两级（本课里是仓库根）。")
    here = Path(__file__).resolve()
    print_ok(f"__file__ → {here}")
    print_ok(f".parent     → {here.parent}     （python_stdlib 文件夹）")
    print_ok(f".parents[1] → {here.parents[1]}  （仓库根 langtest）")

    print_section("用 / 拼接：左边必须是 Path")
    print("    写法: 某个 Path / '文件夹名' / '文件名'")
    print("    / 在这里不是除法，是 Path 的拼接运算符（各系统自己选 \\ 还是 /）。")
    skill = here.parents[1] / "skills" / "translate" / "SKILL.md"
    print_ok(f"拼出来 → {skill}")
    print("    .exists()  文件在不在，返回 True/False，不存在也不报错")
    print("    .name      最后一段文件名    SKILL.md")
    print("    .suffix    后缀              .md")
    print("    .stem      去掉后缀的名字    SKILL")
    print_ok(f"exists={skill.exists()}  name={skill.name}  suffix={skill.suffix}")
    print_trap(
        "用字符串 + 拼路径",
        r"Windows 是反斜杠。漏写分隔符会 silently 拼错。"
        " 永远 Path / 'a' / 'b'。",
    )

    print_section("read_text / write_text / unlink")
    print("    read_text(encoding='utf-8')   整文件读成一个 str。中文必须写 encoding。")
    print("    write_text(内容, encoding=)   覆盖写入。没有这个文件会新建。")
    print("    unlink()                      删除这个文件。不存在会 FileNotFoundError。")
    print("    .stat().st_size               字节大小。")
    if skill.exists():
        text = skill.read_text(encoding="utf-8")
        print_ok(f"read_text 前 40 字: {text[:40]!r}")
    else:
        print_ok("没有这份 SKILL.md 就跳过读（所以先 exists）")

    tmp = here.with_name("_pathlib_demo.txt")  # 同目录、换文件名
    tmp.write_text("hello pathlib\n", encoding="utf-8")
    print_ok(f"write_text 后 exists={tmp.exists()}  size={tmp.stat().st_size}")
    tmp.unlink()
    print_ok(f"unlink 后 exists={tmp.exists()}")

    print_counter(
        "文件不存在还 read_text",
        lambda: Path("_no_such_langtest.md").read_text(encoding="utf-8"),
        FileNotFoundError,
    )

    print_tip(
        "本仓库每课开头的 sys.path.insert(0, Path(__file__).resolve().parents[1])"
        " 就是：找到仓库根，好 import demo_explain。通配符找文件见下一节 glob。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
