"""
Python 标准库教材 ② glob（按「文件长什么样」一次找出一堆路径）

是什么:
  你只知道「这类文件」，不知道每一份的完整路径。
  glob = 按通配符扫目录，返回路径列表（还没读内容）。
  对照终端: Windows 的 dir *.py / Linux 的 ls *.py。
  对照上一节: pathlib 打开「已经知道名字」的那一份；glob 先把名字找出来。

用在:
  1) 只知道文件夹，不知道里面每份文件叫什么 —— 列出 python_basics/pb_1*.py
  2) 本仓库 7_1：扫 skills 下每一层的 SKILL.md（不知道有几个技能、叫什么）
  3) 批量处理：先找出所有 .md / .json，再逐个 read_text

不用:
  已经知道「就是这一份」—— 用 pathlib 直接 Path(...) / read_text
  不是正则：只认识 * ? **，不要拿来解析一段文本

运行: python python_stdlib/ps_2_glob.py
"""
from __future__ import annotations

import glob  # 模块 glob：glob.glob("模式") 返回字符串路径列表
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
    print("② glob：不知道文件名时，按样子一次列出")
    print("=" * 65)

    print_overview(
        "按通配符扫目录，得到路径列表。终端里 dir *.py 做的就是这件事。",
        [
            "只知道文件夹、不知道每份叫什么：列出 python_basics/pb_1*.py",
            "本仓库 7_1：扫 skills 下所有 SKILL.md（技能名事先不知道）",
            "批量处理：先找出全部 .md / .json，再逐个 read_text",
        ],
        "已经知道「就这一份」用 pathlib 直接打开。glob 不是正则，别拿来解析文本。",
    )

    root = Path(__file__).resolve().parents[1]
    basics = root / "python_basics"

    print_section("先对照：你在资源管理器 / 终端里已经会的事")
    print("    Windows:  dir python_basics\\pb_1*.py")
    print("    Linux:    ls python_basics/pb_1*.py")
    print("    意思都是：这个文件夹里，名字像 pb_1 开头、.py 结尾的，全部列出来。")
    print("    你不会手写 pb_1_none.py、pb_10_enumerate.py……因为你记不住、也会新增。")
    print("    Python 里同一件事，写成 Path.glob('pb_1*.py')。")

    print_section("场景 1：教材脚本按前缀批量找到")
    print("    需求: 仓库里「类型课第 1x 节」有哪些文件？")
    print("    不知道完整文件名，只知道前缀 pb_1、后缀 .py。")
    from_path = sorted(basics.glob("pb_1*.py"))
    print_ok(f"basics.glob('pb_1*.py') → {[p.name for p in from_path]}")
    print("    返回值是路径对象列表。还没有读文件。下一行才读第一份的前几个字。")
    if from_path:
        first = from_path[0]
        preview = first.read_text(encoding="utf-8").splitlines()[0]
        print_ok(f"接到 pathlib: {first.name} 第一行 = {preview!r}")

    print_section("场景 2：跨目录找「叫这个名字的文件」（7_1 就是这样）")
    print("    需求: skills 下面每个技能一份 SKILL.md，层数不固定。")
    print("    你不知道有几个技能、文件夹叫什么，只知道叶子文件叫 SKILL.md。")
    print("    7_1 真实写法:  for skill_md in self.root.rglob('SKILL.md')")
    print("    rglob(模式) = 从这一层往下、任意深度，等价 glob('**/SKILL.md')。")
    print("    这里从 skills/ 扫，不要从仓库根扫：根下面还有 .venv，会掺进第三方的 SKILL.md。")
    skills_dir = root / "skills"
    skills = sorted(skills_dir.rglob("SKILL.md"))
    print_ok(f"skills.rglob('SKILL.md') → {[str(p.relative_to(root)) for p in skills]}")

    print_section("两种 API：模块 glob vs Path.glob（结果一样，写法不同）")
    print("    glob.glob(模式字符串)  → list[str]     老写法，模式必须是字符串")
    print("    某个 Path.glob(模式)   → 迭代器[Path]  新写法，推荐；和上一节 Path 一家")
    print("    参数「模式」不是正则。只认识 * ? ** 这几个通配符。")
    from_mod = glob.glob(str(basics / "pb_1*.py"))
    print_ok(f"glob.glob(...) → {[Path(p).name for p in from_mod]}")
    print_ok(f"Path.glob(...) → {[p.name for p in from_path]}")

    print_section("通配符只认识这三个")
    print("    *     本层任意一段字符     pb_*.py     能配 pb_1_none.py 也能配 pb_10_....py")
    print("    ?     恰好一个字符         pb_?_*.py   配 pb_1_none.py，配不到 pb_10_（两位数字）")
    print("    **    跨任意层目录         skills/**/SKILL.md")
    print("    Path.relative_to(root)    把绝对路径裁成相对仓库根，打印好看")
    one_char = sorted(basics.glob("pb_?_*.py"))
    print_ok(f"pb_?_*.py（数字恰好一位）→ {[p.name for p in one_char]}")
    print_ok("pb_10_enumerate.py 不在上面：? 只吃一个字符，10 是两位")

    print_section("反例：以为 * 会进子文件夹")
    print("    recursive=False（默认）只搜这一层，* 不会进子文件夹。")
    print("    要跨目录用 ** 或 Path.rglob，不要指望单层 *。")
    only_here = glob.glob(str(root / "python_basics" / "pb_0*.py"), recursive=False)
    print_ok(f"只搜 python_basics 这一层: {[Path(p).name for p in only_here]}")
    print_trap(
        "写成 skills/*.md，以为能拿到 skills/translate/SKILL.md",
        "单层 * 停在 skills 这一层。要 skills/**/SKILL.md 或 skills.rglob('SKILL.md')。",
    )

    print_tip(
        "记住分工: pathlib 打开「已知的那一份」；glob/rglob 先把「这类文件」找齐。"
        " 扫到路径之后还要自己 path.read_text()。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
