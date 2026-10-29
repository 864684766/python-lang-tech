"""
Harness 教材 ⑬ 先给目录，再按需打开

hs_5 讲的是已经产生的交互记录怎么选取。本课讲外部资料什么时候加载，
两者处理的是不同来源的上下文。系统提示如果本身就是一本百科，第一轮就可能很长。

这一节看常驻提示里放什么。

先看一个现象。三份材料都在：

    地图 = "记笔记: skills/note.md\n架构: docs/rules.md"          # 大约一百字
    note = "先读进度再落笔"
    架构长文 = "先读进度再落笔" * 80                         # 一千多字

启动时如果拼成一个系统提示：

    system = 地图 + note + 架构长文

用户只问笔记规则，那一千多字也在第一轮里。模型还没开始，提示词已经很长。

这里的问题是：每次都载入全部正文，短地图就没用了。模型没法只打开它需要的那一份。

以前维护一个越来越长的说明文件，每次会话从头读到尾。文件越完整，每一轮越满。

这一节常驻的只有地图。正文留在文件里，模型调用工具才打开一份：

    system = 地图
    open_doc("skills/note.md")   # 返回 "先读进度再落笔"
    open_doc("docs/rules.md")    # 才会返回那一千多字；本课的演示不调用这一次

第一轮的工具结果里有笔记规则，没有那段重复几十遍的架构长文。

反面教材：地图只有几行路径，skills/note.md 里是一句「先读进度再落笔」，docs/rules.md 是把这句话重复几十遍的架构长文。启动时把地图和两份正文拼成一个系统提示再发给模型。用户只问笔记规则，架构长文也占满了第一轮。模型即使不再调用 open_doc，也已经读过不该常驻的正文。正确的第一轮只有地图；open_doc("skills/note.md") 的返回里有笔记规则，没有那段架构长文。

影响：Skill 是按需打开的第二层，不必写进每次都在的系统提示。

运行: python harness/hs_13_disclosure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import live_model, print_harness_job
from harness.mini import Runner, called_tools

NOTE_RULE = "RULE-NOTE-先读进度再落笔"
ARCH_BODY = "RULE-ARCH-架构长文。" * 80
FILES = {
    "skills/note.md": NOTE_RULE,
    "docs/rules.md": ARCH_BODY,
}
MAP = """这是地图，不是全文。
记笔记规则: skills/note.md
架构说明: docs/rules.md
需要细则时调用 open_doc，path 必须是上面某一行里的相对路径。
没有打开的文件，就当看不见。
"""


def stuffed() -> str:
    return MAP + "\n" + "\n".join(FILES.values())


@tool
def open_doc(path: str) -> str:
    """按地图上的相对路径读取一个文件。"""
    if path not in FILES:
        return "地图上没有这个路径: " + path
    return FILES[path]


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑬ 先给目录，再按需打开")
    print("=" * 65)

    print_harness_job(
        "第一轮只放短地图。长文留在文件里，等 open_doc 按路径取",
        prop="一份 6 行地图 + 两份正文",
    )

    print_idea(
        phenomenon="规矩、架构、技能各写了一份。全贴进系统提示后，真正的任务被挤到后面。",
        gap="入口是地图：路径和一句话。正文是第二层，调用时才加载。",
        before="维护一个越来越长的说明文件，每次会话从头读到尾。",
        impact="系统提示的字数接近地图本身。架构长文不在第一轮里。",
        counter="启动时把目录下所有文件拼起来。地图这个形式就名存实亡。",
    )

    print_section("字数")
    full = stuffed()
    assert NOTE_RULE not in MAP
    assert "RULE-ARCH" not in MAP
    assert len(MAP) * 5 < len(full)
    print_ok(f"只放地图: {len(MAP)} 字")
    print_ok(f"地图加全部正文: {len(full)} 字")
    print_trap(
        "第一轮就拼完全文",
        "架构长文和笔记规则都进了系统提示，模型还没接任务就已经很满。",
    )

    print_section("真跑：系统提示只有地图")
    turn = Runner(
        live_model(),
        [open_doc],
        system=MAP,
        max_steps=4,
    ).run("记笔记之前要遵守哪条规则？必须调用 open_doc，path 填 skills/note.md，然后只引用工具返回的原文。")
    assert "open_doc" in called_tools(turn), called_tools(turn)
    opened = [item.text for item in turn.items if item.kind == "tool_result"]
    assert any(NOTE_RULE in text for text in opened), opened
    assert all("RULE-ARCH" not in text for text in opened)
    print_ok(f"打开后的工具结果: {opened}")
    print_ok("架构长文没有进入这一轮的工具结果")
    print_ok("Skill 就是这种按需打开的第二层，不必写进每次都在的系统提示")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
