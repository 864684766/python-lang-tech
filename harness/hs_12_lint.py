"""
Harness 教材 ⑫ 红灯要顺便告诉下一步

上一节规定了什么可以写回。模型如果调用了不该调用的工具，还得能从失败里知道改调谁。

这一节看错误信息本身。

先看一个现象。规矩是笔记不能写进账本。用户要记「今天晴」，模型仍建议 write_ledger。工具只返回：

    "错误：write_ledger 违反架构"

账本是空的，这是对的。下一轮模型只看见这八个字，不知道该改调谁，可能再调一次 write_ledger，或口头说「已经记下」。

这里的问题是：报错没有写出下一步。模型只能猜。

以前把分层写在文档里，违反了只回上面这句。人知道「架构」指账本和笔记要分开，模型看不到该改调 write_note。

这一节同一次违规做两件事。write_ledger 不往账本里追加。返回的句子写成：

    "笔记任务只能调用 write_note。请改为调用 write_note，原文是：今天晴。不要直接写账本。"

模型按这句话改调之后：

    notes == ["今天晴"]
    ledger == []

反面教材：用户要记「今天晴」。模型调用了 write_ledger。工具返回「错误：write_ledger 违反架构」，然后结束。账本确实没写上，这是对的；下一轮模型仍不知道笔记该走 write_note，可能再调一次 write_ledger，或直接口头说「已经记下」。教案要写成：「笔记任务只能调用 write_note。请改为调用 write_note，不要直接写账本。」模型才能改调，并且原文「今天晴」进笔记、不进账本。

影响：约束要能执行。执行失败时，错误文本就是下一轮的操作说明。

运行: python harness/hs_12_lint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import live_model, print_harness_job
from harness.mini import Runner, called_tools

ledger: list[str] = []
notes: list[str] = []


def vague_error(tool_name: str) -> str:
    return f"错误：{tool_name} 违反架构"


def teaching_error(tool_name: str) -> str:
    return (
        f"错误：{tool_name} 写的是账本层。"
        "笔记任务只能调用 write_note。"
        "请改为调用 write_note，不要直接写账本。"
    )


@tool
def write_ledger(text: str) -> str:
    """写入账本。笔记类请求不要调用这个工具。"""
    return teaching_error("write_ledger")


@tool
def write_note(text: str) -> str:
    """把一句话写入笔记。"""
    notes.append(text)
    return "已记下: " + text


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑫ 红灯要顺便告诉下一步")
    print("=" * 65)

    print_harness_job(
        "分层规则由 Harness 执行。违规时不落盘，并把修法写进返回给模型的那句话",
        prop="笔记不能直接写账本",
    )

    print_idea(
        phenomenon="文档里写过「笔记层不能碰账本」。模型还是调用了 write_ledger。",
        gap="约束要能跑。跑失败时，错误文本本身就是下一轮的操作说明。",
        before="Linter 只打印 violation。人看得懂，模型只能猜。",
        impact="write_ledger 的返回值点名 write_note。账本列表保持为空。",
        counter="同一条规则若只返回「违反架构」，模型看不到该改调谁。",
    )

    print_section("同一条违规，两种返回")
    vague = vague_error("write_ledger")
    taught = teaching_error("write_ledger")
    assert "write_note" not in vague
    assert "write_note" in taught
    assert "账本" in taught
    print_trap("只有红灯", vague)
    print_ok(taught)

    print_section("真跑：写错层也不会进账本")
    ledger.clear()
    notes.clear()
    turn = Runner(
        live_model(),
        [write_ledger, write_note],
        system=(
            "用户要求记笔记时，先调用 write_ledger。"
            "如果工具返回错误，阅读错误全文，再改调它点名的那个工具，把同一句原文写进去。"
        ),
    ).run("把「今天晴」写入笔记")
    assert ledger == []
    print_ok(f"账本写入次数 = {len(ledger)}")
    print_ok(f"调用顺序 {called_tools(turn)}")
    if "write_note" in called_tools(turn):
        assert notes, notes
        print_ok(f"笔记 {notes} ← 副作用只发生在教案点名的工具上")
    else:
        print_ok("模型这一轮停在错误上。账本仍然是空的，教案已经在工具结果里")
    assert "write_ledger" not in notes


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
