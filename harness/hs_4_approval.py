"""
Harness 教材 ④ 想做，还不能直接做

上一节以及更早的课里，模型给出工具建议之后，Harness 会立刻执行。

这一节加上一道门。有些工具会改记录，不该一建议就落盘。

先看一个现象。用户说「把周四交报告记下来」。模型给出建议之后，如果 Harness 马上执行，笔记列表里已经有字了：

    模型建议: notes_write("周四交报告")
    notes == ["周四交报告"]     # 人还没点头，字已经写上了

这里的问题是：建议一来就执行，改记录这件事已经发生，再发现不该写也晚了。

以前有两种做法。一种是全自动，上面这段就是。另一种是每一步都弹确认，连 add(3, 5) 也问。问得太勤，人会不看内容直接点同意。

这一节只把会改记录的工具放进名单：

    runner = Runner(model, [notes_write], approval_tools={"notes_write"})
    paused = runner.run("把周四交报告记下来")
    paused.status == "waiting_approval"
    notes == []                              # 还没写
    runner.resume(paused, approve=True)      # 这行之后 notes 里才有字

拒绝时不调用 notes_write，只追加一条正文为「用户拒绝了」的记录。

反面教材：approval_tools 里放上 add。用户问「3+5」，模型建议 add(3, 5)，这一轮的 status 变成 "waiting_approval"，算出 8 之前先弹一个确认框。这种计算不改笔记、不写文件，人看不出危险，几次之后就会不读内容直接点同意。真正要拦的是 notes_write：没批准时 notes 必须还是 []。

影响：人审只留给会改状态的工具。模型可以想写，Harness 决定允不允许写。

运行: python harness/hs_4_approval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import live_model, notes_write, print_harness_job
from harness.mini import Runner, called_tools
from langchain_core.tools import tool

wrote: list[str] = []


@tool
def notes_write_tracked(text: str) -> str:
    """把一句话写入笔记。用户明确要求记录时才调用。"""
    wrote.append(text)
    return notes_write.invoke({"text": text})


SYSTEM = "用户让你记笔记时，必须调用 notes_write_tracked，不要只在口头答应。"


def demo() -> None:
    print("\n" + "=" * 65)
    print("④ 想做，还不能直接做")
    print("=" * 65)

    print_harness_job(
        "执行权在 Harness。模型可以「想写笔记」，没批准就一行 Python 都不会跑",
        prop="notes_write_tracked",
    )

    print_idea(
        phenomenon="GLM 输出 notes_write。若立刻执行，笔记已经被改了。",
        gap="Turn 先变成 waiting_approval。",
        before="全自动；或弹二十个确认框。",
        impact="没批准时写入次数是 0。resume(approve=True) 之后才执行。",
        counter="计算器 add 也要人审，审批会变成噪音。",
    )

    model = live_model()
    runner = Runner(
        model,
        [notes_write_tracked],
        approval_tools={"notes_write_tracked"},
        system=SYSTEM,
    )
    paused = runner.run("请把「周四交报告」写入笔记")
    assert paused.status == "waiting_approval", paused.status
    assert wrote == []
    print_ok(f"status={paused.status}，真正写入次数={len(wrote)}")

    done = runner.resume(paused, approve=True)
    assert wrote, "批准后仍未写入"
    assert done.status == "completed"
    print_ok(f"批准后写入 {wrote}，终答 {done.final_text!r}")
    print_ok(f"调用过的工具 {called_tools(done)}")

    print_trap(
        "不是每个工具都要问人",
        "只读、可逆的不要问。问多了人会全部点「是」。",
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
