"""
Harness 教材 ② 工具结果要写回，才能进入下一步

上一节确认了：给了工具，模型建议一次，Harness 执行一次，然后可以终答。

这一节加深一层。有的任务一次做不完，比如先把 3 和 5 相加，再把得数原样复述出来。

先看一个现象。用户说「先把 3 和 5 相加，再把得数原样复述」。
模型第一次回复里只有一份建议，add(3, 5) 还没跑，所以回复里没有 8：

    tool_calls = [{"name": "add", "args": {"a": 3, "b": 5}, "id": "call_1"}]
    content = ""          # 没有文字，也没有数字 8

模型也可能心算猜出 8，但此时还没有 add 的实际执行结果。
Harness 执行 add 之后，记录里多一条它写上的结果，再把这三样一起送回去：

    [
        {"角色": "用户", "内容": "先相加，再复述得数"},
        {"角色": "模型", "建议": "add(3, 5)", "id": "call_1"},
        {"角色": "工具结果", "内容": "8", "对应": "call_1"},   # 这行是 Harness 写的
    ]

第二次提问时模型才看得到 "8"，然后建议 echo("8")。Harness 再执行 echo。

这里的问题是：如果 "8" 只打印在终端上，没有放进上面这个列表，第二次模型仍只看见原问题，就自己编一个数。

以前的做法是人自己调两次接口，把 "8" 复制进第二次的提示词。能用，但容易漏，也看不出「写回再问」是这段循环的职责。

这一节让同一次 run() 里出现两条工具结果：先 add 得到 "8"，写回后再问；模型接着建议 echo，Harness 再执行。

反面教材：add 已经算出 8，但这条结果只 print 在终端上，没有写成 tool_result 再送给模型。下一轮消息仍是「请先相加，再复述得数」。模型看不见 8，就自己编一个数去复述，echo 的参数可能是 7 或 15。人在日志里看得到正确答案，模型的下一步用不上。

影响：终答出现之前，记录里可以有两条工具结果。先是 add 写下的 "8"，再是 echo 写下的复述。

与 hs_1 的区别：hs_1 验证函数有没有真正执行，本课验证后一个工具的参数是否来自
前一个工具的实际结果。它们复用同一个 Runner，不是在本课才新增循环。

运行: python harness/hs_2_tool_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import add, echo, live_model, print_harness_job
from harness.mini import Runner, called_tools


SYSTEM = (
    "你必须按这个顺序使用工具："
    "1) 调用 add 计算 3+5；"
    "2) 调用 echo，把 add 得到的数字原样放进 text；"
    "3) 再用一句话告诉用户。"
    "不要跳过工具，不要心算。"
)


def demo() -> None:
    print("\n" + "=" * 65)
    print("② 做完一件，再决定下一件")
    print("=" * 65)

    print_harness_job(
        "循环本身在 Harness：执行完工具，把观察写回，再问模型下一步",
        prop="先 add 再 echo",
    )

    print_idea(
        phenomenon="「先算 3+5，再把结果原样复述」是两步。",
        gap="缺循环：第一次工具结果必须变成第二次模型输入。",
        before="人肉跑两次 GLM，把第一次输出粘到第二次。",
        impact="Harness 自动把 ToolMessage 接回去。你看到两次 tool_result。",
        counter="执行了 add 却不写回，echo 只能猜。",
    )

    turn = Runner(live_model(), [add, echo], system=SYSTEM).run(
        "先用 add 算 3+5，再用 echo 复述那个数字"
    )
    tools = called_tools(turn)
    assert "add" in tools, f"没有 add: {tools} / {turn.final_text!r}"
    assert "echo" in tools, f"没有 echo: {tools} / {turn.final_text!r}"
    results = [item for item in turn.items if item.kind == "tool_result"]
    add_index = next(i for i, item in enumerate(results) if item.tool == "add")
    echoes = [item for item in results[add_index + 1:] if item.tool == "echo"]
    assert echoes, "echo 必须在 add 执行之后"
    assert echoes[0].args.get("text") == results[add_index].text, "echo 参数必须等于 add 的实际结果"
    kinds = [i.kind for i in turn.items]
    print_ok(f"工具顺序 {tools}")
    print_ok(f"Item 种类 {kinds}")
    print_ok(f"终答 {turn.final_text!r}")

    print_trap(
        "观察必须留在 Thread",
        "tool_result 是下一轮 GLM 的输入。丢掉它，循环就断了。",
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
