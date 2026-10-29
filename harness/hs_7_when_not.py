"""
Harness 教材 ⑦ 步骤已经清楚时，不要套这层

前面的课程已展示工具执行、循环、审批和窗口。本课讨论何时可以直接用确定性程序。

这一节学什么时候不要用它。

先看一个现象。需求是「把 3 和 5 相加，返回和是多少」。步骤只有一次加法，你写代码时已经知道。如果仍写成：

    Runner(model, [add]).run("把 3 和 5 相加，告诉我和是多少")

就要先问模型「你要不要调用 add」。它可能不调，自己说「8」；也可能得到 8 之后再建议一个你没准备的工具。

这里的问题是：缺的不是循环，是一次函数调用。多出来的提问要花钱，模型还可能多走一步。

以前只要用到模型，就先套 Runner 或 create_agent。循环变成默认。

这一节同一道题写成普通函数，中间没有「模型选工具」：

    def pipeline_add(a: int, b: int) -> str:
        return "和是 " + add.invoke({"a": a, "b": b})

    pipeline_add(3, 5)    # 直接得到 "和是 8"，没有 tool_calls

下一步你事先不知道、要模型选，才用 Runner。

对话框里也可以先由程序认明确指令，认不出再交给模型。用户输入「/weather 北京」时：

    if message.startswith("/weather"):
        city = message.removeprefix("/weather").strip()
        return get_weather(city)          # 得到天气，GLM 一次都没出场

流程是：前端收到这句话 → 程序识别 → get_weather("北京") → 结果。

同一句如果丢给模型，它要先判断「用户想查天气」，再建议调用 weather。这一次就要把工具说明和上下文传进去。明确指令走程序，少一次调用，也少一次模型选错工具的机会。

「/weather 北京」是用户明确要求查天气。「北京明天天气怎么样？我下午要出去。」对不上 / 或 @，才交给模型，由它决定要不要调用 weather。

    /weather 北京          → 程序直接 get_weather("北京")
    @file report.pdf       → 程序直接读取这个文件
    @db 查询用户 123       → 程序直接查库
    北京明天天气怎么样？    → Runner，模型自己选工具

输入框里可以约定：/ 是命令，@ 是引用的对象，# 是某一种能力。这些是用户写明了要做什么。

不要为了少调用模型，把所有话都收成斜杠命令。用户说「我明天去东京出差，帮我看看天气和需要带什么」，若强迫他改成「/weather 东京」再加「/packing 东京」，自然对话就没了。

所以按能不能对上明确指令来分：

    能对上 /、@、#     → 程序直接执行。0 次模型，快，结果确定
    对不上             → 交给模型选工具。更贵，能处理多步和推理

反面教材：需求写死为「把 a 和 b 相加，返回和是多少」。函数 pipeline_add 已经能调用 add 并拼出结果。若仍包进 Runner，模型可以不调用 add，自己心算；也可以在得到 8 之后再建议一个并不存在的第四步。步骤没有选择余地时，多出来的循环是模型发明路径的地方。另一条反面是把「我明天去东京出差」也做成必须输入 /weather，省下的那一次调用换来的是用户不会用。

影响：能用程序判断，就不要让模型判断。少花 token 只是结果之一，同时更少等待、更少选错，行为也由代码决定。

运行: python harness/hs_7_when_not.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import MUST_USE_ADD, add, live_model, print_harness_job
from harness.mini import Runner, called_tools


def pipeline_add(a: int, b: int) -> str:
    return "和是 " + add.invoke({"a": a, "b": b})


def get_weather(city: str) -> str:
    return f"{city}：晴"


def read_file(name: str) -> str:
    return f"已读取 {name}"


def dispatch(message: str) -> str | None:
    """对得上 / 或 @ 就直接执行。对不上返回 None，调用方再交给模型。"""
    if message.startswith("/weather"):
        city = message.removeprefix("/weather").strip()
        return get_weather(city)
    if message.startswith("@file"):
        name = message.removeprefix("@file").strip()
        return read_file(name)
    return None


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑦ 先问：步骤清楚吗？")
    print("=" * 65)

    print_harness_job(
        "Harness 是可选层。步骤已经确定时，不要为了「有 Agent」硬套 loop",
        prop="pipeline_add vs Runner",
    )

    print_idea(
        phenomenon="「把 3 和 5 加起来告诉我」——你已经知道只有这一步计算。",
        gap="这里缺的不是 loop，是一次函数调用。",
        before="因为正在学 Agent，所以什么都 create_agent。",
        impact="流水线可以零次 GLM 调用。Harness 留给「下一步由模型决定」。",
        counter="固定的转码→抽帧→分析，还让模型自己决定要不要删文件。",
    )

    print_section("正例：固定步骤，走流水线")
    text = pipeline_add(3, 5)
    assert text == "和是 8"
    print_ok(f"流水线 {text!r}，GLM 一次都没出场")

    print_section("明确指令：程序匹配 / 和 @，不调用模型")
    weather = dispatch("/weather 北京")
    assert weather == "北京：晴"
    print_ok(f"/weather 北京 → {weather!r}")
    opened = dispatch("@file report.pdf")
    assert opened == "已读取 report.pdf"
    print_ok(f"@file report.pdf → {opened!r}")
    assert dispatch("北京明天天气怎么样？我下午要出去。") is None
    print_ok("自然句对不上 / 或 @，dispatch 返回 None，这一句才交给 Runner")

    print_section("对照：同一题交给 Harness + GLM")
    turn = Runner(live_model(), [add], system=MUST_USE_ADD).run("用 add 计算 3+5")
    print_ok(f"Harness 也能做对：工具 {called_tools(turn)} 终答 {turn.final_text!r}")
    print_ok("但它多了模型选择权——这题用不上选择权")

    print_trap(
        "有 GLM 的地方不一定要有 Harness",
        "润色一句提示词，一次 invoke 就够。Loop 是为了不确定的下一步。",
    )
    print_ok("对得上 /、@ → 程序直接执行；对不上 → 模型选工具。")
    print_ok("不要把「我明天去东京出差」收成必须输入 /weather。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
