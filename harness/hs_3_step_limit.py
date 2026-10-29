"""
Harness 教材 ③ 循环自己不会累，所以要有保险丝

上一节的循环会一直问下去：只要模型还在给工具建议，Harness 就执行、写回、再问。

这一节看这个循环的停法。

先看一个现象。模型可以一直建议工具，循环自己不会停：

    第 1 步  模型建议 add → Harness 执行 → 再问
    第 2 步  模型又建议 add → Harness 执行 → 再问
    第 3 步  ...   # 没有人喊停，它就继续

这里的问题是：不写上限的话，最坏会问多少次，事先不知道。没人看着就停不下来。

以前的做法是人盯着终端，不对就按 Ctrl+C。人不在，循环还在。

这一节写成：

    Runner(model, [add], max_steps=2)

一轮里最多问模型 2 次。第 2 次结束还在要工具，就返回 status="failed"，不再问第 3 次。这个次数是代码里的数字，不是模型自己决定的。

反面教材：把 max_steps 设成 1，以为这样最安全。问「用 add 算 3+5」。第 1 步模型建议调用 add，Harness 执行完得到 8，步数已经用完，来不及再问一次「根据结果用中文回答」。这一轮变成 failed，终答不是「和是 8」。正常的「先工具、再开口」至少要两步。保险丝要留得下这一来一回，不是越紧越好。

影响：最坏要问几次，事先就知道。

运行: python harness/hs_3_step_limit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import MUST_USE_ADD, add, live_model, print_harness_job
from harness.mini import Runner, called_tools


def demo() -> None:
    print("\n" + "=" * 65)
    print("③ 循环必须能停")
    print("=" * 65)

    print_harness_job(
        "保险丝在 Harness。模型不会自己说「我调太多次了」",
        prop="max_steps=1 vs 6",
    )

    print_idea(
        phenomenon="「调工具 → 再想」每一步都是一次 GLM 调用。没有上限就会烧钱。",
        gap="缺 max_steps。真模型也会连着 tool_calls。",
        before="开着电脑等到超时，或你去杀进程。",
        impact="上限用完 → status=failed。正常任务把上限设宽一点。",
        counter="上限不是越小越安全。1 步连「先 add 再说话」都做不完。",
    )

    model = live_model()

    print_section("保险丝太紧：max_steps=1")
    tight = Runner(model, [add], max_steps=1, system=MUST_USE_ADD).run("用 add 计算 3+5")
    print_ok(f"max_steps=1 → status={tight.status} 调用了 {called_tools(tight)}")
    if "add" in called_tools(tight):
        assert tight.status == "failed"
        assert "step_limit" in tight.events
        print_ok("第一次就去调工具，1 步不够再开口，于是熔断。这就是保险丝。")
    else:
        print_trap(
            "这次 GLM 没调工具就结束了",
            "保险丝只在「还要再问模型」时触发。下面用 max_steps=6 再跑一遍。",
        )

    print_section("保险丝够用：max_steps=6")
    ok = Runner(model, [add], max_steps=6, system=MUST_USE_ADD).run("用 add 计算 3+5")
    assert ok.status == "completed", ok.status
    assert "add" in called_tools(ok)
    print_ok(f"max_steps=6 → status={ok.status} 终答 {ok.final_text!r}")

    print_trap(
        "上限要大于常见步数",
        "保险丝防灾难，不是防工作。本课用 1 和 6 对照。",
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
