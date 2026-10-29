"""
Harness 教材 ⑤ 不能把整个过去塞进模型

前面的课程着重看工具与执行。本课单独展开 Runner 已有的窗口管理：
它默认就只发送最近 4 条 Item，并非到这一课才开始切窗口。

这一节看记录变长之后怎么办。

先看一个现象。记录里已经有很多条，每次都原样全送：

    全部记录 = ["第一句", "好的", "第二句", "好的", "第三句", "好的", "第四句"]
    送给模型的 = 全部记录          # 7 条都在这一次提示词里

句子再增加，这一次提示词就更长。塞满之后，模型更容易漏掉真正要回答的那句。

这里的问题是：存档可以很长，一次提问不该把存档全文都带上。

以前的做法是从第一句粘到现在。满了就换一个能装更多字的模型。能装的字变多，只是把塞满这件事推迟。

这一节存档仍是上面那 7 条。送给模型的只有最后 4 条：

    Runner(..., context_budget=4)
    送给模型的 = ["好的", "第三句", "好的", "第四句"]
    全部记录   = 原来的 7 条还在     # 没有删，也没有收成一段摘要

「第一句」不在这次提示词里，仍可以在全部记录里查到。把旧对话收成摘要再和最近几条一起送，是 4_1_message_history_demo.py 的 Demo 4，这节不做。

反面教材：context_budget=4 时，把「全部记录」也改成只留最后 4 条。送给模型的列表里没有「第一句」，这是对的。若全部记录里也删掉「第一句」，关掉页面再打开就查不到这句话。正确的切法是：送给 GLM 的列表变短，全部记录里的旧句子还在。

影响：看见的少，记下的还在。砍短的是送给模型的那一份，不是存档。

运行: python harness/hs_5_context.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_tip, print_trap
from harness.live import live_model, print_harness_job
from harness.mini import Item, Runner, items_to_messages
from langchain_core.tools import tool


@tool
def noop() -> str:
    """占位，本课不要求调用。"""
    return "noop"


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑤ 不能把整个过去塞进模型")
    print("=" * 65)

    print_harness_job(
        "窗口由 Harness 切。模型没法要求「把全部历史再喂我一遍」",
        prop="context_budget=4",
    )

    print_idea(
        phenomenon="Thread 里已经有很多旧 Item。每次原样全送，窗口会满。",
        gap="保存用全量，思考用切片。",
        before="从第一句粘到现在。满了换 128K——只会更晚爆炸。",
        impact="本课用最近 4 条演示。生产里还会做摘要、外置大结果。思想一样：有界。",
        counter="Thread 跟着砍。回放、审计、人都没了。",
    )

    history = [
        Item(kind="user", text="第一句"),
        Item(kind="model", text="好的"),
        Item(kind="user", text="第二句"),
        Item(kind="model", text="好的"),
        Item(kind="user", text="第三句"),
        Item(kind="model", text="好的"),
    ]
    runner = Runner(live_model(), [noop], context_budget=4)
    visible = runner._visible(history + [Item(kind="user", text="第四句，只回：看到了")])
    print_ok(f"Thread 将保存 {len(history) + 2} 条左右（含新问题和回答）")
    print_ok(f"实际送给 GLM 的 Item 只有 {len(visible)} 条")
    for i, item in enumerate(visible):
        print_ok(f"  窗口[{i}] {item.kind}: {item.text}")

    turn = runner.run("第四句，只回两个字：看到", history=history)
    assert turn.status == "completed"
    user_lines = [i.text for i in turn.items if i.kind == "user"]
    assert "第一句" in user_lines
    print_ok(f"Thread 里人说过: {user_lines}")
    print_ok(f"GLM 说: {turn.final_text!r}")
    print_ok(f"发给模型的消息数（含 system 则另算）: {len(items_to_messages(visible))}")

    print_trap(
        "看见的少 ≠ 可以扔",
        "砍投影，不砍 Thread。",
    )
    print_tip("这一课只切本轮窗口。进程关掉之后进度在哪，见 hs_9_handoff.py。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
