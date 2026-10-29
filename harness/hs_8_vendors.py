"""
Harness 教材 ⑧ 同一层思想，三座不同的房子

上一节我们会判断：步骤清楚就卸掉循环，下一步要模型选才留下 Harness。

这一节比较架构与选型，不再运行前面已经展示过的加法循环。

先看一个现象。同一句「用 add 算 3+5」，三家记下来的东西名字不同：

    # 本课 Runner，一条条记录
    [Item(kind="user", ...), Item(kind="model", ...), Item(kind="tool_result", text="8")]

    # Deep Agents，一个消息列表
    {"messages": [HumanMessage(...), AIMessage(tool_calls=[...]), ToolMessage(content="8")]}

打开 Codex 的文档会看到 Thread、Turn、Item。打开另一家会看到 Effect。都像在说「只有我这种记法才算 Harness」。

这里的问题是：这些名字只是记录格式。拿只能启动 Codex 自己模型的 SDK 去套本课的 GLM，套不上，因为那个 SDK 启动的是他们的程序。

以前学会一家的对象名，就拿那张图纸盖所有项目。编码工具最怕乱改磁盘，所以做成执行前要批准。自己的业务程序最怕绑死一家模型，所以写成 create_deep_agent(model=你的 GLM)。抄错这层，不是少写几行。

这一节只认三种用法。Codex、Claude Code 那类：人在他们的程序里改代码，模型基本是他们家的。Deep Agents、LangGraph 这类：循环在库里，模型是你传的参数。Cordis 那类：程序不重启也能换工具，本课不写代码，只知道有这种。下一课走进第二种。

反面教材：面试时把「Harness 就是 Item 这个类」背下来。换到 Deep Agents，记录是上面的 messages，没有 Item，于是认为它不是 Harness。再换一家，记录可能叫 Effect，意思是「做了一次、还能撤销」，又对不上这个类名。要对上的是：模型给出建议，外层软件执行并决定是否写回。拿 Codex 的 SDK 去套 GLM 是同一个错误：那个 SDK 启动的是他们的程序和他们的模型。

影响：按这个产品最怕什么来选写法。怕乱改磁盘，就做成执行前要批准。怕绑死一家模型，就把模型写成参数。

运行: python harness/hs_8_vendors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap, print_tip
from harness.live import print_harness_job


def _house(name: str, assume: str, ledger: str, field: str, model: str) -> None:
    print(f"\n    ┌─ {name}")
    print(f"    │  假设: {assume}")
    print(f"    │  账本: {ledger}")
    print(f"    │  主战场: {field}")
    print(f"    │  模型: {model}")
    print("    └")


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑧ 厂家差在房子，不差在「模型会建议」")
    print("=" * 65)

    print_harness_job(
        "每家 Harness 优化的约束不同，所以图纸不同。"
        "相同的仍是：模型只建议，Harness 才执行",
        prop="Codex / Claude Code / 千问 / Deep Agents / Cordis 这些名字",
    )

    print_idea(
        phenomenon="网上文章都在说 Harness，但 Codex 讲 Thread/Turn，Deep Agents 讲 messages，Cordis 讲可撤销 Effect。你会以为只有一种正确架构。",
        gap="缺一张地图：思想同一层，房子按产品假设来盖。",
        before="要么死学 Codex 对象名，要么拿某一家 SDK 去套自己的 GLM。",
        impact="先认房子类型，再决定抄哪张图纸。下一课 Deep Agents 就是「库型」里的一套。",
        counter="把 Turn 或 Effect 背成定义。那是某一家的账本，不是 Harness 本身。",
    )

    print_section("三种架构视角（本课不请求模型）")
    print(
        """
          相同的一层
          模型建议 tool_calls / 文本  →  Harness 执行、写回、停、切窗口

          ┌──────────┐   ┌──────────┐   ┌──────────┐
          │ 产品型   │   │ 库型     │   │ 运行时型 │
          │ 执行面   │   │ 可嵌入   │   │ 组合面   │
          └────┬─────┘   └────┬─────┘   └────┬─────┘
               │              │              │
           他们的 IDE      你的 App      长期活着的
           他们的模型      你换模型      进程，零件热换
        """
    )

    _house(
        "产品型 · Codex / Claude Code / Qwen Code",
        "人在 IDE/CLI 里改代码，进程跑完一个任务即可",
        "Host / Harness / Model；Thread → Turn → Item",
        "沙箱、审批、压缩、和自家模型的专用协议",
        "基本锁自家模型。Codex SDK 启动的是 Codex 程序，不是「给 GLM 套壳」",
    )
    _house(
        "库型 · Deep Agents / LangGraph / Qwen-Agent",
        "你要把循环嵌进自己的产品，模型可能是 GLM / 千问 / DeepSeek",
        "messages 列表，或一张图（节点=模型/工具）",
        "换模型、子 agent、规划、记忆；Host 仍是你写",
        "模型是参数。create_deep_agent(model=你的 GLM) 就是这一座",
    )
    _house(
        "运行时型 · Cordis / DeepSeek Harness",
        "Agent 自己换 Tool / Memory / Runtime 时，不能靠重启进程",
        "Session + 可撤销 Effect；依赖要能观察",
        "热插拔、幽灵资源、组合粒度",
        "不是入门图纸。本课不实现它",
    )

    print_ok("腾讯 WorkBuddy 这类是闭源产品：能用，学不到可复用的图纸。")
    print_ok("阿里千问两边都有：Qwen Code 偏产品型，Qwen-Agent 偏库型。品牌不等于一种架构。")

    print_section("为什么图纸必须不一样")
    print("    约束不同，第一公民就不同：")
    print("    · 编码 IDE 最怕乱改磁盘 → 沙箱和人审做成骨架（产品型）")
    print("    · 业务 App 最怕绑死一家模型 → 模型必须是参数（库型）")
    print("    · 长期运行最怕换零件就崩 → 撤销和热换做成骨架（运行时型）")
    print("    抄错骨架，不是「少写了几行」，是整栋楼盖反了。")

    print_section("怎么选（判断力）")
    print_ok("学思想、面试能讲分层 → 留着手写 mini（本仓库）")
    print_ok("自己的 App，要换 GLM → 库型（下一课 Deep Agents）")
    print_ok("做编码 CLI/IDE → 抄产品型的沙箱/人审/压缩，模型层自己接，别指望 Codex SDK 套 GLM")
    print_ok("零件要热换、不能重启 → 再读运行时型；入门先别实现")

    print_trap(
        "不要把某一家的对象名当成 Harness",
        "Turn、messages、Effect 都是账本。Harness 是「建议 vs 执行」那一层。",
    )
    print_tip("下一课 da_0 走进库型。对照时问：账本换了没有？模型是不是参数？")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
