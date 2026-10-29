"""
Harness 教材 ⓪ 到底什么是 Harness

本课只建立 Host / Harness / Model 和 Thread / Turn / Item 的概念地图，不请求模型。
Host 接收用户请求；Harness 组织上下文、请求模型、执行工具、写回结果并控制停止；
Model 根据输入生成文字或工具调用请求。本地 Python 函数由 Harness 执行。
模型可能自行算出正确数字，但输出正确答案不能证明本地函数执行过。

Thread 是完整交互记录，Item 是其中一条，Turn 是一次请求到暂停或结束的结果。
这些名字是本仓库的约定，不是所有 Harness 都必须使用的类名。
下一课 hs_1 实际运行「不提供 add / 提供 add」对照，并介绍工具消息配对。

运行: python harness/hs_0_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_tip
from harness.live import print_harness_job

def demo() -> None:
    print("\n" + "=" * 65)
    print("⓪ Harness 是一层软件，不是一道题")
    print("=" * 65)

    print_harness_job(
        "模型只能吐「我想调 add」；真正跑 add()、写回结果、决定再问不问，全是 Runner",
        prop="3+5 / add 工具",
    )

    print_idea(
        phenomenon="LangChain 的 create_agent、本课的 Runner、Deep Agents，跑起来都像「问一句、调工具、再回答」。所以你会觉得「不就是个例子」。",
        gap="缺的是分层：模型 = 出主意；Harness = 执行 + 停 + 管上下文。例子只是让这层能跑起来。",
        before="3_2 create_agent 把 Harness 藏在工厂函数里。你看见结果，看不见谁在执行。",
        impact="用结构图分清职责；实际执行和日志在 hs_1 展开。",
        counter="只记住「用了 add」。那是道具。忘掉标签，这套课就废了。",
    )

    print_section("职责地图（不请求模型）")
    print(
        """
          你的脚本 / 以后的产品     ← Host（外壳）
                    │
                    │  「用工具算 3+5」
                    ▼
          ┌─────────────────────┐
          │  Harness = Runner   │  组消息、调模型、跑函数、
          │  本仓库 mini.py     │  人审、步数、窗口
          └─────────┬───────────┘
                    │  问：下一步怎么办？
                    ▼
               GLM（模型）         只会返回 text 或 tool_calls
                                   它碰不到你的 Python 函数
        """
    )
    print_ok("Agent = 模型 + Harness。Runner 执行本地工具，GLM 生成下一步请求。")
    print_ok("Thread / Turn / Item 是 Harness 记账用的三个格子，不是模型的新语法。")

    print_tip("下一课 hs_1 首次运行工具调用；本课只看分层，不重复跑加法。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
