"""
Deep Agents ⓪ 走进库型：循环在库里，模型仍是 GLM

上一节的地图里，第二座房子是库型。循环由库来跑，模型是你传进去的参数。

这一节走进这座房子。

先看一个现象。手写 Runner 已经能看见这四步：

    模型建议 add(3, 5)
    Harness 执行，得到 "8"
    把 "8" 写回记录
    再问一次，模型用中文回答

压缩旧消息、再派一个子助手、读写工作目录里的文件，这些还要自己再写。

这里的问题是：每个项目都从零写一遍这个循环，想法重复，配套跟不上。

以前继续把 mini.py 加厚。能学清楚，但你在重写一套已经有人维护的运行时。

这一节改成把本课的 GLM 和 add 传进库：

    create_deep_agent(model=GLM, tools=[add])

模型仍只建议。变的是记录从一条条 Item 换成 messages。压缩和子助手由库带上。业务是「用户上传视频，排队交给可灵生成」时，这个调用不会自动出现队列和页面。那些仍是你自己的代码。库替代的是再手写一遍上面四步。

反面教材：业务是「用户上传视频，排队交给可灵生成」。代码只写了 create_deep_agent(model=GLM, tools=[add])，就以为生成、队列、页面都在库里。调用之后得到的仍是一轮工具循环：模型建议，库执行你传进去的工具。视频队列、页面和「什么时候算生成成功」还是你自己写的那层代码，不在这个函数里。这个库替代的是再手写一遍上面四步，不是整份产品。

影响：自己的应用要换模型时，写成 create_deep_agent(model=你的 GLM)。不要用只能启动某一家自己模型的 SDK。

运行: python harness/da_0_create.py
需要: pip install deepagents
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import MUST_USE_ADD, add, live_model, print_harness_job
from harness.mini import text_of, tool_calls_of


def demo() -> None:
    print("\n" + "=" * 65)
    print("Deep Agents ⓪ 模型还是 GLM")
    print("=" * 65)

    print_harness_job(
        "这是 hs_8 说的「库型」房子：循环在库里，模型是参数。不是 Codex 那种锁模型的产品型",
        prop="create_deep_agent + add",
    )

    print_idea(
        phenomenon="手写 Runner 已经能驾驭 GLM。但压缩、子 agent、目录工具要自己做。",
        gap="库型 Harness：模型是参数。不是 Codex SDK 那种「启动他们的程序」。",
        before="一直维护自己的 loop。",
        impact="create_deep_agent(model=你的 GLM, tools=你的工具)。思想没变，工程变少。",
        counter="用它当「任意业务框架」却不传 model，或指望它内置可灵。",
    )

    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=live_model(),
        tools=[add],
        system_prompt=MUST_USE_ADD,
    )
    result = agent.invoke({"messages": [HumanMessage(content="用 add 计算 3+5")]})
    messages = result["messages"]
    used = []
    for msg in messages:
        used.extend(c.get("name") for c in tool_calls_of(msg) if c.get("name"))
    final = text_of(messages[-1])
    print_ok(f"Deep Agents 过程中请求过的工具: {used}")
    print_ok(f"终答: {final!r}")
    if "add" not in used:
        raise AssertionError(f"没有调用 add，消息数={len(messages)} 终答={final!r}")

    print_section("和手写 Runner 的关系")
    print_ok("相同：GLM 只建议，Harness 才执行")
    print_ok("不同：账本是 messages，不是 mini 的 Item；另外还带压缩、子 agent、文件系统")
    print_section("在库返回的 messages 中辨认调用与结果")
    for msg in messages:
        calls = tool_calls_of(msg)
        if calls:
            print_ok(f"模型请求: {calls}")
        elif getattr(msg, "type", "") == "tool":
            print_ok(f"工具结果: id={msg.tool_call_id!r}, content={msg.content!r}")
    print_ok("手写 Runner 的工具结果记在 Item 中；库用 ToolMessage。都要保留调用与结果配对。")
    print_ok("上一课三座房子里，你现在走进的是库型")

    print_trap(
        "Deep Agents 替代不了 Host",
        "你的页面、业务规则、视频队列仍要自己写。它只替代「再手搓一遍 loop」。",
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
