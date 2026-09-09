"""
========================================================
LangChain Phase 7-2
标准 MCP Client（不是模拟）
========================================================

先看整条运行时，再看代码为什么出现某一行。

运行时永远是两个进程（stdio 模式）：

  进程 A  本文件 7_2_*.py          = MCP Client（问、调）
  进程 B  math_server 或 npx Figma = MCP Server（声明工具、真正执行）
          中间用 stdin/stdout 传 JSON-RPC，不是 HTTP。

完整流水（Demo 1 每一步都能对上代码）：

  ① math_params()
        只是一张「怎么启动进程 B」的说明书：
        command=当前 Python，args=math_server.py
        此时还没有任何进程，也没有工具。

  ② stdio_client(那张说明书)
        SDK 真正 subprocess 拉起进程 B。
        math_server 里 mcp.run(stdio) 开始听管道。
        得到 (read, write) = 往进程 B 读写的两条流。

  ③ ClientSession(read, write)
        在这对流上发 MCP 规定的 JSON-RPC。
        还没握手，Server 不会理 tools/list。

  ④ session.initialize()
        报文：initialize
        双方确认协议版本、谁有 tools 能力。

  ⑤ session.list_tools()
        报文：tools/list
        Server 回答：我有 add、multiply（来自 @mcp.tool，不是 Client 写的）。
        这一步只拿目录，加法还没算。

  ⑥ session.call_tool("add", {a:3, b:5})
        报文：tools/call
        进程 B 里真正执行 add()，结果从管道传回。

Demo 2 在 ⑤ 之后多一步，不改协议：

  ⑦ load_mcp_tools(session)
        把 list 出来的 MCP 工具包成 LangChain BaseTool。
  ⑧ create_agent(model, tools).ainvoke(用户问题)
        模型选工具名 → adapter 内部还是 call_tool → 再生成回答。

Figma（Demo 3）把进程 B 换成 npx 社区 Server，①～⑧ 不变：
  说明书变成 command=npx、env 里塞 FIGMA_ACCESS_TOKEN。
  list 到的就是 Figma 的工具，call 才会去读设计稿。

为什么要有 math：Figma 依赖令牌和 Node；数学 Server 保证你能先跑通
①～⑥。只关心 Figma 时看 Demo 3，协议与 Demo 1 相同。

FastMCP：只出现在进程 B（math_server.py），用来「实现 Server」。
Client 连别人时不要用 FastMCP。

关于 Figma / GitHub 鉴权：
  · 官方 https://mcp.figma.com/mcp 只走 OAuth，不要 figd_。
  · 本 Demo Figma 用社区 stdio + .env 的 FIGMA_ACCESS_TOKEN。
  · GitHub 官方远程 MCP：https://api.githubcopilot.com/mcp/ + .env 的 GITHUB_PERSONAL_ACCESS_TOKEN
    （本 Demo 用 HTTP，不需要 Docker）。令牌不要写进源码，不要贴进聊天。

运行: python 7_2_mcp_integration_demo.py
依赖: pip install mcp langchain-mcp-adapters
Figma 另需 Node.js（npx）和 .env 里的 FIGMA_ACCESS_TOKEN
GitHub 另需 .env 里的 GITHUB_PERSONAL_ACCESS_TOKEN（远程 HTTP，不必 Docker）
"""
from __future__ import annotations  # 让类型注解里的 | 在旧 Python 也能写

import asyncio  # MCP SDK 是异步的：连接、initialize、list/call 都要用 await
import os  # 读 FIGMA_ACCESS_TOKEN；拼进子进程 env，不要 print 出来
import shutil  # shutil.which("npx")：系统 PATH 里有没有 Node 的 npx
import sys  # sys.executable = 当前 Python，保证子进程用的是同一个解释器/venv
from pathlib import Path  # 定位 math_server.py，避免写死绝对路径

from dotenv import load_dotenv
from langchain.agents import create_agent  # 模型 ↔ 工具循环；MCP 工具桥过来后直接塞这里
from langchain_core.messages import HumanMessage
from mcp import ClientSession, StdioServerParameters
# stdio_client：拉起 Server 子进程，拿到一对 (read, write) 字节流
from mcp.client.stdio import stdio_client
# CallToolResult / TextContent：tools/call 的返回值；内容是块列表，不一定只有一段文字
from mcp.types import CallToolResult, TextContent

from demo_explain import print_lesson, print_run_plan, print_step, print_tip
from model_provider import check_api_key, get_model

load_dotenv()  # 从 .env 读密钥；Figma 令牌也走这里，不要写进源码

# math_server.py 的路径。stdio 模式不会「import 这个文件」，而是另起一个 Python 去跑它
MATH_SERVER = Path(__file__).parent / "mcp_servers" / "math_server.py"


def math_params() -> StdioServerParameters:
    """拼出「启动进程 B」的命令行，还不执行。

    stdio 规定：Server 必须是另一个进程，Client 才能把它的 stdin/stdout 当网线。
    SDK 稍后会拿这张说明书去 spawn，等价于你在终端输入：

        E:\\...\\.venv\\Scripts\\python.exe  E:\\...\\mcp_servers\\math_server.py

    字段怎么读（新手）：
      command  要启动的程序。sys.executable = 「正在跑 7_2 的同一个 Python」
               （含 venv）。若写成 "python" 可能跑到系统自带的、没装 mcp 的那个。
      args     传给该程序的参数，这里只有一项：math_server.py 的路径。
               合起来就是：python  math_server.py

    不新建进程行不行？
      · 继续用 stdio_client → 不行。没有子进程就没有 stdin/stdout 可接，连不上。
      · 换传输：HTTP MCP 连已经在跑的网址，Client 侧不必 spawn。
      · 根本不用 MCP：在本进程里直接 add(3,5)，那就只是普通函数，不是 MCP。
    """
    return StdioServerParameters(
        command=sys.executable,  # 哪个解释器：当前 venv 的 python.exe
        args=[str(MATH_SERVER)],  # 让它执行哪份脚本；list 里可以有多个参数
    )


def _tool_text(result: CallToolResult) -> str:
    """把 tools/call 的结果抠成可读字符串。

    MCP 规定 content 是一块块内容（文字 / 图片 / 资源），不是单一 str。
    本课只演示文字工具，所以取出 TextContent.text 拼起来即可。
    """
    parts = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            # 非文本块（极少见）也打印出来，避免静默丢结果
            parts.append(str(block))
    return "\n".join(parts) or str(result)


def print_runtime_flow() -> None:
    """运行时总图：代码里出现的每个 API 对应哪一步。"""
    print(
        "\n  整条 MCP 运行时（两个进程，管道传 JSON-RPC）\n"
        "\n"
        "  进程 A Client = 本文件                     进程 B Server = math 或 Figma\n"
        "  -------------------------                 ------------------------------\n"
        "  ① math_params() 写启动命令\n"
        "  ② stdio_client(...) 拉起子进程  -------->  mcp.run(stdio) 开始听管道\n"
        "  ③ ClientSession(read, write)\n"
        "  ④ initialize()              ---JSON--->  确认协议版本\n"
        "  ⑤ list_tools()              ---JSON--->  返回 add/multiply（或 Figma 工具）\n"
        "  ⑥ call_tool(name, args)     ---JSON--->  执行函数 / 调 Figma API\n"
        "       <---- 结果从管道回来 ----\n"
        "\n"
        "  Demo 2 额外：⑤ 之后 load_mcp_tools → create_agent，模型替你选 ⑥ 的 name/args。\n"
        "  Demo 3 额外：① 的 command 换成 npx，进程 B 换成 Figma Server，④⑤⑥ 不变。\n"
        "  Demo 4 额外：连 GitHub 远程 HTTP MCP，本机不 spawn，list 工具名。\n"
    )


async def demo_1_protocol() -> None:
    """Demo 1：不经过 LangChain，直接走官方 MCP ①～⑥。"""
    print("\n" + "=" * 65)
    print("① 官方 MCP Client 连本地 Server（stdio）")
    print("=" * 65)
    print("  进程 B: mcp_servers/math_server.py  （FastMCP 实现，不是连 Figma 用的）")
    # stdio_client = 按 math_params 真正 spawn 子进程，并把管道交给你。
    # await/async with 只表示「等子进程起来、退出时再杀掉」——这是协程等 I/O，
    # 不是用协程代替子进程。进程 B 里另有一个 Python 在跑 math_server。
    async with stdio_client(math_params()) as (read, write):
        # ③ 会话：后面所有 initialize/list/call 都走这对管道
        async with ClientSession(read, write) as session:
            await session.initialize()  # ④ 握手。跳过则 list/call 会被拒
            listed = await session.list_tools()  # ⑤ 只问目录，工具仍在进程 B
            print(f"  ⑤ tools/list → {[t.name for t in listed.tools]}")
            added = await session.call_tool("add", {"a": 3, "b": 5})  # ⑥ 进程 B 执行 add
            print(f"  ⑥ tools/call add(3,5) → {_tool_text(added)}")
    print_tip("①～⑥ 就是标准 MCP。Demo 2 只在 ⑤ 后桥成 LangChain Tool。")


async def demo_2_langchain_agent() -> None:
    """Demo 2：同一条 MCP 连接，把工具交给 LangChain Agent。

    load_mcp_tools(session) ≈ 对每个 MCP tool 包一层 LangChain BaseTool：
      模型选出工具名 + 参数 → adapter 转成 session.call_tool → 结果写回 messages。
    业务代码不用自己写 @tool。
    """
    print("\n" + "=" * 65)
    print("② 把 MCP 工具桥成 LangChain Tool，交给 Agent")
    print("=" * 65)
    from langchain_mcp_adapters.tools import load_mcp_tools

    async with stdio_client(math_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # 仍是 ④
            tools = await load_mcp_tools(session)  # ⑦：⑤ 的目录 → LangChain BaseTool
            print(f"  桥接后: {[t.name for t in tools]}")
            agent = create_agent(
                get_model(),
                tools,
                # 强制走工具，避免模型口算「3+5*12」跳过 MCP
                system_prompt="用 MCP 工具计算，不要口算。",
            )
            q = "3 加 5 再乘以 12 等于多少？"
            print(f"  问: {q}")
            # ainvoke：异步跑「想 → 调 MCP → 再想」循环；终答在 messages 最后一条
            result = await agent.ainvoke({"messages": [HumanMessage(content=q)]})
            print(f"  答: {result['messages'][-1].content}")


async def demo_3_figma() -> None:
    """Demo 3：换一个真正的外部 Server（Figma），协议步骤不变。

    注意两种 Figma MCP 不要混：
      · 官方 https://mcp.figma.com/mcp → 只接受浏览器 OAuth，拒绝 figd_ 个人令牌
      · 本 Demo → 社区 npx Server + .env 里的 FIGMA_ACCESS_TOKEN
    没令牌 / 没 Node 就跳过，不让整个课失败。
    """
    print("\n" + "=" * 65)
    print("③ Figma MCP（用你自己的 PAT 测）")
    print("=" * 65)
    token = (os.getenv("FIGMA_ACCESS_TOKEN") or "").strip()
    if not token:
        print("  未配置 FIGMA_ACCESS_TOKEN，跳过真实连接。")
        print("  在 .env 写一行（不要把令牌贴进代码或聊天）：")
        print("      FIGMA_ACCESS_TOKEN=figd_你的新令牌")
        print("  官方 https://mcp.figma.com/mcp 只要 OAuth，不接受 figd_。")
        print("  本 Demo 用社区 stdio Server：npx @nexus2520/figma-mcp-server")
        return
    if shutil.which("npx") is None:
        print("  找不到 npx。Figma 社区 MCP 需要先安装 Node.js。")
        return

    # MultiServerMCPClient 的参数不是 MCP 协议报文，而是本库的「怎么连上」配置。
    # 外层 dict：key = 你起的服务器名（随便取）；value = 下面其中一种连接。
    #
    # transport 可选值（langchain-mcp-adapters）：
    #   "stdio"            拉起本地子进程，用 stdin/stdout（本 Demo）
    #   "http" / "streamable_http" / "streamable-http"  连远程 URL（官方 Figma 那种）
    #   "sse"              较旧的 HTTP+SSE 传输
    #   "websocket"        WebSocket
    # stdio 必填 command、args；HTTP 类必填 url（不要混用）。
    from langchain_mcp_adapters.client import MultiServerMCPClient

    print_step("连接", "stdio + FIGMA_ACCESS_TOKEN（不打印令牌）")
    client = MultiServerMCPClient(
        {
            "figma": {
                # 本条目走子进程。若改官方远程，应改成：
                #   "transport": "http", "url": "https://mcp.figma.com/mcp"
                # （官方要 OAuth，figd_ PAT 无效）
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@nexus2520/figma-mcp-server"],
                "env": {**os.environ, "FIGMA_ACCESS_TOKEN": token},
            }
        }
    )
    try:
        async with client.session("figma") as session:
            listed = await session.list_tools()
            name = listed.tools[0].name
            print(f"  tools/list 第一个: {name}")
            print(f"  直接 call_tool({name!r}, {{}})")
            result = await session.call_tool(name, {})
            print(f"  返回:\n    {_tool_text(result)[:700]}")
    except Exception as exc:
        print(f"  连接或调用失败: {type(exc).__name__}: {exc}")
        print_tip("常见原因：令牌作废、npx 超时、缺 fileKey。可在 .env 加 FIGMA_FILE_KEY=文件密钥。")
        return


async def demo_4_github() -> None:
    """Demo 4：官方 GitHub 远程 MCP。只要看 tools/list，不必在本机起 Docker。

    连 https://api.githubcopilot.com/mcp/ ，用 PAT 放 Authorization: Bearer。
    这是 HTTP 传输：本机不 spawn 子进程。令牌只从环境变量读。
    """
    print("\n" + "=" * 65)
    print("④ GitHub MCP（远程 HTTP，列出有哪些工具）")
    print("=" * 65)
    token = (os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    if not token:
        print("  未配置 GITHUB_PERSONAL_ACCESS_TOKEN，跳过真实连接。")
        print("  在 .env 写一行（不要把令牌贴进代码或聊天）：")
        print("      GITHUB_PERSONAL_ACCESS_TOKEN=github_pat_你的新令牌")
        return

    from langchain_mcp_adapters.client import MultiServerMCPClient

    print_step("连接", "https://api.githubcopilot.com/mcp + Bearer PAT（不打印令牌）")
    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )
    try:
        async with client.session("github") as session:
            print('  直接 call_tool("get_me", {})')
            result = await session.call_tool("get_me", {})
            print(f"  返回:\n    {_tool_text(result)[:700]}")
    except Exception as exc:
        print(f"  连接或调用失败: {type(exc).__name__}: {exc}")
        print_tip("常见原因：令牌已撤销、权限不够、或远程 MCP 要 OAuth。")
        return


async def main() -> None:
    print_lesson(__file__)
    print(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║     Phase 7-2: 标准 MCP Client                            ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )
    print_run_plan(
        [
            ("运行时总图", "两个进程 + ①～⑥，对上每一行 API"),
            ("协议跑通", "本地 math：initialize / list / call"),
            ("桥接 Agent", "load_mcp_tools → create_agent（仍是 ⑥）"),
            ("Figma", "同一套 ①～⑥，进程 B 换成 npx；没令牌就跳过"),
            ("GitHub", "远程 HTTP list 工具；令牌只放 .env"),
        ]
    )
    print_runtime_flow()
    check_api_key()
    # await demo_1_protocol()
    # await demo_2_langchain_agent()
    # await demo_3_figma()
    await demo_4_github()
    print("\n🎉 Phase 7-2 完成。Skill 是说明书，MCP 才是外部工具。\n")


if __name__ == "__main__":
    # MCP 全是 async；asyncio.run 开一个事件循环把 main 跑完
    asyncio.run(main())
