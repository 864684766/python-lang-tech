# -*- coding: utf-8 -*-
"""
LangGraph Phase 5-5: LangGraph Platform / CLI 概念 Demo

学习目标:
  · langgraph dev 本地开发服务器
  · langgraph.json 项目配置概念
  · 无需 API Key，纯本地说明 + CLI 检测

运行: python lg_5_5_platform_demo.py
"""

import shutil
import subprocess
from demo_explain import print_lesson, print_step, print_tip


PLATFORM_GUIDE = """
LangGraph Platform 本地开发流程:

1. 安装 CLI
   pip install langgraph-cli

2. 项目根目录创建 langgraph.json
   {
     "graphs": {
       "agent": "./lg_2_2_prebuilt_agent_demo.py:agent"
     },
     "env": ".env"
   }

3. 启动开发服务器
   langgraph dev
   → 提供 REST API + LangGraph Studio UI
   中文 Windows 需先设置用户环境变量 PYTHONUTF8=1（官方 issue 8060），否则会按 GBK 读 UTF-8 文件。

4. 打包部署
   langgraph build

典型目录结构:
  my_project/
  ├── langgraph.json
  ├── .env
  ├── graphs/
  │   └── agent.py      # compile() 后的 graph 导出
  └── requirements.txt
"""


def check_cli():
    print("=" * 65)
    print("LangGraph 5-5: Platform / CLI")
    print("=" * 65)

    print_step(1, "检测 CLI", "有 langgraph 命令才能 langgraph dev；没有也不影响理解配置")
    cli = shutil.which("langgraph")
    if cli:
        print(f"  ✅ langgraph CLI 已安装: {cli}")
        try:
            out = subprocess.run(
                ["langgraph", "--help"],
                capture_output=True, text=True, timeout=10,
            )
            print(out.stdout[:400])
        except Exception as exc:
            print(f"  运行 --help 失败: {exc}")
    else:
        print("  ❌ langgraph CLI 未安装")
        print("  安装: pip install langgraph-cli")

    print_step(2, "配置样例", "graphs 里的值是「文件路径:导出的 compiled graph 变量名」")
    print(PLATFORM_GUIDE)
    print_tip("把 6_2 里的 Graph compile() 后导出，再写进 langgraph.json 就能当 API 跑")


if __name__ == "__main__":
    print_lesson(__file__)
    check_cli()
