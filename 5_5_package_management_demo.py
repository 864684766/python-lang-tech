# -*- coding: utf-8 -*-
"""
========================================================
LangChain Phase 5-5
包管理与依赖控制 教材级 Demo
========================================================

📚 学习目标：
1. 理解 LangChain 生态「拆包」后的安装策略
2. 只安装项目真正需要的 integration package
3. 检测当前环境已安装的核心包及版本
4. 避免 pip install langchain 拉入大量无用依赖

本 Demo 无需 API Key，纯本地运行。

版本基准：LangChain 1.3.14
"""

import importlib
import subprocess
import sys
from typing import Optional
from demo_explain import print_lesson, print_run_plan


# ============================================================================
# 推荐的最小依赖矩阵（按场景）
# ============================================================================

INSTALL_PROFILES = {
    "core_lcel": {
        "desc": "纯 LCEL 管道（Prompt → Model → Parser）",
        "packages": ["langchain-core", "python-dotenv"],
    },
    "zhipu_chat": {
        "desc": "智谱 GLM 聊天模型",
        "packages": ["langchain-core", "langchain-community", "zhipuai", "python-dotenv"],
    },
    "rag_local": {
        "desc": "本地 RAG（Chroma + 智谱 Embedding）",
        "packages": [
            "langchain-core", "langchain-community", "langchain-text-splitters",
            "langchain-chroma", "chromadb", "zhipuai", "python-dotenv",
        ],
    },
    "agent": {
        "desc": "Agent + Tool Calling",
        "packages": [
            "langchain", "langchain-core", "langchain-community",
            "langgraph", "zhipuai", "python-dotenv",
        ],
    },
    "full_stack": {
        "desc": "LangChain + LangGraph + RAG + Agent（本项目）",
        "packages": [
            "langchain", "langchain-core", "langchain-community",
            "langchain-text-splitters", "langchain-chroma",
            "langgraph", "chromadb", "zhipuai", "python-dotenv", "pydantic",
        ],
    },
}


# import_name → pip package name 映射（部分包 import 名与 pip 名不同）
IMPORT_TO_PIP = {
    "langchain": "langchain",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_chroma": "langchain-chroma",
    "langgraph": "langgraph",
    "chromadb": "chromadb",
    "zhipuai": "zhipuai",
    "dotenv": "python-dotenv",
    "pydantic": "pydantic",
    "langsmith": "langsmith",
}


def get_version(import_name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "已安装(无 __version__)")
    except ImportError:
        return None


# ============================================================================
# Demo 1：生态包结构说明
# ============================================================================

def demo_1_package_structure():
    print("\n" + "=" * 65)
    print("Demo 1: LangChain 生态包结构")
    print("=" * 65)
    print("""
    LangChain 1.x 采用「拆包」架构，各包职责清晰：

    langchain-core      核心抽象（Runnable, Prompt, Message, Tool）
    langchain           meta-package + create_agent 等高层 API
    langchain-community 第三方集成（Loaders, ChatModel, Cache...）
    langchain-openai    OpenAI 专用集成（按需安装）
    langchain-chroma    Chroma 向量库集成
    langchain-text-splitters  文本分块器
    langgraph           有状态流程编排（Agent 底层引擎）
    langsmith           Tracing & Evaluation（可选）

    ⚠️  避免：pip install langchain[all] 或盲目装全量 extras
    ✅  推荐：按场景最小化安装（见下方 INSTALL_PROFILES）
    """)


# ============================================================================
# Demo 2：检测当前环境
# ============================================================================

def demo_2_check_installed():
    print("\n" + "=" * 65)
    print("Demo 2: 当前环境包检测")
    print("=" * 65)

    check_list = [
        "langchain", "langchain_core", "langchain_community",
        "langchain_text_splitters", "langchain_chroma",
        "langgraph", "chromadb", "zhipuai", "langsmith", "pydantic",
    ]

    print(f"Python: {sys.version.split()[0]}\n")
    print(f"{'包名':<30} {'版本':<20} {'状态'}")
    print("-" * 60)

    for name in check_list:
        ver = get_version(name)
        status = "✅" if ver else "❌ 未安装"
        print(f"{name:<30} {ver or '-':<20} {status}")


# ============================================================================
# Demo 3：按场景推荐安装命令
# ============================================================================

def demo_3_install_profiles():
    print("\n" + "=" * 65)
    print("Demo 3: 按场景推荐安装命令")
    print("=" * 65)

    for profile_id, info in INSTALL_PROFILES.items():
        print(f"\n【{profile_id}】{info['desc']}")
        cmd = "pip install " + " ".join(info["packages"])
        print(f"  {cmd}")


# ============================================================================
# Demo 4：验证核心 API 可用性
# ============================================================================

def demo_4_api_smoke_test():
    print("\n" + "=" * 65)
    print("Demo 4: 核心 API 导入冒烟测试")
    print("=" * 65)

    tests = [
        ("langchain_core.runnables", "Runnable 协议"),
        ("langchain_core.prompts", "Prompt Template"),
        ("langchain_core.tools", "@tool 装饰器"),
        ("langchain.agents", "create_agent"),
        ("langgraph.graph", "StateGraph"),
    ]

    for module, desc in tests:
        try:
            importlib.import_module(module)
            print(f"  ✅ {desc} ({module})")
        except ImportError as exc:
            print(f"  ❌ {desc} ({module}): {exc}")


# ============================================================================
# Demo 5：依赖冲突排查建议
# ============================================================================

def demo_5_troubleshooting():
    print("\n" + "=" * 65)
    print("Demo 5: 依赖冲突排查")
    print("=" * 65)
    print("""
    常见问题与解决：

    1. ImportError: langchain.agents.create_agent
       → pip install -U langchain langgraph

    2. ChatZhipuAI 找不到
       → pip install langchain-community zhipuai

    3. 版本不兼容
       → pip show langchain langchain-core langgraph
       → 确保 langchain >= 1.0, langgraph >= 0.2

    4. 查看已安装版本:
       pip list | grep -i langchain

    5. 干净环境重建:
       python -m venv .venv && pip install -r requirements.txt
    """)

    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True, text=True, timeout=30,
        )
        lang_pkgs = [
            line for line in out.stdout.splitlines()
            if "lang" in line.lower() or "zhipu" in line.lower()
        ]
        if lang_pkgs:
            print("当前 lang 相关包:")
            for line in lang_pkgs[:15]:
                print(f"  {line}")
    except Exception:
        pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_lesson(__file__)
    print("""
╔══════════════════════════════════════════════════════════╗
║     LangChain Phase 5-5: 包管理与依赖控制                 ║
║     无需 API Key                                         ║
╚══════════════════════════════════════════════════════════╝
""")
    print_run_plan([
        ('包结构', 'langchain / langchain-core / 厂商包怎么拆'),
        ('检查已装版本', '对照教材目标 1.3.x'),
        ('安装档位', '最小集 vs 完整教材'),
        ('API 冒烟', '关键 import 是否还能用'),
        ('排错', '常见版本冲突'),
    ])

    demo_1_package_structure()
    demo_2_check_installed()
    demo_3_install_profiles()
    demo_4_api_smoke_test()
    demo_5_troubleshooting()

    print("\n🎉 Phase 5-5 完成！下一步: 5_6_structured_output_demo.py\n")
