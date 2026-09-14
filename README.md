# langtest

动手学 **LangChain / LangGraph** 的工程，顺带补 Python 类型和 FastAPI（给会 Koa、Python 还不熟的人）。

每个 demo 都是可运行的脚本。跑起来后会先打印「本例子要表达什么」，再执行。语言 / 标准库 / FastAPI 每课文件头还有「是什么、用在、不用」。路线图在 [`learning_roadmaps/`](./learning_roadmaps/learning_roadmap.md)，按仓库里**实际有的文件**排，不按过期博客猜。

## 你能学到什么

| 一块 | 内容 | 目录 / 文件 |
|------|------|-------------|
| Python 类型 | `None` / `bool` / 数字 / `str` / `list` / `tuple` / `dict` / `set` / `range`，外加 `enumerate`、标准库 `array`、NumPy `ndarray`，带正例和反例 | `python_basics/` |
| Python 语言 | 面向对象、typing 标注、`with` / contextlib、asyncio | `python_lang/` |
| Python 标准库 | json、pathlib、glob、copy、os/sys、datetime、logging | `python_stdlib/` |
| FastAPI | 路由、校验（422）、中间件与 `Depends`（对照 Koa） | `fastapi_basics/` |
| LangChain | Runnable、LCEL、Prompt、RAG、Tool、Agent、记忆、工程优化 | `1-*` ~ `6_2_*` |
| LangGraph | 图、条件边、人审、多 Agent、Checkpointer | `lg_*`、`langgraph_demo_examples.py` |
| 扩展 | Agent Skill、MCP、Hooks | `7_1` ~ `7_3`、`6_3` |

**LangChain 造零件，LangGraph 拼流程。** Skill 是给模型看的说明书，不是 `@tool`；MCP 用来接 Figma / GitHub 这类外部 Server。

## 怎么开始

在仓库根目录（不要在 `learning_roadmaps/` 里跑命令）。

1. Python 3.10+，建议用已有虚拟环境 `.venv`。
2. 复制一份环境变量（自己建 `.env`，**不要把密钥写进源码或贴到聊天里**）：

```env
ZHIPUAI_API_KEY=你的密钥
ZHIPUAI_API_BASE=          # 走代理/千帆时再填
ZHIPUAI_MODEL=glm-5.2

# 主模型挂了会试 OpenAI 兼容接口（可选）
OPENAI_API_KEY=
OPENAI_API_BASE=
OPENAI_MODEL=gpt-4o-mini

# 可选：MCP Demo
FIGMA_ACCESS_TOKEN=
GITHUB_PERSONAL_ACCESS_TOKEN=

# 可选：LangSmith 追踪
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=
```

3. 不需要模型的部分可以先跑：

```bash
python python_basics/pb_run_all.py
python python_lang/pl_run_all.py
python python_stdlib/ps_run_all.py
python fastapi_basics/fb_run_all.py
```

4. 需要模型时：

```bash
python 1-1runnable_demo.py
python langgraph_demo_examples.py
```

模型入口在 `model_provider.py`，各 demo 共用。RAG 向量有时要单独的 Embedding Key，以报错提示为准。

## 建议学习顺序

完整表格见 [学习路线总入口](./learning_roadmaps/learning_roadmap.md)。

```
Python 类型 → Python 语言 → Python 标准库 → FastAPI → LangChain Phase 1 → LangGraph 阶段零
  → RAG / Tool → LangGraph Agent → 扩展（Skill / MCP / Hooks）
```

会 Koa、Python 不熟：先类型，再语言和标准库，再 FastAPI。已经会类型：LangChain 第一段和 LangGraph 总览可以交叉，不必等 LangChain 全部学完。

| 路线 | 文档 |
|------|------|
| 总入口 | [learning_roadmaps/learning_roadmap.md](./learning_roadmaps/learning_roadmap.md) |
| Python | [python_basics_learning_roadmap.md](./learning_roadmaps/python_basics_learning_roadmap.md) |
| Python 语言 | [python_lang_learning_roadmap.md](./learning_roadmaps/python_lang_learning_roadmap.md) |
| Python 标准库 | [python_stdlib_learning_roadmap.md](./learning_roadmaps/python_stdlib_learning_roadmap.md) |
| FastAPI | [fastapi_learning_roadmap.md](./learning_roadmaps/fastapi_learning_roadmap.md) |
| LangChain | [langchain_learning_roadmap.md](./learning_roadmaps/langchain_learning_roadmap.md) |
| LangGraph | [langgraph_learning_roadmap.md](./learning_roadmaps/langgraph_learning_roadmap.md) |
| Skill / MCP / Hooks | [extensions_learning_roadmap.md](./learning_roadmaps/extensions_learning_roadmap.md) |

## 目录在干什么

```
python_basics/          类型教材（pb_0 ~ pb_12）
fastapi_basics/         FastAPI 教材（fb_0 ~ fb_2）
learning_roadmaps/      学习路线（只放 md）
skills/                 Agent Skill（SKILL.md）
mcp_servers/            本地 MCP Server 示例
demo_explain.py         所有 demo 共用的「本例子要表达什么」
model_provider.py       读 .env、选模型
1-1 … 1-5               LangChain 核心
2_*                     RAG
3_*                     Tool / Agent
4_*                     Memory
5_*                     缓存、流式、异步、LangSmith…
6_*                     和 LangGraph 怎么分工
7_*                     Skill / MCP / Hooks
lg_*                    LangGraph 分节
langgraph_demo_examples.py   LangGraph 五件套总览
langgraph.json          Studio / langgraph dev
```

Windows 中文终端跑 `langgraph dev` 时建议先设 `PYTHONUTF8=1`。

## 约定

- 命令一律在仓库**根目录**执行。
- 密钥只放 `.env`，不要提交 git、不要写进 demo。
- 类型和 FastAPI 教材里的【反例】是故意跑失败，用来对照报错，不是程序坏了。
