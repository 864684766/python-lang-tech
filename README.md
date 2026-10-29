# python-lang-tech

动手学 **LangChain / LangGraph** 的工程，顺带补 Python 类型和 FastAPI（给会 Koa、Python 还不熟的人）。

每个 demo 都是可运行的脚本，多数会先打印「本例子要表达什么」，再执行；部分概念课直接打印教学说明。语言 / 标准库 / FastAPI 每课文件头还有「是什么、用在、不用」。路线图在 [`learning_roadmaps/`](./learning_roadmaps/learning_roadmap.md)，按仓库里**实际有的文件**排，不按过期博客猜。

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
| Harness | 模型外面那一层：建议 vs 执行；再加人审 / 保险丝 / 窗口。后半是进化专题：文件交接、按条改手册、环外验收。加法只是道具 | `harness/` |

**LangChain 造零件，LangGraph 拼流程。** Skill 是给模型看的说明书，不是 `@tool`；MCP 用来接 Figma / GitHub 这类外部 Server。

## 怎么开始

在仓库根目录（不要在 `learning_roadmaps/` 里跑命令）。

1. Python 3.13+（以 `pyproject.toml` 的要求为准），建议用已有虚拟环境 `.venv`。
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

4. 需要模型时（Harness 的在线执行课也要 Key，概念地图无需 Key）：

```bash
python harness/hs_1_cannot_act.py
python 1-1runnable_demo.py
python langgraph_demo_examples.py
```

模型入口在 `model_provider.py`，各 demo 共用。RAG 向量有时要单独的 Embedding Key，以报错提示为准。

## 怎样学：运行、手动调试、带着问题问 AI，再验证

本项目适合边运行边学。第一次看不懂，不必马上通读整个框架；先选一个小例子，沿着输入、模型回复、工具执行和最终状态追踪一遍。

1. **先读现象与问题。** 用自己的话说出“原来哪里不对”“这次增加了什么机制”，再运行单个脚本。先别用总入口一次跑完所有在线示例。
2. **手动打断点。** 在 IDE 中选择项目的 `.venv` 解释器，在 `invoke()` 前后、工具函数第一行、验收判断处暂停。用“单步跳过”先看本层变量；需要追踪调用关系时再“单步进入”。
3. **先猜，再看。** 例如先猜 `ai` 是完整历史还是单条回复，再查看 `type(ai)`、`ai.content`、`ai.tool_calls`。记录预期与实际的差异，这往往就是最有价值的问题。
4. **带代码和现象问 AI。** 提供文件名、相关代码、输入、实际结果和自己的理解，让 AI 对照当前实现解释。不要只问“这个是什么意思”，也不要直接要求它把失败改成通过。
5. **回到代码验证解释。** 用一个小输入或单处修改验证结论。例如对比 `model.invoke()` 与绑定工具后的调用，或在 `hs_14` 中查看修改前后的真实工具顺序。模型解释也可能有错，应以当前代码和运行证据核对。
6. **写一句学习结论。** 记下“现象 → 原因 → 解决办法 → 不这样做的后果”，并标注哪些是真实执行、哪些是固定提案或模拟数据。

可以直接使用这个提问模板：

```text
我正在调试【文件名 / 函数名】。
这段代码是：【只贴相关部分】。
输入是：【输入】，实际返回是：【删去密钥和私人数据后的结果】。
我原以为会：【预期】，但实际看到：【现象】。
我的理解是：【自己的判断】。
请对照代码说明原因，并给一个我能单步验证的小例子。
还请区分：实际发生了什么、作者想演示什么、这个例子没有实现什么。
```

针对本项目常见疑问，继续读 [调试与答疑指南](./learning_roadmaps/debugging_and_questions.md)。其中包含断点位置、消息与历史、Tool Calling、Skill 渐进披露及 Python/JavaScript 对照。

## 建议学习顺序

完整表格见 [学习路线总入口](./learning_roadmaps/learning_roadmap.md)。

```
Python 类型 → Python 语言 → Python 标准库 → FastAPI → Harness（真 GLM）
  → LangChain Phase 1 → LangGraph 阶段零 → RAG / Tool → Deep Agents
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
| Harness / Deep Agents | [harness_learning_roadmap.md](./learning_roadmaps/harness_learning_roadmap.md) |

## 目录在干什么

```
python_basics/          类型教材（pb_0 ~ pb_12）
fastapi_basics/         FastAPI 教材（fb_0 ~ fb_2）
harness/                Harness 教程（课表见学习路线；执行机制、Deep Agents、持久化与改进）
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
