# LangGraph 学习路线

编排层：图、状态、条件边、人审、多 Agent、持久化。  
零件（Prompt/RAG/Tool）在 [LangChain](./langchain_learning_roadmap.md)。

部分 demo 需要仓库根目录 `.env` 的 `ZHIPUAI_API_KEY`。命令在仓库根目录执行。  
Windows 中文终端跑 Studio / `langgraph dev` 时先设 `PYTHONUTF8=1`。

## 阶段零：五个词 + 总览 Demo

**跑：** `python langgraph_demo_examples.py`

先记住这五个，再往下拆文件。细节以 Demo 输出为准，不要对着过期博客抄。

| 特性 | 作用 | 何时用 |
|------|------|--------|
| **State** | 节点之间传递的那份 dict | 任何多步流程 |
| **stream_mode** | `values` 完整状态 / `updates` 只返回变更 | 看进度、调试 |
| **Reducer** | 默认覆盖；`Annotated[..., operator.add]` 改成追加 | 消息列表、计数 |
| **Checkpointer** | 按 `thread_id` 存快照 | 多轮、断点恢复；`interrupt` 必须配 |
| **interrupt** | 节点里暂停，等人；`Command(resume=...)` 继续 | 人工审核 |

补充：`messages` 模式可流 token（打字机）；时间旅行见 `lg_2_3`。

## 第一阶段：动态控制流

| 节 | 文件 | 学什么 |
|----|------|--------|
| 1-1 | `lg_1_1_conditional_edges_demo.py` | `add_conditional_edges`，按 state 选下一跳 |
| 1-2 | `lg_1_2_command_goto_demo.py` | 节点内 `Command(goto=..., update=...)` |
| 1-3 | `lg_1_3_self_correction_demo.py` | 合法回环 + `recursion_limit` |

```
python lg_1_1_conditional_edges_demo.py
```

## 第二阶段：Agent

前置：LangChain `3_1` ~ `3_4`。

| 节 | 文件 | 学什么 |
|----|------|--------|
| 2-1 | `lg_2_1_tool_calling_agent_demo.py` | 手写 ReAct 图：agent ↔ tools |
| 2-2 | `lg_2_2_prebuilt_agent_demo.py` | `create_agent`（不要用已废弃的 `create_react_agent`） |
| 2-3 | `lg_2_3_time_travel_demo.py` | `interrupt`、改 state、再 `resume` |

## 第三阶段：多智能体

| 节 | 文件 | 学什么 |
|----|------|--------|
| 3-1 | `lg_3_1_subgraphs_demo.py` | 子图嵌套 |
| 3-2 | `lg_3_2_supervisor_demo.py` | 主管路由员工 |
| 3-3 | `lg_3_3_send_api_demo.py` | `Send` 并行 Map，Reducer 拼回来 |

## 第四阶段：持久化与长期记忆

| 节 | 文件 | 学什么 |
|----|------|--------|
| 4-1 | `lg_4_1_checkpointer_demo.py` | 同一 `thread_id` 接续；内存 vs SQLite |
| 4-2 | `lg_4_2_store_api_demo.py` | Checkpointer 管一轮对话；Store 按用户跨线程 |

SQLite 后端：`pip install langgraph-checkpoint-sqlite`。

## 第五阶段：工程与平台

| 节 | 文件 | 学什么 |
|----|------|--------|
| 5-1 | `lg_5_1_structured_output_demo.py` | 图节点里 `with_structured_output` |
| 5-2 | `lg_5_2_error_fallback_demo.py` | 节点失败 `Command(goto='fallback')` |
| 5-3 | `lg_5_3_token_optimization_demo.py` | 少把原文重复塞进每次 prompt |
| 5-4 | `lg_5_4_langsmith_tracing_demo.py` | 图同样走 LangSmith |
| 5-5 | `lg_5_5_platform_demo.py` | `langgraph.json` + `langgraph dev` |
| 调试 | `lg_studio_debug_demo.py` | 给 Studio 用的分流图 |

## 推荐顺序

```
langgraph_demo_examples.py
  → lg_1_1 → lg_1_2 → lg_1_3
  → lg_2_1 → lg_2_2 → lg_2_3
  → lg_3_1 → lg_3_2 → lg_3_3
  → lg_4_1 → lg_4_2
  → lg_5_1 ~ lg_5_5
  → 混合架构 6_2_hybrid_architecture_demo.py
```

和 LangChain 的交叉：`lg_2_*` 需要 `@tool`；`lg_5_1` 对应 `5_6`；长期记忆对应 `4_2`。

Skill / MCP / Hooks → [扩展](./extensions_learning_roadmap.md)。
