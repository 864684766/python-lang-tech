# LangChain 学习路线

零件层：Prompt、模型、解析、RAG、Tool、记忆、工程优化。  
流程图、人审、多 Agent 在 [LangGraph](./langgraph_learning_roadmap.md)。  
Skill / MCP / Hooks 在 [扩展](./extensions_learning_roadmap.md)。

多数 demo 需要仓库根目录 `.env` 里的 `ZHIPUAI_API_KEY`。命令在仓库根目录执行。

前置建议：[Python 类型](./python_basics_learning_roadmap.md)。

## Phase 1 核心抽象

| 节 | 文件 | 学什么 |
|----|------|--------|
| 1-1 | `1-1runnable_demo.py` | `invoke` / `stream` / `batch` / `with_retry` / `with_fallbacks` |
| 1-2 | `1-2lcel_demo.py` | `\|` 管道；Parallel / Passthrough / 分支 |
| 1-3 | `1-3prompt_template_demo.py` | `ChatPromptTemplate`、角色消息、变量槽 |
| 1-4 | `1-4output_parser_demo.py` | 文本抠成 list / JSON / Pydantic |
| 1-5 | `1-5model_layer_demo.py` | 消息类型、`bind_tools`、结构化输出入口 |

验证：能自己串 `prompt | model | parser`。

```bash
python 1-1runnable_demo.py
```

## Phase 2 RAG

| 节 | 文件 | 学什么 |
|----|------|--------|
| 2-1 | `2_1_document_loaders.py` | 文档 → `Document` |
| 2-2 | `2_2_text_splitters.py` | 切块 |
| 2-3 | `2_3_embeddings_teacher.py` | 文本 → 向量 |
| 2-4 | `2_4_vector_store_teacher.py` | 存和搜 |
| 2-5 | `2_5_retriever_teacher.py` | Retriever 进阶 |
| 2-6 | `2_6_rag_pipeline_demo.py` | Load → Split → Embed → Retrieve → 生成 |

`rag_pipeline_demo.py` 是早期总览，新学习以 `2_6` 为准。

## Phase 3 Tools & Agents

| 节 | 文件 | 学什么 |
|----|------|--------|
| 3-1 | `3_1_tool_decorator_demo.py` | `@tool`，docstring 即描述 |
| 3-2 | `3_2_create_agent_demo.py` | `create_agent(model, tools)` |
| 3-3 | `3_3_react_agent_demo.py` | ReAct：想 → 调工具 → 再想 |
| 3-4 | `3_4_function_calling_demo.py` | 模型吐 `tool_calls`，不是自己编格式 |
| 3-5 | `3_5_tool_error_handling_demo.py` | 工具失败、重试、主备 |

图里手写 ReAct 见 LangGraph `lg_2_1` / `lg_2_2`。

## Phase 4 Memory

| 节 | 文件 | 学什么 |
|----|------|--------|
| 4-1 | `4_1_message_history_demo.py` | 会话内消息历史 |
| 4-2 | `4_2_memory_layers_demo.py` | 窗口 / 摘要 / 长期向量 |
| 4-3 | `4_3_memory_in_chain_demo.py` | Memory 接到 LCEL 上 |

多轮图状态用 LangGraph Checkpointer，不要和这里混成一套 API。

## Phase 5 工程

| 节 | 文件 | 学什么 |
|----|------|--------|
| 5-1 | `5_1_cache_demo.py` | 相同入参不打 API |
| 5-2 | `5_2_streaming_demo.py` | `stream` |
| 5-3 | `5_3_async_demo.py` | `ainvoke` / `abatch` |
| 5-4 | `5_4_langsmith_demo.py` | Tracing（需 LangSmith 配置） |
| 5-5 | `5_5_package_management_demo.py` | 包怎么拆着装 |
| 5-6 | `5_6_structured_output_demo.py` | `with_structured_output` |

图节点里的结构化输出见 `lg_5_1`。

## Phase 6 和 LangGraph 怎么分工

| 节 | 文件 | 学什么 |
|----|------|--------|
| 6-1 | `6_1_architecture_selection_demo.py` | 线性用 LCEL，有状态/人审用图 |
| 6-2 | `6_2_hybrid_architecture_demo.py` | 图管流程，节点里仍可用 LCEL |

`6_3_hook_manager_demo.py` 算横切，放到 [扩展路线](./extensions_learning_roadmap.md)，和 `7_3` 一起看。

原则：LangGraph 管流程，LCEL 管节点里的数据处理；`@tool` 在 LangChain 声明，在图里调度。

## 和别的路线怎么接

- 学完 Phase 1 → 可以去 LangGraph 阶段零 + `lg_1_*`（条件边）
- 学完 Phase 3 → 去 `lg_2_*`（图里的 Agent）
- 要对外 HTTP → [FastAPI](./fastapi_learning_roadmap.md)
- 要接 Figma/GitHub 或 Skill 说明书 → [扩展](./extensions_learning_roadmap.md)
