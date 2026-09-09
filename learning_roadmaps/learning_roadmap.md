# 学习路线总入口

文档在 `learning_roadmaps/`。按仓库里**现有能跑的 demo**排，不按过期文档猜。密钥只放 `.env`，不要写进源码。

下面所有 `python ...` 命令都在**仓库根目录**执行（不是本文件夹）。

## 建议顺序

| 顺序 | 路线 | 文件 | 要不要模型 Key |
|------|------|------|----------------|
| 0 | [Python 类型](./python_basics_learning_roadmap.md) | `python_basics/` | 否 |
| 1 | [FastAPI](./fastapi_learning_roadmap.md) | `fastapi_basics/` | 否 |
| 2 | [LangChain](./langchain_learning_roadmap.md) | `1-*` ~ `6_2_*` | 多数要 |
| 3 | [LangGraph](./langgraph_learning_roadmap.md) | `lg_*`、`langgraph_demo_examples.py` | 部分要 |
| 4 | [扩展：Skill / MCP / Hooks](./extensions_learning_roadmap.md) | `7_1` ~ `7_3`、`6_3` | 7-1/7-3 要；7-2 可选令牌 |

会 Koa、Python 不熟：先 0 再 1。  
已经会类型：LangChain Phase 1 和 LangGraph 阶段零可以交叉，不必等 LangChain 全部学完。

## 怎么跑

在仓库根目录：

```bash
python python_basics/pb_run_all.py
python fastapi_basics/fb_run_all.py
python 1-1runnable_demo.py
python langgraph_demo_examples.py
```

每个 demo 开头会打印「本例子要表达什么」（`demo_explain.py`）。

需要模型的 demo：`.env` 里配置 `ZHIPUAI_API_KEY`（以及你实际用的厂商 Key）。
