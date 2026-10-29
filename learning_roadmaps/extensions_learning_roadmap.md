# 扩展学习路线：Skill、MCP、Hooks

给模型加「说明书、外部工具、横切逻辑」。  
不要和 LangChain 的 `@tool` 混成一个词。

前置：LangChain Phase 1 + 3（会 `create_agent` / `@tool`）；Hooks 那节会对照 FastAPI 中间件。命令在仓库根目录执行。

## 三者各管什么

| | Skill | MCP | Hooks |
|--|-------|-----|-------|
| 是什么 | 给模型看的操作规程（`SKILL.md`） | 连外部 Server 的协议 | 在调用前后插一刀（观测或拦截） |
| 不是什么 | 不是可执行 Tool | 不是「再写一套假函数」 | 不是业务节点本身 |
| Demo | `7_1_skill_registry_demo.py` | `7_2_mcp_integration_demo.py` | `7_3_hooks_system_demo.py`、`6_3_hook_manager_demo.py` |

## 7-1 Agent Skill

**跑：** `python 7_1_skill_registry_demo.py`（要模型 Key）

三层披露：

1. 扫描 `skills/**/SKILL.md`，只把 `name` + `description` 给模型（目录）
2. 模型选中后，再注入该文件正文
3. 需要时再读取参考资料、模板等资源，或通过 `run_python` / `run_shell` 执行配套脚本。脚本也可以调用外部 API；Skill 本身是说明和资源，动作由工具执行，外部工具也可通过 MCP 接入。

复杂 Skill 内部也可以有目录，按任务分支继续读取文档，这与 `harness/hs_13_disclosure.py` 的渐进披露机制一致。区别在于 `hs_13` 只演示手写目录和按指定路径取正文；`7_1` 还实现扫描、模型选技能、激活和资源执行。

建议依次在 `catalog_text()`、`pick_skill()`、`activated_skill_prompt()`、`read_skill_file()` / `run_python()` 打断点：观察程序内存中有哪些资料，以及本次真正发送给模型的是哪部分。这两件事不能混为一谈。

不要把工具中的路径检查、工作目录设置当成完整安全沙箱；尤其 shell 命令的能力不受“当前目录”自动限制。更完整的机制对照和提问方法见 [调试与答疑指南](./debugging_and_questions.md)。

仓库里已有 `skills/weather`、`skills/translate`、`skills/docx` 等，往文件夹丢新技能就能被扫到。

## 7-2 MCP

**跑：** `python 7_2_mcp_integration_demo.py`

协议：Client `initialize` → `tools/list` → `tools/call`。本仓库 demo 用标准 MCP，不是模拟。

| Demo 段 | 连接 | 说明 |
|---------|------|------|
| 本地 math | stdio 拉起 `mcp_servers/math_server.py` | 看协议本身，`FastMCP` 是**写 Server** |
| Figma | `npx` 社区 Server + `.env` 的 `FIGMA_ACCESS_TOKEN` | 官方 `https://mcp.figma.com/mcp` 是 OAuth，不是 PAT |
| GitHub | HTTP `https://api.githubcopilot.com/mcp/` + Bearer | `.env`：`GITHUB_PERSONAL_ACCESS_TOKEN` 或 `GITHUB_TOKEN` |

令牌只放 `.env`。桥成 LangChain Tool 时用 `langchain_mcp_adapters`。

## 7-3 / 6-3 Hooks

**跑：**

```bash
python 7_3_hooks_system_demo.py
python 6_3_hook_manager_demo.py
```

| 文件 | 模式 | 和 FastAPI 的直觉 |
|------|------|-------------------|
| `7_3` | LangChain **Callbacks**（看 Token，不能拦）+ **HookManager**（pre 可返回 False 拦住） | Callbacks ≈ 日志中间件；HookManager ≈ 能 `401` 的 Depends |
| `6_3` | 图里加 audit 节点 / `node_wrapper` | 不改业务节点，只包一层 |

盯：`7_3` 里安全/成本 Hook 返回 `False` 时主流程被挡住；Callbacks 只观测。

## 建议顺序

```
7_1 技能说明书
  → 7_2 外部工具协议（先 math，再按需 Figma/GitHub）
  → 7_3 观测 vs 拦截
  → 6_3 图内 hook 节点
```

总顺序见 [总入口](./learning_roadmap.md)。
