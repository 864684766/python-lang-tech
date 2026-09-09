# -*- coding: utf-8 -*-
"""
所有教材 Demo 共用的「本例子要表达什么」运行时说明。

用法:
    from demo_explain import print_lesson, print_step, print_tip
    print_lesson(__file__)
"""

from __future__ import annotations

from pathlib import Path

LESSONS: dict[str, dict[str, str]] = {
    # ── LangChain Phase 1 ──
    "1-1runnable_demo.py": {
        "what": "Runnable 是 LangChain 的统一调用协议：invoke / stream / batch / with_retry / with_fallbacks。",
        "why": "后面所有链、Agent、图节点，本质都是 Runnable。先掌握这套接口，后面不用反复学调用方式。",
        "watch": "对比 invoke（一次拿齐）和 stream（边生成边吐）；with_fallbacks 是主模型失败换备胎。",
        "flow": "Prompt → Model.invoke → 完整回答",
    },
    "1-2lcel_demo.py": {
        "what": "LCEL 用 | 把 Prompt、Model、Parser 串成一条流水线。",
        "why": "线性任务不必上图。Prompt→模型→解析，一条链就能跑通。",
        "watch": "chain.invoke({变量}) 时数据从左流到右；RunnableParallel / Branch 是链上的分支。",
        "flow": "dict 入参 → Prompt 填槽 → LLM → Parser → 字符串/结构",
    },
    "1-3prompt_template_demo.py": {
        "what": "PromptTemplate / ChatPromptTemplate 把固定话术和 {变量} 分开。",
        "why": "提示词不要散落在字符串拼接里，模板可复用、可测试。",
        "watch": "system / human 角色消息如何拼成 messages 列表。",
        "flow": "模板 + 变量 → messages → 交给模型",
    },
    "1-4output_parser_demo.py": {
        "what": "Output Parser 把模型的自由文本抠成 JSON / 列表 / Pydantic 对象。",
        "why": "下游代码不能靠「猜」模型说了什么，要可解析的结构。",
        "watch": "软解析（失败能兜底）和硬约束 structured_output 的差别，后者见 5-6 / lg_5_1。",
        "flow": "LLM 文本 → Parser → Python 对象",
    },
    "1-5model_layer_demo.py": {
        "what": "Chat Model 层：temperature、bind_tools、不同厂商模型的替换点。",
        "why": "业务代码应依赖「聊天模型接口」，而不是写死某一家 HTTP。",
        "watch": "本项目 get_model() 已封装主备切换，见 model_provider.py。",
        "flow": "messages → ChatModel.invoke → AIMessage",
    },
    # ── Phase 2 RAG ──
    "2_6_rag_pipeline_demo.py": {
        "what": "完整 RAG：Loader → Split → Embed → VectorStore → Retriever → Prompt+LLM。",
        "why": "模型没有你的私有文档。先检索再生成，回答才能接地气。",
        "watch": "索引阶段（离线）和查询阶段（每次提问）是两条时间线。",
        "flow": "文档入库 → 问题检索片段 → 片段+问题进 Prompt → 回答",
    },
    "rag_pipeline_demo.py": {
        "what": "早期 RAG 总览（与 2_6 同类）。新学习请以 2_6_rag_pipeline_demo.py 为准。",
        "why": "对照完整流水线各环节职责。",
        "watch": "Loader / Splitter / Embedding / Retriever 各自输入输出。",
        "flow": "同 2_6",
    },
    # ── Phase 3 Tools & Agents ──
    "3_1_tool_decorator_demo.py": {
        "what": "@tool 把普通函数变成 LLM 可调用的工具（name / description / 参数 schema）。",
        "why": "Agent 不会凭空访问天气或计算器，必须把能力声明成 Tool。",
        "watch": "docstring 会变成工具描述；LLM 靠描述决定调不调、怎么填参。",
        "flow": "函数 → @tool → BaseTool → bind_tools / create_agent",
    },
    "3_2_create_agent_demo.py": {
        "what": "create_agent(model, tools) 一行得到「模型选工具 → 执行 → 再思考」循环。",
        "why": "手写 ReAct 图太长；标准 Agent 用工厂函数即可。",
        "watch": "invoke 入参是 {messages: [...]}；最终回答在 messages 最后一条。",
        "flow": "用户问题 → model 节点 ↔ tools 节点 → 最终 AIMessage",
    },
    "3_3_react_agent_demo.py": {
        "what": "ReAct：Reason（想）+ Act（调工具），多轮直到能直接回答。",
        "why": "复杂问题往往要先查再算，不能一轮生成完。",
        "watch": "中间会出现 tool_calls，然后才是自然语言终答。",
        "flow": "思考 → 工具 → 观察 → 再思考 → … → 回答",
    },
    "3_4_function_calling_demo.py": {
        "what": "Function Calling：模型输出结构化 tool_calls，而不是自己编造调用格式。",
        "why": "bind_tools 后，模型按 JSON schema 填参数，代码才能可靠执行。",
        "watch": "AIMessage.tool_calls 里的 name / args / id。",
        "flow": "bind_tools → invoke → tool_calls → 执行 → ToolMessage 回灌",
    },
    "3_5_tool_error_handling_demo.py": {
        "what": "工具失败时的层次：内部返回字符串、ToolException、Retry/Error 中间件、主备工具。",
        "why": "网络抖动不该让整个 Agent 崩溃。",
        "watch": "ToolException 必须 handle_tool_error=True，否则图会直接炸。",
        "flow": "工具抛错 → 中间件/错误消息 → LLM 看到错误再改策略",
    },
    # ── Phase 4 Memory ──
    "4_1_message_history_demo.py": {
        "what": "会话记忆：InMemoryChatMessageHistory + RunnableWithMessageHistory。",
        "why": "多轮对话必须带上前文，否则模型「失忆」。",
        "watch": "session_id 区分不同用户/会话。",
        "flow": "本次输入 + 历史 messages → 模型 → 新消息写回 history",
    },
    "4_2_memory_layers_demo.py": {
        "what": "记忆分层：窗口内近期消息 vs 摘要 vs 长期向量。",
        "why": "把全部历史塞进上下文会爆 token、又贵又慢。",
        "watch": "trim_messages 截断；摘要节点压缩旧对话。",
        "flow": "短记忆直接带 / 长记忆先摘要或检索再带",
    },
    "4_3_memory_in_chain_demo.py": {
        "what": "把 Memory 接到 LCEL 链上，而不是自己维护列表。",
        "why": "链复用 invoke/stream 时，记忆注入应是管道的一环。",
        "watch": "history 如何作为 chain 的一个输入变量。",
        "flow": "load history → 填进 Prompt → 调用 → 保存新轮",
    },
    # ── Phase 5 工程 ──
    "5_1_cache_demo.py": {
        "what": "LLM 缓存：相同入参命中缓存则不再打 API。",
        "why": "省钱、降延迟；演示/回归测试也能更稳。",
        "watch": "第一次慢、第二次快；改 prompt 后缓存失效。",
        "flow": "invoke → 查 cache → 未命中才请求模型",
    },
    "5_2_streaming_demo.py": {
        "what": "stream() 按 token/块增量输出，适合聊天 UI。",
        "why": "用户不想等整段生成完才看到第一个字。",
        "watch": "for chunk in chain.stream(...) 每次打印一小段。",
        "flow": "模型边生成边回调 → 前端逐字显示",
    },
    "5_3_async_demo.py": {
        "what": "ainvoke / abatch 异步并发，吞吐比同步循环高。",
        "why": "服务端同时处理多个请求时不要阻塞线程。",
        "watch": "asyncio.gather 同时跑多条链。",
        "flow": "多个 ainvoke 并发 → 一起等待结果",
    },
    "5_4_langsmith_demo.py": {
        "what": "LangSmith：把每次链/节点调用上报，便于排查。",
        "why": "Agent 多跳失败时，没有 trace 只能盲猜。",
        "watch": "环境变量 LANGCHAIN_TRACING_V2 + API Key；Dashboard 里看耗时。",
        "flow": "invoke → 自动埋点 → smith.langchain.com",
    },
    "5_5_package_management_demo.py": {
        "what": "依赖与版本：langchain / langgraph / 厂商包如何拆。",
        "why": "API 经常变，锁版本才能让教材和代码对得上。",
        "watch": "本仓库目标 LangChain 1.3.x；create_react_agent 已废弃。",
        "flow": "pip 包 → import 路径 → 对应官方文档版本",
    },
    "5_6_structured_output_demo.py": {
        "what": "with_structured_output：用 Pydantic/TypedDict 强制模型吐 JSON。",
        "why": "比 OutputParser 更硬，字段缺失会校验失败而不是 silently 乱答。",
        "watch": "Field(description=) 会进 schema，模型靠描述填字段。",
        "flow": "schema → 模型约束输出 → Pydantic 对象",
    },
    # ── Phase 6 混合架构 ──
    "6_1_architecture_selection_demo.py": {
        "what": "选型：线性用 LCEL，有状态/人审/多 Agent 用 LangGraph。",
        "why": "不是所有问题都要上图，也不是所有问题一条链就够。",
        "watch": "决策树打印；对照后面 6_2 混合示例。",
        "flow": "需求复杂度 → 选 LCEL / Graph / 混合",
    },
    "6_2_hybrid_architecture_demo.py": {
        "what": "混合架构：LangGraph 管流程，LCEL 管单节点里的 Prompt|Model|Parser。",
        "why": "图负责状态机，链负责数据处理，各干各的。",
        "watch": "节点内部仍是 LCEL；跨节点才用 State。",
        "flow": "START → 若干 Graph 节点（内部 LCEL）→ END",
    },
    "6_3_hook_manager_demo.py": {
        "what": "在图里加 audit_hook 节点：不改业务节点，横切日志/校验。",
        "why": "权限、审计不要写进每个业务函数。",
        "watch": "和 7_3 对比：6_3 是图内节点，7_3 还有 Callbacks + 可阻断的 HookManager。",
        "flow": "业务节点前后插 hook 节点 / wrapper",
    },
    # ── Phase 7 扩展 ──
    "7_1_skill_registry_demo.py": {
        "what": "按 Anthropic Agent Skills 规范：Level 1 只注入 name - description；命中后 Level 2 注入 SKILL.md 正文；Level 3 按需读文件/跑脚本。",
        "why": "技能调用标准是 Anthropic 的 Agent Skills（agentskills.io），不是自拟「Cursor 同类」提示词。说明书即规程。",
        "watch": "1+1 不选技能、直接用模型答；工具失败要说明；GraphRecursionError 捕获后友好收尾，demo 继续。",
        "flow": "metadata catalog → 选 skill → 注入 SKILL.md instructions → tools 对应 filesystem/bash",
    },
    "7_2_mcp_integration_demo.py": {
        "what": "标准 MCP Client：stdio 连接 Server，initialize → tools/list → tools/call；再桥成 LangChain Tool。",
        "why": "Figma 等外部能力走同一套协议，不要为每个源手写假 Server。",
        "watch": "math 看协议；Figma/GitHub 令牌只放 .env。切勿把 PAT 写进源码或聊天。",
        "flow": "拉起 Server 进程 → initialize → list_tools → call_tool（Agent 路径多一步 load_mcp_tools）",
    },
    "7_3_hooks_system_demo.py": {
        "what": "三重 Hook：Callbacks 观测 Token；HookManager 可拦截敏感请求。",
        "why": "观测（看）和管控（拦）要分开：Callbacks 不能阻断，pre-hook 可以。",
        "watch": "Demo 3 安全/成本 Hook 返回 False 时主流程被挡住。",
        "flow": "请求 → pre hooks → 业务 LLM → post hooks / callbacks",
    },
    # ── LangGraph ──
    "lg_1_1_conditional_edges_demo.py": {
        "what": "条件边：节点跑完后，按 state['intent'] 选择下一跳（翻译/总结/未知）。",
        "why": "固定 add_edge 只能走一条路；业务要根据分类结果分支。",
        "watch": "route 函数返回的字符串必须出现在映射表的 key 里。",
        "flow": "classify → (条件边) → translate | summarize | unknown → END",
    },
    "lg_1_2_command_goto_demo.py": {
        "what": "节点内部用 Command(goto=..., update=...) 自己决定下一跳并改 state。",
        "why": "有时路由逻辑就在打分节点里，不必再写外部 route 函数。",
        "watch": "分数低且未达次数上限会 goto='generate' 重写；否则 goto='finalize'。",
        "flow": "generate → score ─┬─ 低分 → generate（循环）\n                    └─ 过关 → finalize",
    },
    "lg_1_3_self_correction_demo.py": {
        "what": "图上的合法回环：生成 SQL → 校验失败 → 带着 error 再生成；recursion_limit 防死循环。",
        "why": "自我修正是 Agent 常用模式，但必须有退出条件。",
        "watch": "attempts 用 operator.add 累加；超 recursion_limit 会抛 GraphRecursionError。",
        "flow": "generate_sql → run_sql ─┬─ 非法 → generate_sql\n                         └─ 通过 → END",
    },
    "lg_2_1_tool_calling_agent_demo.py": {
        "what": "手写 ReAct 图：agent 节点 bind_tools，tools 节点执行 tool_calls，条件边决定是否继续。",
        "why": "看懂 create_agent 内部在干什么，才能自定义。",
        "watch": "invoke_with_logged_retry：模型超时会打印等待秒数再重试。",
        "flow": "agent ─┬─ 有 tool_calls → tools → 回到 agent\n        └─ 无 tool_calls → END",
    },
    "lg_2_2_prebuilt_agent_demo.py": {
        "what": "create_agent 预构建 vs 手写图；ModelRetryMiddleware / with_retry 处理超时重试。",
        "why": "生产环境 API 会 429/超时，必须自动重试，间隔用指数退避。",
        "watch": "Demo 3 只打印等待时间表；Demo 2 才是真的超时再请求。",
        "flow": "create_agent(model, tools) → invoke；失败则 backoff 后重试",
    },
    "lg_2_3_time_travel_demo.py": {
        "what": "Human-in-the-loop：interrupt 暂停给人看；update_state 改草稿；Command(resume=) 把人的决定填回 interrupt 返回值。",
        "why": "审批、改稿不能让图一口气跑完。",
        "watch": "interrupt({...}) 是给人看的说明书；resume 的键必须对上 decision.get(...)；config.thread_id 标识这一条审批单。",
        "flow": "write → review(interrupt) → [人改 state / 点批准] → finalize",
    },
    "lg_3_1_subgraphs_demo.py": {
        "what": "子图：调研(search→extract)产出摘要，写作(draft→polish)必须读这份摘要再成文。",
        "why": "父图只做调度+字段翻译；子图封装流程。旧版 write 没传 research_result 会跑题。",
        "watch": "只填 task；Trace 里 research 的 notes/summary 与 write 的 outline/article 应同一主题。",
        "flow": "task → research(子图) → research_result → write(子图) → article",
    },
    "lg_3_2_supervisor_demo.py": {
        "what": "主管-员工：supervisor 看任务决定路由，data/format worker 干活，merge 汇总。",
        "why": "多专员协作时，要有一个节点做任务拆解，而不是所有人抢着上。",
        "watch": "route=both 时先 both_entry 再串行 data→format→merge。",
        "flow": "supervisor → (条件边) → workers → merge",
    },
    "lg_3_3_send_api_demo.py": {
        "what": "Send：运行时为列表每一项派一个并行 worker（Map），reducer operator.add 把结果拼回去（Reduce）。",
        "why": "标题数量运行前不知道，不能写死 3 条边。",
        "watch": "Send 是调度指令不是直接函数调用；并行完成顺序可能和列表顺序不同（日志会乱序）。",
        "flow": "START → dispatch 返回 N 个 Send → N 个 summarize_one 并行 → 汇总 summaries",
    },
    "lg_4_1_checkpointer_demo.py": {
        "what": "Checkpointer：同一 thread_id 多次 invoke 会接续 state；内存版重启丢失，SQLite 可持久化。",
        "why": "多轮对话、interrupt 恢复都依赖把图状态存下来。",
        "watch": "config={'configurable': {'thread_id': '...'}} 就是这条会话的钥匙。",
        "flow": "invoke(轮1, thread) → invoke(轮2, 同 thread) → messages 累加",
    },
    "lg_4_2_store_api_demo.py": {
        "what": "Store vs Checkpointer：thread_id 隔离一轮对话；Store 按 user_id 跨线程记长期档案。",
        "why": "用户换一个聊天窗口仍应记住「我叫张三、喜欢简洁」。",
        "watch": "thread-A 写入 profile，thread-B 同一 user_id 仍能读到。",
        "flow": "save_profile(Store.put) → chat(Store.get + LLM)",
    },
    "lg_5_1_structured_output_demo.py": {
        "what": "图节点里 with_structured_output，让情感只能是 positive/neutral/negative，下游按字段路由。",
        "why": "自由文本无法 if/else；结构化字段才能可靠分支。",
        "watch": "Literal 三选一；model_dump() 把 Pydantic 转成 dict 写入 State。",
        "flow": "analyze(结构化) → route(根据 sentiment 定 action)",
    },
    "lg_5_2_error_fallback_demo.py": {
        "what": "节点 try/except 后 Command(goto='fallback')，LLM/API 挂了走降级文案而不是整图崩溃。",
        "why": "生产必须软着陆。",
        "watch": "问题含「崩溃」会模拟超时；正常问题走主路径。",
        "flow": "llm_safe ─┬─ 成功 → END\n           └─ 异常 → fallback → END",
    },
    "lg_5_3_token_optimization_demo.py": {
        "what": "LLM 按送进去的字数计费。图里每调一次模型就付一次；原文不该在每次 prompt 里再出现一遍。",
        "why": "第 1 次摘要必须看原文；第 2 次回答只带摘要，少付一截。对话则用 trim 丢掉旧消息。",
        "watch": "对比 [answer] 实际 prompt 字数 vs「若再塞原文」的字数；trim 前后各留下哪些消息。",
        "flow": "summarize(原文→短摘要) → answer(只把摘要放进 prompt)\n对话: 12 条消息 → trim 留最近 4 条",
    },
    "lg_5_4_langsmith_tracing_demo.py": {
        "what": "LangGraph 同样走 LangSmith：config 里 tags/metadata 会出现在 Dashboard。",
        "why": "图比单链更需要可视化每一步。",
        "watch": "未配 KEY 时只本地跑；配上 TRACING 后可在网站回放 DAG。",
        "flow": "double → add_ten，结果 value=20；可选上报 LangSmith",
    },
    "lg_studio_debug_demo.py": {
        "what": "专门给 LangGraph Studio 用的调试图：route 后按问题分流到 short / detail。",
        "why": "先看清 Studio 怎么点，再决定要不要配 LangSmith Key 上报 Trace。",
        "watch": "question 含「详细/为什么/怎么」走 detail；Studio 点节点看 state。Key 写 .env，不要发给别人。",
        "flow": "route ─┬─ short → END\n        └─ detail → END",
    },
    "lg_5_5_platform_demo.py": {
        "what": "LangGraph Platform / CLI 概念：langgraph.json 声明图，langgraph dev 起 API 和 Studio。",
        "why": "本地 demo 要变成可部署服务时用这套，而不是自己搓 Flask。",
        "watch": "中文 Windows 先设 PYTHONUTF8=1 再 langgraph dev；官方包未指定 encoding。",
        "flow": "编写 graph.compile() → 写入 langgraph.json → langgraph dev",
    },
    "pb_run_all.py": {
        "what": "Python 类型教材总入口：先认有哪些类型，再按类型各跑一节。",
        "why": "新手先建立类型地图，再学每种类型的字面量和常用操作。",
        "watch": "从 pb_0 总览开始；①～⑨ 可单独跑。",
        "flow": "总览 → None → bool → 数字 → str → list → tuple → dict → set → range",
    },
    "pb_0_types_overview.py": {
        "what": "Python 常用内置类型一张表：None/bool/int/float/str/list/tuple/dict/set/range。",
        "why": "先知道有哪些，再按节打开，避免把 list 和 tuple、{} 和 set 搞混。",
        "watch": "type(x) 看类型；可变 vs 不可变。",
        "flow": "值 → type() → 对照 JS",
    },
    "pb_1_none.py": {
        "what": "None 表示没有值；用 is None 判断。",
        "why": "函数没 return、dict.get 找不到键，都会得到 None。",
        "watch": "is None，不要 == None。",
        "flow": "赋值 None → is None → 函数默认返回 None",
    },
    "pb_2_bool.py": {
        "what": "True/False；以及 if 里什么算假。",
        "why": "空列表空字典在 Python 是假，和 JS 的 [] 不同。",
        "watch": "关键字是 True/False，首字母大写。",
        "flow": "比较得到 bool → and/or/not → bool(空值)",
    },
    "pb_3_int_float.py": {
        "what": "int 整数、float 小数；/ 和 // 的差别。",
        "why": "Python 不像 JS 把所有数字都叫 number。",
        "watch": "除法 / 永远是 float；整除用 //。",
        "flow": "四则运算 → 类型转换",
    },
    "pb_4_str.py": {
        "what": "字符串：拼接、f-string、strip、split/join、切片。",
        "why": "str 不可变，方法返回新串。",
        "watch": "s = s.strip()，否则原串不变。",
        "flow": "字面量 → 方法 → 新字符串",
    },
    "pb_5_list.py": {
        "what": "list 可变列表：下标、append、遍历、复制。",
        "why": "日常一排数据几乎都用 list。",
        "watch": "b=a 不是复制；要两份用 a.copy()。",
        "flow": "创建 → 改原地 → 遍历",
    },
    "pb_6_tuple.py": {
        "what": "tuple 不可变；解包；单元素必须写逗号。",
        "why": "固定几个槽位（多返回值）用 tuple。",
        "watch": "('hello') 不是元组，('hello',) 才是。",
        "flow": "创建 → 解包 → 和 list 互转",
    },
    "pb_7_dict.py": {
        "what": "dict 键值对：[]、.get、in 查键、** 合并。",
        "why": "配置、接口参数几乎都是 dict。",
        "watch": "键不存在用 .get，避免 KeyError。",
        "flow": "写入 → 读取 → 遍历 items",
    },
    "pb_8_set.py": {
        "what": "set 去重、无顺序；并交差。",
        "why": "查「在不在」和去重时用。",
        "watch": "{} 是空字典，空集合写 set()。",
        "flow": "创建 → add → 集合运算",
    },
    "pb_9_range.py": {
        "what": "range 用来数次数，含头不含尾。",
        "why": "for i in range(n) 是固定次数循环的标准写法。",
        "watch": "range(3) 是 0,1,2，没有 3。",
        "flow": "range → for 循环 → list(range) 查看",
    },
    "fb_run_all.py": {
        "what": "FastAPI 三节：最小接口、校验、中间件/Depends。给会 Koa 的人对照着看。",
        "why": "HTTP 思路能迁移，写法差在「没有 ctx」和「类型即校验」。",
        "watch": "用 TestClient，不用起服务。每节都有【反例】。",
        "flow": "hello → 校验 422 → 中间件洋葱 / Depends 401",
    },
    "fb_0_hello.py": {
        "what": "最小 FastAPI：装饰器注册路由，return dict 即 JSON。",
        "why": "对照 Koa 的 ctx.body，先建立「这就是一个 HTTP 接口」。",
        "watch": "错方法 405、错路径 404。",
        "flow": "TestClient.get → 路由函数 → JSON",
    },
    "fb_1_validation.py": {
        "what": "路径/查询/Body 靠类型和 Pydantic 校验，错了 422。",
        "why": "Koa 默认不校验；FastAPI 把校验写进签名。",
        "watch": "不写类型就不会校验；Field(ge=0) 会拦负数。",
        "flow": "请求 → 校验 → 进函数 或 422",
    },
    "fb_2_middleware.py": {
        "what": "HTTP 中间件是洋葱；Depends 是按路由注入；忘了 call_next 路由不跑。",
        "why": "全局日志用中间件，登录态用 Depends，不要把校验塞进中间件。",
        "watch": "进出顺序；401 vs 缺 Header 的 422。",
        "flow": "中间件进 → 路由/Depends → 中间件出",
    },
    "langgraph_demo_examples.py": {
        "what": "LangGraph 五件套总览：stream 模式、Checkpointer、interrupt、Reducer、图里接真实 LLM。",
        "why": "按编号拆开的 lg_* 之前，先用这一份把核心原语跑一遍。",
        "watch": "values vs updates；同一 thread_id 消息累加；interrupt 必须配 checkpointer。",
        "flow": "stream → persistence → interrupt → reducer → AI 节点",
    },
}


def print_lesson(file: str) -> None:
    """运行时打印：这个例子要表达什么。"""
    name = Path(file).name
    lesson = LESSONS.get(name)
    print()
    print("=" * 68)
    print(f"  [本例子要表达什么]  {name}")
    print("=" * 68)
    if not lesson:
        print("  （未登记说明，请看文件顶部 docstring）")
        print()
        return
    print(f"  【要解决的问题】{lesson['what']}")
    print(f"  【为什么这样写】{lesson['why']}")
    print(f"  【运行时请盯】  {lesson['watch']}")
    print("  【数据/控制流】")
    for line in lesson["flow"].split("\n"):
        print(f"      {line}")
    print()


def print_step(n: int | str, title: str, detail: str = "") -> None:
    """关键步骤日志。"""
    print(f"\n── 步骤 {n}: {title} ──")
    if detail:
        print(f"   {detail}")


def print_tip(msg: str) -> None:
    print(f"   提示: {msg}")


def print_section(title: str) -> None:
    """小节标题，避免一长串 print 糊在一起。"""
    print(f"\n  · {title}")


def print_ok(msg: str) -> None:
    print(f"  【正例】{msg}")


def print_trap(title: str, detail: str) -> None:
    """能跑、但不该这么写。"""
    print(f"  【反例·能跑但是错】{title}")
    print(f"    {detail}")


def print_counter(title: str, fn, expect: type | tuple[type, ...] = Exception) -> None:
    """故意跑一段会失败的代码，把报错打印出来。"""
    print(f"  【反例】{title}")
    try:
        result = fn()
        print(f"    （没有报错，得到 {result!r}。反例可能写错了。）")
    except expect as e:
        print(f"    报错 {type(e).__name__}: {e}")


def print_run_plan(items: list[tuple[str, str]]) -> None:
    """打印本文件本次会跑哪些小节（以及为什么跳过别的）。"""
    print("── 本次会跑 ──")
    for i, (title, why) in enumerate(items, 1):
        print(f"  {i}. {title}")
        if why:
            print(f"     {why}")
    print()
