# Harness

这组教程研究模型外部的运行与管理层：工具执行、循环控制、审批、上下文、持久化与修改验收。
加法和笔记只是演示工具。

完整课表和每课的独立问题见 [学习路线](../learning_roadmaps/harness_learning_roadmap.md)。
建议同时打开 [调试与答疑指南](../learning_roadmaps/debugging_and_questions.md)，逐个案例打断点，带着真实输入和返回值向 AI 提问，再用代码验证解释。
先读概念地图，再运行第一课：

```bash
python harness/hs_0_map.py
python harness/hs_1_cannot_act.py
```

工具调用接口说明集中在 `hs_1`；多步工具的数据依赖在 `hs_2`。
Deep Agents 接入和手写实现对照集中在 `da_0_create.py`。
重复课程已合并，文件保留原编号，按学习路线的课表顺序学习。

`hs_0`、`hs_8`、`hs_15` 为概念课，不调用在线模型。
其余课程包含在线演示，需要配置模型 Key；`hs_11_verifier.py --offline` 可以只看固定提案，不调用 API。需要工具循环的实践课使用 `mini.py`；部分课直接调用模型，`hs_11` 的离线部分使用普通 Python。模型和共用工具配置在 `live.py`。

`hs_9` 到 `hs_14` 分别研究持久化、规则增量修改、回归验收、错误反馈、按需加载和效果评估。
`hs_11` 用运费修复演示权限与测试；`hs_14` 用真实工具调用记录比较旧手册和候选手册。
`hs_15` 结合前面的错误反馈、资料加载等案例说明修改对象，不实现自动搜索或整库修改。

读案例时要区分三个层次：工具说明是给模型看的；`tool_calls` 是模型提出的请求；普通 Python 工具函数由 Harness 真正执行。控制台打印不一定发送给模型，返回的 `AIMessage` 也不等于完整历史。

```bash
python harness/hs_run_all.py
```
