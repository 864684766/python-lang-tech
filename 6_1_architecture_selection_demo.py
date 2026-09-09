"""
========================================================
LangChain Phase 6-1
架构选择指南 教材级 Demo
========================================================

📚 学习目标：
1. 根据需求复杂度选择 LCEL / LangGraph / 混合架构
2. 理解六种典型场景的技术选型
3. 用最小代码示例对比 RunnableBranch vs StateGraph
4. 掌握 Phase 6 六大设计原则

本 Demo 部分示例需要 API Key（LCEL/LangGraph 对比 Demo）。
"""
import os
from numbers import Number
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langgraph.graph import END, START, StateGraph

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def demo_1_decision_tree():
    print('\n' + '=' * 65)
    print('Demo 1: 架构选择决策树')
    print('=' * 65)
    print('\n    需求复杂度低（线性流水线）          → 纯 LCEL\n      例: prompt | model | parser\n\n    需求含分支但无需持久化              → LCEL + RunnableBranch\n      例: 诗歌/代码/闲聊 三分支\n\n    需求有状态、需多轮记忆              → LangGraph（轻）\n      例: Checkpointer + thread_id\n\n    需求是多 Agent、人工审核、子流程     → LangGraph（重）\n      例: interrupt + Command(resume)\n\n    需求是动态 Skill / MCP 工具         → LangChain + LangGraph 混合\n      例: Tool Registry + StateGraph 路由\n\n    需求需要 Hooks 生命周期管理          → LangGraph interrupt + callback\n      例: 审计日志、权限校验中间件\n\n    需求对外暴露 API、长时间挂起         → LangGraph Platform\n      例: 生产部署、Human-in-the-loop\n    ')

def demo_2_pure_lcel():
    print('\n' + '=' * 65)
    print('Demo 2: 纯 LCEL — 线性流水线')
    print('=' * 65)
    chain = ChatPromptTemplate.from_messages([('human', '翻译为英文：{text}')]) | get_model() | StrOutputParser()
    result = chain.invoke({'text': '架构选择很重要'})
    print(f'结果: {result}')
    print('💡 无分支、无状态 → LCEL 足够，代码最简')

def demo_3_runnable_branch():
    print('\n' + '=' * 65)
    print('Demo 3: LCEL + RunnableBranch — 条件分支')
    print('=' * 65)

    def classify(text: str) -> str:
        if any((k in text for k in ['代码', 'python', '函数'])):
            return 'code'
        if any((k in text for k in ['诗', '散文', '写一首'])):
            return 'poem'
        return 'chat'
    code_chain = ChatPromptTemplate.from_template('你是程序员，用代码或伪代码回答：{text}') | get_model() | StrOutputParser()
    poem_chain = ChatPromptTemplate.from_template('你是诗人，写简短诗句：{text}') | get_model() | StrOutputParser()
    chat_chain = ChatPromptTemplate.from_template('友好聊天：{text}') | get_model() | StrOutputParser()
    router = RunnableBranch((lambda x: classify(x['text']) == 'code', code_chain), (lambda x: classify(x['text']) == 'poem', poem_chain), chat_chain)
    for q in ['写一个 Python 快排', '写一首关于春天的诗', '你好']:
        route = classify(q)
        ans = router.invoke({'text': q})
        print(f'\n[{route}] Q: {q}\nA: {ans[:80]}...')
    print('\n💡 有分支但无跨轮状态 → RunnableBranch，不必上 LangGraph')

class CounterState(TypedDict):
    count: int
    last_input: str

def demo_4_langgraph_state():
    print('\n' + '=' * 65)
    print('Demo 4: LangGraph — 有状态计数器')
    print('=' * 65)

    def increment(state: CounterState):
        res = {'count': state['count'] + 1, 'last_input': state.get('last_input', '')}
        print(f'res={res}')
        return res
    graph = StateGraph(CounterState)
    graph.add_node('inc', increment)
    graph.add_edge(START, 'inc')
    graph.add_edge('inc', END)
    app = graph.compile()
    state = {'count': 0, 'last_input': ''}
    for i in range(3):
        state = app.invoke({**state, 'last_input': f'step-{i}'})
    print(f"最终 count: {state['count']}")
    print('💡 需要跨节点/跨轮累积状态 → LangGraph StateGraph')

def demo_5_principles():
    print('\n' + '=' * 65)
    print('Demo 5: LangChain × LangGraph 六大原则')
    print('=' * 65)
    print('\n    1. LangGraph 管流程，LCEL 管数据处理\n       · 节点内用 LCEL 链处理数据，节点间用边控制流转\n\n    2. Tool 在 LangChain 定义，在 LangGraph 调度\n       · @tool 放独立模块，Graph 节点 bind_tools / create_agent\n\n    3. Memory 分层管理\n       · 会话内: Checkpointer + thread_id\n       · 会话间: VectorStore + RAG 用户画像\n\n    4. Observability 统一走 LangSmith\n       · LCEL 和 LangGraph 共用 LANGCHAIN_TRACING_V2\n\n    5. Skill/MCP 动态注册到 Tool Registry\n       · 运行时 load_skill()，不写死 tool list\n\n    6. Hooks 作为横切关注点\n       · 审计/权限/成本用 Middleware 或 Hook，不侵入业务节点\n    ')

def demo_6_selection_table():
    print('\n' + '=' * 65)
    print('Demo 6: 本项目 Demo 文件与架构对照')
    print('=' * 65)
    print('\n    Phase 1 (1-x demo)     → 纯 LCEL 基础\n    Phase 2 (2_x demo)     → LCEL RAG 管道\n    Phase 3 (3_x demo)     → Tool + create_agent\n    Phase 4 (4_x demo)     → Memory（LCEL + Checkpointer）\n    Phase 5 (5_x demo)     → 工程优化\n    Phase 6 (6_2 demo)     → LangGraph 编排 LCEL + Agent\n\n    下一步: 6_2_hybrid_architecture_demo.py 看完整混合架构实战\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 6-1: 架构选择指南                    ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('决策树', '什么时候用 LCEL / Graph / 混合'),
        ('设计原则', '六条选型原则'),
        ('对照表', '场景 → 架构'),
        ('纯 LCEL', '线性翻译链（需 API）'),
        ('RunnableBranch', '链上的 if/else（需 API）'),
        ('LangGraph State', '有状态才上图（需 API）'),
    ])
    demo_1_decision_tree()
    demo_5_principles()
    demo_6_selection_table()
    check_api_key()
    try:
        demo_2_pure_lcel()
        demo_3_runnable_branch()
        demo_4_langgraph_state()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 6-1 完成！下一步: 6_2_hybrid_architecture_demo.py\n')