"""
========================================================
Phase 6-3
Hook Manager 教材级 Demo（基于 6_2 混合架构扩展）
========================================================

📚 学习目标：
1. 理解 Hook 是「横切关注点」—— 审计/日志/Token 统计不侵入业务节点
2. 在 merge 前插入 audit_hook 节点，记录每次请求的轨迹
3. 用 Reducer 累积 audit_logs 列表

🔗 架构（在 6_2 基础上加一层）：
  START → router → rag_node / agent_node → audit_hook → merge → END
                                              ↑
                                    审计日志 / Token 估算

运行:
  python 6_3_hook_manager_demo.py

需要 API Key（agent 分支会调模型；纯 RAG 关键词走 mock 可不耗 Token）
"""
import os
import time
from numbers import Number
from typing import Annotated, Literal, TypedDict
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_step, print_run_plan
load_dotenv()
_AUDIT_SINK: list[dict] = []

class HookHybridState(TypedDict):
    question: str
    route: str
    rag_answer: str
    agent_answer: str
    final_answer: str
    audit_logs: Annotated[list[dict], operator.add]

class HookManager:
    """
    Hook Manager：统一管理 pre/post 钩子。

    设计原则（路线图 Phase 6）：
      · 业务节点（router/rag/agent/merge）只关心业务逻辑
      · 审计、Token 统计、权限校验由 Hook 层负责
      · 新增 Hook 不改业务代码，只注册到 manager
    """

    def __init__(self):
        self.pre_hooks: list = []
        self.post_hooks: list = []

    def register_pre(self, fn):
        self.pre_hooks.append(fn)
        return fn

    def register_post(self, fn):
        self.post_hooks.append(fn)
        return fn

    def run_pre(self, state: dict, node_name: str) -> list[dict]:
        logs = []
        for hook in self.pre_hooks:
            entry = hook(state, node_name=node_name, phase='pre')
            if entry:
                logs.append(entry)
        return logs

    def run_post(self, state: dict, node_name: str) -> list[dict]:
        logs = []
        for hook in self.post_hooks:
            entry = hook(state, node_name=node_name, phase='post')
            if entry:
                logs.append(entry)
        return logs
hook_manager = HookManager()

@hook_manager.register_pre
def audit_pre_hook(state: dict, *, node_name: str, phase: str) -> dict:
    """Pre-Hook：节点执行前记录"""
    return {'ts': time.strftime('%H:%M:%S'), 'phase': phase, 'node': node_name, 'event': 'enter', 'route': state.get('route', '')}

@hook_manager.register_post
def token_estimate_hook(state: dict, *, node_name: str, phase: str) -> dict:
    """Post-Hook：估算本轮回答的 Token（Demo 用字符数/4 近似）"""
    if node_name != 'audit_hook':
        return {}
    answer = state.get('rag_answer') or state.get('agent_answer') or ''
    est_tokens = len(answer) // 4
    return {'ts': time.strftime('%H:%M:%S'), 'phase': phase, 'node': node_name, 'event': 'token_estimate', 'est_tokens': est_tokens, 'answer_len': len(answer)}

@hook_manager.register_post
def audit_post_hook(state: dict, *, node_name: str, phase: str) -> dict:
    """Post-Hook：merge 前记录完整快照"""
    if node_name != 'audit_hook':
        return {}
    return {'ts': time.strftime('%H:%M:%S'), 'phase': phase, 'node': node_name, 'event': 'before_merge', 'question': state.get('question', '')[:50], 'route': state.get('route')}

def audit_hook_node(state: HookHybridState) -> dict:
    """
    audit_hook 节点 —— 在 merge 之前运行所有 Hook。

    这是 6_2 混合架构的自然扩展点：
      rag_node / agent_node 完成后 → 先过 audit_hook → 再 merge
    """
    print('  📋 audit_hook: 运行 Pre/Post Hooks...')
    logs: list[dict] = []
    logs.extend(hook_manager.run_pre(state, 'audit_hook'))
    logs.extend(hook_manager.run_post(state, 'audit_hook'))
    for entry in logs:
        _AUDIT_SINK.append(entry)
        print(f"     · [{entry.get('event')}] {entry}")
    return {'audit_logs': logs}

def router_node(state: HookHybridState) -> dict:
    q = state['question'].lower()
    route = 'agent' if any((k in q for k in ['计算', '+', '时间'])) else 'rag'
    print(f'  🔀 Router → {route}')
    return {'route': route}

def rag_node(state: HookHybridState) -> dict:
    print('  📚 RAG Node (mock)')
    mock = 'LangChain 提供 LCEL；LangGraph 提供有状态编排。'
    return {'rag_answer': f'[RAG] {mock}'}

def agent_node(state: HookHybridState) -> dict:
    print('  🔧 Agent Node')
    llm = get_model()
    ans = llm.invoke([HumanMessage(content=f"简短回答: {state['question']}")]).content
    return {'agent_answer': f'[Agent] {ans}'}

def merge_node(state: HookHybridState) -> dict:
    final = state.get('rag_answer') or state.get('agent_answer') or ''
    print('  ✅ Merge Node')
    return {'final_answer': final}

def route_after_router(state: HookHybridState) -> Literal['rag_node', 'agent_node']:
    return 'rag_node' if state['route'] == 'rag' else 'agent_node'

def build_graph_with_hooks():
    g = StateGraph(HookHybridState)
    g.add_node('router', router_node)
    g.add_node('rag_node', rag_node)
    g.add_node('agent_node', agent_node)
    g.add_node('audit_hook', audit_hook_node)
    g.add_node('merge', merge_node)
    g.add_edge(START, 'router')
    g.add_conditional_edges('router', route_after_router, {'rag_node': 'rag_node', 'agent_node': 'agent_node'})
    g.add_edge('rag_node', 'audit_hook')
    g.add_edge('agent_node', 'audit_hook')
    g.add_edge('audit_hook', 'merge')
    g.add_edge('merge', END)
    return g.compile()
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║  Phase 6-3: Hook Manager（6_2 混合架构扩展）              ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('audit_hook 节点', '业务节点前后横切日志，不改 RAG/Agent 函数本身'),
    ])
    check_api_key()
    app = build_graph_with_hooks()
    cases = ['LangChain 和 LangGraph 区别？', '计算 10 + 20 等于多少？']
    for i, q in enumerate(cases, 1):
        print_step(i, '跑图', q)
        print(f"\n{'═' * 55}")
        print(f'问题: {q}')
        print('═' * 55)
        _AUDIT_SINK.clear()
        result = app.invoke({'question': q, 'route': '', 'rag_answer': '', 'agent_answer': '', 'final_answer': '', 'audit_logs': []})
        print(f"\n  最终回答: {result['final_answer'][:120]}...")
        print(f"  本轮 audit_logs ({len(result['audit_logs'])} 条):")
        for log in result['audit_logs']:
            print(f'    {log}')
    print('\n💡 要点:\n  · Hook 放在独立节点（audit_hook），merge 前统一执行\n  · HookManager 可注册多个 pre/post 钩子，互不影响业务节点\n  · 生产环境还可接: 权限校验 Hook、成本阈值 Hook、LangSmith trace\n')