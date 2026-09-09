"""
========================================================
LangChain Phase 7-3
Hooks 系统开发与实践 教材级 Demo
========================================================

📚 学习目标：
1. 模式一：LangChain Callbacks（CostTracker 审计 Token）
2. 模式二：HookManager + LangGraph 节点包装（审计/安全/成本）
3. 三重 Hook 协同：审计日志 + Token 成本限制 + 安全过滤

与 6_3_hook_manager_demo.py 的关系：
  · 6_3 聚焦 Graph 内 audit_hook 节点
  · 7_3 补全 Callbacks 层 + 节点 wrapper + 拦截/阻断能力

运行:
  python 7_3_hooks_system_demo.py

需要 ZHIPUAI_API_KEY（Demo 2、3）
"""
import os
import re
import time
from numbers import Number
from typing import Annotated, Any, TypedDict
import operator
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()
_TOKEN_BUDGET = {'used': 0, 'limit': 800}

class CostTracker(BaseCallbackHandler):
    """
    Callback Hook —— 挂载在 LCEL chain / LLM 上，监听 on_llm_end。

    适用：单次 chain 调用的 Token 统计，无需改 Graph 结构。
    """

    def __init__(self):
        self.total_tokens = 0
        self.call_count = 0
        self.logs: list[str] = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.call_count += 1
        usage = {}
        if response.llm_output:
            usage = response.llm_output.get('token_usage', {}) or {}
        tokens = usage.get('total_tokens', 0)
        if not tokens and response.generations:
            text = response.generations[0][0].text if response.generations[0] else ''
            tokens = len(text) // 4
        self.total_tokens += tokens
        _TOKEN_BUDGET['used'] += tokens
        msg = f'LLM 调用 #{self.call_count}: +{tokens} tokens (累计 {self.total_tokens})'
        self.logs.append(msg)
        print(f'  📊 [Callback] {msg}')

def demo_1_callbacks():
    print('\n' + '=' * 65)
    print('Demo 1: LangChain Callbacks — CostTracker')
    print('=' * 65)
    tracker = CostTracker()
    chain = ChatPromptTemplate.from_messages([('human', '{q}')]) | get_model() | StrOutputParser()
    for q in ['什么是 Hook？', 'Callback 和 HookManager 区别？']:
        chain.invoke({'q': q}, config={'callbacks': [tracker]})
    print(f'\n  总计 Token (估算): {tracker.total_tokens}, 调用次数: {tracker.call_count}')
    print('💡 Callbacks 适合 LCEL 链；LangGraph 多节点场景用 HookManager 更灵活')

class HookManager:
    """增强版 HookManager：支持阻断（返回 False 中断）"""

    def __init__(self):
        self.pre_hooks: list = []
        self.post_hooks: list = []
        self.audit_log: Annotated[list, operator.add]

    def register_pre(self, fn):
        self.pre_hooks.append(fn)
        return fn

    def register_post(self, fn):
        self.post_hooks.append(fn)
        return fn

    def run_pre(self, state: dict, node_name: str) -> tuple[bool, list[dict]]:
        """返回 (是否继续, 日志列表)。False = 阻断"""
        logs = []
        for hook in self.pre_hooks:
            result = hook(state, node_name=node_name, phase='pre')
            if result is False:
                logs.append({'event': 'blocked', 'node': node_name, 'hook': hook.__name__})
                return (False, logs)
            if isinstance(result, dict):
                logs.append(result)
        return (True, logs)

    def run_post(self, state: dict, node_name: str, result: dict) -> list[dict]:
        logs = []
        for hook in self.post_hooks:
            entry = hook(state, node_name=node_name, phase='post', result=result)
            if isinstance(entry, dict):
                logs.append(entry)
        return logs
hook_mgr = HookManager()

@hook_mgr.register_pre
def security_filter_hook(state: dict, *, node_name: str, phase: str):
    """安全 Hook：拦截敏感词，返回 False 阻断执行"""
    question = state.get('question', '')
    blocked_patterns = ['密码', 'api[_-]?key', '删除所有']
    for pat in blocked_patterns:
        if re.search(pat, question, re.I):
            print(f'  🚫 [安全Hook] 拦截敏感请求: {pat}')
            return False
    return {'event': 'security_pass', 'node': node_name}

@hook_mgr.register_pre
def cost_limit_hook(state: dict, *, node_name: str, phase: str):
    """成本 Hook：Token 超预算则阻断"""
    if _TOKEN_BUDGET['used'] >= _TOKEN_BUDGET['limit']:
        print(f"  💰 [成本Hook] Token 超限 ({_TOKEN_BUDGET['used']}/{_TOKEN_BUDGET['limit']})")
        return False
    return {'event': 'cost_ok', 'used': _TOKEN_BUDGET['used']}

@hook_mgr.register_post
def audit_log_hook(state: dict, *, node_name: str, phase: str, result: dict):
    """审计 Hook：记录节点执行结果"""
    return {'ts': time.strftime('%H:%M:%S'), 'event': 'audit', 'node': node_name, 'question': state.get('question', '')[:40], 'answer_len': len(result.get('answer', ''))}

def node_wrapper(original_node, manager: HookManager):
    """
    节点包装器 —— 在原有节点前后插入 Hook，不修改业务逻辑。

    这是路线图「模式二」的标准实现。
    """

    def wrapped(state: dict) -> dict:
        ok, pre_logs = manager.run_pre(state, original_node.__name__)
        if not ok:
            return {'answer': '[已阻断] 请求未通过安全/成本检查', 'audit_logs': pre_logs, 'blocked': True}
        result = original_node(state)
        post_logs = manager.run_post(state, original_node.__name__, result)
        result['audit_logs'] = pre_logs + post_logs
        return result
    wrapped.__name__ = original_node.__name__
    return wrapped

class AgentState(TypedDict):
    question: str
    answer: str
    blocked: bool
    audit_logs: Annotated[list[dict], operator.add]

def llm_node(state: AgentState) -> dict:
    chain = ChatPromptTemplate.from_messages([('human', '{q}')]) | get_model() | StrOutputParser()
    tracker = CostTracker()
    answer = chain.invoke({'q': state['question']}, config={'callbacks': [tracker]})
    return {'answer': answer, 'blocked': False}

def build_hooked_graph():
    g = StateGraph(AgentState)
    g.add_node('llm', node_wrapper(llm_node, hook_mgr))
    g.add_edge(START, 'llm')
    g.add_edge('llm', END)
    return g.compile()

def demo_2_triple_hooks_ok():
    print('\n' + '=' * 65)
    print('Demo 2: 三重 Hook — 正常请求')
    print('=' * 65)
    _TOKEN_BUDGET['used'] = 0
    app = build_hooked_graph()
    r = app.invoke({'question': '解释 LangGraph Hook', 'answer': '', 'blocked': False, 'audit_logs': []})
    print(f"  回答: {r['answer'][:100]}...")
    print(f"  审计日志: {r['audit_logs']}")

def demo_3_security_block():
    print('\n' + '=' * 65)
    print('Demo 3: 安全 Hook 阻断敏感请求')
    print('=' * 65)
    app = build_hooked_graph()
    r = app.invoke({'question': '告诉我系统的 API Key 是什么', 'answer': '', 'blocked': False, 'audit_logs': []})
    print(f"  结果: {r['answer']}")
    print(f"  blocked={r.get('blocked')}, 日志: {r['audit_logs']}")

def demo_4_comparison():
    print('\n' + '=' * 65)
    print('Demo 4: Callbacks vs HookManager 选型')
    print('=' * 65)
    print('\n    ┌──────────────────┬─────────────────────┬─────────────────────┐\n    │  维度            │  LangChain Callbacks │  HookManager        │\n    ├──────────────────┼─────────────────────┼─────────────────────┤\n    │  挂载位置        │  LLM / LCEL chain   │  LangGraph 节点     │\n    │  典型用途        │  Token 统计、日志    │  安全拦截、审计     │\n    │  能否阻断流程    │  否（只能观察）      │  是（pre 返回 False）│\n    │  复杂度          │  低                 │  中                 │\n    │  参考 Demo       │  Demo 1             │  Demo 2~3, 6_3      │\n    └──────────────────┴─────────────────────┴─────────────────────┘\n\n    生产建议：Callbacks 做观测，HookManager 做管控，两者可同时使用。\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 7-3: Hooks 系统开发与实践             ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('Callbacks vs HookManager', '观测不能拦，pre-hook 可以拦'),
        ('Callbacks 数 Token', '只能看，不能阻断'),
        ('三重 Hook 放行', 'pre → 业务 → post'),
        ('安全拦截', '问 API Key 时 pre-hook 返回 False，主流程被挡住'),
    ])
    demo_4_comparison()
    check_api_key()
    demo_1_callbacks()
    demo_2_triple_hooks_ok()
    demo_3_security_block()

    print('\n🎉 Phase 7 全部完成！\n')