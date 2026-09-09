"""
LangGraph Phase 2-2: 预构建 Agent + 模型重试 教材级 Demo

学习目标:
  · LangChain 1.3.14 推荐: langchain.agents.create_agent
  · 对比手写 ReAct (lg_2_1) vs 预构建工厂函数的代码量
  · ModelRetryMiddleware —— Agent 层模型超时/限流自动重试
  · model.with_retry() —— LCEL 层通用重试
  · 理解指数退避 (Exponential Backoff)

运行: python lg_2_2_prebuilt_agent_demo.py

说明:
  · Demo 1 需要 ZHIPUAI_API_KEY（真实 API）
  · Demo 2~4 用模拟超时模型，无需 API
"""
import os
import time
from numbers import Number
from typing import Any, Optional
import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware._retry import calculate_delay
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_step, print_tip, print_run_plan
load_dotenv()

@tool
def search_kb(query: str) -> str:
    """搜索知识库"""
    kb = {'langgraph': 'LangGraph 是有状态图编排框架', 'agent': 'Agent 可自主调用工具'}
    for k, v in kb.items():
        if k in query.lower():
            return v
    return '未找到'

class FlakyChatModel(BaseChatModel):
    """
    教学用 ChatModel：前 max_fails 次调用抛出 ReadTimeout，之后返回固定回答。

    用来观察 ModelRetryMiddleware / with_retry 的重试行为，无需消耗 API 配额。
    """
    max_fails: int = 2
    success_reply: str = 'LangGraph 是有状态图编排框架（模拟模型返回）'

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]]=None, run_manager: Any=None, **kwargs: Any) -> ChatResult:
        attempt = self._attempt_count + 1
        self._attempt_count = attempt
        if attempt <= self.max_fails:
            print(f'  ⏱️  模拟 ReadTimeout（第 {attempt} 次 model 调用）')
            raise httpx.ReadTimeout(f'模拟超时，第 {attempt} 次')
        print(f'  ✅ 第 {attempt} 次 model 调用成功')
        msg = AIMessage(content=self.success_reply)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def reset_attempts(self) -> None:
        self._attempt_count = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> 'FlakyChatModel':
        """create_agent 内部会 bind_tools；教学模型直接返回 self 即可。"""
        return self

    @property
    def _llm_type(self) -> str:
        return 'flaky-demo'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._attempt_count = 0

def demo_1_basic_agent():
    """一行 create_agent 替代 lg_2_1 手写 ReAct Graph"""
    print_step(1, "create_agent 预构建", "对照 lg_2_1：同样 ReAct，工厂函数省掉 agent/tools/条件边")
    print('\n' + '=' * 65)
    print('Demo 1: create_agent 预构建 vs 手写 ReAct')
    print('=' * 65)
    print('\n    手写 ReAct (lg_2_1):  ~50 行 (agent节点 + tools节点 + 条件边)\n    预构建 create_agent:  ~3 行\n\n    注意: langgraph.prebuilt.create_react_agent 已废弃，\n          LangChain 1.3.14 请用 langchain.agents.create_agent\n    ')
    check_api_key()
    agent = create_agent(get_model(), [search_kb], system_prompt='你是助手，查知识用 search_kb')
    try:
        r = agent.invoke({'messages': [HumanMessage(content='LangGraph 是什么？')]})
        print(f"🤖 回答: {r['messages'][-1].content}")
    except httpx.HTTPStatusError as exc:
        print(f'  ⚠️  API 返回 {exc.response.status_code}，跳过本 Demo（可稍后重试）')
        print('  💡 生产环境可对 429 配置 ModelRetryMiddleware(retry_on=...)')

def demo_2_model_retry_middleware():
    """
    ModelRetryMiddleware 拦截 Agent 内部的 model 调用，
    遇到 timeout / 限流等异常时按指数退避重试。
    """
    print_step(2, "ModelRetryMiddleware", "教学模型前两次故意超时，看 Agent 层自动退避重试")
    print('\n' + '=' * 65)
    print('Demo 2: ModelRetryMiddleware —— 超时后自动重试')
    print('=' * 65)
    flaky = FlakyChatModel(max_fails=2)
    retry_middleware = ModelRetryMiddleware(max_retries=3, retry_on=(httpx.TimeoutException,), initial_delay=0.5, backoff_factor=2.0, max_delay=10.0, jitter=False, on_failure='error')
    agent = create_agent(flaky, [search_kb], system_prompt='你是助手', middleware=[retry_middleware])
    print('  配置: max_fails=2, max_retries=3, initial_delay=0.5, backoff_factor=2.0')
    print('  预期: 第 1、2 次超时 → 等待 0.5s、1s → 第 3 次成功\n')
    t0 = time.perf_counter()
    r = agent.invoke({'messages': [HumanMessage(content='LangGraph 是什么？')]})
    elapsed = time.perf_counter() - t0
    print(f'\n  总耗时: {elapsed:.2f}s（含退避等待）')
    print(f"  🤖 回答: {r['messages'][-1].content}")
    print('\n💡 要点:')
    print('   · middleware=[ModelRetryMiddleware(...)] 挂在 create_agent 上')
    print('   · max_retries 是「首次调用之后」的重试次数')
    print('   · retry_on 应精确指定异常类型，避免对逻辑错误盲目重试')

def demo_3_backoff_schedule():
    """打印指数退避公式与各次重试的等待时间"""
    print_step(3, "指数退避时间表", "只打印公式，不打真实 API；等 1s、2s、4s、8s")
    print('\n' + '=' * 65)
    print('Demo 3: 指数退避 (Exponential Backoff) 时间表')
    print('=' * 65)
    initial_delay = 1.0
    backoff_factor = 2.0
    max_retries = 4
    print(f'\n    公式: delay = initial_delay × (backoff_factor ^ retry_number)\n          本例: initial_delay={initial_delay}, backoff_factor={backoff_factor}\n\n    retry_number 是 0-indexed（第几次「重试前」的等待）:\n    ')
    print(f"  {'重试序号':<10} {'计算公式':<22} {'等待(秒)':<10}")
    print('  ' + '-' * 44)
    for i in range(max_retries):
        delay = calculate_delay(i, backoff_factor=backoff_factor, initial_delay=initial_delay, max_delay=60.0, jitter=False)
        formula = f'{initial_delay} × {backoff_factor}^{i}'
        print(f'  第 {i + 1} 次重试前  {formula:<22} {delay:.1f}s')
    print('\n    「指数」= 等待时间按幂次增长 (1 → 2 → 4 → 8 …)\n    「退避」= 遇到故障先退一步、暂缓再试，避免压垮服务端\n    生产环境建议 jitter=True，在 delay 上加 ±25% 随机抖动，防止惊群效应\n    ')

def demo_4_model_with_retry():
    """
    不经过 Agent，直接在 ChatModel 上挂 with_retry。
    适合纯 LCEL 链；Agent 场景更推荐 Demo 2 的 ModelRetryMiddleware。
    """
    print_step(4, "model.with_retry()", "不经过 Agent，直接挂在 ChatModel 上；适合纯 LCEL 链")
    print('\n' + '=' * 65)
    print('Demo 4: model.with_retry() —— LCEL 层重试')
    print('=' * 65)
    flaky = FlakyChatModel(max_fails=1)
    model_with_retry = flaky.with_retry(stop_after_attempt=3, retry_if_exception_type=(httpx.TimeoutException,), wait_exponential_jitter=False)
    print('  配置: max_fails=1, stop_after_attempt=3')
    print('  预期: 第 1 次超时 → 第 2 次成功\n')
    msg = model_with_retry.invoke([HumanMessage(content='你好')])
    print(f'  🤖 回答: {msg.content}')
    print('\n    对比:\n    ┌─────────────────────┬──────────────────────────┐\n    │ ModelRetryMiddleware│ model.with_retry()       │\n    ├─────────────────────┼──────────────────────────┤\n    │ create_agent 专用   │ 任意 Runnable / Model    │\n    │ 可配 on_failure     │ 失败直接抛 RetryError    │\n    │ 指数退避参数更丰富   │ 基于 tenacity，API 略不同 │\n    └─────────────────────┴──────────────────────────┘\n    ')


if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 2-2: create_agent + 模型重试')
    print('=' * 65)
    # Demo 1 打真实 API；2~4 用 FlakyChatModel，没 Key 也能看懂重试
    has_key = bool(os.getenv('ZHIPUAI_API_KEY'))
    plan = []
    if has_key:
        plan.append(('Demo 1 create_agent', '需要 API Key，对比手写图的代码量'))
    else:
        print('\nDemo 1 跳过：未设置 ZHIPUAI_API_KEY（2~4 不依赖真实模型）')
    plan.extend([
        ('Demo 2 ModelRetryMiddleware', '模拟超时，观察 Agent 层重试'),
        ('Demo 3 退避时间表', '只打印公式，不发请求'),
        ('Demo 4 with_retry', 'LCEL 层重试，对比 Demo 2'),
    ])
    print_run_plan(plan)
    if has_key:
        demo_1_basic_agent()
    demo_2_model_retry_middleware()
    demo_3_backoff_schedule()
    demo_4_model_with_retry()
    print_tip('Agent 场景优先 ModelRetryMiddleware；纯链用 model.with_retry()。')
    print('\n🎉 Phase 2-2 完成！下一步: lg_2_3_time_travel_demo.py')
