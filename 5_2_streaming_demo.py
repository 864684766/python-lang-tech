"""
========================================================
LangChain Phase 5-2
Streaming 全链路 教材级 Demo
========================================================

📚 学习目标：
1. LLM token-level streaming：model.stream()
2. LCEL chain-level streaming：chain.stream()
3. 事件级 streaming：chain.astream_events()（async）
4. Agent stream：create_agent 的 stream(updates)

运行前提：
  pip install langchain langchain-core langchain-community langgraph zhipuai python-dotenv
"""
import asyncio
import os
from numbers import Number
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def demo_1_model_stream():
    print('\n' + '=' * 65)
    print('Demo 1: Model.stream() Token 级流式')
    print('=' * 65)
    model = get_model()
    print('输出: ', end='', flush=True)
    full = ''
    for chunk in model.stream('用三句话介绍 Python'):
        text = chunk.content if chunk.content else ''
        full += text
        print(text, end='', flush=True)
    print(f'\n\n共 {len(full)} 字符')
    print('💡 适合聊天 UI 打字机效果')

def demo_2_chain_stream():
    print('\n' + '=' * 65)
    print('Demo 2: chain.stream() 管道级流式')
    print('=' * 65)
    prompt = ChatPromptTemplate.from_messages([('human', '写一首关于编程的短诗，4行即可')])
    chain = prompt | get_model() | StrOutputParser()
    print('chain.stream 输出:\n')
    for chunk in chain.stream({}):
        print(chunk, end='', flush=True)
    print('\n\n💡 StrOutputParser 使 stream 直接产出字符串 chunk')

async def demo_3_astream_events_async():
    """
    astream_events(version='v2') 产出细粒度事件：
    on_chat_model_stream / on_chain_start / on_chain_end 等。
    适合构建调试面板或 SSE 推送。
    """
    print('\n' + '=' * 65)
    print('Demo 3: astream_events 事件流')
    print('=' * 65)
    prompt = ChatPromptTemplate.from_messages([('human', '{q}')])
    chain = prompt | get_model() | StrOutputParser()
    event_types: dict[str, int] = {}
    token_chunks = 0
    async for event in chain.astream_events({'q': '说一个技术冷笑话'}, version='v2'):
        kind = event.get('event', '')
        event_types[kind] = event_types.get(kind, 0) + 1
        if kind == 'on_chat_model_stream':
            token_chunks += 1
    print('事件统计:')
    for k, v in sorted(event_types.items()):
        print(f'  · {k}: {v}')
    print(f'Token stream 事件数: {token_chunks}')
    print('💡 前端 SSE/WebSocket 可订阅 on_chat_model_stream 实时渲染')

def demo_3_astream_events():
    asyncio.run(demo_3_astream_events_async())

@tool
def echo_tool(text: str) -> str:
    """回显输入文本"""
    return f'[echo] {text}'

def demo_4_agent_stream():
    print('\n' + '=' * 65)
    print('Demo 4: Agent stream(updates)')
    print('=' * 65)
    agent = create_agent(get_model(), [echo_tool])
    question = '用 echo_tool 回显：streaming works'
    for i, chunk in enumerate(agent.stream({'messages': [HumanMessage(content=question)]}, stream_mode='updates'), 1):
        print(f'  update [{i}]: {list(chunk.keys())}')
    print('💡 Agent 流式可实时展示「思考→调工具→回答」各阶段')

def demo_5_comparison():
    print('\n' + '=' * 65)
    print('Demo 5: stream vs invoke 选型')
    print('=' * 65)
    print('\n    ┌─────────────┬────────────────────┬─────────────────────────┐\n    │  方法       │  返回方式           │  适用场景                │\n    ├─────────────┼────────────────────┼─────────────────────────┤\n    │  invoke     │  阻塞，一次返回      │  批处理、后端同步 API    │\n    │  stream     │  迭代器，逐块返回    │  聊天 UI、打字机效果     │\n    │  astream    │  异步迭代器          │  FastAPI/WebSocket      │\n    │  astream_events │  细粒度事件      │  调试、Observability    │\n    └─────────────┴────────────────────┴─────────────────────────┘\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 5-2: Streaming 全链路                ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('model.stream', '逐 token 打出'),
        ('chain.stream', '整条 LCEL 链流式'),
        ('astream_events', '按事件类型过滤（on_chat_model_stream）'),
        ('agent.stream', 'ReAct 中间步骤也能流'),
        ('对比表', 'invoke vs stream vs events'),
    ])
    check_api_key()
    try:
        demo_1_model_stream()
        demo_2_chain_stream()
        demo_3_astream_events()
        demo_4_agent_stream()
        demo_5_comparison()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 5-2 完成！下一步: 5_3_async_demo.py\n')