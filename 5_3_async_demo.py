"""
========================================================
LangChain Phase 5-3
异步编程 教材级 Demo
========================================================

📚 学习目标：
1. chain.ainvoke() — 单条异步调用
2. chain.abatch() — 批量并行
3. chain.astream() — 异步流式
4. 理解高并发场景下 async 的延迟优势

运行前提：
  pip install langchain langchain-core langchain-community zhipuai python-dotenv
"""
import asyncio
import os
import time
from numbers import Number
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def build_qa_chain():
    prompt = ChatPromptTemplate.from_messages([('system', '用一句话简洁回答。'), ('human', '{question}')])
    return prompt | get_model() | StrOutputParser()
QUESTIONS = ['什么是 Docker？', '什么是 Kubernetes？', '什么是 LangChain？']

async def demo_1_ainvoke():
    print('\n' + '=' * 65)
    print('Demo 1: chain.ainvoke() 单条异步')
    print('=' * 65)
    chain = build_qa_chain()
    result = await chain.ainvoke({'question': 'Python 是什么？'})
    print(f'回答: {result}')
    print('💡 在 async 路由（FastAPI）中直接用 await chain.ainvoke(...)')

async def demo_2_abatch():
    print('\n' + '=' * 65)
    print('Demo 2: chain.abatch() 批量并行')
    print('=' * 65)
    chain = build_qa_chain()
    inputs = [{'question': q} for q in QUESTIONS]
    t0 = time.perf_counter()
    results = await chain.abatch(inputs)
    t_async = time.perf_counter() - t0
    for q, r in zip(QUESTIONS, results):
        print(f'  Q: {q}\n  A: {r[:80]}...\n')
    print(f'abatch 总耗时: {t_async:.2f}s（3 个问题并行）')
    print('💡 abatch 内部并发调度，比串行 invoke 快 2~5 倍（取决于 API 限流）')

async def demo_3_sync_vs_async():
    print('\n' + '=' * 65)
    print('Demo 3: 串行 invoke vs 并行 abatch 耗时对比')
    print('=' * 65)
    chain = build_qa_chain()
    inputs = [{'question': q} for q in QUESTIONS]
    t0 = time.perf_counter()
    for inp in inputs:
        chain.invoke(inp)
    t_sync = time.perf_counter() - t0
    t1 = time.perf_counter()
    await chain.abatch(inputs)
    t_async = time.perf_counter() - t1
    print(f'串行 invoke 3 次: {t_sync:.2f}s')
    print(f'并行 abatch 3 次: {t_async:.2f}s')
    if t_async > 0:
        print(f'加速比约: {t_sync / t_async:.1f}x')

async def demo_4_astream():
    print('\n' + '=' * 65)
    print('Demo 4: chain.astream() 异步流式')
    print('=' * 65)
    chain = build_qa_chain()
    print('输出: ', end='', flush=True)
    async for chunk in chain.astream({'question': '介绍 asyncio'}):
        print(chunk, end='', flush=True)
    print('\n💡 WebSocket handler 中用 async for chunk in chain.astream(...)')

async def demo_5_gather():
    print('\n' + '=' * 65)
    print('Demo 5: asyncio.gather + ainvoke')
    print('=' * 65)
    chain = build_qa_chain()
    tasks = [chain.ainvoke({'question': q}) for q in QUESTIONS]
    results = await asyncio.gather(*tasks)
    print(f'gather 完成 {len(results)} 个任务')
    print('💡 gather 与 abatch 类似；abatch 是 LangChain 内置的批量抽象')

async def main():
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 5-3: 异步编程                        ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('ainvoke', '单次异步调用'),
        ('abatch', '一批输入并发'),
        ('同步 vs 异步耗时', '循环 invoke 会串行卡住'),
        ('astream', '异步流式'),
        ('asyncio.gather', '自己拼并发任务'),
    ])
    check_api_key()
    await demo_1_ainvoke()
    await demo_2_abatch()
    await demo_3_sync_vs_async()
    await demo_4_astream()
    await demo_5_gather()
    print('\n🎉 Phase 5-3 完成！下一步: 5_4_langsmith_demo.py\n')
if __name__ == '__main__':
    print_lesson(__file__)
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise