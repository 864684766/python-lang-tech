"""
Runnable 协议详解 Demo

本文件演示 LangChain 中 Runnable 协议的核心方法：
  .invoke() / .stream() / .batch() / .with_fallbacks() / .with_config() / .with_retry()

适合初学者学习：每一段代码都配有详细的中文注释。

运行前提：
  pip install langchain langchain-openai
  export ZHIPUAI_API_KEY="你的智谱 API Key"

注意：环境变量名是 ZHIPUAI_API_KEY（中间有 AI），不是 ZHIPU_API_KEY。
"""
import os
import time
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_step, print_run_plan

def demo_invoke():
    """
    .invoke(input) 是 Runnable 最基本的调用方式。

    工作原理：
      1. 你把输入（字符串/字典/消息列表）传进去
      2. Runnable 阻塞等待，直到模型返回完整结果
      3. 返回一个完整的输出对象

    类比：就像打电话问一个问题，等对方回答完再挂断。

    适用场景：单次查询、不需要流式输出的简单任务。
    """
    print('\n' + '=' * 60)
    print('Demo 1: .invoke() — 一次调用，一次性返回完整结果')
    print('=' * 60)
    model = get_model()
    print('\n📡 正在向模型发送请求（invoke 模式，等待完整回复）...')
    start_time = time.time()
    result = model.invoke('请用一句话解释什么是 Runnable？')
    elapsed = time.time() - start_time
    answer_text = result.content
    print(f'\n⏱️  耗时: {elapsed:.2f} 秒')
    print(f'🤖  模型回答:\n{answer_text}')
    print(f'\n💡 要点: invoke() 返回的是完整的一次性结果，类型为 {type(result).__name__}')

def demo_stream():
    """
    .stream(input) 让模型的回答像打字机一样逐块输出。

    工作原理：
      1. 你发起请求
      2. 模型每生成一小段文字（Chunk），就会立即推送给你
      3. 你可以用 for 循环实时处理每一小块

    类比：就像看直播 vs 看录播。invoke 是录播（等全部结束再看），
          stream 是直播（边说边听）。

    适用场景：聊天界面实时显示、需要降低用户感知延迟的场景。
    """
    print('\n' + '=' * 60)
    print('Demo 2: .stream() — 流式输出，一个字一个字地打印')
    print('=' * 60)
    model = get_model()
    print('\n📡 正在流式接收模型回答（请注意文字是逐步出现的）:')
    print('-' * 40)
    full_answer = ''
    for chunk in model.stream('请写一首五言绝句，主题是「编程」'):
        piece = chunk.content
        full_answer += piece
        print(piece, end='', flush=True)
    print('\n' + '-' * 40)
    print(f'\n💡 要点: stream() 返回迭代器，共收到 {len(full_answer)} 个字符')
    print('   好处：用户可以立刻看到开头，不用苦等整篇回答完成。')

def demo_batch():
    """
    .batch(inputs) 一次性向模型发送多个独立的请求。

    工作原理：
      1. 传入一个输入列表，比如 ["问题A", "问题B", "问题C"]
      2. Runnable 会对每个输入分别发起调用（内部尽量并行）
      3. 返回一个结果列表，顺序与输入一一对应

    类比：就像餐厅后厨同时炒三盘菜，而不是做完一盘再做下一盘。

    适用场景：需要对大量输入做同样的处理（比如翻译一批句子、分析一堆评论）。

    注意：max_concurrency 参数控制最大并发数；
          return_exceptions=True 时，某个请求失败不会导致整个批次崩溃。
    """
    print('\n' + '=' * 60)
    print('Demo 3: .batch() — 批量发送多个请求，并行处理')
    print('=' * 60)
    model = get_model()
    questions = ['什么是 Python？用一句话回答。', '什么是 JavaScript？用一句话回答。', '什么是 Rust？用一句话回答。']
    print(f'\n📡 批量发送 {len(questions)} 个问题（batch 模式）:')
    for i, q in enumerate(questions, 1):
        print(f'   [{i}] {q}')
    print('\n⏳ 等待模型回答...')
    start_time = time.time()
    results = model.batch(questions, max_concurrency=3, return_exceptions=False)
    elapsed = time.time() - start_time
    print(f'\n⏱️  总耗时: {elapsed:.2f} 秒\n')
    for i, (question, result) in enumerate(zip(questions, results), 1):
        if isinstance(result, BaseException):
            print(f'   ❌ [{i}] 出错: {result}')
        else:
            print(f'   📝 [{i}] Q: {question}')
            print(f'        A: {result.content}\n')
    print('💡 要点: batch() 比串行调用快很多（3个问题同时飞出去，一起等）。')

def demo_chain_pipeline():
    """
    演示 Runnable 的真正威力：用 |（管道操作符）串联多个步骤。

    流程：提示词模板 → 模型 → 输出解析器

    因为每一步都是 Runnable，串联后的整体也是一个 Runnable！
    这意味着 .invoke() / .stream() / .batch() 可以直接用在链上，
    而不需要单独在每一步之间写胶水代码。

    工作原理图解：
        用户输入 ──▶ [模板填入变量] ──▶ [发给LLM] ──▶ [提取纯文本] ──▶ 最终结果

    这叫做 LCEL（LangChain Expression Language）。
    """
    print('\n' + '=' * 60)
    print('Demo 4: LCEL 管道 — 模板 → 模型 → 解析，三步无缝串联')
    print('=' * 60)
    model = get_model()
    prompt_template = ChatPromptTemplate.from_messages([('system', '你是一个友好的百科全书助手。'), ('human', '请简要介绍 {topic}，控制在 100 字以内。')])
    chain = prompt_template | model | StrOutputParser()
    topic = '机器学习'
    print(f'\n🔗 正在通过管道查询主题: 「{topic}」\n')
    answer = chain.invoke({'topic': topic})
    print(f'📖 {answer}\n')
    print('🔄 同一个管道换成 .stream() 流式输出:')
    print('-' * 40)
    for chunk in chain.stream({'topic': '深度学习'}):
        print(chunk, end='', flush=True)
    print('\n' + '-' * 40)
    print('\n💡 要点: 一条链 = 一套完整流水线，复用 invoke/stream/batch 三种调用方式。')

def demo_with_fallbacks():
    """
    演示“质量驱动的软回退（Quality-driven Fallback）”：
    当主模型的回答不符合预期时，主动触发异常，让备用模型接管。
    """
    print('\n' + '=' * 60)
    print('Demo 5: 质量校验不达标 → 自动切换备用模型')
    print('=' * 60)
    primary_model = get_model()
    fallback_model = get_model()

    def ultimate_fallback(inputs):
        return '⚠️ 抱歉，当前系统繁忙，所有AI服务暂时不可用。请稍后再试！'
    fallback_text_runnable = RunnableLambda(ultimate_fallback)

    def quality_validator(ai_message):
        if '抱歉' in ai_message.content or '不知道' in ai_message.content:
            raise ValueError(f'主模型回答质量不达标，触发回退！原始回答：{ai_message.content}')
        return ai_message
    primary_chain = primary_model | quality_validator
    fallback_chain = fallback_model
    robust_chain = primary_chain.with_fallbacks([fallback_chain, fallback_text_runnable])
    print('\n🛡️  已配置: 主模型(带质检) → 备用模型（软回退机制）')
    print('📡 正在通过带 fallback 的链提问...')
    result = robust_chain.invoke('请解释什么是「降级策略」（Fallback Strategy），一句话即可。')
    print(f'\n✅ 最终回答: {result.content}')
    print('\n💡 核心要点:')
    print('   · 默认情况下，只有抛出异常（如网络超时）才会触发 fallback')
    print("   · 通过在链末尾加 RunnableLambda 校验器，可以实现'软回退'")
    print("   · 当主模型回答敷衍（如包含'抱歉'）时，主动 raise ValueError 触发备胎")

def demo_advanced_config():
    """
    两个重要的增强方法：

    ① .with_config(config)  → 覆盖运行时配置（超时、标签、元数据等）
    ② .with_retry()         → 遇到特定错误时自动重试

    这两个方法通常结合使用，为生产环境做好准备。
    """
    print('\n' + '=' * 60)
    print('Demo 6: .with_config() + .with_retry() — 高级运行时配置')
    print('=' * 60)
    model = get_model()
    configured_model = model.with_config(tags=['my-demo-tag'], metadata={'experiment': 'beginner_tutorial'})
    print('\n🎛️  已添加运行时配置: tags + metadata')
    result = configured_model.invoke('你好，我是你的第一个配置化调用。请打个招呼。')
    print(f'👋 {result.content}')
    resilient_model = model.with_retry(wait_exponential_jitter=1, stop_after_attempt=3)
    print('\n🔄 已添加自动重试机制（最多 3 次）')
    result = resilient_model.invoke('重试机制已生效吗？请确认一下。')
    print(f'✅ 回答: {result.content}')
    print('\n💡 要点:')
    print('   · with_config() 用于打标签和传递元数据，便于调试与监控')
    print('   · with_retry() 应对不稳定网络或 API 限流，自动重试直到成功')
    print('   · 两者可以叠加: model.with_config(...).with_retry(...)')

def show_summary_table():
    """打印一个直观的方法对比表，帮助初学者快速回顾"""
    table = '\n┌──────────────────┬────────────────┬─────────────────────────────┬──────────────────────────┐\n│     方法          │   返回值类型    │           用途              │        典型场景           │\n├──────────────────┼────────────────┼─────────────────────────────┼──────────────────────────┤\n│ .invoke(input)   │ 单个完整结果    │ 同步调用，拿到全部结果再返回  │ 简单问答、脚本批处理       │\n│ .stream(input)   │ 迭代器 (chunks) │ 增量返回，边生成分边推送     │ 聊天 UI 实时显示          │\n│ .batch(inputs)   │ 结果列表        │ 并行处理多个独立请求         │ 翻译/分类大批量文本        │\n│ .with_fallbacks()│ Runnable 自身   │ 主方案失败时自动切换备胎     │ 高可用生产系统            │\n│ .with_config()   │ Runnable 自身   │ 注入标签、元数据、超时等配置 │ 可观测性与调试            │\n│ .with_retry()    │ Runnable 自身   │ 遇到特定错误时自动重试       │ 对抗网络抖动/API 限流      │\n└──────────────────┴────────────────┴─────────────────────────────┴──────────────────────────┘\n\n关键认知：\n  • invoke / stream / batch → 是「怎么调用」\n  • with_fallbacks / with_config / with_retry → 是「如何增强健壮性」\n  • 它们可以任意组合，比如 chain.with_retry().invoke(x)\n'
    print(table)
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║                                                          ║\n║          LangChain Runnable 协议完整教程                  ║\n║          (智谱 GLM-4-Flash 驱动)                         ║\n║                                                          ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('Demo 1 invoke', '一次调用拿完整结果。同文件还有 stream/batch/fallback/retry，可单独调用对应函数'),
    ])
    check_api_key()
    try:
        demo_invoke()
    except Exception as e:
        print(f'\n❌ 运行出错: {e}')
        print('提示：请检查网络连接和 ZHIPUAI_API_KEY 是否正确设置。')
        raise
    show_summary_table()
    print('\n🎉 所有 Demo 执行完毕！建议你修改 prompt 来体验不同效果。\n')