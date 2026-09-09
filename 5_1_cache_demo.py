"""
========================================================
LangChain Phase 5-1
缓存机制 教材级 Demo
========================================================

📚 学习目标：
1. 理解 LLM 缓存的作用：省钱 + 提速
2. 掌握 InMemoryCache（内存级，进程内有效）
3. 了解 SQLiteCache（持久化，重启后仍命中）
4. 判断何时该用缓存

🔗 缓存原理：
  相同 prompt + 相同模型参数 → 相同 key → 直接返回缓存，不调 API

运行前提：
  pip install langchain langchain-core langchain-community zhipuai python-dotenv
"""
import os
import tempfile
import time
from numbers import Number
from dotenv import load_dotenv
from langchain_core.caches import InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def build_translation_chain():
    """固定 prompt 的翻译链 —— 最适合缓存的场景"""
    prompt = ChatPromptTemplate.from_messages([('system', '你是翻译器，只输出译文，不要解释。'), ('human', '将以下文本翻译为英文：{text}')])
    return prompt | get_model(temperature=0) | StrOutputParser()

def demo_1_in_memory_cache():
    print('\n' + '=' * 65)
    print('Demo 1: InMemoryCache 内存缓存')
    print('=' * 65)
    set_llm_cache(InMemoryCache())
    chain = build_translation_chain()
    text = '你好，世界'
    t0 = time.perf_counter()
    r1 = chain.invoke({'text': text})
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    r2 = chain.invoke({'text': text})
    t3 = time.perf_counter()
    print(f'第 1 次（API）: {r1!r}  耗时 {t1 - t0:.2f}s')
    print(f'第 2 次（缓存）: {r2!r}  耗时 {t3 - t2:.4f}s')
    print(f'结果一致: {r1 == r2}')
    print(f'缓存类型: {type(get_llm_cache()).__name__}')
    print('\n💡 第 2 次命中缓存，通常快 10~100 倍且零 Token 消耗')

def demo_2_cache_miss_hit():
    print('\n' + '=' * 65)
    print('Demo 2: 缓存命中 vs 未命中')
    print('=' * 65)
    set_llm_cache(InMemoryCache())
    chain = build_translation_chain()
    r_a = chain.invoke({'text': '早上好'})
    r_b = chain.invoke({'text': '晚上好'})
    r_a2 = chain.invoke({'text': '早上好'})
    print(f'「早上好」第1次: {r_a}')
    print(f'「晚上好」（miss）: {r_b}')
    print(f'「早上好」第2次（hit）: {r_a2}')
    print('\n💡 缓存 key = prompt 序列化 + 模型参数；任一变化都会 miss')

def demo_3_sqlite_cache():
    """
    SQLiteCache 来自 langchain_community.cache。
    缓存写入 SQLite 文件，进程重启后仍可命中。
    """
    print('\n' + '=' * 65)
    print('Demo 3: SQLiteCache 持久化缓存')
    print('=' * 65)
    from langchain_community.cache import SQLiteCache
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'llm_cache.db')
        set_llm_cache(SQLiteCache(database_path=db_path))
        chain = build_translation_chain()
        r1 = chain.invoke({'text': 'LangChain 缓存测试'})
        print(f'写入缓存: {r1}')
        set_llm_cache(None)
        set_llm_cache(SQLiteCache(database_path=db_path))
        chain2 = build_translation_chain()
        r2 = chain2.invoke({'text': 'LangChain 缓存测试'})
        print(f'重启后读取: {r2}')
        print(f'持久化命中: {r1 == r2}')
        print(f'数据库文件: {db_path}')
    print('\n💡 适合固定 prompt 批处理、翻译、分类等重复任务')

def demo_4_when_to_cache():
    print('\n' + '=' * 65)
    print('Demo 4: 缓存使用决策指南')
    print('=' * 65)
    print('\n    ✅ 适合缓存：\n       · prompt 模板固定，仅输入变量变化范围有限\n       · 相同问题会被多次询问（FAQ、文档摘要）\n       · 开发调试阶段反复跑同一条链\n       · temperature=0 的确定性输出\n\n    ❌ 不适合缓存：\n       · 需要实时数据（股价、天气）\n       · 每次 prompt 都唯一（带 timestamp）\n       · 高 temperature 创意写作（同 prompt 也可能不同输出）\n       · 安全敏感场景（缓存可能泄露他人查询结果 → 需隔离 key）\n\n    成本估算示例：\n       某翻译 API 1000 次/天重复率 40%\n       → 缓存可节省约 40% Token 费用\n    ')

def demo_5_clear_cache():
    print('\n' + '=' * 65)
    print('Demo 5: 清理缓存')
    print('=' * 65)
    cache = InMemoryCache()
    set_llm_cache(cache)
    chain = build_translation_chain()
    chain.invoke({'text': '测试清理'})
    cache.clear()
    set_llm_cache(None)
    print('✅ 已 clear() 并 set_llm_cache(None)')
    print('💡 模型升级或 prompt 变更后应清理旧缓存')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 5-1: 缓存机制                        ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('何时该缓存', '相同 prompt 才命中。同文件还有 InMemory / SQLiteCache 可单独调用'),
        ('清空缓存', '改 prompt 或参数后必须清，否则会拿到旧答案'),
    ])
    check_api_key()
    try:
        demo_4_when_to_cache()
        demo_5_clear_cache()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    finally:
        set_llm_cache(None)
    print('\n🎉 Phase 5-1 完成！下一步: 5_2_streaming_demo.py\n')