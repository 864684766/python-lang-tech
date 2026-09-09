"""
========================================================
LangChain Phase 4-1
Message History 管理 教材级 Demo
========================================================

📚 学习目标：
1. 理解 LCEL 层面对话历史的存储与读取
2. 掌握 InMemoryChatMessageHistory（LangChain 1.3.14 推荐）
3. 掌握 trim_messages 实现「窗口截断」与「Token 截断」
4. 了解 langchain_classic 旧版 Memory 类的迁移关系

🔗 现代 vs 经典 API：
┌────────────────────────────┬──────────────────────────────────────┐
│  路线图/经典名称              │  LangChain 1.3.14 推荐替代            │
├────────────────────────────┼──────────────────────────────────────┤
│  ConversationBufferWindow  │  InMemoryChatMessageHistory + trim     │
│  ConversationSummaryBuffer │  定期用 LLM 摘要 + trim_messages       │
│  TokenBuffer               │  trim_messages(max_tokens=...)         │
└────────────────────────────┴──────────────────────────────────────┘

运行前提：
  pip install langchain langchain-core langchain-community zhipuai python-dotenv
"""
import os
from numbers import Number
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def demo_1_basic_history():
    """
    BaseChatMessageHistory 是对话历史的统一抽象。
    InMemoryChatMessageHistory 是最简单的内存实现（进程重启后丢失）。
    """
    print('\n' + '=' * 65)
    print('Demo 1: InMemoryChatMessageHistory 基础操作')
    print('=' * 65)
    history = InMemoryChatMessageHistory()
    history.add_user_message('我叫小明，喜欢 Python')
    history.add_ai_message('你好小明！Python 是很棒的语言。')
    history.add_user_message('我还喜欢 LangChain')
    print(f'消息数量: {len(history.messages)}')
    for i, msg in enumerate(history.messages, 1):
        print(f'  [{i}] {msg.type}: {msg.content}')
    print('\n💡 add_user_message / add_ai_message 是对 add_message 的便捷封装')

def demo_2_window_trim():
    """
    等价于经典 ConversationBufferWindowMemory(k=N) 的现代实现：
    用 trim_messages + token_counter=len 按「消息条数」截断。
    """
    print('\n' + '=' * 65)
    print('Demo 2: 窗口截断（保留最近 4 条消息）')
    print('=' * 65)
    long_history = [SystemMessage(content='你是助手'), HumanMessage(content='第1轮：你好'), AIMessage(content='你好！'), HumanMessage(content='第2轮：今天天气'), AIMessage(content='请告诉我城市'), HumanMessage(content='第3轮：北京'), AIMessage(content='北京晴天'), HumanMessage(content='第4轮：谢谢'), AIMessage(content='不客气'), HumanMessage(content='第5轮：再见')]
    print(f'原始消息数: {len(long_history)}')
    trimmed = trim_messages(long_history, max_tokens=4, token_counter=len, strategy='last', include_system=True, start_on='human')
    print(f'截断后消息数: {len(trimmed)}')
    for msg in trimmed:
        print(f'  · {msg.type}: {msg.content[:40]}')
    print('\n💡 token_counter=len 时，max_tokens 表示「消息条数上限」')

def demo_3_token_trim():
    """
    等价于 TokenBuffer：用 trim_messages 的 token_counter='approximate'
    按近似 Token 数截断，适合控制 API 成本。
    """
    print('\n' + '=' * 65)
    print('Demo 3: 近似 Token 截断')
    print('=' * 65)
    messages = [SystemMessage(content='你是简洁助手'), HumanMessage(content='请详细解释什么是机器学习' * 5), AIMessage(content='机器学习是...' * 10), HumanMessage(content='那深度学习呢？')]
    trimmed = trim_messages(messages, max_tokens=100, token_counter='approximate', strategy='last', include_system=True)
    for msg in trimmed:
        print(f'  · {msg.type}: {msg.content[:40]}')
    print(f'原始 {len(messages)} 条 → 截断后 {len(trimmed)} 条')
    total_chars = sum((len(m.content) for m in trimmed if isinstance(m.content, str)))
    print(f'截断后总字符约: {total_chars}')
    print("\n💡 'approximate' 比精确 Token 计数快，适合运行时热路径")

def demo_4_summary_compress():
    """
    经典 ConversationSummaryBufferMemory 的思路：
    历史太长时，用 LLM 把早期对话压缩成摘要，再与近期消息合并。
    """
    print('\n' + '=' * 65)
    print('Demo 4: 摘要压缩（模拟 SummaryBuffer）')
    print('=' * 65)
    early_messages = [HumanMessage(content='我叫张三，在北京工作'), AIMessage(content='你好张三，北京是个很棒的城市！'), HumanMessage(content='我是做后端开发的'), AIMessage(content='后端开发涉及 API、数据库等。')]
    recent_messages = [HumanMessage(content='LangChain 的 Memory 怎么用？')]
    summarize_prompt = ChatPromptTemplate.from_messages([('system', '将以下对话压缩为 2~3 句摘要，保留关键事实（姓名、职业等）。'), ('human', '{conversation}')])
    summarize_chain = summarize_prompt | get_model() | StrOutputParser()
    conversation_text = '\n'.join((f'{m.type}: {m.content}' for m in early_messages))
    summary = summarize_chain.invoke({'conversation': conversation_text})
    print(f'📋 早期对话摘要:\n   {summary}')
    merged = [SystemMessage(content=f'对话背景摘要：{summary}'), *recent_messages]
    print(f'\n合并后消息数: {len(merged)}（原早期 {len(early_messages)} 条已压缩为 1 条摘要）')
    response = get_model().invoke(merged)
    print(f'🤖 模型回答: {response.content[:150]}...')

def demo_5_messages_placeholder():
    """演示如何在 Prompt 模板中预留 history 插槽"""
    print('\n' + '=' * 65)
    print('Demo 5: MessagesPlaceholder 注入历史')
    print('=' * 65)
    prompt = ChatPromptTemplate.from_messages([('system', '你是多轮对话助手，请参考历史上下文回答。'), MessagesPlaceholder(variable_name='history'), ('human', '{input}')])
    chain = prompt | get_model() | StrOutputParser()
    history = InMemoryChatMessageHistory()
    history.add_user_message('我叫李四')
    history.add_ai_message('你好李四！')
    result = chain.invoke({'history': history.messages, 'input': '我叫什么名字？'})
    print(f'问题: 我叫什么名字？')
    print(f'回答: {result}')
    print('\n💡 MessagesPlaceholder 是 LCEL 链中注入动态历史的标准方式')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 4-1: Message History 管理             ║\n║     版本: LangChain 1.3.14                               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('摘要压缩', '长对话先摘要再带进上下文。同文件还有 InMemoryHistory / trim 可单独调用'),
        ('MessagesPlaceholder', '把历史插进 Prompt 模板'),
    ])
    check_api_key()
    try:
        demo_4_summary_compress()
        demo_5_messages_placeholder()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 4-1 完成！下一步: 4_2_memory_layers_demo.py\n')