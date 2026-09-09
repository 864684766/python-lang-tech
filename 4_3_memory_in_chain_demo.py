"""
========================================================
LangChain Phase 4-3
Memory 注入 Chain 教材级 Demo
========================================================

📚 学习目标：
1. 将对话历史注入 LCEL 链（路线图 4-3 核心场景）
2. 掌握 RunnableWithMessageHistory 自动管理 session 历史
3. 搭建支持多轮对话的记忆聊天机器人

🔗 路线图示例的现代实现：
  经典写法（langchain_classic）:
    memory_buffer.load_memory_variables(x)["history"]

  LangChain 1.3.14 推荐:
    RunnableWithMessageHistory 自动读写 InMemoryChatMessageHistory

运行前提：
  pip install langchain langchain-core langchain-community zhipuai python-dotenv
"""
import os
from numbers import Number
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()
_session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

def demo_1_manual_injection():
    """
    路线图原始模式的手动实现：
    chain 输入 dict 包含 history + input，Prompt 用 MessagesPlaceholder 接收 history。
    """
    print('\n' + '=' * 65)
    print('Demo 1: 手动注入 history 到 LCEL 链')
    print('=' * 65)
    prompt = ChatPromptTemplate.from_messages([('system', '你是友好的中文助手，请结合对话历史回答。'), MessagesPlaceholder(variable_name='history'), ('human', '{input}')])
    chain = prompt | get_model() | StrOutputParser()
    history = InMemoryChatMessageHistory()
    history.add_user_message('我叫王五')
    history.add_ai_message('你好王五，很高兴认识你！')
    answer = chain.invoke({'history': history.messages, 'input': '我叫什么？'})
    print(f'问: 我叫什么？\n答: {answer}')
    history.add_user_message('我叫什么？')
    history.add_ai_message(answer)
    print(f'\n历史消息数: {len(history.messages)}')

def demo_2_runnable_with_history():
    """
    RunnableWithMessageHistory 包装链后：
    - invoke 时自动从 get_session_history 加载历史
    - 执行完毕后自动将本轮 Q&A 写入历史
    - 通过 config["configurable"]["session_id"] 区分会话
    """
    print('\n' + '=' * 65)
    print('Demo 2: RunnableWithMessageHistory 自动管理')
    print('=' * 65)
    prompt = ChatPromptTemplate.from_messages([('system', '你是记忆助手，能记住用户在本次会话中说的信息。'), MessagesPlaceholder(variable_name='history'), ('human', '{input}')])
    base_chain = prompt | get_model() | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(base_chain, get_session_history, input_messages_key='input', history_messages_key='history')
    config = {'configurable': {'session_id': 'chat-session-001'}}
    turns = ['我住在杭州，职业是产品经理', '我住哪个城市？', '我的职业是什么？']
    for user_input in turns:
        answer = chain_with_history.invoke({'input': user_input}, config=config)
        print(f'\n👤 {user_input}')
        print(f'🤖 {answer}')
    stored = get_session_history('chat-session-001')
    print(f'\n📦 会话共保存 {len(stored.messages)} 条消息')

def demo_3_windowed_chatbot():
    """
    综合实战：RunnableWithMessageHistory + trim_messages 窗口控制。
    防止历史无限增长超出模型上下文。
    """
    print('\n' + '=' * 65)
    print('Demo 3: 带窗口截断的记忆聊天机器人')
    print('=' * 65)
    WINDOW_SIZE = 6

    def trim_history(history: list) -> list:
        return trim_messages(history, max_tokens=WINDOW_SIZE, token_counter=len, strategy='last', start_on='human')
    prompt = ChatPromptTemplate.from_messages([('system', '你是简洁的中文助手。'), MessagesPlaceholder(variable_name='history'), ('human', '{input}')])

    def chain_with_trim(inputs: dict) -> str:
        trimmed = trim_history(inputs.get('history', []))
        return (prompt | get_model() | StrOutputParser()).invoke({'history': trimmed, 'input': inputs['input']})
    from langchain_core.runnables import RunnableLambda
    bot = RunnableWithMessageHistory(RunnableLambda(chain_with_trim), get_session_history, input_messages_key='input', history_messages_key='history')
    config = {'configurable': {'session_id': 'windowed-chat-001'}}
    for i in range(5):
        q = f'这是第 {i + 1} 轮测试消息'
        bot.invoke({'input': q}, config=config)
    msgs = get_session_history('windowed-chat-001').messages
    print(f'5 轮对话后历史消息数: {len(msgs)}')
    print('💡 实际送入模型的 history 会被 trim 到 WINDOW_SIZE 条以内')

def demo_4_multi_session():
    """不同 session_id 的历史完全隔离"""
    print('\n' + '=' * 65)
    print('Demo 4: 多 session 隔离')
    print('=' * 65)
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name='history'), ('human', '{input}')])
    chain = RunnableWithMessageHistory(prompt | get_model() | StrOutputParser(), get_session_history, input_messages_key='input', history_messages_key='history')
    chain.invoke({'input': '我的代号是 Alpha'}, config={'configurable': {'session_id': 's1'}})
    chain.invoke({'input': '我的代号是 Beta'}, config={'configurable': {'session_id': 's2'}})
    r1 = chain.invoke({'input': '我的代号是什么？'}, config={'configurable': {'session_id': 's1'}})
    r2 = chain.invoke({'input': '我的代号是什么？'}, config={'configurable': {'session_id': 's2'}})
    print(f'session s1 回答: {r1}')
    print(f'session s2 回答: {r2}')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 4-3: Memory 注入 Chain              ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('手动注入', '自己把 history 塞进 Prompt'),
        ('RunnableWithMessageHistory', '链自动读写 history'),
        ('窗口聊天', 'trim 后再注入'),
        ('多 session', 'session_id 隔离用户'),
    ])
    check_api_key()
    try:
        demo_1_manual_injection()
        demo_2_runnable_with_history()
        demo_3_windowed_chatbot()
        demo_4_multi_session()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 4 全部完成！下一步: 5_1_cache_demo.py\n')