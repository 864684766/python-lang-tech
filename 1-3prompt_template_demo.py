"""
Prompt Template 系统详解 Demo (1-3)

本文件演示 LangChain 的提示词模板系统，包括：
  - PromptTemplate：纯文本模板，适合简单单行输入
  - ChatPromptTemplate：消息列表模板，支持 system/human/ai/tool 角色
  - MessagesPlaceholder：占位符，用于注入历史对话消息
  - StringPromptTemplate + 自定义格式化函数

适合初学者：每一步都有详细的中文注释和图解。

运行前提：
  pip install langchain langchain-core zhipuai python-dotenv
  设置环境变量 ZHIPUAI_API_KEY

注意：环境变量名是 ZHIPUAI_API_KEY（中间有 AI），不是 ZHIPU_API_KEY。
"""
import os
from typing import List
from urllib import response
from dotenv import load_dotenv
from numbers import Number
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, StringPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatZhipuAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan

def demo_prompt_template():
    print('\n' + '=' * 65)
    print('Demo 1: PromptTemplate - Pure Text Template')
    print('=' * 65)
    template = PromptTemplate(template='Please summarize the following {name} and extract 3 key points:\n\n{article}', input_variables=['article', 'name'])
    article_content = '\n    LangChain is a framework for developing applications powered by language models.\n    It provides two core components: the LangChain library and the LangChain CLI.\n    The library is mainly used to build LLM applications by composing various components.\n    The CLI manages the lifecycle of LangChain applications including creation, testing and deployment.\n    '
    formatted_prompt = template.format(article=article_content, name='Big Y')
    print('\n[Template] Formatted prompt:')
    print(formatted_prompt[:150] + '...')
    model = get_model()
    print('\n[Model] Sending to model for summary...\n')
    result = model.invoke(formatted_prompt).content
    print(f'[Result]\n{result}')
    print('\n[NOTE]:')
    print('   - PromptTemplate is suitable for simple single-line text scenarios')
    print('   - input_variables is optional but recommended for validation')
    print('   - format() returns a plain string, not a message object')

def demo_chat_prompt_template():
    print('\n' + '=' * 65)
    print('Demo 2: ChatPromptTemplate - Message List Template')
    print('=' * 65)
    chat_template = ChatPromptTemplate.from_messages([('system', 'You are a professional technical document translator.'), ('human', 'Please translate the following {source_lang} text to {target_lang}:\n\n{source_text}')])
    input_data = {'source_lang': 'Chinese', 'target_lang': 'English', 'source_text': 'LangChain provides rich components for building AI applications.'}
    messages = chat_template.format_messages(**input_data)
    print('\n[Messages] Generated message list:')
    for i, msg in enumerate(messages):
        print(f'  [{i}] Role: {msg.type} | Content: {msg.content[:50]}...')
    model = get_model()
    print('\n[Model] Sending to model for translation...\n')
    result = model.invoke(messages).content
    print(f'[Result]\n{result}')
    print('\n[NOTE]:')
    print('   - ChatPromptTemplate is the core tool for building conversational apps')
    print('   - format_messages() returns a message list that can be passed directly to LLM')
    print('   - from_messages() accepts multiple formats including lists and single messages')

def demo_messages_placeholder():
    print('\n' + '=' * 65)
    print('Demo 3: MessagesPlaceholder - Dynamic History Injection')
    print('=' * 65)
    chat_template_with_history = ChatPromptTemplate.from_messages([('system', 'You are a patient programming teacher.'), MessagesPlaceholder(variable_name='chat_history'), ('human', '{user_question}')])
    chat_history = [HumanMessage(content='What is Python?'), AIMessage(content='Python is a high-level programming language known for its concise syntax.'), HumanMessage(content='What can it be used for?'), AIMessage(content='Python can be used for web development, data analysis, AI and more.')]
    current_question = 'What is the difference between Java and Python?'
    messages = chat_template_with_history.format_messages(chat_history=chat_history, user_question=current_question)
    print(f'\n[Messages] Final message list ({len(messages)} total):')
    for i, msg in enumerate(messages):
        preview = msg.content[:40] + '...' if len(msg.content) > 40 else msg.content
        print(f'  [{i}] {msg.type}: {preview}')
    model = get_model()
    print('\n[Model] Sending to model with full history...\n')
    result = model.invoke(messages).content
    print(f'[Result]\n{result}')
    print('\n[NOTE]:')
    print('   - MessagesPlaceholder enables dynamic injection of conversation history')
    print('   - variable_name specifies the dict key to pass when calling format()')
    print('   - This is a core technique for building multi-turn dialogue bots')

def demo_string_prompt_template():
    """
    StringPromptTemplate 允许你完全控制模板的格式化过程。

    【关于 Pydantic 验证错误的解答】

    错误信息：
      ValidationError: input_variables Field required

    原因分析：
      1. StringPromptTemplate 继承自 Pydantic BaseModel（通过 BasePromptTemplate）
      2. Pydantic v2 要求所有字段都必须有值（要么有默认值，要么是可选的）
      3. 父类 BasePromptTemplate 定义了两个关键字段：
         - template: str          （模板字符串）
         - input_variables: List[str]  （输入变量列表）
      4. 当你在子类中写 `template: str` **没有默认值**时，
         Pydantic v2 会认为这是一个**新的必填字段**
      5. 但初始化时你只传了 template="..."，没传 input_variables，所以报错

    【解决方案】

    ❌ 错误写法（会报错）：
       class CustomPromptTemplate(StringPromptTemplate):
           template: str                    # 没有默认值 -> Pydantic 认为是必填新字段
           suffix: str = "..."              # 有默认值 -> OK
           # input_variables 没声明 -> 报错！

    ✅ 正确写法（推荐）：
       class CustomPromptTemplate(StringPromptTemplate):
           input_variables: List[str] = []  # 必须声明，给默认空列表
           suffix: str = "..."              # 你的自定义字段
           # 不要重新声明 template！父类已经处理好了

    【为什么 template: str 不需要写？】
       - template 字段已经由父类 BasePromptTemplate 定义了
       - 你在初始化时传的 template="..." 会被父类自动接收和处理
       - 如果你在子类中再次声明 template: str（没有默认值），Pydantic 会认为
         你要覆盖父类的定义，并且变成一个新的必填字段
       - 这就是为什么报错说 "Field required"
    """
    print('\n' + '=' * 65)
    print('Demo 4: StringPromptTemplate - Custom Template Class')
    print('=' * 65)

    class CustomPromptTemplate(StringPromptTemplate):
        """
        自定义提示词模板类
        【关键修复点】
        必须声明 input_variables 并给默认值！
        不要重新声明 template 字段（父类已处理）
        """
        template: str = ''
        input_variables: List[str] = []
        suffix: str = 'Please answer within 200 words.'

        def format(self, **kwargs) -> str:
            """
            重写格式化方法，实现自定义逻辑

            参数:
              **kwargs：格式化的变量字典（如 topic="机器学习"）

            返回:
              格式化后的完整提示词字符串
            """
            base_prompt = self.template.format(**kwargs)
            return f'{base_prompt}\n\n{self.suffix}'
    custom_template = CustomPromptTemplate(template='请简要介绍一下 {topic}', suffix='要求：易于理解，适合初学者')
    print('\n[Template] Custom template structure:')
    print(f'  Suffix: {custom_template.suffix}')
    print(f'  Input variables: {custom_template.input_variables}')
    result = custom_template.format(topic='机器学习')
    print(f'\n[Formatted] Result:\n{result}')
    model = get_model()
    print('\n[Model] Sending to model...\n')
    answer = model.invoke(result).content
    print(f'[Result]\n{answer}')
    print('\n[NOTE]:')
    print('   - 自定义模板需要继承 StringPromptTemplate')
    print('   - 您必须声明 input_variables: List[str] = []，以避免 Pydantic 错误')
    print('   - 不要重新声明模板字段（父类已处理）')
    print('   - 仅声明您自己的自定义字段（如后缀）')

def demo_full_conversation_system():
    print('\n' + '=' * 65)
    print('演示 5: 完整的多轮对话系统')
    print('=' * 65)
    model = get_model()
    system_prompt = ChatPromptTemplate.from_messages([('system', '你是一个聪明且友好的 AI 助手。\n\n你的职责：\n1. 清晰、简洁地回答用户问题\n2. 针对技术问题提供具体的示例\n3. 保持友好和鼓励的态度\n4. 遇到不知道的问题时保持诚实')])
    conversation_history = []
    questions = ['你好，我想学习 Python，应该如何入门？', '谢谢！你能推荐一些学习资源吗？', '好的，你认为 Python 最难的部分是什么？']
    print('\n[对话] 开始模拟多轮对话:\n')
    for i, question in enumerate(questions, 1):
        print(f"{'-' * 50}")
        print(f'第 {i} 轮:')
        print(f'用户: {question}\n')
        system_msg = system_prompt.format_messages()
        messages = system_msg.copy()
        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
        messages.append(HumanMessage(content=question))
        response = model.invoke(messages)
        conversation_history.append({'role': 'user', 'content': question})
        conversation_history.append({'role': 'assistant', 'content': response.content})
        print(f'助手: {response.content}\n')
        print(f'  (已保存 {len(conversation_history)} 条历史记录)')
    print('\n[注意]:')
    print('   - 完整的对话系统需要维护消息历史记录')
    print('   - 每一轮都需要将历史消息重新发送给模型')
    print('   - 可以使用字典或专用的存储后端来保存历史')

def show_cheat_sheet():
    card = '\n\n==============================================================\n  Prompt Template Cheat Sheet\n==============================================================\n\n[PromptTemplate] Simple text template\n  template = PromptTemplate(\n      template="Summarize: {text}",\n      input_variables=["text"]\n  )\n\n[ChatPromptTemplate] Conversation template\n  template = ChatPromptTemplate.from_messages([\n      ("system", "You are an assistant"),\n      ("human", "{question}")\n  ])\n\n[MessagesPlaceholder] History placeholder\n  MessagesPlaceholder(variable_name="history")\n  # Usage: format(chat_history=[...])\n\n[StringPromptTemplate] Custom template (FIXED!)\n  class MyTemplate(StringPromptTemplate):\n      input_variables: List[str] = []  # REQUIRED! Give default empty list\n      suffix: str = "..."              # Your custom field\n      # DON\'T re-declare \'template\' - parent class handles it!\n      def format(self, **kwargs):\n          return super().format(**kwargs) + "\\nextra"\n\nKEY DIFFERENCES:\n  - PromptTemplate.format() -> returns str\n  - ChatPromptTemplate.format_messages() -> returns List[BaseMessage]\n\n==============================================================\n'
    print(card)

def demo_conversation_chain():
    """
    演示 LangGraph 的多轮对话与多用户隔离机制。

    【背景对比】
    在旧版 LangChain 的 ConversationChain 中，如果不显式传入 session_id，
    所有对话历史会默认存在同一个内存对象中，导致多用户场景下历史记录混杂（串台）。

    【新版优势】
    LangGraph 通过 Checkpointer（状态检查点）和 thread_id 机制，
    强制要求按会话隔离状态。只要传入不同的 thread_id，每个用户的会话历史
    都会被严格独立存储，彻底解决了串台问题。
    """
    prompt = ChatPromptTemplate.from_messages([('system', '你是一个暴躁的女助手，喜欢用不耐烦的语气回答问题。'), MessagesPlaceholder(variable_name='messages')])
    model = get_model()

    def call_model(state: MessagesState):
        chain = prompt | model
        response = chain.invoke(state)
        return {'messages': response}
    builder = StateGraph(MessagesState)
    builder.add_node('chatbot', call_model)
    builder.set_entry_point('chatbot')
    memory = InMemorySaver()
    app = builder.compile(checkpointer=memory)
    config = {'configurable': {'thread_id': 'abc-123'}}
    print('第一轮：')
    res1 = app.invoke({'messages': [HumanMessage(content='你好，我叫张三')]}, config)
    print(res1['messages'][-1].content)
    print('\n第二轮：')
    res2 = app.invoke({'messages': [HumanMessage(content='你还记得我叫什么吗？')]}, config)
    print(res2['messages'][-1].content)
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n+----------------------------------------------------------+\n|                                                          |\n|       LangChain Prompt Template Tutorial                 |\n|       (Section 1-3)                                      |\n|                                                          |\n|  Covers: PromptTemplate / ChatPromptTemplate /           |\n|          MessagesPlaceholder / Custom Template           |\n|                                                          |\n+----------------------------------------------------------+\n')
    print_run_plan([
        ('对话链', 'ChatPromptTemplate + MessagesPlaceholder。同文件还有基础 PromptTemplate 可单独调用'),
    ])
    check_api_key()
    try:
        demo_conversation_chain()
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        print('Hint: Check network connection and ZHIPUAI_API_KEY.')
        raise
    show_cheat_sheet()
    print('\n[SUCCESS] Prompt Template tutorial complete!\n')