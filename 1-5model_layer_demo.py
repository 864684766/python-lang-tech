"""
Model Layer（模型层）统一接口详解 Demo (1-5)

本文件演示 LangChain 对各家大模型的统一包装层，包括：
  - ChatModel.invoke(messages) / .stream(messages) —— 统一调用入口
  - AIMessage / HumanMessage / SystemMessage / ToolMessage —— 消息类型体系
  - model.with_structured_output(SchemaClass) —— 强制结构化输出
  - model.bind_tools([tool1, tool2]) —— 绑定工具供 LLM 自主调用

适合初学者：每一步都有详细的中文注释和图解。

运行前提：
  pip install langchain langchain-core pydantic zhipuai python-dotenv
  设置环境变量 ZHIPUAI_API_KEY

注意：环境变量名是 ZHIPUAI_API_KEY（中间有 AI），不是 ZHIPU_API_KEY。
"""
import os
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from numbers import Number
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan

def demo_basic_invoke_stream():
    """
    演示 LLM 最基本的两种调用方式：同步调用和流式调用。

    【核心概念】
    ChatModel 是所有聊天模型的统一接口，它提供两个核心方法：
      - invoke(messages)：同步调用，等待完整结果后返回
      - stream(messages)：流式调用，逐块返回生成的内容

    【工作流程图】

    invoke() 模式:
    ┌─────────────┐     ┌──────────┐     ┌─────────────┐
    │  构建消息   │ ──▶ │  LLM推理 │ ──▶ │ 完整结果    │
    │  列表       │     │  等待    │     │  一次性返回 │
    └─────────────┘     └──────────┘     └─────────────┘

    stream() 模式:
    ┌─────────────┐     ┌──────────┐     ┌─────────────┐
    │  构建消息   │ ──▶ │  LLM推理 │ ──▶ │ 逐个 chunk  │
    │  列表       │     │          │     │  边生成边   │
    └─────────────┘     └──────────┘     │  推送返回   │
                                         └─────────────┘
    """
    print('\n' + '=' * 65)
    print('Demo 1: 基本调用 — invoke() vs stream()')
    print('=' * 65)
    model = get_model()
    print('\n📡 方式一：invoke() 同步调用\n')
    messages = [SystemMessage(content='你是一个友好的中文助手，请用简洁的语言回答问题。'), HumanMessage(content='请用三句话介绍一下什么是 Python？')]
    print('发送的消息:')
    for msg in messages:
        role_emoji = '🤖' if msg.type == 'system' else '👤'
        print(f'   {role_emoji} [{msg.type}]: {msg.content[:30]}...')
    response = model.invoke(messages)
    print(f'\n⏱️  调用耗时: {type(response).__name__}')
    print(f'📝 回复内容:\n   {response.content}\n')
    print('\n📡 方式二：stream() 流式调用\n')
    stream_messages = [SystemMessage(content='你是一个诗歌爱好者。'), HumanMessage(content='请写一首关于秋天的五言绝句。')]
    print('🎙️  流式输出效果（模拟打字机）:\n')
    full_text = ''
    for chunk in model.stream(stream_messages):
        content = chunk.content if chunk.content else ''
        full_text += content
        print(content, end='', flush=True)
    print('\n' + '-' * 50)
    print(f'💡 流式输出共收到 {len(full_text)} 个字符\n')
    print('💡 要点:')
    print('   · invoke() 适合不需要实时显示的简单场景')
    print('   · stream() 适合聊天界面、需要边生成边显示的场景')
    print('   · 两者都返回 AIMessage 或迭代器，内部处理相同')

def demo_message_types():
    """
    演示 LangChain 的四类核心消息对象。

    【核心概念】
    LangChain 定义了四种标准消息类型，对应对话中的不同角色：

    1. SystemMessage：系统指令，设定 AI 的角色和行为准则
       - 通常放在消息列表的最前面
       - 用户不可见，仅影响模型行为
       - 类似"你是..."、"你的职责是..."

    2. HumanMessage：用户输入，代表真实用户的提问或输入
       - 这是对话的主要驱动者
       - 可以多次出现，形成多轮对话

    3. AIMessage：助手回复，代表 AI 的回答
       - 由模型自动生成
       - 在多轮对话中保持上下文连贯性

    4. ToolMessage：工具执行结果
       - 当 AI 调用外部工具后的返回结果
       - 用于让 AI 知道工具的执行情况

    【消息流转图】

    用户输入 → HumanMessage
                 │
                 ▼
    ┌──────────────────────────────────────┐
    │         System 指令（静默）            │
    │         Human 输入                    │
    │         AI 历史回复                   │
    │              │                        │
    │              ▼                        │
    │         LLM 推理                      │
    │              │                        │
    │              ▼                        │
    │         AIMessage（回复）             │
    │              │                        │
    │              ├─→ 直接返回给用户        │
    │              │                        │
    │              └─→ 调用工具              │
    │                         │             │
    │                         ▼             │
    │                  Tool 执行            │
    │                         │             │
    │                         ▼             │
    │                  ToolMessage          │
    │                         │             │
    │                         ▼             │
    │                  再次进入 LLM         │
    └──────────────────────────────────────┘
    """
    print('\n' + '=' * 65)
    print('Demo 2: 消息类型 — 四种核心消息对象')
    print('=' * 65)
    model = get_model()
    conversation = [SystemMessage(content='你是一位专业的编程教师，擅长用简单的例子解释复杂概念。\n你的回答应该：\n1. 准确且易于理解\n2. 包含具体的代码示例\n3. 鼓励学生继续提问'), HumanMessage(content='请问 Python 中的装饰器是什么？能举个例子吗？'), AIMessage(content="装饰器是 Python 中一种高级功能，它允许你在不修改原函数代码的情况下，动态地增强函数的功能。就像给函数套上一个'外衣'。\n\n例如，我们可以定义一个简单的计时装饰器：\n\n```python\nimport time\n\n@timer\ndef my_function():\n    time.sleep(1)\n\ndef timer(func):\n    def wrapper():\n        start = time.time()\n        func()\n        end = time.time()\n        print(f'耗时: {end - start:.2f}秒')\n    return wrapper\n```\n\n这就是装饰器的基本原理！"), HumanMessage(content='明白了！那这个装饰器具体是怎么工作的呢？')]
    print('\n📋 构建的消息列表（共 {} 条）:'.format(len(conversation)))
    for i, msg in enumerate(conversation, 1):
        icon_map = {'system': '⚙️', 'human': '👤', 'ai': '🤖', 'tool': '🔧'}
        icon = icon_map.get(msg.type, '❓')
        content_preview = msg.content[:50] + '...' if len(msg.content) > 50 else msg.content
        print(f'   {i}. {icon} [{msg.type}]: {content_preview}')
    print('\n🤖 正在调用模型生成回复...\n')
    response = model.invoke(conversation)
    print(f'✅ 模型回复:\n{response.content}')
    print('\n💡 要点:')
    print('   · SystemMessage 通常放在最前面，只出现一次')
    print('   · HumanMessage 和 AIMessage 交替出现，形成对话')
    print('   · ToolMessage 在工具调用场景中使用')
    print('   · 消息顺序非常重要，决定了 AI 的理解上下文')

def demo_structured_output():
    """
    演示如何使用 with_structured_output() 强制 LLM 返回结构化数据。

    【核心概念】
    with_structured_output() 是 LangChain 的一个强大功能，它可以让 LLM
    严格按照你定义的 Pydantic 模型格式返回数据。

    工作原理：
    1. 你定义一个 Pydantic 模型（定义输出的数据结构）
    2. 调用 model.with_structured_output(MyModel) 创建一个专用链
    3. LLM 被要求在提示词中以 JSON 格式输出，并符合你的 schema
    4. 返回的结果自动解析为 Pydantic 模型实例

    【适用场景】
    - 需要提取固定字段的数据
    - 需要保证输出格式的稳定性
    - 需要后续进行数据验证和处理

    【流程对比】

    普通调用:
    LLM → 自由文本 → 需要用 Parser 手动解析 → 容易出错

    with_structured_output:
    LLM → 结构化 JSON → 自动解析为 Pydantic 对象 → 类型安全
    """
    print('\n' + '=' * 65)
    print('Demo 3: with_structured_output() — 强制结构化输出')
    print('=' * 65)

    class ArticleAnalysis(BaseModel):
        """文章分析结果的标准化结构"""
        title: str = Field(..., description='文章的主标题')
        main_topic: str = Field(..., description='文章的主要主题，如 AI、技术、商业等')
        key_points: List[str] = Field(..., description='文章的核心要点，3-5条')
        sentiment: str = Field(..., description='文章情感：正面/负面/中性')
        word_count: int = Field(..., description='文章的预估字数')
        readability: str = Field(..., description='阅读难度：入门/进阶/专业')
    structured_model = get_model().with_structured_output(ArticleAnalysis)
    test_article = '\n    LangChain 2.0 重磅发布，带来多项革命性更新。\n    新版引入了全新的 Agent 架构，支持更复杂的推理任务和多步骤工作流。\n    RAG（检索增强生成）性能提升显著，响应速度提高 50%。\n    社区贡献者表示，新版本在生产环境中的稳定性大幅改善。\n    主要功能包括：\n    - 增强的工具调用能力\n    - 更好的多模态支持\n    - 企业级安全特性\n    预计将在 2024 年第三季度完成全面部署。\n    '
    print('\n📄 待分析文章:\n')
    print(test_article[:100] + '...')
    print('\n🤖 正在分析文章结构...\n')
    result = structured_model.invoke(f'请分析以下文章的结构和内容：\n\n{test_article}')
    print('✅ 分析结果 (Pydantic 模型实例):')
    print(f'   类型: {type(result).__name__}')
    print(f'   标题: {result.title}')
    print(f'   主题: {result.main_topic}')
    print(f'   关键要点 ({len(result.key_points)} 条):')
    for i, point in enumerate(result.key_points, 1):
        print(f'     {i}. {point}')
    print(f'   情感倾向: {result.sentiment}')
    print(f'   预估字数: {result.word_count}')
    print(f'   难度等级: {result.readability}')
    print('\n💡 要点:')
    print('   · with_structured_output() 让 LLM 输出严格符合你的数据结构')
    print('   · Pydantic 模型自动验证数据类型，出错时会有明确提示')
    print('   · 适合需要将 LLM 输出接入后端系统的场景')
    print('   · 返回的是模型实例，可以直接访问字段属性')

def demo_bind_tools():
    """
    演示如何使用 bind_tools() 让 LLM 自主决定调用哪些工具。

    【核心概念】
    bind_tools() 将一组工具绑定到模型上，LLM 会根据用户的问题
    自主判断是否需要调用工具，以及调用哪个工具。

    工作流程：
    1. 定义工具（使用 @tool 装饰器）
    2. 使用 bind_tools() 将工具绑定到模型
    3. LLM 会根据需要返回工具调用指令
    4. 你需要执行工具并将结果返回给 LLM

    【典型应用场景】
    - 天气查询：LLM 调用天气工具获取实时数据
    - 计算器：LLM 调用计算工具进行精确运算
    - 搜索：LLM 调用搜索工具获取最新信息
    - 数据库：LLM 调用数据库工具查询数据
    """
    print('\n' + '=' * 65)
    print('Demo 4: bind_tools() — 工具绑定与调用')
    print('=' * 65)

    @tool
    def calculate(expression: str) -> str:
        """执行数学表达式计算，支持加减乘除和括号

        Args:
            expression: 数学表达式，如 "2 + 3 * 4"
        Returns:
            计算结果字符串
        """
        try:
            result = eval(expression, {'__builtins__': {}}, {})
            return str(result)
        except Exception as e:
            return f'计算错误: {e}'

    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的天气信息

        Args:
            city: 城市名称，如 "北京"、"上海"
        Returns:
            天气信息字符串
        """
        weather_data = {'北京': '晴天，温度 25°C，风速 3级', '上海': '多云，温度 22°C，湿度 70%', '广州': '小雨，温度 28°C，湿度 85%'}
        return weather_data.get(city, f'暂不支持查询 {city} 的天气')

    @tool
    def translate(text: str, target_lang: str) -> str:
        """将文本翻译成目标语言

        Args:
            text: 要翻译的原文
            target_lang: 目标语言，如 "英文"、"日文"
        Returns:
            翻译结果字符串
        """
        translations = {'Hello World': {'英文': 'Hello World', '日文': 'ハローワールド', '韩文': '헬로 월드'}, '谢谢': {'英文': 'Thank you', '日文': 'ありがとう', '韩文': '감사합니다'}}
        key = text
        if key not in translations:
            key = next((k for k in translations if text in k), None)
        if key and target_lang in translations[key]:
            return translations[key][target_lang]
        return f'[模拟翻译] {text} 的 {target_lang} 翻译'
    tools = [calculate, get_weather, translate]
    model_with_tools = create_agent(get_model(), tools)
    print('\n📦 已绑定的工具:')
    for i, t in enumerate(tools, 1):
        print(f'   {i}. {t.name}: {t.description[:40]}...')
    test_questions = ['帮我计算一下 (15 + 25) * 3 等于多少？', '北京今天天气怎么样？', "请把 '你好世界' 翻译成英文。", '今天天气不错，帮我算一下 100 减 37 是多少？']
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 50}")
        print(f'问题 [{i}]: {question}\n')
        response = model_with_tools.invoke({'messages': [HumanMessage(content=question)]})
        print(f'🤖 模型响应:')
        print(f'   内容: ')
        for msg in response['messages']:
            msg.pretty_print()

def execute_tool(tool_call: dict, tools: list) -> str:
    """执行工具调用并返回结果

    Args:
        tool_call: 工具调用字典，包含 name, args, id
        tools: 可用工具列表
    Returns:
        工具执行结果字符串
    """
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    for tool in tools:
        if tool.name == tool_name:
            return tool.invoke(tool_args)
    return f'未找到工具: {tool_name}'

def demo_intelligent_assistant():
    """
    综合应用：构建一个带有工具调用的智能助手系统。

    功能：
    1. 接收用户问题
    2. 根据问题类型决定是否调用工具
    3. 调用合适的工具获取信息
    4. 将工具结果反馈给模型生成最终回答
    """
    print('\n' + '=' * 65)
    print('Demo 5: 综合应用 — 智能助手系统')
    print('=' * 65)

    @tool
    def search_knowledge(query: str) -> str:
        """在知识库中搜索相关信息

        Args:
            query: 搜索关键词
        Returns:
            相关知识片段
        """
        knowledge_base = {'Python': 'Python 是一种高级编程语言，以简洁易读著称，广泛用于数据分析、AI 开发等领域。', 'LangChain': 'LangChain 是用于构建 LLM 应用的框架，提供提示模板、工具调用、记忆管理等组件。', 'RAG': 'RAG（检索增强生成）是一种结合检索系统和生成模型的技术，能提供更准确的回答。'}
        for key, value in knowledge_base.items():
            if key.lower() in query.lower() or query.lower() in key.lower():
                return f'【知识片段】{value}'
        return '【知识片段】未找到相关知识，请尝试其他关键词'

    @tool
    def perform_calculation(expr: str) -> str:
        """执行数学计算

        Args:
            expr: 数学表达式
        Returns:
            计算结果
        """
        try:
            result = eval(expr, {'__builtins__': {}}, {})
            return f'【计算结果】{expr} = {result}'
        except Exception as e:
            return f'【计算错误】{e}'
    tools = [search_knowledge, perform_calculation]
    assistant_model = create_agent(get_model(), tools)
    print('\n🤖 智能助手已启动！')
    print('支持的查询：')
    print('   - Python 相关问题')
    print('   - LangChain 相关问题')
    print('   - 数学计算问题\n')
    print("输入 '退出' 结束程序\n")
    test_scenarios = ['什么是 LangChain？', '帮我计算一下 256 * 3.14', 'Python 有什么特点？', '请计算 (100 + 200) * 2']
    for i, user_input in enumerate(test_scenarios, 1):
        print(f"{'─' * 50}")
        print(f'用户: {user_input}\n')
        conversation = {'messages': [SystemMessage(content='你是一个智能助手，可以搜索知识和执行计算。'), HumanMessage(content=user_input)]}
        response = assistant_model.invoke(conversation)
        conversation['messages'].append(response['messages'][1:])
        print(f'🤖 助手:\n')
        for msg in response['messages']:
            msg.pretty_print()

def show_cheat_sheet():
    """打印一张速查卡片"""
    card = '\n\n╔═══════════════════════ Model Layer 速查卡片 ═══════════════════════════╗\n\n║                                                                            ║\n║  📌 基本调用                                                                ║\n║     response = model.invoke(messages)    # 同步调用                        ║\n║     for chunk in model.stream(messages):  # 流式调用                        ║\n║                                                                            ║\n║  📌 消息类型                                                                ║\n║     SystemMessage("系统指令")              ← 设定角色和行为                ║\n║     HumanMessage("用户输入")               ← 用户提问                       ║\n║     AIMessage("AI回复")                    ← 助手回答                       ║\n║     ToolMessage("工具结果")                ← 工具执行结果                   ║\n║                                                                            ║\n║  📌 结构化输出                                                              ║\n║     structured_model = model.with_structured_output(MyModel)               ║\n║     result = structured_model.invoke(input)                                 ║\n║                                                                            ║\n║  📌 工具绑定                                                                ║\n║     model_with_tools = model.bind_tools([tool1, tool2])                    ║\n║     response = model_with_tools.invoke(messages)                           ║\n║     # 检查 response.tool_calls 获取工具调用                                 ║\n║                                                                            ║\n║  🔑 最佳实践：                                                             ║\n║     • 始终在消息列表最前面添加 SystemMessage                               ║\n║     • 使用 with_structured_output 保证输出格式稳定                         ║\n║     • 使用 bind_tools 让模型自主选择工具                                   ║\n║     • 记得处理工具调用结果并返回给模型                                     ║\n║                                                                            ║\n╚════════════════════════════════════════════════════════════════════════════╝\n'
    print(card)
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║                                                          ║\n║          LangChain Model Layer 教程                       ║\n║          (1-5 节)                                        ║\n║                                                          ║\n║  涵盖: invoke/stream / 消息类型 / 结构化输出 / 工具绑定   ║\n║                                                          ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('智能助手综合', '本入口跑综合示例。同文件还有 temperature/bind_tools 等小节可单独调用'),
    ])
    check_api_key()
    try:
        demo_intelligent_assistant()
    except Exception as e:
        print(f'\n❌ 运行出错: {e}')
        import traceback
        traceback.print_exc()
        print('提示：请检查网络连接和 ZHIPUAI_API_KEY 是否正确设置。')
        raise
    show_cheat_sheet()
    print('\n🎉 Model Layer 教程完成！你现在已经掌握了 LangChain 的模型层核心概念。\n')