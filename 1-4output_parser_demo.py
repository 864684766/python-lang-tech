"""
Output Parser（输出解析）详解 Demo (1-4)

本文件演示 LangChain 的输出解析器，用于将 LLM 的自由文本输出转换为结构化的数据：
  - StrOutputParser()：提取纯文本字符串
  - CommaSeparatedListOutputParser：逗号分隔列表
  - PydanticOutputParser：直接反序列化为 Pydantic model
  - JsonOutputParser：解析 JSON 格式输出
  - parse_with_retry：解析失败时自动重试

适合初学者：每一步都有详细的中文注释和图解。

运行前提：
  pip install langchain langchain-core pydantic zhipuai python-dotenv
  设置环境变量 ZHIPUAI_API_KEY

注意：环境变量名是 ZHIPUAI_API_KEY（中间有 AI），不是 ZHIPU_API_KEY。
"""
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from numbers import Number
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain_community.chat_models import ChatZhipuAI

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan

def clean_markdown_json(text: str) -> str:
    """去掉 Markdown 代码块标记"""
    if not text:
        return text
    text = text.strip()
    match = re.search('```(?:json)?\\s*(.*?)\\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def parse_with_retry(inputs: dict) -> dict:
    """
    自定义 JSON 解析重试逻辑。
    inputs 必须包含: raw_output (str), format_instructions (str)
    :param inputs:
    :return:
    """
    instructions = inputs['format_instructions']
    max_retries = inputs.get('max_retries', 2)
    raw = None
    for attempt in range(max_retries + 1):
        if not raw:
            raw = clean_markdown_json(inputs.get('raw_output', ''))
        else:
            raw = clean_markdown_json(raw)
        try:
            res = json.loads(raw)
            return res
        except (OutputParserException, ValueError) as e:
            if attempt == max_retries:
                raise
            fix_prompt = f'你之前的 JSON 输出格式有误，解析时报错：\n\n            错误信息：{str(e)}\n        \n            你的原始输出：\n            {raw}\n        \n            格式要求：\n            {instructions}\n        \n            请修正输出，确保严格符合 JSON 格式。只输出修正后的 JSON 内容，不要任何解释：'
            model = get_model()
            raw = model.invoke(fix_prompt).content
    return json.loads(raw)

def demo_str_output_parser():
    """
    StrOutputParser 是默认的解析器，用于从 LLM 响应中提取纯文本。

    【为什么需要解析器？】
    LLM 返回的是 AIMessage 对象，它包含多个属性：
      - content：实际的文本内容
      - type：消息类型（如 "ai"）
      - additional_kwargs：额外的元数据

    StrOutputParser 的作用就是提取 .content 字段，返回一个干净的字符串。

    【工作流程图】

    LLM 输出: AIMessage(content="LangChain 是...", type="ai", ...)
                          │
                          ▼
              ┌───────────────────────┐
              │   StrOutputParser     │  ← 只取 .content 字段
              └───────────────────────┘
                          │
                          ▼
              纯字符串: "LangChain 是..."
    """
    print('\n' + '=' * 65)
    print('Demo 1: StrOutputParser — 提取纯文本')
    print('=' * 65)
    model = get_model()
    parser = StrOutputParser()
    chain = ChatPromptTemplate.from_messages([('system', '你是一个简洁的回答助手，请用一句话回答问题。'), ('human', '{question}')]) | model | parser
    question = '什么是人工智能？'
    print(f'\n📝 问题: {question}')
    print('🤖 等待模型回答...\n')
    result = chain.invoke({'question': question})
    print(f'✅ 解析结果 (类型: {type(result).__name__}):')
    print(f'   {result}')
    print(f'   字符数: {len(result)}')
    print('\n💡 要点:')
    print('   · StrOutputParser 是默认选择，适用于绝大多数场景')
    print('   · 它会把 AIMessage 转换为普通的 Python str 类型')
    print('   · 如果你不需要结构化输出，直接用这个就够了')

def demo_comma_separated_list_parser():
    """
    CommaSeparatedListOutputParser 专门用于提取逗号分隔的列表。

    【使用场景】
    当 LLM 被要求生成一个列表时，这个解析器可以帮你自动提取成 Python 列表。

    【工作流程图】

    LLM 可能输出:
    "Python, Java, JavaScript, C++, Rust, Go"
              │
              ▼
    ┌──────────────────────────────┐
    │ CommaSeparatedListParser     │  ← 按逗号和空格分割
    └──────────────────────────────┘
              │
              ▼
    Python 列表: ["Python", "Java", "JavaScript", "C++", "Rust", "Go"]
    """
    print('\n' + '=' * 65)
    print('Demo 2: CommaSeparatedListOutputParser — 提取列表')
    print('=' * 65)
    model = get_model()
    list_parser = CommaSeparatedListOutputParser()
    format_instructions = list_parser.get_format_instructions()
    print(f'\n📋 格式化指令:\n{format_instructions}\n')
    chain = ChatPromptTemplate.from_messages([('system', '你是一个编程语言专家。'), ('human', f'请列出 5 种常见的编程语言。{format_instructions}')]) | model | list_parser
    print('🤖 等待模型生成列表...\n')
    result = chain.invoke({})
    print(f'✅ 解析结果 (类型: {type(result).__name__}):')
    for i, lang in enumerate(result, 1):
        print(f'   {i}. {lang}')
    print('\n💡 要点:')
    print('   · get_format_instructions() 返回 LLM 应该遵循的输出格式说明')
    print('   · 解析器会智能处理各种分隔方式（逗号、换行等）')
    print('   · 返回的是 Python 原生的 list[str] 类型')

def demo_pydantic_output_parser():
    """
    PydanticOutputParser 是最高级的解析器，可以直接将 LLM 输出解析为 Pydantic 模型。

    【核心概念】
    Pydantic 是 Python 的数据验证库，允许你用类定义数据结构。
    PydanticOutputParser 会自动：
      1. 从 Pydantic 模型生成 JSON Schema 描述
      2. 在提示词中要求 LLM 按此格式输出
      3. 将 LLM 的 JSON 输出解析为模型实例
      4. 自动进行数据类型验证

    【工作流程图】

    Step 1: 定义 Pydantic 模型
    ┌─────────────────────────────────┐
    │ class ArticleSummary(BaseModel):│
    │   title: str                     │
    │   summary: str                   │
    │   keywords: List[str]            │
    └─────────────────────────────────┘
              │
              ▼
    Step 2: 生成 JSON Schema 描述
    ┌─────────────────────────────────┐
    │ {"title": str,                │
    │  "summary": str,              │
    │  "keywords": [str]}           │
    └─────────────────────────────────┘
              │
              ▼
    Step 3: LLM 按格式输出 JSON
    ┌─────────────────────────────────┐
    │ {                               │
    │   "title": "LangChain 教程",    │
    │   "summary": "这是一篇...",      │
    │   "keywords": ["AI", ...]       │
    │ }                               │
    └─────────────────────────────────┘
              │
              ▼
    Step 4: 解析为 Pydantic 对象
    ┌─────────────────────────────────┐
    │ ArticleSummary(                 │
    │   title="LangChain 教程",       │
    │   summary="这是一篇...",        │
    │   keywords=["AI", ...]         │
    │ )                               │
    └─────────────────────────────────┘
    """
    print('\n' + '=' * 65)
    print('Demo 3: PydanticOutputParser — 结构化数据解析')
    print('=' * 65)

    class ArticleSummary(BaseModel):
        """文章摘要的结构化表示"""
        title: str = Field(..., description='文章的主标题')
        summary: str = Field(..., description='文章的简要摘要，不超过 200 字')
        keywords: List[str] = Field(..., description='提取 3-5 个关键标签')
        difficulty: int = Field(..., description='文章难度等级 (1-5)')
    parser = PydanticOutputParser(pydantic_object=ArticleSummary)
    format_instructions = parser.get_format_instructions()
    print(f'\n📋 Pydantic 模型的 JSON Schema:')
    print(format_instructions)
    chain = ChatPromptTemplate.from_messages([('system', '你是一位专业的技术编辑，擅长提炼文章要点。'), ('human', '请阅读以下文章内容，并按照要求的格式提取信息：\n\n【文章内容】\n{text}\n\n请提取文章信息，格式要求：\n{format_instructions}\n')]) | get_model() | parser
    test_article = '\n    LangChain 是一个开源框架，用于开发由大语言模型（LLM）驱动的应用程序。\n    它提供了多种组件，包括模型_io、内存管理、检索增强生成（RAG）、代理（Agent）等。\n    LangChain 支持多种 LLM，如 OpenAI、Anthropic、Google PaLM 等。\n    该框架的目标是简化 LLM 应用的开发流程，让开发者能够专注于业务逻辑而非底层细节。\n    '
    print('\n🤖 等待模型分析文章...\n')
    result = chain.invoke({'text': test_article, 'format_instructions': format_instructions})
    print('✅ 解析成功！结果为 Pydantic 模型实例:')
    print(f'   类型: {type(result).__name__}')
    print(f'   标题: {result.title}')
    print(f'   摘要: {result.summary}')
    print(f"   关键词: {', '.join(result.keywords)}")
    print(f'   难度: {result.difficulty}/5')
    print('\n💡 要点:')
    print('   · Pydantic 模型提供强类型定义和数据验证')
    print('   · get_format_instructions() 自动生成 JSON Schema 说明')
    print('   · 解析失败时会自动抛出异常，便于错误处理')
    print('   · 适合需要严格结构化输出的生产环境')

def demo_json_output_parser():
    """
    JsonOutputParser 用于解析 LLM 输出的 JSON 文本。

    【与 PydanticOutputParser 的区别】
    - JsonOutputParser：只负责解析 JSON，不做类型验证
    - PydanticOutputParser：先定义模型，再解析并验证类型

    【适用场景】
    当你只需要解析 JSON，或者 JSON 结构不固定时，使用 JsonOutputParser 更灵活。
    """
    print('\n' + '=' * 65)
    print('Demo 4: JsonOutputParser — 通用 JSON 解析')
    print('=' * 65)
    model = get_model()
    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()
    chain = ChatPromptTemplate.from_messages([('system', '你是一位数据分析助手，擅长将信息整理为结构化格式。'), ('human', '请将以下信息整理为 JSON 格式：\n\n{data}\n\n格式要求：\n{format_instructions}\n')]) | model | json_parser
    raw_data = '\n    产品：智能手表\n    价格：2999元\n    功能：心率监测、GPS定位、防水\n    颜色：黑色、白色、蓝色\n    库存：500台\n    '
    print('\n📝 原始数据:')
    print(raw_data)
    print('\n🤖 等待模型解析为 JSON...\n')
    result = chain.invoke({'data': raw_data, 'format_instructions': format_instructions})
    print('✅ 解析结果:')
    print(f'   类型: {type(result).__name__}')
    print(f'   内容: {json.dumps(result, ensure_ascii=False, indent=2)}')
    print('\n💡 要点:')
    print('   · JsonOutputParser 解析结果为 Python dict')
    print('   · 不做类型验证，灵活但需要自行检查结构')
    print('   · 适合输出格式不固定或动态变化的场景')

def demo_retry_from_parsing():
    """
    parse_with_retry 在解析失败时自动让 LLM 重试。

    【为什么需要重试？】
    LLM 有时会输出格式不完全正确的 JSON，比如：
    - 缺少闭合括号
    - 使用了单引号而不是双引号
    - 包含额外的解释文本

    与其直接报错，不如让 LLM 看到错误提示后重新生成，提高成功率。

    【工作流程图】

    LLM 输出（格式错误）:
    "{"title": "LangChain", summary: "..."}"  ← 缺少引号
              │
              ▼
    解析失败: JSONDecodeError
              │
              ▼
    ┌──────────────────────────────┐
    │ parse_with_retry         │  ← 捕获异常
    │ 发送给 LLM 修正提示           │
    └──────────────────────────────┘
              │
              ▼
    LLM 重新输出（修正后）:
    "{"title": "LangChain", "summary": "..."}"  ← 正确格式
              │
              ▼
    解析成功，返回结果
    """
    print('\n' + '=' * 65)
    print('Demo 5: parse_with_retry — 解析失败自动重试')
    print('=' * 65)
    model = get_model()
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([('system', '你是一位专业的数据分析师。'), ('human', '请分析以下文本并提取关键信息，以 JSON 格式输出：\n\n{text}\n\n格式要求：\n{format_instructions}\n')])
    chain = RunnableParallel(raw_output=prompt | model | (lambda x: x.content), format_instructions=lambda x: parser.get_format_instructions(), max_retries=lambda x: 2) | RunnableLambda(parse_with_retry)
    text_to_analyze = '\n    LangChain 是一个强大的框架，支持多种功能：\n    - 文档加载和分块\n    - 向量存储和检索\n    - 对话记忆管理\n    - 工具调用和代理执行\n    \n    最新版本已发布，新增了对多模态模型的支持。\n    '
    print('\n🤖 发送请求给 LLM（带自动重试机制）...\n')
    result = chain.invoke({'text': text_to_analyze, 'format_instructions': parser.get_format_instructions()})
    print('✅ 解析成功:')
    print(f'   类型: {type(result).__name__}')
    print(f'   内容: {json.dumps(result, ensure_ascii=False, indent=2)}')
    print('\n💡 要点:')
    print('   · parse_with_retry 可以显著提高 JSON 解析成功率')
    print('   · 通过 parse_exception_types 控制哪些错误触发重试')
    print('   · max_retries 限制重试次数，防止无限循环')

def demo_full_pipeline():
    """
    综合应用：构建一个完整的文章摘要生成系统。

    功能：
    1. 接收任意长文本
    2. 调用 LLM 提取标题、摘要、关键词
    3. 使用 Pydantic 保证输出结构一致
    4. 添加重试机制确保稳定性
    """
    print('\n' + '=' * 65)
    print('Demo 6: 综合应用 — 智能摘要生成器')
    print('=' * 65)

    class SummaryResult(BaseModel):
        """摘要结果的统一结构"""
        original_title: str = Field(..., description='原文标题')
        generated_title: str = Field(..., description='生成的精简标题')
        abstract: str = Field(..., description='核心摘要')
        key_points: List[str] = Field(..., description='关键要点列表')
        sentiment: str = Field(..., description='文章情感倾向：正面/负面/中性')
    parser = PydanticOutputParser(pydantic_object=SummaryResult)
    prompt = ChatPromptTemplate.from_messages([('system', '你是一位资深的文章编辑，擅长提炼文章核心内容。\n    任务：对输入文章进行全面分析，提取关键信息。'), ('human', '请分析以下文章，并按照要求格式输出：\n                【文章标题】\n                {title}\n                【文章内容】\n                {content}                \n                输出格式要求：\n                {format_instructions}\n                ')])
    model = get_model()
    chain = RunnableParallel(raw_output=prompt | model | (lambda x: x.content), format_instructions=lambda x: parser.get_format_instructions(), max_retries=lambda x: 2) | RunnableLambda(parse_with_retry)
    test_content = '\n    LangChain 发布 v1.0 正式版，带来多项重大更新。\n    该版本引入了全新的 Agent 架构，支持更复杂的推理任务。\n    同时优化了 RAG（检索增强生成）性能，响应速度提升 40%。\n    社区反馈显示，新版本在实际生产环境中表现稳定。\n    主要贡献者表示，未来将继续加强多模态支持和企业级功能。\n    '
    print('\n📄 分析文章摘要...\n')
    result = chain.invoke({'title': 'LangChain v1.0 正式发布', 'content': test_content, 'format_instructions': parser.get_format_instructions()})
    print('✅ 摘要生成完成:')
    print(f"   原标题: {result['original_title']}")
    print(f"   精简标题: {result['generated_title']}")
    print(f"   核心摘要: {result['abstract']}")
    print(f'   关键要点:')
    for i, point in enumerate(result['key_points'], 1):
        print(f'     {i}. {point}')
    print(f"   情感倾向: {result['sentiment']}")

def show_cheat_sheet():
    """打印一张速查卡片"""
    card = '\n\n╔═══════════════════════ Output Parser 速查卡片 ═══════════════════════════╗\n\n║                                                                            ║\n║  📌 StrOutputParser                                                        ║\n║     最简单，只提取纯文本                                                    ║\n║     用法：| StrOutputParser()                                             ║\n║                                                                            ║\n║  📌 CommaSeparatedListOutputParser                                         ║\n║     提取逗号分隔的列表                                                      ║\n║     用法：                                                                  ║\n║       parser = CommaSeparatedListOutputParser()                             ║\n║       instructions = parser.get_format_instructions()                       ║\n║                                                                            ║\n║  📌 PydanticOutputParser （推荐）                                          ║\n║     最强类型安全，自动验证                                                   ║\n║     用法：                                                                  ║\n║       parser = PydanticOutputParser(pydantic_object=MyModel)               ║\n║       instructions = parser.get_format_instructions()                       ║\n║                                                                            ║\n║  📌 JsonOutputParser                                                       ║\n║     灵活 JSON 解析，无类型约束                                              ║\n║     用法：                                                                  ║\n║       parser = JsonOutputParser()                                          ║\n║       instructions = parser.get_format_instructions()                       ║\n║                                                                            ║\n║  📌 parse_with_retry                                                   ║\n║     解析失败时自动重试                                                       ║\n║     用法：retry_parser = parse_with_retry(parser, max_retries=2)        ║\n║                                                                            ║\n║  🔑 最佳实践：                                                             ║\n║     • 需要严格类型 → PydanticOutputParser                                  ║\n║     • 只需字符串 → StrOutputParser                                         ║\n║     • 需要灵活 → JsonOutputParser                                          ║\n║     • 生产环境 → 始终加 parse_with_retry                               ║\n║                                                                            ║\n╚════════════════════════════════════════════════════════════════════════════╝\n'
    print(card)
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║                                                          ║\n║          LangChain Output Parser 教程                     ║\n║          (1-4 节)                                        ║\n║                                                          ║\n║  涵盖: Str / List / Pydantic / JSON / Retry              ║\n║                                                          ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('StrOutputParser', '把 AIMessage 抠成纯字符串'),
        ('CommaSeparatedListParser', '逗号列表 → Python list'),
        ('PydanticOutputParser', '软解析成对象，失败可兜底'),
        ('JsonOutputParser', 'JSON 文本 → dict'),
        ('Retry 解析', '解析失败再问模型一次'),
        ('完整管道', 'Prompt | Model | Parser 串起来'),
    ])
    check_api_key()
    try:
        demo_str_output_parser()
        demo_comma_separated_list_parser()
        demo_pydantic_output_parser()
        demo_json_output_parser()
        demo_retry_from_parsing()
        demo_full_pipeline()
    except Exception as e:
        print(f'\n❌ 运行出错: {e}')
        import traceback
        traceback.print_exc()
        print('提示：请检查网络连接和 ZHIPUAI_API_KEY 是否正确设置。')
        raise
    show_cheat_sheet()
    print('\n🎉 Output Parser 教程完成！你现在可以优雅地处理各种 LLM 输出了。\n')