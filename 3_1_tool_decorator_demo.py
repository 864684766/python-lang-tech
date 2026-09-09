"""
========================================================
LangChain Phase 3-1
@tool 装饰器 教材级 Demo
========================================================

📚 学习目标：
1. 理解 LangChain 中「工具（Tool）」是什么、为什么 Agent 需要它
2. 掌握 @tool 装饰器的用法与参数含义
3. 理解 docstring / args_schema 如何变成 LLM 可见的工具描述
4. 学会直接调用 tool.invoke() 做单元测试

🔗 核心概念：
┌──────────────────────────────────────────────────────────────┐
│  普通 Python 函数                                              │
│       │                                                        │
│       ▼  @tool 装饰器                                           │
│  LangChain BaseTool 对象                                       │
│       │  • name        → 工具名称（LLM 调用时使用）             │
│       │  • description → 工具描述（LLM 决定是否调用）           │
│       │  • args_schema → 参数 JSON Schema（LLM 填参依据）       │
│       ▼                                                        │
│  可被 bind_tools() / create_agent() 绑定到模型                  │
└──────────────────────────────────────────────────────────────┘

运行前提：
  pip install langchain langchain-core pydantic zhipuai python-dotenv
  设置环境变量: ZHIPUAI_API_KEY=你的key

版本说明：本 Demo 基于 LangChain 1.3.14，API 以 langchain_core.tools.tool 为准。
"""
import json
import os
from numbers import Number
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def demo_0_concept():
    """
    【概念导读】

    在 LangChain Agent 体系中，Tool 是 LLM 与外部世界交互的「手」：

    - LLM 本身只能生成文本，无法查天气、算数学、读数据库
    - 我们把 Python 函数包装成 Tool，LLM 通过 tool_calls 请求调用
    - @tool 是最常用的包装方式：一行装饰器，函数变工具

    类比：
      函数 = 员工的具体技能
      @tool = 把技能登记进公司通讯录（含职责说明）
      LLM = 项目经理，看通讯录决定派谁干活
    """
    print('\n' + '=' * 65)
    print('Demo 0: 概念导读 —— Tool 是什么？')
    print('=' * 65)
    print('\n    普通函数:  def add(a, b): return a + b\n                    ↓ @tool\n    LangChain Tool:\n      · name = "add"\n      · description = 函数 docstring（LLM 读这个决定是否调用）\n      · args_schema = {"a": number, "b": number}（LLM 按 schema 填参）\n\n    下一步 Demo 会亲手创建并 inspect 这些字段。\n    ')

def demo_1_basic_tool():
    """
    【核心规则】
    @tool 装饰器会把函数变成 BaseTool 对象。
    函数的 docstring（文档字符串）会被 LLM 当作工具描述。

    LLM 不会看到函数源码，只会看到：
      - 工具名
      - 描述（docstring）
      - 参数 schema（从类型注解推断）
    """
    print('\n' + '=' * 65)
    print('Demo 1: 最基础的 @tool')
    print('=' * 65)

    @tool
    def get_current_time(location: str) -> str:
        """获取指定地点的当前时间（模拟数据，教学用）"""
        mock_times = {'北京': '2026-08-20 17:30:00 CST', '东京': '2026-08-20 18:30:00 JST', '纽约': '2026-08-20 05:30:00 EDT'}
        return mock_times.get(location, f'暂无 {location} 的时区数据')
    print(f'\n📦 工具名称: {get_current_time.name}')
    print(f'📝 工具描述: {get_current_time.description}')
    print(f'🔧 参数 Schema:\n{json.dumps(get_current_time.args, ensure_ascii=False, indent=2)}')
    result = get_current_time.invoke({'location': '北京'})
    print(f'\n✅ 直接调用结果: {result}')
    print('\n💡 要点:')
    print('   · docstring 写清楚「做什么、何时用」，LLM 才能正确选择工具')
    print('   · 参数必须有类型注解，LangChain 才能生成 JSON Schema')
    print('   · tool.invoke({...}) 可单独测试，无需消耗 LLM Token')

def demo_2_custom_description():
    """
    有时函数 docstring 面向开发者，而 LLM 需要更简洁/更面向任务的描述。
    此时用 @tool(description="...") 显式指定。
    """
    print('\n' + '=' * 65)
    print('Demo 2: 自定义 description')
    print('=' * 65)

    @tool(description='搜索互联网获取最新信息。当用户问实时新闻、股价、天气时使用。')
    def web_search(query: str) -> str:
        """
        内部实现：模拟搜索引擎。
        （这段 docstring 会被 description 参数覆盖，LLM 看不到）
        """
        database = {'langchain 最新版本': 'LangChain 1.3.x 系列，推荐使用 create_agent 构建 Agent。', 'python 3.13': 'Python 3.13 于 2024 年发布，改进了 GIL 和 typing 体验。'}
        for key, value in database.items():
            if key in query.lower() or query.lower() in key:
                return value
        return f'未找到与「{query}」相关的模拟搜索结果'
    print(f'📝 LLM 看到的描述: {web_search.description}')
    print(f"🔍 搜索结果: {web_search.invoke({'query': 'langchain 最新版本'})}")
    print('\n💡 要点:')
    print('   · description 参数优先级高于 docstring')
    print('   · 面向 LLM 的描述应说明「能力边界 + 使用场景」')

class SearchInputSchema(BaseModel):
    """联网搜索工具的输入参数（Pydantic 模型）"""
    query: str = Field(description='搜索关键词，尽量具体，如「2026 Python 新特性」')
    max_results: int = Field(default=3, ge=1, le=10, description='最多返回几条结果')

def demo_3_args_schema():
    """
    复杂工具建议用 Pydantic BaseModel 作为 args_schema：
    - 每个字段可写 Field(description=...) 给 LLM 更细的填参说明
    - 自动做类型校验（max_results 必须是 1~10 的整数）
    """
    print('\n' + '=' * 65)
    print('Demo 3: args_schema（Pydantic 精确定义参数）')
    print('=' * 65)

    @tool(args_schema=SearchInputSchema)
    def advanced_search(query: str, max_results: int=3) -> str:
        """在知识库中执行结构化搜索"""
        hits = [f'结果{i + 1}: 关于「{query}」的模拟条目' for i in range(max_results)]
        return '\n'.join(hits)
    schema = advanced_search.args_schema.model_json_schema()
    print('📋 自动生成的 JSON Schema:')
    print(json.dumps(schema, ensure_ascii=False, indent=2))
    result = advanced_search.invoke({'query': 'RAG 检索增强', 'max_results': 2})
    print(f'\n✅ 调用结果:\n{result}')
    print('\n💡 要点:')
    print('   · args_schema 让参数约束更严格，减少 LLM 乱填参数')
    print('   · Field(description=...) 会进入 schema，帮助 LLM 理解每个字段')

def demo_4_advanced_params():
    """
    演示 @tool 的其他常用参数（LangChain 1.3.14 支持）：
    - return_direct=True：工具结果直接作为 Agent 最终答案，不再回 LLM 加工
    - parse_docstring=True：从 Google Style docstring 解析参数描述
    """
    print('\n' + '=' * 65)
    print('Demo 4: @tool 高级参数')
    print('=' * 65)

    @tool(return_direct=True)
    def format_answer(text: str) -> str:
        """格式化最终答案，加粗标题后直接返回给用户"""
        return f'【最终答复】\n{text}'
    print(f'return_direct = {format_answer.return_direct}')
    print('含义: Agent 调用此工具后，不再让 LLM 二次总结，直接输出工具返回值')

    @tool(parse_docstring=True)
    def calculate(expression: str) -> str:
        """执行数学表达式计算。

        Args:
            expression: 数学表达式，支持 + - * / 和括号，如 "(2+3)*4"

        Returns:
            计算结果的字符串
        """
        try:
            result = eval(expression, {'__builtins__': {}}, {})
            return str(result)
        except Exception as exc:
            return f'计算错误: {exc}'
    print(f"\n计算 (2+3)*4 = {calculate.invoke({'expression': '(2+3)*4'})}")
    print('\n💡 parse_docstring=True 时，Args 段会写入参数 schema 的 description')

def demo_5_typed_params():
    """演示多种类型注解如何映射到工具 schema"""
    print('\n' + '=' * 65)
    print('Demo 5: 多参数类型注解 → JSON Schema')
    print('=' * 65)

    @tool
    def translate(text: str, target_lang: Literal['英文', '日文', '韩文'], formal: bool=False) -> str:
        """将文本翻译为目标语言。

        Args:
            text: 待翻译原文
            target_lang: 目标语言，只能是 英文/日文/韩文
            formal: 是否使用正式语气
        """
        tone = '正式' if formal else '口语'
        return f'[{tone}翻译→{target_lang}] {text}'
    print('参数 Schema:')
    print(json.dumps(translate.args, ensure_ascii=False, indent=2))
    print(f"\n示例: {translate.invoke({'text': '你好', 'target_lang': '英文', 'formal': True})}")
    print("\n💡 Literal['a','b'] 会生成 enum 约束，限制 LLM 只能选枚举值")

def demo_6_bind_preview():
    """
    预览：工具创建完成后，下一步就是 bind_tools / create_agent。
    这里只演示 bind_tools 后模型「看到」的工具列表格式。
    """
    print('\n' + '=' * 65)
    print('Demo 6: bind_tools 预览（需要 API Key）')
    print('=' * 65)

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气（模拟）"""
        data = {'北京': '晴 25°C', '上海': '多云 22°C'}
        return data.get(city, f'暂无 {city} 数据')
    model = get_model()
    model_with_tools = model.bind_tools([get_weather])
    print('✅ 模型已绑定工具，后续 3-2 / 3-4 节将演示完整 Agent 调用流程')
    print(f'   绑定工具数量: 1（{get_weather.name}）')
    print(f'   返回类型: {type(model_with_tools).__name__}')

def show_cheat_sheet():
    card = '\n╔════════════════════ @tool 装饰器速查（LangChain 1.3.14）══════════════════╗\n║                                                                            ║\n║  基础用法:                                                                  ║\n║    @tool                                                                   ║\n║    def my_func(arg: str) -> str:                                           ║\n║        docstring = LLM 看到的工具描述                                       ║\n║        return "..."                                                        ║\n║                                                                            ║\n║  常用参数:                                                                  ║\n║    description="..."     → 覆盖 docstring 作为工具描述                      ║\n║    args_schema=MyModel   → Pydantic 精确定义参数                            ║\n║    return_direct=True    → 工具结果直接作为最终答案                          ║\n║    parse_docstring=True  → 从 Google Style docstring 解析参数说明           ║\n║                                                                            ║\n║  测试工具:                                                                  ║\n║    result = my_func.invoke({"arg": "value"})                               ║\n║                                                                            ║\n║  关键原则:                                                                  ║\n║    · 参数必须有类型注解                                                      ║\n║    · docstring 写清「做什么 + 何时用」                                       ║\n║    · 先 invoke 单测，再交给 Agent                                           ║\n║                                                                            ║\n╚════════════════════════════════════════════════════════════════════════════╝\n'
    print(card)
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 3-1: @tool 装饰器教程                 ║\n║     版本: LangChain 1.3.14                               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('概念导读', 'Tool 是 LLM 的「手」'),
        ('基础 @tool', 'docstring 会变成工具描述'),
        ('自定义 description', '覆盖默认 docstring'),
        ('args_schema', 'Pydantic 精确参数'),
        ('高级参数', 'return_direct 等'),
        ('类型注解', 'Literal / 复杂类型'),
        ('bind_tools 预览', '需要 API Key，失败会跳过'),
    ])
    demo_0_concept()
    demo_1_basic_tool()
    demo_2_custom_description()
    demo_3_args_schema()
    demo_4_advanced_params()
    demo_5_typed_params()
    try:
        check_api_key()
        demo_6_bind_preview()
    except RuntimeError as err:
        print(f'\n⚠️  Demo 6 跳过: {err}')
    show_cheat_sheet()
    print('\n🎉 Phase 3-1 完成！下一步请学习 3_2_create_agent_demo.py\n')