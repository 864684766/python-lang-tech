"""
========================================================
LangChain Phase 3-4
Function Calling（原生工具调用）教材级 Demo
========================================================

📚 学习目标：
1. 理解 Function Calling 与传统 ReAct 文本格式的区别
2. 掌握 bind_tools 如何将工具转为 OpenAI Function 格式
3. 学会 inspect AIMessage.tool_calls 结构
4. 了解智谱 GLM-4 的原生工具调用能力（LangChain 1.3.14）

🔗 Function Calling 数据流：
┌──────────────────────────────────────────────────────────────┐
│  Python @tool 函数                                            │
│       │ convert_to_openai_tool()                             │
│       ▼                                                       │
│  OpenAI Function Schema（JSON）                               │
│       │ 随 API 请求发送给模型                                  │
│       ▼                                                       │
│  模型返回 AIMessage.tool_calls = [{name, args, id}, ...]      │
│       │ Agent / 手动循环执行                                   │
│       ▼                                                       │
│  ToolMessage(content=结果, tool_call_id=id) 回传模型           │
└──────────────────────────────────────────────────────────────┘

与传统 ReAct 文本格式对比：
  · Function Calling：结构化 JSON，Token 更少，解析更可靠
  · 文本 ReAct：模型输出 Thought/Action/Observation 纯文本（旧方案）

智谱 GLM-4 原生支持 OpenAI 兼容的 tools 参数，LangChain 通过 bind_tools 封装。

运行前提：
  pip install langchain langchain-core langchain-community zhipuai python-dotenv
  设置环境变量: ZHIPUAI_API_KEY=你的key
"""
import json
import os
from numbers import Number
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

@tool
def get_stock_price(symbol: str) -> str:
    """查询股票实时价格（模拟数据）。当用户问股价、股票时使用。

    Args:
        symbol: 股票代码，如 AAPL、TSLA
    """
    prices = {'AAPL': '185.20 USD', 'TSLA': '242.50 USD', 'BABA': '78.30 USD'}
    return prices.get(symbol.upper(), f'暂无 {symbol} 行情')

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """查询两种货币之间的汇率（模拟）。

    Args:
        from_currency: 源货币，如 USD
        to_currency: 目标货币，如 CNY
    """
    rates = {('USD', 'CNY'): 7.25, ('CNY', 'USD'): 0.138, ('EUR', 'CNY'): 7.85}
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate:
        return f'1 {from_currency.upper()} = {rate} {to_currency.upper()}'
    return f'暂无 {from_currency}/{to_currency} 汇率'

@tool
def summarize_text(text: str, max_words: int=50) -> str:
    """将长文本压缩为简短摘要（模拟）。

    Args:
        text: 待摘要文本
        max_words: 摘要最大字数
    """
    words = text[:max_words]
    return f'【摘要】{words}...'
FC_TOOLS = [get_stock_price, get_exchange_rate, summarize_text]
FC_TOOL_MAP = {t.name: t for t in FC_TOOLS}

def demo_1_openai_schema():
    """
    bind_tools 内部调用 convert_to_openai_tool()，
    把 Python 函数转成模型 API 能理解的 JSON Schema。
    """
    print('\n' + '=' * 65)
    print('Demo 1: @tool → OpenAI Function Schema')
    print('=' * 65)
    for t in FC_TOOLS:
        schema = convert_to_openai_tool(t)
        print(f'\n工具: {t.name}')
        print(json.dumps(schema, ensure_ascii=False, indent=2))
    print('\n💡 这就是发给 GLM-4 API 的 tools 字段内容')
    print('   LLM 根据 function.description 和 parameters 决定调用哪个')

def demo_2_inspect_tool_calls():
    """
    单次 model.invoke() 后，检查 AIMessage 上的 tool_calls 字段。
    这是 Function Calling 的核心返回值结构。
    """
    print('\n' + '=' * 65)
    print('Demo 2: 检查 AIMessage.tool_calls 结构')
    print('=' * 65)
    model = get_model().bind_tools(FC_TOOLS, tool_choice='auto')
    question = '苹果(AAPL)现在股价多少？另外 1 美元等于多少人民币？'
    print(f'\n问题: {question}\n')
    response: AIMessage = model.invoke([HumanMessage(content=question)])
    print(f'response 类型: {type(response).__name__}')
    print(f"content（文本部分）: {response.content or '(空，模型选择直接调工具)'}")
    if response.tool_calls:
        print(f'\n🔧 tool_calls 数量: {len(response.tool_calls)}')
        for i, tc in enumerate(response.tool_calls, 1):
            print(f'\n  调用 [{i}]:')
            print(f"    name : {tc['name']}")
            print(f"    args : {json.dumps(tc['args'], ensure_ascii=False)}")
            print(f"    id   : {tc['id']}")
            print(f"    type : {tc.get('type', 'tool_call')}")
    else:
        print('模型未发起工具调用（可能直接文本回答了）')
    print('\n💡 tool_calls 是结构化 dict 列表，无需正则解析 ReAct 文本')

def demo_3_full_roundtrip():
    """
    演示 Function Calling 的完整一轮：
      HumanMessage → AIMessage(tool_calls) → 执行 → ToolMessage → AIMessage(最终答案)
    """
    print('\n' + '=' * 65)
    print('Demo 3: Function Calling 完整往返')
    print('=' * 65)
    model = get_model().bind_tools(FC_TOOLS)
    messages = [HumanMessage(content='TSLA 股价是多少？')]
    ai_msg: AIMessage = model.invoke(messages)
    messages.append(ai_msg)
    print('第一轮 — 模型 tool_calls:')
    for tc in ai_msg.tool_calls:
        print(f"  → {tc['name']}({tc['args']})")
        result = FC_TOOL_MAP[tc['name']].invoke(tc['args'])
        messages.append(ToolMessage(content=result, tool_call_id=tc['id']))
        print(f'  ← 工具返回: {result}')
    final: AIMessage = model.invoke(messages)
    print(f'\n第二轮 — 最终回答:\n  {final.content}')
    print('\n💡 create_agent 就是把上述循环自动化了')

def demo_4_parallel_tool_calls():
    """
    现代模型（含 GLM-4）一次响应可返回多个 tool_calls，
    实现并行工具调用，比串行 ReAct 更高效。
    """
    print('\n' + '=' * 65)
    print('Demo 4: 一次响应中的多工具调用')
    print('=' * 65)
    model = get_model().bind_tools(FC_TOOLS)
    response: AIMessage = model.invoke([HumanMessage(content='查 AAPL 股价、USD 兑 CNY 汇率，并摘要：LangChain 是 LLM 应用框架。')])
    if response.tool_calls:
        print(f'模型一次发起了 {len(response.tool_calls)} 个工具调用:')
        for tc in response.tool_calls:
            print(f"  · {tc['name']} ← {tc['args']}")
    else:
        print('本次未触发多工具调用（取决于模型判断）')
    print('\n💡 Function Calling 优势：结构化、可并行、Token 效率高')

def demo_5_tool_choice():
    """
    ChatZhipuAI.bind_tools 的 tool_choice 目前仅支持 "auto"（LangChain 1.3.14）。
    了解此限制，避免使用 OpenAI 文档中其他 tool_choice 值导致报错。
    """
    print('\n' + '=' * 65)
    print('Demo 5: tool_choice 与智谱 GLM 兼容性')
    print('=' * 65)
    model = get_model()
    bound = model.bind_tools(FC_TOOLS, tool_choice='auto')
    print("✅ tool_choice='auto' — 智谱 GLM-4 支持（模型自主决策）")
    print('\n⚠️  智谱 ChatZhipuAI 当前限制（langchain-community 源码）:')
    print('   · glm-4v 不支持 tool calling')
    print("   · tool_choice 仅支持 'auto'，其他值会 ValueError")
    print('   · 如需强制调用特定工具，可在 system_prompt 中约束行为')
    resp = bound.invoke([HumanMessage(content='BABA 股价')])
    print(f'\n验证 invoke 成功，tool_calls: {len(resp.tool_calls or [])} 个')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 3-4: Function Calling 原生工具调用    ║\n║     版本: LangChain 1.3.14                               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('OpenAI schema', '工具如何变成 JSON schema'),
        ('inspect tool_calls', 'name / args / id'),
        ('完整 roundtrip', '调工具 → ToolMessage 回灌'),
        ('并行 tool_calls', '一轮里多个调用'),
        ('tool_choice', '强制或不强制调用'),
    ])
    demo_1_openai_schema()
    check_api_key()
    try:
        demo_2_inspect_tool_calls()
        demo_3_full_roundtrip()
        demo_4_parallel_tool_calls()
        demo_5_tool_choice()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 3-4 完成！下一步请学习 3_5_tool_error_handling_demo.py\n')