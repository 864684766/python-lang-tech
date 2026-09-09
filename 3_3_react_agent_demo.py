"""
========================================================
LangChain Phase 3-3
Agent 执行流原理（ReAct）教材级 Demo
========================================================

📚 学习目标：
1. 理解 ReAct（Reasoning + Acting）循环的完整执行流程
2. 用 stream 模式观察 Agent 每一步的状态变化
3. 掌握 recursion_limit 防止无限循环
4. 理解 Agent 的「步（step）」概念

🔗 ReAct 执行流：
┌─────────────────────────────────────────────────────────────────┐
│  User Input                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────┐    有 tool_calls?    ┌──────────┐                 │
│  │ LLM思考 │ ────── 是 ─────────▶ │ 执行Tool │                 │
│  └─────────┘                      └──────────┘                 │
│      │                                 │                        │
│      │ 否                              │ ToolMessage            │
│      ▼                                 ▼                        │
│  最终答案 ◀────────────────────── 回到 LLM 再次思考             │
└─────────────────────────────────────────────────────────────────┘

运行前提：
  pip install langchain langchain-core langchain-community langgraph zhipuai python-dotenv
  设置环境变量: ZHIPUAI_API_KEY=你的key
"""
import os
from numbers import Number
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

@tool
def lookup(term: str) -> str:
    """在知识库中查找术语定义"""
    db = {'react': 'ReAct = Reasoning（推理）+ Acting（行动），Agent 交替思考与调用工具。', 'agent': 'Agent 是 LLM 在循环中调用工具直到任务完成的系统。'}
    return db.get(term.lower(), f'未找到「{term}」')

@tool
def add_numbers(a: float, b: float) -> str:
    """将两个数字相加"""
    return f'{a} + {b} = {a + b}'
DEMO_TOOLS = [lookup, add_numbers]

def demo_1_react_concept():
    """纯概念讲解，不调用 API"""
    print('\n' + '=' * 65)
    print('Demo 1: ReAct 执行流原理')
    print('=' * 65)
    print('\n    ReAct 论文核心思想：让 LLM 不仅「说」，还要「做」。\n\n    每一步（step）Agent 图执行一次「model → tools? → model」：\n\n    Step 1: LLM 收到用户问题 "ReAct 是什么？另外 3+5=?"\n            → 决定调用 lookup("react") 和 add_numbers(3,5)\n\n    Step 2: 工具返回结果 → 作为 ToolMessage 追加到对话\n            → LLM 再次阅读全部上下文，组织最终回答\n\n    Step 3: LLM 不再发起 tool_calls → 输出最终答案 → END\n\n    关键术语：\n      · step（步）     = LangGraph 图的一次节点流转\n      · iteration      = 一次完整的「思考→行动」循环\n      · recursion_limit = 最多允许多少步，防止死循环\n    ')

def demo_2_stream_updates():
    """
    create_agent 返回的是 LangGraph CompiledStateGraph。
    使用 stream(..., stream_mode="updates") 可以看到每个节点执行后的增量更新。

    典型节点名：
      - "model" 或类似：LLM 推理节点
      - "tools"：工具执行节点
    """
    print('\n' + '=' * 65)
    print('Demo 2: stream(updates) 观察 ReAct 每一步')
    print('=' * 65)
    agent = create_agent(get_model(), DEMO_TOOLS, system_prompt='你是教学助手，回答前先查知识库或计算。')
    question = 'ReAct 是什么？另外帮我算 3+5'
    print(f'\n问题: {question}\n')
    for chunk in agent.stream({'messages': [HumanMessage(content=question)]}, stream_mode='messages'):
        print(chunk[0].content)
    print("💡 stream_mode='updates' 适合调试 Agent 决策过程")

def demo_3_stream_values():
    """
    stream_mode="values" 每步返回完整 state（含全部 messages）。
    与 updates 的区别：values 看全貌，updates 看增量。
    """
    print('\n' + '=' * 65)
    print('Demo 3: stream(values) 完整状态快照')
    print('=' * 65)
    agent = create_agent(get_model(), DEMO_TOOLS)
    question = 'agent 是什么？'
    print(f'问题: {question}\n')
    for i, state in enumerate(agent.stream({'messages': [HumanMessage(content=question)]}, stream_mode='values'), 1):
        msg_count = len(state.get('messages', []))
        print(f'  快照 [{i}]: messages 列表长度 = {msg_count}')
    print('\n💡 values 模式：每步看到累积后的完整对话历史')

def demo_4_recursion_limit():
    """
    LangGraph 用 recursion_limit 限制最大步数（默认 25）。
    超出会抛出 GraphRecursionError。

    在 Agent 场景下，这等价于路线图中的 max_iterations 防护。
    """
    print('\n' + '=' * 65)
    print('Demo 4: recursion_limit 防护')
    print('=' * 65)
    agent = create_agent(get_model(), DEMO_TOOLS)
    low_limit = 2
    print(f'设置 recursion_limit={low_limit}（故意设很低以演示保护机制）\n')
    try:
        agent.invoke({'messages': [HumanMessage(content='查 react 定义并计算 1+2+3+4+5')]}, config={'recursion_limit': low_limit})
        print('未触发限制（任务可能在限制内完成了）')
    except GraphRecursionError as exc:
        print(f'⚠️  捕获 GraphRecursionError（预期行为）:')
        print(f'   {str(exc)[:200]}...')
    print('\n💡 生产环境建议:')
    print('   · 正常任务: recursion_limit=25~50')
    print('   · 复杂多步任务: 适当提高，同时监控 Token 消耗')
    print("   · 配置方式: agent.invoke(input, config={'recursion_limit': N})")

def demo_5_count_iterations():
    """分析 messages 列表，统计 LLM 发起了几次工具调用"""
    print('\n' + '=' * 65)
    print('Demo 5: 统计 ReAct 轮次')
    print('=' * 65)
    agent = create_agent(get_model(), DEMO_TOOLS)
    question = '解释 agent 并计算 10+20'
    result = agent.invoke({'messages': [HumanMessage(content=question)]})
    messages = result['messages']
    ai_with_tools = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    print(f'问题: {question}')
    print(f'📊 消息总数: {len(messages)}')
    print(f'📊 LLM 发起工具调用轮次: {len(ai_with_tools)}')
    print(f'📊 工具执行次数: {len(tool_results)}')
    print(f'📊 最终回答: {messages[-1].content[:150]}...')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 3-3: ReAct Agent 执行流原理           ║\n║     版本: LangChain 1.3.14                               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('ReAct 概念', '想→调工具→观察→再想'),
        ('stream values', '边跑边看完整 state'),
        ('recursion_limit', '防止工具循环打转'),
        ('统计迭代次数', '一轮 ReAct 走了几跳'),
    ])
    demo_1_react_concept()
    check_api_key()
    try:
        demo_3_stream_values()
        demo_4_recursion_limit()
        demo_5_count_iterations()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 3-3 完成！下一步请学习 3_4_function_calling_demo.py\n')