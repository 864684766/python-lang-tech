"""
========================================================
LangChain Phase 3-2
create_agent 与 bind_tools 教材级 Demo
========================================================

📚 学习目标：
1. 理解两种 Agent 创建方式的区别与适用场景
2. 掌握 create_agent() 开箱即用工厂（LangChain 1.3.14 推荐）
3. 掌握 bind_tools() 底层绑定 + 手动执行工具调用
4. 构建「计算器 + 联网搜索」双工具 Agent 综合实战

🔗 两种方式对比：
┌────────────────────┬──────────────────────┬─────────────────────────┐
│  方式              │  代码量               │  适用场景                │
├────────────────────┼──────────────────────┼─────────────────────────┤
│  create_agent      │  少（内部 ReAct 循环） │  快速原型、标准 Agent    │
│  bind_tools 手动   │  多（自己写循环）      │  自定义流程、LangGraph   │
└────────────────────┴──────────────────────┴─────────────────────────┘

运行前提：
  pip install langchain langchain-core langchain-community langgraph \\
              zhipuai python-dotenv
  设置环境变量: ZHIPUAI_API_KEY=你的key

版本说明：LangChain 1.3.14 中 create_react_agent 已迁移至 langchain.agents.create_agent。
"""
import json
import os
from numbers import Number
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

@tool
def calculator(expression: str) -> str:
    """执行数学表达式计算。当用户问数学、算术、公式计算时使用。

    Args:
        expression: 数学表达式，如 "(15 + 25) * 3"
    """
    try:
        result = eval(expression, {'__builtins__': {}}, {})
        return f'{expression} = {result}'
    except Exception as exc:
        return f'计算失败: {exc}'

@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。当用户问新闻、版本、概念定义等非计算问题时使用。

    Args:
        query: 搜索关键词
    """
    knowledge = {'kubernetes': 'Kubernetes 是容器编排平台，Pod 是最小调度单元。', 'langchain': 'LangChain 1.3.x 提供 create_agent 作为 Agent 工厂函数。', 'rag': 'RAG（检索增强生成）通过检索外部知识增强 LLM 回答准确性。', 'docker': 'Docker 是容器化平台，镜像和容器是其核心概念。'}
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f'【搜索结果】{value}'
    return f'【搜索结果】未找到「{query}」的模拟数据，请换关键词'
TOOLS = [calculator, web_search]
TOOL_MAP = {t.name: t for t in TOOLS}

def demo_1_create_agent():
    """
    create_agent(model, tools) 是 LangChain 1.3.14 的标准 Agent 工厂。

    内部做了什么（你不需要手写）：
    1. 将 tools 绑定到 model（bind_tools）
    2. 构建 LangGraph 状态图：model 节点 ↔ tools 节点 循环
    3. compile 后返回可 invoke / stream 的 CompiledStateGraph

    调用格式：
        agent.invoke({"messages": [HumanMessage(...)]})
    """
    print('\n' + '=' * 65)
    print('Demo 1: create_agent 开箱即用')
    print('=' * 65)
    agent = create_agent(get_model(), TOOLS, system_prompt='你是智能助手，可计算数学或搜索知识。根据问题类型选择合适的工具。')
    questions = ['帮我算一下 (15 + 25) * 3 等于多少？', '什么是 RAG？请简单介绍。']
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 50}")
        print(f'问题 [{i}]: {question}')
        result = agent.invoke({'messages': [HumanMessage(content=question)]})
        final_msg = result['messages'][-1]
        print(f'\n🤖 最终回答: {final_msg.content}')
        tool_calls_count = sum((1 for msg in result['messages'] if isinstance(msg, AIMessage) and msg.tool_calls))
        print(f'🔧 模型发起工具调用次数: {tool_calls_count}')
    print('\n💡 要点:')
    print('   · create_agent 返回 LangGraph 编译图，直接 invoke 即可')
    print('   · system_prompt 参数设定 Agent 角色，等价于 SystemMessage')

def run_manual_tool_loop(messages: list, max_steps: int=5) -> list:
    """
    手动 ReAct 循环：模拟 create_agent 内部的核心逻辑。

    流程：
      LLM 思考 → 有 tool_calls? → 执行工具 → 结果追加到 messages → 再思考
      无 tool_calls → 结束，返回最终文本

    这就是 LangGraph 里 model 节点 + tools 节点的最小实现。
    """
    model = get_model().bind_tools(TOOLS)
    messages = list(messages)
    for step in range(1, max_steps + 1):
        print(f'\n   --- ReAct 第 {step} 步 ---')
        ai_msg: AIMessage = model.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            print('   ✅ 模型决定不再调用工具，循环结束')
            break
        for tc in ai_msg.tool_calls:
            tool_name = tc['name']
            tool_args = tc['args']
            print(f'   🔧 调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')
            selected = TOOL_MAP.get(tool_name)
            if selected is None:
                observation = f'错误: 未知工具 {tool_name}'
            else:
                observation = selected.invoke(tool_args)
            print(f'   📤 工具返回: {observation[:80]}...')
            messages.append(ToolMessage(content=str(observation), tool_call_id=tc['id']))
    else:
        print('   ⚠️ 达到 max_steps 上限，强制停止')
    return messages

def demo_2_bind_tools_manual():
    """
    bind_tools() 只负责「告诉模型有哪些工具」，不会自动执行工具。

    你必须自己：
    1. 检查 response.tool_calls
    2. 执行对应工具
    3. 把结果包装成 ToolMessage 塞回 messages
    4. 再次 invoke，直到没有 tool_calls

    优点：完全可控，适合嵌入自定义 LangGraph 节点。
    """
    print('\n' + '=' * 65)
    print('Demo 2: bind_tools 手动 ReAct 循环')
    print('=' * 65)
    question = '北京和上海温差如果上海22度，北京25度，差几度？'
    print(f'\n问题: {question}')
    messages = [SystemMessage(content='你是助手，数学问题用 calculator，知识问题用 web_search。'), HumanMessage(content=question)]
    final_messages = run_manual_tool_loop(messages)
    print(f'\n🤖 最终回答: {final_messages[-1].content}')
    print('\n💡 要点:')
    print('   · bind_tools 返回 RunnableBinding，invoke 后检查 .tool_calls')
    print('   · ToolMessage 必须带 tool_call_id，与 AIMessage.tool_calls 对应')
    print('   · 手动循环 = 你在 LangGraph 里自己写 AI 节点的原理')

def demo_3_comparison():
    """并排对比 create_agent 与 bind_tools 手动方式的差异"""
    print('\n' + '=' * 65)
    print('Demo 3: create_agent vs bind_tools 对比')
    print('=' * 65)
    comparison = '\n    ┌──────────────────┬─────────────────────────┬─────────────────────────┐\n    │  维度            │  create_agent           │  bind_tools 手动         │\n    ├──────────────────┼─────────────────────────┼─────────────────────────┤\n    │  代码行数        │  ~5 行                  │  ~30 行（含循环）        │\n    │  工具执行        │  自动（LangGraph 内置）  │  手动 for 循环           │\n    │  流式/持久化     │  支持 stream/checkpoint │  需自己实现              │\n    │  中间件          │  支持 ToolRetry 等      │  需自己写                │\n    │  自定义流程      │  有限（改图需 LangGraph）│  完全自由                │\n    │  学习价值        │  生产首选               │  理解底层原理            │\n    └──────────────────┴─────────────────────────┴─────────────────────────┘\n\n    建议路径：\n      1. 先用 create_agent 快速跑通\n      2. 再用 bind_tools 手动实现一遍，理解 ReAct 循环\n      3. 复杂流程回到 LangGraph 自定义 StateGraph\n    '
    print(comparison)

def demo_4_dual_tool_agent():
    """
    路线图 Phase 3-5 要求的综合场景：
    计算器 + 联网搜索，LLM 自主判断使用哪个工具。
    """
    print('\n' + '=' * 65)
    print('Demo 4: 综合实战 — 计算器 + 搜索 双工具 Agent')
    print('=' * 65)
    agent = create_agent(get_model(temperature=0), TOOLS, system_prompt='你是全能助手。\n- 纯数学计算 → 使用 calculator\n- 概念/知识查询 → 使用 web_search\n- 可以同时使用多个工具\n回答要简洁准确。')
    scenarios = ['256 乘以 3.14 等于多少？', 'LangChain 是什么？用一句话说明。', '先告诉我 Docker 是什么，再算 100 除以 4。']
    for i, q in enumerate(scenarios, 1):
        print(f"\n{'═' * 50}")
        print(f'场景 [{i}]: {q}')
        print('═' * 50)
        result = agent.invoke({'messages': [HumanMessage(content=q)]})
        for msg in result['messages']:
            if isinstance(msg, HumanMessage):
                print(f'👤 用户: {msg.content}')
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"🤖 模型决定调用: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f'🤖 模型回答: {msg.content}')
            elif isinstance(msg, ToolMessage):
                print(f'🔧 工具结果: {msg.content[:100]}')
    print('\n💡 观察: LLM 会根据问题语义自动选择 calculator 或 web_search')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 3-2: create_agent & bind_tools       ║\n║     版本: LangChain 1.3.14                               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('create_agent', '一行得到 ReAct 循环'),
        ('手动 bind_tools', '看 tool_calls 长什么样'),
        ('对比表', '工厂函数 vs 手写'),
        ('双工具 Agent', '模型自己选 calculator 或 search'),
    ])
    check_api_key()
    try:
        demo_1_create_agent()
        demo_2_bind_tools_manual()
        demo_3_comparison()
        demo_4_dual_tool_agent()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 3-2 完成！下一步请学习 3_3_react_agent_demo.py\n')