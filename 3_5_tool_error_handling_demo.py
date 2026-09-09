"""
========================================================
LangChain Phase 3-5
Tool 错误处理与降级 教材级 Demo
========================================================

📚 学习目标：
1. 理解工具执行失败时 Agent 应如何优雅降级
2. 掌握 ToolException + handle_tool_error=True 可恢复错误
3. 掌握 ToolErrorMiddleware / ToolRetryMiddleware（LangChain 1.3.14）
4. 实现主备工具 fallback 模式

🔗 错误处理层次：
┌──────────────────────────────────────────────────────────────┐
│  层次 1: 工具内部 try/except → 返回友好字符串                  │
│  层次 2: raise ToolException + handle_tool_error=True → error ToolMessage │
│  层次 3: ToolRetryMiddleware → 自动重试 transient 错误       │
│  层次 4: ToolErrorMiddleware → 捕获异常，反馈给 LLM 重试       │
│  层次 5: 主备工具 fallback → 主搜索失败换备用数据源             │
└──────────────────────────────────────────────────────────────┘

运行前提：
  pip install langchain langchain-core langchain-community langgraph zhipuai python-dotenv
  设置环境变量: ZHIPUAI_API_KEY=你的key
"""
import os
from numbers import Number
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware, ToolRetryMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException, tool

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()
_PRIMARY_SEARCH_FAIL = True

@tool
def safe_calculator(expression: str) -> str:
    """安全计算器。遇到非法表达式时返回错误说明而非崩溃。

    Args:
        expression: 数学表达式
    """
    allowed = set('0123456789+-*/(). ')
    if not all((c in allowed for c in expression)):
        return '错误：表达式包含非法字符，仅支持数字和 +-*/()'
    try:
        result = eval(expression, {'__builtins__': {}}, {})
        return f'{expression} = {result}'
    except Exception as exc:
        return f'计算错误（已捕获）: {exc}'

def demo_1_graceful_return():
    """
    最简单策略：工具内部捕获异常，返回描述性字符串。
    LLM 看到 ToolMessage 中的错误说明，可调整策略或告知用户。
    """
    print('\n' + '=' * 65)
    print('Demo 1: 工具内部优雅返回（try/except → 字符串）')
    print('=' * 65)
    cases = ['(2+3)*4', 'import os', '1/0']
    for expr in cases:
        result = safe_calculator.invoke({'expression': expr})
        print(f'  输入: {expr!r}  →  {result}')
    print('\n💡 优点: 简单、Agent 不会崩溃')
    print('   缺点: 无法区分「业务错误」和「系统错误」，.retry .middleware 看不到异常')

def _make_strict_divide_tool():
    """
    创建带错误处理的除法工具。

    LangChain 1.3.14 关键细节：
      · @tool 生成的 BaseTool 默认 handle_tool_error=False
      · 此时 raise ToolException 会在 ToolNode 层直接抛出，Agent 整图崩溃
      · 必须设置 handle_tool_error=True，ToolException 才会转为
        status='error' 的 ToolMessage，LLM 才能读取并继续回答

    替代方案：不改工具，改用 ToolErrorMiddleware（见 Demo 4）
    """

    @tool
    def strict_divide(a: float, b: float) -> str:
        """两数相除。除数为 0 时抛出 ToolException，由 Agent 框架处理。

        Args:
            a: 被除数
            b: 除数
        """
        if b == 0:
            raise ToolException('除数不能为 0，请提供非零除数后重试')
        return f'{a} / {b} = {a / b}'
    strict_divide.handle_tool_error = True
    return strict_divide

def demo_2_tool_exception():
    """ToolException + handle_tool_error=True → status='error' 的 ToolMessage"""
    print('\n' + '=' * 65)
    print('Demo 2: ToolException 可恢复错误')
    print('=' * 65)
    strict_divide = _make_strict_divide_tool()
    agent = create_agent(get_model(), [strict_divide])
    result = agent.invoke({'messages': [HumanMessage(content='请计算 10 除以 0')]})
    for msg in result['messages']:
        if hasattr(msg, 'status') and msg.status == 'error':
            print(f'  ⚠️  error ToolMessage: {msg.content}')
    print(f"\n  最终回答: {result['messages'][-1].content}")
    print('\n💡 要点:')
    print('   · 仅 raise ToolException 不够，还需 handle_tool_error=True')
    print('   · 否则 create_agent 的 ToolNode 不会捕获，整图 invoke 失败')
    print('   · 或用 ToolErrorMiddleware 在 Agent 层统一捕获（Demo 4）')
_call_count = 0

@tool
def flaky_api(query: str) -> str:
    """模拟不稳定的远程 API：前两次调用失败，第三次成功。

    Args:
        query: 查询关键词
    """
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError(f'模拟网络超时（第 {_call_count} 次）')
    return f'【API 成功】关于「{query}」的数据'

def demo_3_tool_retry():
    """ToolRetryMiddleware 在工具抛异常时自动重试"""
    print('\n' + '=' * 65)
    print('Demo 3: ToolRetryMiddleware 自动重试')
    print('=' * 65)
    global _call_count
    _call_count = 0
    retry_middleware = ToolRetryMiddleware(max_retries=3, retry_on=(ConnectionError,), backoff_factor=0.1, initial_delay=0.1)
    agent = create_agent(get_model(), [flaky_api], middleware=[retry_middleware])
    result = agent.invoke({'messages': [HumanMessage(content='查询 langchain 相关信息')]})
    print(f'  工具实际被调用次数: {_call_count}')
    print(f"  最终回答: {result['messages'][-1].content}")
    print('\n💡 适合 transient 错误（网络抖动、限流），不要对逻辑错误盲目重试')

@tool
def sensitive_operation(user_id: str) -> str:
    """执行敏感操作（模拟权限校验失败）。

    Args:
        user_id: 用户 ID
    """
    if user_id != 'admin':
        raise PermissionError('权限不足：仅 admin 可执行此操作')
    return '操作成功'

def demo_4_tool_error_middleware():
    """
    ToolErrorMiddleware.on_error 捕获工具执行异常，
    可选择转为 ToolMessage 反馈给 LLM，或 re-raise 终止运行。
    """
    print('\n' + '=' * 65)
    print('Demo 4: ToolErrorMiddleware 自定义错误处理')
    print('=' * 65)

    def on_error(exc: Exception, request) -> str | None:
        tool_name = request.tool_call['name']
        if isinstance(exc, PermissionError):
            return f'工具 `{tool_name}` 权限被拒绝：请使用 admin 账户或联系管理员'
        return None
    agent = create_agent(get_model(), [sensitive_operation], middleware=[ToolErrorMiddleware(on_error=on_error)])
    result = agent.invoke({'messages': [HumanMessage(content='用 user_id=guest 执行敏感操作')]})
    print(f"  最终回答: {result['messages'][-1].content}")
    print('\n💡 on_error 返回 str → 转为 error ToolMessage；返回 None → 异常继续传播')

@tool
def primary_web_search(query: str) -> str:
    """主搜索服务（模拟：可能不可用）。优先使用此工具获取最新信息。

    Args:
        query: 搜索词
    """
    global _PRIMARY_SEARCH_FAIL
    if _PRIMARY_SEARCH_FAIL:
        raise ConnectionError('主搜索服务暂时不可用 (503)')
    return f'【主搜索】{query} 的最新结果'

@tool
def backup_knowledge_search(query: str) -> str:
    """备用本地知识库搜索。当主搜索失败时使用。

    Args:
        query: 搜索词
    """
    local_kb = {'python': 'Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建...', 'agent': 'Agent 是 LLM 循环调用工具的系统，核心组件包括...', 'langchain': 'LangChain 是一个用于构建 LLM 应用的开发框架，核心功能包括链、工具、记忆...'}
    for key, val in local_kb.items():
        if key in query.lower():
            return f'【备用知识库】{val}'
    return f'【备用知识库】未找到「{query}」，但服务可用'

def demo_5_fallback_tools():
    """
    主备工具策略（路线图 with_fallbacks 思想在 Tool 层的实现）：

    1. 注册 primary + backup 两个工具
    2. system_prompt 告知 LLM：主搜索失败时改用 backup
    3. ToolErrorMiddleware 把主搜索错误转为可读 ToolMessage，引导 LLM 切换

    注意：LangChain 1.3.14 的 BaseTool 没有 .with_fallbacks()，
    降级逻辑通过「多工具 + prompt + middleware」组合实现。
    """
    print('\n' + '=' * 65)
    print('Demo 5: 主备工具 Fallback（计算器 + 搜索综合场景）')
    print('=' * 65)

    @tool
    def calculator(expression: str) -> str:
        """数学计算工具"""
        try:
            return str(eval(expression, {'__builtins__': {}}, {}))
        except Exception as exc:
            return f'计算错误: {exc}'

    def on_search_error(exc: Exception, request) -> str | None:
        if isinstance(exc, ConnectionError):
            return f'主搜索服务失败 ({type(exc).__name__})，请改用 backup_knowledge_search 工具查询本地知识库'
        return None
    agent = create_agent(get_model(), [calculator, primary_web_search, backup_knowledge_search], system_prompt='你是助手。\n1. 数学问题 → calculator\n2. 知识查询 → 先试 primary_web_search\n3. 若主搜索报错 → 立即改用 backup_knowledge_search\n4. 回答简洁\n5. 不要重复调用同一个工具', middleware=[ToolErrorMiddleware(on_error=on_search_error)])
    scenarios = ['计算 (50 + 50) * 2', 'LangChain 是什么？']
    for q in scenarios:
        print(f"\n{'─' * 50}")
        print(f'问题: {q}')
        result = agent.invoke({'messages': [HumanMessage(content=q)]})
        for msg in result['messages']:
            t = type(msg).__name__
            if t == 'AIMessage' and getattr(msg, 'tool_calls', None):
                for tc in msg.tool_calls:
                    print(f"  🤖 调用: {tc['name']}")
            elif t == 'ToolMessage':
                print(f'  🔧 工具: {msg.content[:100]}')
        print(f"  ✅ 回答: {result['messages'][-1].content[:150]}")
    print('\n💡 Fallback 模式：主工具失败 → middleware 提示 → LLM 切换备用工具')

def demo_6_runnable_fallback_note():
    """
    路线图提到的 with_fallbacks 适用于 Runnable 链（见 1-1runnable_demo）。
    LangChain 1.3.14 中 Agent 层降级优先用 middleware，而非 Runnable.with_fallbacks。
    """
    print('\n' + '=' * 65)
    print('Demo 6: with_fallbacks 与 Agent 的关系（概念说明）')
    print('=' * 65)
    print('\n    Runnable.with_fallbacks([backup])  — 用于 LCEL 链 / ChatModel 实例\n      例: model.with_fallbacks([backup_model])\n\n    Agent 工具层降级 — 使用 middleware 或多工具策略：\n      · ToolRetryMiddleware   → 同一工具重试\n      · ToolErrorMiddleware   → 异常转 ToolMessage，让 LLM 决策\n      · 主备双工具 + prompt    → LLM 自主切换数据源\n\n    两者不要混用：BaseTool 在 1.3.14 没有 with_fallbacks 方法。\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 3-5: Tool 错误处理与降级              ║\n║     版本: LangChain 1.3.14                               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('主备工具 fallback', '本入口只跑层次 5。同文件还有 ToolException / Retry / Error 中间件可单独调用'),
    ])
    check_api_key()
    try:
        demo_5_fallback_tools()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 3 全部完成！你已掌握 Tools & Agents 核心能力。\n')