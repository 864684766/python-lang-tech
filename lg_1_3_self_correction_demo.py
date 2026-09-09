"""
LangGraph Phase 1-3: 循环与自我修正 (Cycles) 教材级 Demo

学习目标:
  · 合法回环: 生成 → 试运行 → 报错 → 修正 → 再试
  · recursion_limit 防止死循环

运行: python lg_1_3_self_correction_demo.py
"""
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()
FORBIDDEN = {'DROP', 'DELETE', 'INSERT', 'UPDATE'}


class SQLState(TypedDict):
    question: str
    sql: str
    error: str
    # operator.add：每次节点 return attempts=1，都会累加，而不是覆盖
    attempts: Annotated[int, operator.add]
    success: bool


def generate_sql(state: SQLState) -> dict:
    """根据问题（以及上次 error）生成 SQL。attempts 只返回 1，由 reducer 累加。"""
    llm = get_model()
    feedback = f"\n上次错误: {state['error']}" if state.get('error') else ''
    prompt = f"根据问题生成 SQLite 查询语句，只输出 SQL，不要解释。\n问题: {state['question']}{feedback}"
    sql = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    sql = sql.replace('```sql', '').replace('```', '').strip()
    print(f'  📝 生成 SQL (attempt+1): {sql[:60]}...')
    return {'sql': sql, 'attempts': 1}


def run_sql(state: SQLState):
    """试运行：命中禁词或不是 SELECT → Command 跳回 generate_sql，并把 error 写进 State。"""
    sql_upper = state['sql'].upper()
    if any((kw in sql_upper for kw in FORBIDDEN)):
        return Command(goto='generate_sql', update={'error': '不允许写操作，请只用 SELECT', 'success': False})
    if 'SELECT' not in sql_upper:
        return Command(goto='generate_sql', update={'error': '必须是 SELECT 语句', 'success': False})
    print('  ✅ SQL 试运行通过')
    return Command(goto=END, update={'error': '', 'success': True})


def build_graph():
    g = StateGraph(SQLState)
    g.add_node('generate_sql', generate_sql)
    g.add_node('run_sql', run_sql)
    g.add_edge(START, 'generate_sql')
    g.add_edge('generate_sql', 'run_sql')
    # 回环不画在 add_edge 上：由 run_sql 的 Command(goto='generate_sql') 动态跳回
    return g.compile()


if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 1-3: 自修复循环 + recursion_limit')
    print('=' * 65)
    app = build_graph()
    print_step(1, "生成 → 校验", "失败会带着 error 回到 generate_sql；attempts 用 operator.add 计数")
    try:
        r = app.invoke(
            {'question': '查询所有用户', 'sql': '', 'error': '', 'attempts': 0, 'success': False},
            config={'recursion_limit': 10},
        )
        print(f"\n成功: {r['success']}, 尝试次数: {r['attempts']}, SQL: {r['sql'][:80]}")
        print_tip("recursion_limit 是步数上限，防止 generate↔run 永远打转。")
    except GraphRecursionError as e:
        print(f'⚠️  recursion_limit 触发: {e}')
        print_tip("把 recursion_limit 调小就能稳定复现 GraphRecursionError。")