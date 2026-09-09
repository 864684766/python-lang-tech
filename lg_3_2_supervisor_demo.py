"""
LangGraph Phase 3-2: Supervisor Pattern 主管-员工模式 Demo

学习目标:
  · 主管节点拆解任务 → 条件边分发 → 专员节点 → 汇总

运行: python lg_3_2_supervisor_demo.py
"""
from typing import Annotated, Literal, TypedDict
import operator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from model_provider import get_model
from demo_explain import print_lesson, print_step, print_tip
load_dotenv()

class ReportState(TypedDict):
    task: str
    route: str
    data_section: str
    format_section: str
    final_report: Annotated[list[str], operator.add]

def supervisor_node(state: ReportState) -> dict:
    """主管只做路由，不写正文。看任务关键词决定派哪些专员。"""
    task = state['task'].lower()
    needs_data = any((k in task for k in ['数据', '指标', '周报', 'report']))
    needs_format = any((k in task for k in ['格式', '排版', '周报', 'report']))
    if needs_data and needs_format:
        route = 'both'
    elif needs_data:
        route = 'data_only'
    else:
        route = 'format_only'
    print(f'  👔 主管路由: {route}')
    return {'route': route}


def data_worker(state: ReportState) -> dict:
    """数据专员：只填 data_section。"""
    llm = get_model()
    text = llm.invoke([HumanMessage(content=f"提取任务中的数据要点（模拟）: {state['task']}")]).content
    return {'data_section': text}


def format_worker(state: ReportState) -> dict:
    """排版专员：读 data_section，写成 Markdown 段落。"""
    llm = get_model()
    text = llm.invoke([HumanMessage(content=f"将以下内容格式化为 Markdown 周报段落（100字内）:\n数据: {state.get('data_section', '无')}")]).content
    return {'format_section': text}


def merge_node(state: ReportState) -> dict:
    """汇总：把各专员产出拼成一份报告。final_report 用 operator.add 追加。"""
    report = f"# 周报\n\n## 数据\n{state.get('data_section', '')}\n\n## 正文\n{state.get('format_section', '')}"
    return {'final_report': [report]}


def route_supervisor(state: ReportState) -> Literal['data_worker', 'format_worker', 'both']:
    """条件边映射：both 先进入 both_entry，再串行 data → format。"""
    r = state['route']
    if r == 'both':
        return 'both'
    if r == 'data_only':
        return 'data_worker'
    return 'format_worker'


def both_entry(state: ReportState) -> dict:
    """空节点：只为了让「两条专员都要」时有一个固定入口，再接到 data_worker。"""
    return {}


def build_graph():
    g = StateGraph(ReportState)
    g.add_node('supervisor', supervisor_node)
    g.add_node('data_worker', data_worker)
    g.add_node('format_worker', format_worker)
    g.add_node('both_entry', both_entry)
    g.add_node('merge', merge_node)
    g.add_edge(START, 'supervisor')
    g.add_conditional_edges(
        'supervisor',
        route_supervisor,
        {'data_worker': 'data_worker', 'format_worker': 'format_worker', 'both': 'both_entry'},
    )
    g.add_edge('both_entry', 'data_worker')
    g.add_edge('data_worker', 'format_worker')
    g.add_edge('format_worker', 'merge')
    g.add_edge('merge', END)
    return g.compile()
if __name__ == '__main__':
    print_lesson(__file__)
    print('=' * 65)
    print('LangGraph 3-2: Supervisor 主管-员工模式')
    print('=' * 65)
    app = build_graph()
    print_step(1, "主管拆任务", "含「周报」→ route=both → data_worker → format_worker → merge")
    r = app.invoke({'task': '写一份包含数据和排版的周报', 'route': '', 'data_section': '', 'format_section': '', 'final_report': []})
    print(f"\n📄 最终周报:\n{r['final_report'][-1][:300]}...")
    print_tip("主管只做路由，专员只做自己那一段，merge 负责拼装。")