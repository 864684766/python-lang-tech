"""
LangGraph Phase 3-1: 子图嵌套 Subgraphs 教材级 Demo

故事线（数据必须一路传下去）:
  task「是否存在外星人」
    → 调研子图: search 收集与 topic 有关的笔记 → extract 压成 summary
    → 父图 research_result = summary
    → 写作子图: draft 根据 research_summary 列大纲 → polish 写成短文
    → 父图 article

运行: python lg_3_1_subgraphs_demo.py

Studio:
  langgraph.json → graph。Input 只填 task，其余留空。
  Trace 里可展开 research 子图的 search / extract。
"""
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from demo_explain import print_lesson, print_step, print_tip
from model_provider import get_model

load_dotenv()


# ── 调研子图 State：只关心「题目 → 笔记 → 摘要」──
class ResearchState(TypedDict):
    topic: str
    notes: str
    summary: str


# ── 写作子图 State：必须吃到上一步的 research_summary ──
class WriteState(TypedDict):
    topic: str
    research_summary: str
    outline: str
    article: str


# ── 父图 State：对外只暴露 task / research_result / article ──
class MainState(TypedDict):
    task: str
    research_result: str
    article: str


def search_node(state: ResearchState) -> dict:
    """模拟检索：笔记内容必须和 topic 一致，不能写死别的主题。"""
    topic = state["topic"]
    notes = (
        f"【检索词】{topic}\n"
        f"【资料A】关于「{topic}」，常见讨论点包括：是否有可观测证据、"
        f"科学界的共识与争议、以及公众为何关心这个话题。\n"
        f"【资料B】目前没有单一结论能回答「{topic}」；"
        f"不同学科会从各自角度给出阶段性判断。"
    )
    print(f"  [search] topic={topic!r} → notes {len(notes)} 字")
    return {"notes": notes}


def extract_node(state: ResearchState) -> dict:
    """用 LLM 把笔记压成一句调研结论，供写作子图使用。"""
    topic = state["topic"]
    llm = get_model()
    summary = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"下面是与「{topic}」有关的调研笔记。请用 2~3 句话写调研结论，"
                    f"只谈这个主题，不要引入 LangGraph 等无关技术名词。\n\n"
                    f"{state['notes']}"
                )
            )
        ]
    ).content
    print(f"  [extract] summary: {summary[:80]}...")
    return {"summary": summary}


def build_research_subgraph():
    g = StateGraph(ResearchState)
    g.add_node("search", search_node)
    g.add_node("extract", extract_node)
    g.add_edge(START, "search")
    g.add_edge("search", "extract")
    g.add_edge("extract", END)
    return g.compile(name="research_subgraph")


def draft_node(state: WriteState) -> dict:
    """大纲必须引用 research_summary，不能只看 topic 瞎写。"""
    topic = state["topic"]
    summary = state["research_summary"]
    outline = (
        f"标题: {topic}\n"
        f"依据调研: {summary}\n"
        f"结构: 1. 为什么问这个问题  2. 调研要点  3. 一句结论"
    )
    print(f"  [draft] 大纲已绑定调研摘要（{len(summary)} 字）")
    return {"outline": outline}


def polish_node(state: WriteState) -> dict:
    """成文时同时看 topic、调研摘要、大纲，避免和 research 脱节。"""
    topic = state["topic"]
    llm = get_model()
    article = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"请写一篇约 120 字的短文。\n"
                    f"主题必须是「{topic}」。\n"
                    f"必须基于下面调研摘要展开，不要跑题：\n{state['research_summary']}\n\n"
                    f"可按此大纲组织：\n{state['outline']}"
                )
            )
        ]
    ).content
    print(f"  [polish] article: {article[:80]}...")
    return {"article": article}


def build_write_subgraph():
    g = StateGraph(WriteState)
    g.add_node("draft", draft_node)
    g.add_node("polish", polish_node)
    g.add_edge(START, "draft")
    g.add_edge("draft", "polish")
    g.add_edge("polish", END)
    return g.compile(name="write_subgraph")


research_subgraph = build_research_subgraph()
write_subgraph = build_write_subgraph()


def run_research(state: MainState) -> dict:
    """父图字段翻译: task → 子图 topic；子图 summary → research_result。"""
    topic = state["task"]
    print(f"\n  📦 调研子图  主图 task={topic!r} → 子图 topic")
    result = research_subgraph.invoke({"topic": topic, "notes": "", "summary": ""})
    summary = result["summary"]
    print(f"  📦 调研子图结束 → 主图 research_result")
    return {"research_result": summary}


def run_write(state: MainState) -> dict:
    """写作子图必须传入 research_result，否则 article 会和调研对不上。"""
    topic = state["task"]
    summary = state["research_result"]
    if not summary.strip():
        raise ValueError("research_result 为空，请先跑完 research 节点")
    print(f"\n  📦 写作子图  topic={topic!r} + research_summary({len(summary)}字)")
    result = write_subgraph.invoke(
        {
            "topic": topic,
            "research_summary": summary,
            "outline": "",
            "article": "",
        }
    )
    print(f"  📦 写作子图结束 → 主图 article")
    return {"article": result["article"]}


def build_main_graph():
    g = StateGraph(MainState)
    g.add_node("research", run_research)
    g.add_node("write", run_write)
    g.add_edge(START, "research")
    g.add_edge("research", "write")
    g.add_edge("write", END)
    return g.compile(name="subgraphs")


graph = build_main_graph()


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("LangGraph 3-1: 子图嵌套 — task 一路传到 article")
    print("=" * 65)
    print_step(1, "填 task", "整条链只认这一个输入；后面字段由节点写入")
    task = "是否存在外星人"
    print(f"  task = {task!r}\n")
    result = graph.invoke({"task": task, "research_result": "", "article": ""})
    print("\n── 最终 State ──")
    print(f"  research_result: {result['research_result']}")
    print(f"  article: {result['article']}")
    print_tip("Studio Trace 里展开 research，应看到 search/extract 都围绕同一 topic。")
