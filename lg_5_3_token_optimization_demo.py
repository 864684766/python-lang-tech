"""
LangGraph Phase 5-3: Token 精简 —— 少喂字，少花钱

这个例子只讲一件事：

    LLM 按「送进去的字数」计费。
    图有几个节点，就可能调几次模型。
    如果每次都把整篇长文档塞进 prompt，同一份原文会被付费很多次。

本文件用两种办法把后面几次调用变短：

  办法 1（图节点）: 先摘要，后面的节点 prompt 只带摘要，不再带原文
  办法 2（对话）  : trim_messages 只留最近几条，旧聊天丢掉

注意：LangGraph 的 State 里 raw_document 字段还在，并没有删掉。
省的不是「图里存不存原文」，而是「第二次 invoke 时 prompt 里有没有原文」。

运行: python lg_5_3_token_optimization_demo.py
"""
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, StateGraph

from demo_explain import print_lesson, print_step, print_tip
from model_provider import get_model

load_dotenv()


class DocState(TypedDict):
    raw_document: str  # 原文一直留在 State 里，方便对照；但不该再塞进后续 prompt
    summary: str
    answer: str


def _chars(text: str) -> int:
    return len(text)


def summarize_node(state: DocState) -> dict:
    """第一次调用：必须看原文，这一次的 token 省不掉。"""
    doc = state["raw_document"]
    prompt = f"用不超过 50 字总结下面文档的主要内容，不要重复原文：\n{doc}"
    print(f"  [summarize] 本次 prompt 约 {_chars(prompt)} 字（含原文，这次必须花）")
    summary = get_model().invoke([HumanMessage(content=prompt)]).content
    print(f"  [summarize] 得到摘要 {_chars(summary)} 字: {summary[:80]}")
    return {"summary": summary}


def answer_node(state: DocState) -> dict:
    """第二次调用：故意只把 summary 放进 prompt，不放 raw_document。"""
    prompt = (
        "根据摘要回答：这份文档的主要内容是什么？只答一两句。\n"
        f"摘要：{state['summary']}"
    )
    wasted = (
        "根据摘要回答：这份文档的主要内容是什么？只答一两句。\n"
        f"原文：{state['raw_document']}"
    )
    print(f"  [answer]    本次 prompt 约 {_chars(prompt)} 字（只带摘要）")
    print(f"  [对照]      若把原文再塞一遍，prompt 会变成约 {_chars(wasted)} 字")
    print(f"  [对照]      第二次调用少喂了约 {_chars(wasted) - _chars(prompt)} 字")
    answer = get_model().invoke([HumanMessage(content=prompt)]).content
    return {"answer": answer}


def demo_1_summarize_then_answer() -> None:
    print_step(
        1,
        "先摘要，再回答",
        "两个节点都会调 LLM。省钱发生在第 2 次：prompt 不再带原文。",
    )
    g = StateGraph(DocState)
    g.add_node("summarize", summarize_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "summarize")
    g.add_edge("summarize", "answer")
    g.add_edge("answer", END)
    app = g.compile()

    # 重复句子只是为了凑出「看起来很长」的原文，方便对比字数
    long_doc = "LangGraph 用状态图编排多步 Agent。" * 40
    print(f"  原文长度: {_chars(long_doc)} 字")
    print_tip("State 里仍然有 raw_document；answer 节点只是没用它去拼 prompt。")

    result = app.invoke({"raw_document": long_doc, "summary": "", "answer": ""})
    print(f"\n  最终回答: {result['answer']}")


def demo_2_trim_chat_history() -> None:
    print_step(
        2,
        "对话太长就截断",
        "和文档摘要是另一条线：聊天 12 条全塞进模型，不如只留最近几条。",
    )
    history = [
        SystemMessage(content="你是助手"),
        *[HumanMessage(content=f"用户第 {i} 句") for i in range(1, 11)],
        AIMessage(content="上一轮回复"),
    ]
    print("  截断前:")
    for msg in history:
        print(f"    - {type(msg).__name__}: {msg.content}")

    # token_counter=len 表示「一条消息算 1」，所以 max_tokens=4 就是最多留 4 条
    # 生产环境应换成真实 tokenizer，按真正的 token 数截
    trimmed = trim_messages(
        history,
        max_tokens=4,
        token_counter=len,
        strategy="last",
    )
    print(f"\n  截断后 ({len(history)} 条 → {len(trimmed)} 条，strategy='last' 留尾巴):")
    for msg in trimmed:
        print(f"    - {type(msg).__name__}: {msg.content}")
    print_tip("旧消息被丢掉了，模型不再为第 1～N 句付费；代价是它会「忘记」更早的内容。")


if __name__ == "__main__":
    print_lesson(__file__)
    print("=" * 65)
    print("这个例子要说明：少把字送进 LLM，就能少花钱、少排队。")
    print("=" * 65)
    demo_1_summarize_then_answer()
    demo_2_trim_chat_history()
    print()
    print_tip("第三种常见做法（本文件不跑模型）：小模型做分类/路由，大模型只处理难问题。")
