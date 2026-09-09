#!/usr/bin/env python3
"""LangGraph 核心特性实际运行示例

这个文件演示了 LangGraph 的5个核心功能：
1. stream_mode — 流式输出（values / updates）
2. 持久化 — Checkpointer + thread_id 实现多轮对话记忆
3. Interrupt — 人工审核流程（人机协作）
4. Reducer — 用 Annotated + operator.add 实现状态累加/追加
5. AI节点 — 接入智谱 GLM-4 真实大模型作为图中的一个处理节点

用法:
    python langgraph_demo_examples.py                          # 运行所有示例
    python -c "from langgraph_demo_examples import demo_stream_mode; demo_stream_mode()"  # 单独运行某个示例

前置条件:
    pip install langchain langchain-community zhipuai langgraph
    设置环境变量 ZHIPUAI_API_KEY (仅示例5需要，注意变量名是 ZHIPUAI_API_KEY)
"""
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator
# StateGraph: 有状态的有向图构建器，用来编排节点执行流程
# START:     图的虚拟入口点（固定常量，代表"从哪开始"）
# END:       图的虚拟出口点（固定常量，代表"到哪结束"）
from langgraph.graph import StateGraph, START, END
# interrupt: 让图在运行时暂停，等待外部输入（人工介入/审批）
# Command:   恢复中断后传入的指令对象，携带 resume 数据继续执行
from langgraph.types import interrupt, Command
# InMemorySaver: 内存中的检查点保存器，用于持久化图的状态
from langgraph.checkpoint.memory import InMemorySaver
# ChatZhipuAI: 智谱 AI 的大模型封装（langchain-community 版）
from langchain_community.chat_models import ChatZhipuAI
# HumanMessage: LangChain 的消息类型之一，表示用户的输入消息
from langchain_core.messages import HumanMessage
from demo_explain import print_lesson, print_step, print_tip


# ============================================================
# 共享资源: 在需要持久化的示例中复用同一个 checkpointer
# ============================================================
# _shared_checkpointer: 全局唯一的 InMemorySaver 实例，避免每次新建丢失历史数据
_shared_checkpointer = None

def _get_checkpointer():
    """懒加载单例 checkpointer —— 首次调用时创建，之后复用同一个实例
    
    为什么用单例？因为中断恢复等场景需要同一个 checkpointer 才能读取之前的快照。
    """
    global _shared_checkpointer
    if _shared_checkpointer is None:
        # InMemorySaver 是内存级持久化；生产环境可换成 SqliteSaver / PostgresSaver
        _shared_checkpointer = InMemorySaver()
    return _shared_checkpointer


# ============================================================
# 示例 1: stream_mode 对比
# ============================================================
# LangGraph 概念: 编译后的图对象支持 .stream() 方法，可以边执行边产出中间结果。
#   stream_mode="values"  : 每跑完一个节点，返回完整的当前状态字典
#   stream_mode="updates": 每跑完一个节点，只返回该节点修改的字段增量
def demo_stream_mode():
    print("=" * 80)
    print("示例1: stream_mode 对比")
    print("=" * 80)

    # 【State Schema】定义图的状态结构。这里用 TypedDict 声明三个字段及其类型。
    # 节点之间通过共享这份 state 来传递数据，每个节点只需要返回它要更新的字段。
    class DemoState(TypedDict):
        question: str  # 用户输入的问题
        step1: str     # 第一步的处理结果
        step2: str     # 第二步的最终结果

    def node1(state):
        """第一个处理节点：读取 state 里的 question，写入 step1 字段"""
        return {"step1": f"处理:{state['question']}"}

    def node2(state):
        """第二个处理节点：读取 step1，写入 step2 字段"""
        return {"step2": f"完成:{state['step1']}"}

    # StateGraph(DemoState): 创建一个以 DemoState 为状态的有向图
    graph = StateGraph(DemoState)

    # add_node(名称, 函数): 把普通 Python 函数注册为图中的处理节点
    graph.add_node("n1", node1)
    graph.add_node("n2", node2)

    # add_edge(上游, 下游): 定义固定流转路径。
    # START → "n1"  表示 n1 是图的入口节点
    # "n1" → "n2"   表示 n1 完成后一定进入 n2
    # "n2" → END    表示 n2 完成后整张图结束
    graph.add_edge(START, "n1")
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", END)

    # compile(): 把蓝图编译成可执行的 CompiledGraph 对象
    app = graph.compile()

    # ---------- values 模式 ----------
    # 每个节点执行完后，吐出完整的状态快照
    print("\n【values模式】每个节点后返回完整状态:")
    for chunk in app.stream({"question": "你好"}, stream_mode="values"):
        print(f"  {chunk}")

    # ---------- updates 模式 ----------
    # 每个节点执行完后，只吐出该节点的变更量，格式为 {节点名: 变更字典}
    print("\n【updates模式】每个节点只返回变更部分:")
    for chunk in app.stream({"question": "你好"}, stream_mode="updates"):
        for node_name, updates in chunk.items():
            print(f"  节点{node_name}: {updates}")


# ============================================================
# 示例 2: 持久化 (Checkpointer) -- 多轮对话
# ============================================================
# LangGraph 概念: Checkpointer（检查点）+ thread_id = 跨 invoke 保留状态。
#   每次调用 compile(checkpointer=...) 并带上相同的 thread_id，图会先恢复上次状态再执行。
#   配合 Annotated[list, operator.add] 做消息队列的累加。
def demo_persistence():
    print("\n" + "=" * 80)
    print("示例2: 持久化 (Checkpointer) -- 多轮对话")
    print("=" * 80)

    # messages 使用 operator.add reducer，新消息会被追加到列表末尾而不是覆盖旧值
    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]  # Reducer: 每次合并时用 list.extend
        response: str                             # 普通的覆盖模式

    def chat_node(state):
        """模拟聊天机器人：取出最后一条用户消息，构造助手的回复，追加到 messages 列表中"""
        last_msg = state["messages"][-1]["content"] if state["messages"] else ""
        return {
            # operator.add 会让这条消息拼接到已有列表上
            "messages": [{"role": "assistant", "content": f"我收到了:{last_msg}"}],
            "response": f"已回复:{last_msg}"
        }

    checkpointer = _get_checkpointer()

    chat_graph = StateGraph(ChatState)
    chat_graph.add_node("chat", chat_node)
    chat_graph.add_edge(START, "chat")
    chat_graph.add_edge("chat", END)

    # 编译时注入 checkpointer，开启状态持久化能力
    chat_app = chat_graph.compile(checkpointer=checkpointer)

    # thread_id 用来标识一个会话线程。同一次 thread_id 的多次 invoke 会共享持久化状态
    thread_config = {"configurable": {"thread_id": "user_session_001"}}

    print("\n第一轮对话:")
    result1 = chat_app.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config=thread_config  # 带着线程配置，状态被保存到 checkpointer
    )
    print(f"  消息历史: {len(result1['messages'])}条")
    print(f"  最后一条: {result1['messages'][-1]['content']}")

    print("\n第二轮对话 (状态自动保留):")
    # 不用传之前历史，checkpointer 已经存了上一轮快照，自动恢复后再执行
    result2 = chat_app.invoke(
        {"messages": [{"role": "user", "content": "再见"}]},
        config=thread_config
    )
    print(f"  消息历史: {len(result2['messages'])}条")
    print(f"  所有内容: {[m['content'] for m in result2['messages']]}")

    print("\n第三轮对话 (继续保留):")
    result3 = chat_app.invoke(
        {"messages": [{"role": "user", "content": "下次见"}]},
        config=thread_config
    )
    print(f"  消息历史: {len(result3['messages'])}条")
    print(f"  所有内容: {[m['content'] for m in result3['messages']]}")


# ============================================================
# 示例 3: Interrupt -- 人工审核流程
# ============================================================
# LangGraph 概念: interrupt() 可以在任意节点中暂停图的执行，
#   抛出中断标记供外部决策，然后再次 invoke(Command(resume=...)) 恢复后续节点。
#   前提是 compile 时必须带 checkpointer（否则没有持久化位置存放断点）。
def demo_interrupt():
    print("\n" + "=" * 80)
    print("示例3: Interrupt -- 人工审核流程")
    print("=" * 80)

    # ApprovalState: 文档审批流程需要的字段
    class ApprovalState(TypedDict):
        draft: str      # 草稿内容
        approved: bool  # 是否已通过审批
        feedback: str   # 审核意见
        final: str      # 最终发布文本

    def write_draft(state):
        """节点1——写手：起草一份初稿，填充 draft 字段"""
        print("  [写手] 正在起草文档...")
        return {"draft": "这是一份AI生成的初稿", "approved": False}

    def review_node(state):
        """节点2——审核人：调用 interrupt() 暂停执行，等待人类决定
        
        interrupt({dict}) 会把 dict 内容暴露给调用方，图在此处挂起。
        当外部再 invoke 一次 Command(resume=...) 时，interrupt() 就返回 resume 的数据。
        """
        print("  [审核] 等待人工审批...")
        decision = interrupt({
            "message": "请审批草稿",
            "draft": state["draft"],
            "action_required": True
        })
        return {
            "approved": decision.get("approve", False),
            "feedback": decision.get("feedback", "")
        }

    def finalize(state):
        """节点3——发布：根据审批结果生成最终文本"""
        print("  [发布] 生成最终结果...")
        if state["approved"]:
            return {"final": f"[APPROVED] {state['draft']}"}
        else:
            return {"final": f"[REJECTED] {state['feedback']}"}

    approval_graph = StateGraph(ApprovalState)
    approval_graph.add_node("write", write_draft)
    approval_graph.add_node("review", review_node)
    approval_graph.add_node("finalize", finalize)
    approval_graph.add_edge(START, "write")
    approval_graph.add_edge("write", "review")
    approval_graph.add_edge("review", "finalize")
    approval_graph.add_edge("finalize", END)

    # interrupt 依赖 checkpointer 存储断点，必须传入
    checkpointer = _get_checkpointer()
    approval_app = approval_graph.compile(checkpointer=checkpointer)

    # 专属的 thread_id，隔离不同审批任务
    config = {"configurable": {"thread_id": "approval_001"}}

    print("\n第一步: 执行到审核节点会中断")
    # 首次 invoke: 会走完 write → 停在 review 的 interrupt()
    result = approval_app.invoke(
        {"draft": "", "approved": False, "feedback": "", "final": ""},
        config=config
    )
    print(f"  结果类型: {type(result)}")

    # __interrupt__ 是 LangGraph 内部约定的中断标记键
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        print(f"  检测到中断! 需要人工介入")
        print(f"  中断内容: {interrupts[0].value}")

        print("\n第二步: 用户批准后恢复执行")
        # 用 Command(resume={...}) 给 interrupt() 提供返回值，然后从断点后继续
        user_decision = Command(resume={"approve": True, "feedback": "批准通过"})
        final_result = approval_app.invoke(user_decision, config=config)
        print(f"  最终结果: {final_result.get('final', '')}")
    else:
        # 如果 checkpointer 里已经存在之前审批过的快照，图不会中断而是直接出结果
        print("  无中断，正常完成")
        print(f"  结果: {result}")


# ============================================================
# 示例 4: Reducer (累加器)
# ============================================================
# LangGraph 概念: Annotated[field_type, reducer_func]
#   默认情况下，节点 return 的字典会用 update() 语义覆盖同级字段。
#   如果加了 reducer（如 operator.add），多个节点的返回值会按 reducer 规则合并。
#   operator.add 对 list 就是拼接，对 int 就是求和。
def demo_reducer():
    print("\n" + "=" * 80)
    print("示例4: Reducer (累加器)")
    print("=" * 80)

    class CounterState(TypedDict):
        visited_nodes: Annotated[list, operator.add]   # 追加模式: [] + ["A"] + ["B"]
        total_count: Annotated[int, operator.add]       # 累加模式: 0 + 10 + 5
        current_step: str                                # 覆盖模式: 后写的值替换前写的值

    def node_a(state):
        print("  [节点A] 执行...")
        return {
            "visited_nodes": ["A"],     # reducer 把它拼接到现有列表
            "total_count": 10,          # reducer 把它加到现有数值
            "current_step": "a"         # 普通覆盖模式
        }

    def node_b(state):
        print("  [节点B] 执行...")
        return {
            "visited_nodes": ["B"],     # 拼接到 A 后面 → ['A','B']
            "total_count": 5,           # 加到 10 上面 → 15
            "current_step": "b"         # 覆盖 a → b
        }

    counter_graph = StateGraph(CounterState)
    counter_graph.add_node("a", node_a)
    counter_graph.add_node("b", node_b)
    counter_graph.add_edge(START, "a")
    counter_graph.add_edge("a", "b")
    counter_graph.add_edge("b", END)
    counter_app = counter_graph.compile()

    result = counter_app.invoke(
        {"visited_nodes": [], "total_count": 0, "current_step": "start"}
    )
    print(f"\n最终状态:")
    print(f"  visited_nodes: {result['visited_nodes']}    # A + B = ['A', 'B']")
    print(f"  total_count: {result['total_count']}        # 10 + 5 = 15")
    print(f"  current_step: {result['current_step']}      # B覆盖了A → 'b'")


# ============================================================
# 示例 5: 真实 AI 处理节点 (智谱 GLM-4)
# ============================================================
# LangGraph 概念: 图中节点可以是任何可调用的 Python 函数。
#   这里我们接入真实 LLM，让大模型的推理成为图流水线中的一环。
#   数据流向:  问题 → AI回答节点 → AI摘要节点 → 结束
def demo_ai_node():
    load_dotenv()
    print("\n" + "=" * 80)
    print("示例5: 真实 AI 处理节点 (智谱 GLM-4)")
    print("=" * 80)

    # AIState: 记录问题、原始答案、一句话摘要
    class AIState(TypedDict):
        question: str     # 用户提的问题
        ai_answer: str    # 大模型的完整回答
        summary: str      # 大模型生成的摘要

    # ── 获取 API Key ──
    # ChatZhipuAI 要求的环境变量名是 ZHIPUAI_API_KEY（注意中间有 AI）
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("  [跳过] 未找到环境变量 ZHIPUAI_API_KEY，示例5无法运行。")
        print("  请先设置: set ZHIPUAI_API_KEY=your_key_here  (Windows cmd)")
        print("  或       $env:ZHIPUAI_API_KEY=\"your_key_here\" (PowerShell)")
        return

    # 初始化大模型客户端，显式传入 api_key
    llm = ChatZhipuAI(model="glm-4", zhipuai_api_key=api_key)

    def ai_process(state):
        """节点1——让 AI 回答问题
        
        参数:
            state: 当前状态字典，包含 question 字段
        返回:
            只更新 ai_answer 字段，其余字段不变
        """
        print(f"\n  [AI] 正在思考: {state['question']}...")
        # llm.invoke() 接收消息列表；HumanMessage 包装用户提问
        response = llm.invoke([HumanMessage(content=state["question"])])
        # response.content 是大模型输出的纯文本字符串
        return {"ai_answer": response.content}

    def ai_summarize(state):
        """节点2——让 AI 对上一步的回答生成一句话摘要
        
        节点可以直接读取前面节点写入的 state 字段（这里是 ai_answer），
        体现了图中数据沿 edge 向后流动的机制。
        """
        prompt = f"请用一句话总结以下回答:\n{state['ai_answer']}"
        print(f"  [AI] 正在生成摘要...")
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"summary": response.content}

    # 构建图: START → answer → summarize → END
    graph = StateGraph(AIState)
    graph.add_node("answer", ai_process)       # 注册回答节点
    graph.add_node("summarize", ai_summarize)  # 注册摘要节点
    graph.add_edge(START, "answer")            # 入口连到 answer
    graph.add_edge("answer", "summarize")      # 答完问题后去总结
    graph.add_edge("summarize", END)           # 总结完则结束
    app = graph.compile()

    # 用真实的 LangGraph 问题提问，观察 AI 如何工作
    result = app.invoke({
        "question": "什么是LangGraph？用100字以内解释。",
        "ai_answer": "",
        "summary": ""
    })

    print(f"\n  原始回答:\n    {result['ai_answer']}")
    print(f"\n  AI摘要:\n    {result['summary']}")


# ============================================================
# Main 入口: 依次运行所有示例
# ============================================================
if __name__ == "__main__":
    print_lesson(__file__)
    print_step(1, "stream_mode", "values=完整快照；updates=每个节点的增量")
    demo_stream_mode()
    print_step(2, "Checkpointer", "同一 thread_id 多次 invoke，messages 用 reducer 累加")
    demo_persistence()
    print_step(3, "interrupt", "必须配 checkpointer；resume 的键对准 decision.get(...)")
    demo_interrupt()
    print_step(4, "Reducer", "list 拼接、int 相加；没 Annotated 的字段是覆盖")
    demo_reducer()
    print_step(5, "真实 LLM 节点", "图节点就是普通函数，里面可以 invoke 模型")
    demo_ai_node()
    print("\n" + "=" * 80)
    print("所有示例运行完毕!")
    print("=" * 80)
    print_tip("按主题深入请看编号文件：lg_2_3 interrupt、lg_4_1 checkpointer、lg_3_3 Send。")
