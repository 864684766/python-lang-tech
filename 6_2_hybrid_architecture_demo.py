"""
========================================================
LangChain Phase 6-2
LangChain × LangGraph 混合架构 实战 Demo
========================================================

📚 学习目标：
1. LangGraph 做全局流程编排（路由 → 分支 → 汇总）
2. LCEL 链作为 Graph 节点内的数据处理单元（RAG Pipeline）
3. create_agent 作为 Graph 节点内的 Tool Agent
4. 理解路线图「典型混合架构」的完整实现

🔗 混合架构图：
┌─────────────────────────────────────────────────────────────┐
│  LangGraph (全局编排)                                        │
│                                                             │
│  START → [Router Node] ──┬──→ [RAG Node (LCEL)]            │
│                          │         ↓                        │
│                          └──→ [Agent Node (create_agent)]   │
│                                    ↓                        │
│                          [Merge Node] → END                 │
└─────────────────────────────────────────────────────────────┘

运行前提：
  pip install langchain langchain-core langchain-community \\
              langchain-chroma langgraph zhipuai python-dotenv
"""
import os
from numbers import Number
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from model_provider import check_api_key, get_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

def build_rag_chain():
    """
    纯 LCEL 构建的 RAG 链 —— 作为 LangGraph 节点内的「技能管道」。
    节点只负责 invoke，不关心编排逻辑。
    """
    from langchain_chroma import Chroma
    from langchain_community.embeddings import ZhipuAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    docs = [Document(page_content='LangChain 提供 Runnable、Prompt、Tool 等组件，用 LCEL 管道组合。', metadata={'source': 'lc.txt'}), Document(page_content='LangGraph 基于 StateGraph 编排有状态流程，适合 Agent 和多步任务。', metadata={'source': 'lg.txt'}), Document(page_content='RAG 通过检索外部知识增强 LLM，适合企业知识库问答。', metadata={'source': 'rag.txt'})]
    splits = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_documents(docs)
    vectorstore = Chroma.from_documents(splits, ZhipuAIEmbeddings(api_key=get_api_key()), collection_name='hybrid_demo_kb')
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

    def format_docs(docs_list):
        return '\n\n'.join((f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs_list))
    rag_prompt = ChatPromptTemplate.from_messages([('system', '基于以下资料回答问题，不知道就说不知道。\n\n{context}'), ('human', '{question}')])
    return {'context': retriever | format_docs, 'question': RunnablePassthrough()} | rag_prompt | get_model() | StrOutputParser()

@tool
def calculator(expression: str) -> str:
    """数学计算。当问题涉及算数、公式时使用。"""
    try:
        return str(eval(expression, {'__builtins__': {}}, {}))
    except Exception as exc:
        return f'错误: {exc}'

@tool
def get_system_time() -> str:
    """获取当前系统时间（模拟）"""
    return '2026-08-20 18:00:00 CST'

def build_tool_agent():
    """create_agent 工厂 —— 作为 LangGraph 节点内的 MCP/Tool Agent 替身"""
    return create_agent(get_model(), [calculator, get_system_time], system_prompt='你是工具助手，数学用 calculator，时间用 get_system_time。')

class HybridState(TypedDict):
    question: str
    route: str
    rag_answer: str
    agent_answer: str
    final_answer: str

def router_node(state: HybridState) -> dict:
    """
    SkillRouter：根据问题关键词决定走 RAG 还是 Tool Agent。
    生产环境可换 LLM 分类或 embedding 相似度路由。
    """
    q = state['question'].lower()
    calc_keywords = ['计算', '等于', '加', '减', '乘', '除', '+', '-', '*', '/']
    time_keywords = ['时间', '几点', '日期']
    if any((k in q for k in calc_keywords + time_keywords)):
        route = 'agent'
    else:
        route = 'rag'
    print(f"  🔀 Router: 「{state['question'][:30]}...」 → {route}")
    return {'route': route}

def rag_node(state: HybridState) -> dict:
    """RAG Node：内部调用 LCEL 链，LangGraph 不管检索细节"""
    print('  📚 RAG Node: 执行 LCEL 检索增强链...')
    chain = build_rag_chain()
    answer = chain.invoke(state['question'])
    return {'rag_answer': answer}

def agent_node(state: HybridState) -> dict:
    """Agent Node：内部调用 create_agent，LangGraph 不管 ReAct 循环"""
    print('  🔧 Agent Node: 执行 Tool Agent...')
    agent = build_tool_agent()
    result = agent.invoke({'messages': [HumanMessage(content=state['question'])]})
    answer = result['messages'][-1].content
    return {'agent_answer': answer}

def merge_node(state: HybridState) -> dict:
    """汇总 Node：根据 route 选取对应分支的答案"""
    if state['route'] == 'rag':
        final = state.get('rag_answer', '')
    else:
        final = state.get('agent_answer', '')
    return {'final_answer': final}

def route_after_router(state: HybridState) -> Literal['rag_node', 'agent_node']:
    return 'rag_node' if state['route'] == 'rag' else 'agent_node'

def build_hybrid_graph():
    graph = StateGraph(HybridState)
    graph.add_node('router', router_node)
    graph.add_node('rag_node', rag_node)
    graph.add_node('agent_node', agent_node)
    graph.add_node('merge', merge_node)
    graph.add_edge(START, 'router')
    graph.add_conditional_edges('router', route_after_router, {'rag_node': 'rag_node', 'agent_node': 'agent_node'})
    graph.add_edge('rag_node', 'merge')
    graph.add_edge('agent_node', 'merge')
    graph.add_edge('merge', END)
    return graph.compile()

def demo_1_concept():
    print('\n' + '=' * 65)
    print('Demo 1: 混合架构核心思想')
    print('=' * 65)
    print('\n    LangGraph 节点 = 编排单元（何时做什么）\n    LangChain 链/Agent = 执行单元（怎么做）\n\n    本 Demo 中：\n      router_node   → 纯 Python 路由逻辑\n      rag_node      → 内部 invoke LCEL RAG chain（Phase 2 产物）\n      agent_node    → 内部 invoke create_agent（Phase 3 产物）\n      merge_node    → 汇总输出\n\n    好处：\n      · RAG 链可独立测试（2_6_rag_pipeline_demo.py）\n      · Agent 可独立测试（3_2_create_agent_demo.py）\n      · Graph 只负责「问知识库还是调工具」的编排\n    ')

def demo_2_end_to_end():
    print('\n' + '=' * 65)
    print('Demo 2: 混合架构端到端运行')
    print('=' * 65)
    app = build_hybrid_graph()
    test_cases = ['LangChain 和 LangGraph 有什么区别？', '帮我计算 (100 + 50) * 2', '什么是 RAG？', '现在几点了？']
    for question in test_cases:
        print(f"\n{'═' * 55}")
        print(f'问题: {question}')
        print('═' * 55)
        result = app.invoke({'question': question, 'route': '', 'rag_answer': '', 'agent_answer': '', 'final_answer': ''})
        print(f"  路由: {result['route']}")
        print(f"  ✅ 最终回答: {result['final_answer'][:200]}...")

def demo_3_stream():
    print('\n' + '=' * 65)
    print('Demo 3: stream(updates) 观察混合流程')
    print('=' * 65)
    app = build_hybrid_graph()
    question = 'RAG 是什么？'
    for i, chunk in enumerate(app.stream({'question': question, 'route': '', 'rag_answer': '', 'agent_answer': '', 'final_answer': ''}, stream_mode='updates'), 1):
        nodes = list(chunk.keys())
        print(f'  Step {i}: 节点 {nodes}')
    print('💡 可实时展示「路由→RAG/Agent→汇总」各阶段')

def demo_4_extensions():
    print('\n' + '=' * 65)
    print('Demo 4: 混合架构扩展方向')
    print('=' * 65)
    print('\n    在本 Demo 基础上可扩展：\n\n    · Hook Manager 节点：merge 前插入审计日志 / Token 统计\n      → 见 6_3_hook_manager_demo.py\n    · Checkpointer：多轮对话记住上下文（Phase 4）\n    · Skill Registry：router 动态加载技能链（Phase 7-1）\n    · MCP Tool Agent：agent_node 替换为 MCP 工具集（Phase 7-2）\n    · interrupt：敏感操作前人工审核（langgraph_demo_examples.py）\n\n    原则不变：Graph 0 业务逻辑，LCEL 0 流程控制。\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 6-2: 混合架构实战                     ║\n║     LangGraph 编排 + LCEL RAG + Tool Agent               ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('概念图', '图管流程，链管节点内部'),
        ('扩展点', '还可以接 MCP / Hook'),
        ('端到端', '路由 → RAG 或 Agent → 合并（需 API）'),
        ('stream', '混合图的流式输出（需 API）'),
    ])
    demo_1_concept()
    demo_4_extensions()
    check_api_key()
    try:
        pass
        demo_2_end_to_end()
        demo_3_stream()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n╔══════════════════════════════════════════════════════════╗\n║  🎉 Phase 4~6 全部 Demo 完成！                            ║\n║                                                          ║\n║  Phase 4: Memory        → 4_1 ~ 4_3                      ║\n║  Phase 5: 工程实践       → 5_1 ~ 5_6                      ║\n║  Phase 6: 混合架构       → 6_1 ~ 6_2                      ║\n║                                                          ║\n║  下一步: Phase 7 Skill / MCP / Hooks（路线图 Step 14~16）║\n╚══════════════════════════════════════════════════════════╝\n')