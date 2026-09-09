"""
========================================================
LangChain Phase 5-6
结构化输出 教材级 Demo
========================================================

📚 学习目标：
1. model.with_structured_output() 强制 JSON Schema 输出
2. Pydantic Model vs TypedDict 两种 schema 定义
3. 在 RAG 场景标准化回答格式（summary + sources + confidence）
4. 对比 Output Parser 软解析 vs structured_output 硬约束

运行前提：
  pip install langchain langchain-core pydantic zhipuai python-dotenv
"""
import json
import os
from numbers import Number
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from model_provider import check_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()

class ArticleAnalysis(BaseModel):
    """文章分析结构化输出（Pydantic）"""
    title: str = Field(description='文章标题或主题')
    summary: str = Field(description='2~3 句摘要')
    keywords: List[str] = Field(description='3~5 个关键词')
    sentiment: str = Field(description='情感倾向：positive/neutral/negative')

class RAGAnswer(TypedDict):
    """RAG 回答标准格式（TypedDict，路线图 5-6 示例）"""
    summary: str
    confidence: float
    sources: List[str]

def demo_1_pydantic():
    print('\n' + '=' * 65)
    print('Demo 1: Pydantic Model + with_structured_output')
    print('=' * 65)
    structured_llm = get_model().with_structured_output(ArticleAnalysis)
    text = '\n    LangChain 是一个开源框架，用于构建大语言模型应用。\n    它提供了 Runnable、Prompt、Tool、Agent 等核心组件。\n    开发者可以用 LCEL 管道快速组装 AI 应用。\n    '
    result: ArticleAnalysis = structured_llm.invoke(f'分析以下文本：\n{text}')
    print(f'类型: {type(result).__name__}')
    print(f'title    : {result.title}')
    print(f'summary  : {result.summary}')
    print(f'keywords : {result.keywords}')
    print(f'sentiment: {result.sentiment}')
    print('\n💡 返回 Pydantic 实例，字段自动类型验证')

def demo_2_typed_dict():
    print('\n' + '=' * 65)
    print('Demo 2: TypedDict + with_structured_output（RAG 场景）')
    print('=' * 65)
    structured_llm = get_model().with_structured_output(RAGAnswer)
    context = '\n    [来源: docker_doc.txt] Docker 是一个开源容器化平台。\n    [来源: k8s_doc.txt] Kubernetes 用于容器编排。\n    '
    result: RAGAnswer = structured_llm.invoke(f'基于以下资料回答「Docker 和 K8s 的关系」：\n{context}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('\n💡 TypedDict 返回 dict，适合直接 JSON 序列化给前端/API')

def demo_3_rag_structured_chain():
    print('\n' + '=' * 65)
    print('Demo 3: RAG + 结构化输出 LCEL 链')
    print('=' * 65)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    rag_prompt = ChatPromptTemplate.from_messages([('system', '你是知识库助手。基于 context 回答，并严格按 schema 输出。'), ('human', '资料:\n{context}\n\n问题: {question}')])
    structured_model = get_model().with_structured_output(RAGAnswer)

    def build_input(question: str) -> dict:
        ctx = '[来源: rag.txt] RAG 通过检索增强 LLM 回答，减少幻觉。'
        return {'context': ctx, 'question': question}
    full_chain = RunnableLambda(build_input) | rag_prompt | structured_model
    result = full_chain.invoke('什么是 RAG？')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('💡 Phase 2 RAG 输出可标准化为 {summary, confidence, sources}')

def demo_4_vs_parser():
    print('\n' + '=' * 65)
    print('Demo 4: structured_output vs Output Parser')
    print('=' * 65)
    print('\n    ┌─────────────────────┬──────────────────────┬─────────────────────┐\n    │  方式               │  可靠性               │  适用场景             │\n    ├─────────────────────┼──────────────────────┼─────────────────────┤\n    │  with_structured_   │  高（模型 function   │  API 返回、数据入库  │\n    │  output             │  calling 硬约束）     │  RAG 标准化          │\n    ├─────────────────────┼──────────────────────┼─────────────────────┤\n    │  PydanticOutput     │  中（依赖 LLM 遵守   │  快速原型、格式要求  │\n    │  Parser             │  JSON 文本格式）      │  不严格的场景        │\n    ├─────────────────────┼──────────────────────┼─────────────────────┤\n    │  JsonOutputParser   │  低~中（无类型约束）  │  灵活 JSON 提取      │\n    └─────────────────────┴──────────────────────┴─────────────────────┘\n\n    智谱 GLM-4 支持 function_calling 模式的 with_structured_output。\n    生产环境优先 structured_output，Parser 作为降级方案。\n    ')

def demo_5_include_raw():
    print('\n' + '=' * 65)
    print('Demo 5: include_raw 调试原始响应')
    print('=' * 65)
    structured_llm = get_model().with_structured_output(ArticleAnalysis, include_raw=True)
    result = structured_llm.invoke('分析：今天天气很好，心情愉快。')
    print(f'keys: {list(result.keys())}')
    if 'parsed' in result:
        print(f"parsed.title: {result['parsed'].title}")
    if result.get('parsing_error'):
        print(f"parsing_error: {result['parsing_error']}")
    print('💡 include_raw=True 返回 {raw, parsed, parsing_error} 便于调试')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n╔══════════════════════════════════════════════════════════╗\n║     LangChain Phase 5-6: 结构化输出                      ║\n╚══════════════════════════════════════════════════════════╝\n')
    print_run_plan([
        ('include_raw', '本入口只跑这一节：结构化对象 + 原始 AIMessage。同文件还有 Pydantic/TypedDict 可单独调用'),
    ])
    check_api_key()
    try:
        demo_5_include_raw()
    except Exception as exc:
        print(f'\n❌ 运行出错: {exc}')
        import traceback
        traceback.print_exc()
        raise
    print('\n🎉 Phase 5 全部完成！下一步: 6_1_architecture_selection_demo.py\n')