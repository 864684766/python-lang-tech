"""
========================================================

LangChain RAG 完整 Pipeline 教材级 Demo
（Phase 2-1 至 2-6 端到端整合）


本文件演示 LangChain 的完整 RAG（检索增强生成）流程：

  Phase 2-1: Document Loaders   → 从各种来源加载文档
  Phase 2-2: Text Splitters     → 将长文档切分为合适的块
  Phase 2-3: Embeddings         → 将文本转换为语义向量
  Phase 2-4: Vector Stores      → 存储和检索向量（FAISS / Chroma）
  Phase 2-5: Retrievers         → 高级检索策略（MMR / BM25 / Ensemble...）
  Phase 2-6: Full RAG Pipeline  → 端到端 LCEL 管道构建


📚 学习目标：
1. 理解 RAG 的完整数据流
2. 掌握每个环节的原理和使用方法
3. 学会用 LCEL（LangChain Expression Language）构建可组合管道
4. 理解各个环节的参数调优要点


🔗 RAG 完整流程图：
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 数据流全景图                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    【索引阶段】(离线，一次性)                  │   │
│  │                                                              │   │
│  │  原始文档 ──▶ Loader ──▶ Splitter ──▶ Embedding ──▶ Vector Store  │
│  │   (PDF/网页/    (提取内容)  (切分成块)   (转成向量)   (存入数据库)   │
│  │    TXT/CSV)                                                           │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    【检索阶段】(在线，每次查询)                │   │
│  │                                                              │   │
│  │  用户问题 ──▶ Retriever ──▶ 相关文档片段                      │   │
│  │   ("什么是RAG？")   (相似度搜索)    (从向量库召回)              │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    【生成阶段】(在线，每次查询)                │   │
│  │                                                              │   │
│  │  问题+文档 ──▶ Prompt模板 ──▶ LLM ──▶ 最终回答               │   │
│  │   (组合上下文)  (结构化提示)   (智谱/Qwen/OpenAI)  (基于知识回答) │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


运行前提：
  pip install langchain langchain-core langchain-community             langchain-text-splitters chromadb zhipuai python-dotenv
  设置环境变量: export ZHIPUAI_API_KEY='your-key'
"""
import os
import tempfile
from numbers import Number
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

from model_provider import check_api_key, get_api_key, get_model
from demo_explain import print_lesson, print_run_plan
load_dotenv()
embeddings = ZhipuAIEmbeddings(api_key=get_api_key())

def demo_document_loaders():
    """
    演示如何使用不同的 Document Loader 加载各种格式的文档。
    
    【核心概念】
    Document Loader 是 RAG  pipeline 的第一步，负责从各种数据源读取内容，
    并统一转换为 LangChain 的 Document 对象。
    
    ┌──────────────────────────────────────────────────────────────┐
    │  Document 对象结构                                            │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │  page_content: "文档的实际文本内容..."                  │  │
    │  │  metadata: {"source": "file.pdf", "page": 1, ...}      │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                              │
    │  page_content → 会被 Embedding 向量化                        │
    │  metadata     → 用于过滤和溯源                               │
    └──────────────────────────────────────────────────────────────┘
    
    【常见 Loader 及使用场景】
    ┌──────────────────┬───────────────────────────┬─────────────────┐
    │  Loader          │  适用文件格式               │  典型场景        │
    ├──────────────────┼───────────────────────────┼─────────────────┤
    │ TextLoader       │ .txt, .md                 │ 纯文本知识库     │
    │ PyPDFLoader      │ .pdf                      │ 技术手册、论文   │
    │ WebBaseLoader    │ URL                       │ 官网文档抓取     │
    │ DirectoryLoader  │ 整个目录                  │ 批量导入文档库   │
    │ CSVLoader        │ .csv                      │ 表格数据         │
    │ UnstructuredLoader│ .docx, .pptx, .xlsx     │ Office 文档      │
    └──────────────────┴───────────────────────────┴─────────────────┘
    
    【工作流程】
    
    原始文档 📄 ──▶ Loader 加载 ──▶ List[Document] ──▶ 后续处理
                    (解析、清洗)      (page_content   (切分/向量化
                                       + metadata)     /存储)
    """
    print('\n' + '=' * 70)
    print('Demo 1: Document Loaders - 文档加载器')
    print('=' * 70)
    print('\n[1] 加载单个文本文档')
    print('    适用场景：单份技术文档、FAQ 文档、政策文件')
    temp_file = os.path.join(tempfile.gettempdir(), 'sample.txt')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write('LangChain 是一个用于开发由大语言模型（LLM）驱动的应用程序的开源框架。\n\n核心特性：\n1. 丰富的组件库：提供提示模板、记忆管理、检索增强生成（RAG）、工具调用等组件\n2. 灵活的编排能力：支持链式调用、条件分支、循环等复杂流程\n3. 多模型支持：兼容 OpenAI、Anthropic、智谱 GLM 等多种大模型\n4. 可扩展性：易于自定义和扩展，满足各种业务需求\n\n典型应用场景：\n- 智能客服系统：基于知识库的自动化问答\n- 文档问答机器人：上传 PDF，自动回答相关问题\n- 数据分析助手：结合数据库和自然语言查询\n- 代码生成工具：根据需求描述自动生成代码\n')
    loader = TextLoader(temp_file, encoding='utf-8')
    documents = loader.load()
    print(f'    ✅ 加载了 {len(documents)} 个文档')
    print(f'    内容预览: {documents[0].page_content[:80]}...')
    print(f'    元数据:   {documents[0].metadata}')
    print(f'    💡 metadata 自动记录了 source 文件路径，便于溯源')
    print('\n[2] 批量加载目录下所有文档')
    print('    适用场景：公司知识库、产品文档库、论文集合')
    sample_dir = os.path.join(tempfile.gettempdir(), 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(3):
        with open(os.path.join(sample_dir, f'doc_{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(f'This is document #{i + 1} about LangChain framework.\n')
            f.write('LangChain provides rich components for building AI applications.\n')
            f.write('It supports multiple LLMs and vector databases with great extensibility.\n')
    directory_loader = DirectoryLoader(sample_dir, glob='**/*.txt', loader_cls=TextLoader)
    dir_documents = directory_loader.load()
    print(f'    ✅ 加载了 {len(dir_documents)} 个文档')
    for i, doc in enumerate(dir_documents[:3], 1):
        print(f"    [{i}] 来源: {doc.metadata['source']}")
        print(f'        内容: {doc.page_content[:50]}...')
    print('\n[3] 加载网页内容')
    print('    适用场景：抓取官方文档、技术博客、新闻文章')
    print('    注意：WebBaseLoader 需要联网，此处仅展示用法')
    print('\n    示例代码：\n    from langchain_community.document_loaders import WebBaseLoader\n    loader = WebBaseLoader(["https://www.cnblogs.com/llcdbk"])\n    documents = loader.load()\n    ')
    os.remove(temp_file)
    for f in os.listdir(sample_dir):
        os.remove(os.path.join(sample_dir, f))
    os.rmdir(sample_dir)
    print('\n[NOTE] Loader 选择指南:')
    print('   • 单个文件      → TextLoader / PyPDFLoader')
    print('   • 整个目录      → DirectoryLoader（递归扫描）')
    print('   • 网页内容      → WebBaseLoader（需要联网）')
    print('   • 所有 Loader 都返回 List[Document]，格式统一')

def demo_text_splitters():
    """
    演示如何将长文档切分为合适的块（Chunking）。
    
    【为什么需要分块？】
    这是 RAG 系统中非常关键的一步，直接关系到检索质量和成本。
    
    ┌────────────────────────────────────────────────────────────────┐
    │  原因一：Token 限制                                             │
    │  大模型有输入长度限制（如 4K、8K、32K tokens）                   │
    │  如果文档太长，无法一次性处理和检索                              │
    │                                                                │
    │  原因二：检索效率                                                │
    │  小块更容易精确匹配，搜索速度更快                                │
    │  大块可能包含过多噪音，降低检索精度                              │
    │                                                                │
    │  原因三：成本控制                                                │
    │  只检索相关片段，而不是整篇文档                                  │
    │  减少 Token 消耗，降低 LLM 调用成本                              │
    └────────────────────────────────────────────────────────────────┘
    
    【Chunking 参数详解】
    ┌─────────────────┬──────────────────────────────────────────┐
    │  参数           │  说明                                    │
    ├─────────────────┼──────────────────────────────────────────┤
    │  chunk_size     │  每个块的最大字符数                        │
    │                 │  推荐：中文 500-1000 字，英文 200-500 词  │
    ├─────────────────┼──────────────────────────────────────────┤
    │  chunk_overlap  │  相邻块之间的重叠字符数                    │
    │                 │  推荐：50-100 字符                         │
    │                 │  作用：避免关键信息被截断在块的边界         │
    ├─────────────────┼──────────────────────────────────────────┤
    │  separator      │  优先在哪些字符处切分                     │
    │                 │  默认：["

", "
", " ", ""]           │
    │                 │  优先级从高到低，先尝试在大块切分           │
    └─────────────────┴──────────────────────────────────────────┘
    
    【分块不当的后果】
    ❌ chunk_size 太大：
       • 检索结果冗长，包含大量无关信息
       • 浪费 Token，增加成本
       • LLM 可能被噪音干扰产生幻觉
    
    ❌ chunk_size 太小：
       • 丢失上下文，检索结果断章取义
       • 同一个概念被拆散在多块中
       • 检索结果碎片化，难以理解
    
    ❌ chunk_overlap 为 0：
       • 关键句子可能被截断在两块的边界
       • 边界处的信息不完整
    """
    print('\n' + '=' * 70)
    print('Demo 2: Text Splitters - 文本分块策略')
    print('=' * 70)
    long_text = 'LangChain 是一个用于开发由大语言模型驱动的应用程序的框架。\n\n第一章：基础概念\nLangChain 的核心思想是将复杂的 AI 应用分解为可组合的组件。每个组件都可以独立开发和测试。\n主要组件包括：\n- Prompt Templates：管理提示词模板\n- Models：封装各种大语言模型\n- Chains：将多个组件连接成工作流\n- Agents：让模型能够调用工具和做出决策\n- Memory：管理对话历史\n- Indexes：处理文档和检索\n\n第二章：RAG 架构\n检索增强生成（RAG）是 LangChain 的核心应用场景之一。它通过以下步骤实现：\n1. 加载文档：使用 Document Loader 从各种来源读取内容\n2. 分块处理：使用 Text Splitter 将长文档切分为小块\n3. 生成嵌入：使用 Embeddings 模型将文本转换为向量\n4. 存储向量：将向量存入 Vector Store\n5. 检索相关文档：根据查询召回最相关的文档片段\n6. 生成回答：将检索结果与用户问题一起发送给 LLM\n\n第三章：工具调用\nLangChain 支持强大的工具调用功能：\n- @tool 装饰器：快速定义工具函数\n- bind_tools：将工具绑定到模型\n- Agent：自动决定何时调用工具\n- Multi-agent：多代理协作\n\n第四章：最佳实践\n- 选择合适的分块大小（通常 500-1000 字符）\n- 保持适当的重叠（通常 50-100 字符）\n- 使用元数据过滤提高检索精度\n- 结合重排序提升相关性\n'
    print('\n[1] RecursiveCharacterTextSplitter')
    print('    原理：按字符递进切分，优先在换行符处断开')
    print('    参数：chunk_size=500, chunk_overlap=50')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    docs = text_splitter.create_documents([long_text])
    print(f'    原始文本长度: {len(long_text)} 字符')
    print(f'    切分后得到:   {len(docs)} 个文档块')
    print(f'\n    第一个块（前 200 字符）:')
    print(f'    {docs[0].page_content[:200]}...')
    print(f'\n    第二个块（注意与前一块的重叠部分）:')
    print(f'    {docs[1].page_content[:200]}...')
    print(f'    💡 重叠的作用：确保边界处的关键信息不被丢失')
    print('\n[2] MarkdownTextSplitter')
    print('    原理：按照 Markdown 的标题层级（#、##、###）进行切分')
    print('    适用：README、技术文档、论文等结构化内容')
    markdown_text = '# LangChain 教程\n\n## 第一章：基础概念\n\n### 1.1 什么是 LangChain\nLangChain 是一个...\n\n### 1.2 核心组件\n- Prompt Templates\n- Models\n- Chains\n\n## 第二章：RAG 架构\n\n### 2.1 检索流程\n1. 加载文档\n2. 分块处理\n3. 生成嵌入\n'
    md_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=30)
    md_docs = md_splitter.create_documents([markdown_text])
    print(f'    Markdown 切分后得到: {len(md_docs)} 个文档块')
    for i, doc in enumerate(md_docs[:3], 1):
        print(f'\n    [{i}] 前 100 字符: {doc.page_content[:100]}...')
    print('    💡 MarkdownSplitter 能保持章节完整性，适合结构化文档')
    print('\n[3] 不同分块策略对比')
    small_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    small_docs = small_splitter.create_documents([long_text])
    print(f'\n    小块模式: chunk_size=200, overlap=20')
    print(f'    结果: {len(small_docs)} 个块')
    print(f'    优点: 检索精度高，每条结果聚焦')
    print(f'    缺点: 可能丢失上下文，需要更多结果才完整')
    large_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    large_docs = large_splitter.create_documents([long_text])
    print(f'\n    大块模式: chunk_size=1000, overlap=100')
    print(f'    结果: {len(large_docs)} 个块')
    print(f'    优点: 保留更多上下文，检索结果更完整')
    print(f'    缺点: 可能引入噪音，Token 消耗更多')
    print('\n[NOTE] 参数选择建议:')
    print('   • chunk_size: 中文文档推荐 500-1000 字符')
    print('   • chunk_overlap: 推荐 50-100 字符（约为 chunk_size 的 10%）')
    print('   • 重叠的作用: 防止关键句子被截断在两个块的边界处')
    print('   • 经验法则: 从 chunk_size=800, overlap=80 开始，根据效果调整')

def demo_embeddings():
    """
    演示如何使用 Embedding 模型将文本转换为向量。
    
    【什么是 Embedding？】
    Embedding 是将文本映射到高维向量空间的技术。
    
    核心思想：
    • 语义相近的文本 → 向量在空间中距离接近
    • 语义无关的文本 → 向量在空间中距离较远
    
    举例：
    "猫" 和 "猫咪" 的向量距离 < "猫" 和 "汽车" 的向量距离
    
    【工作流程】
    
    文本："LangChain 是一个 AI 框架"
         │
         ▼
    ┌─────────────────┐
    │  Embedding 模型  │  ← 神经网络（通常是 Transformer）
    │  (1024 维)      │
    └────────┬────────┘
             │
             ▼
    向量：[0.12, -0.45, 0.78, ..., 0.23]  ← 1024 个浮点数
             │
             ▼
    存入 Vector Store，供后续相似度搜索使用
    
    【模型选择指南】
    ┌─────────────────────┬────────┬──────────┬──────────┬──────────┐
    │  模型               │ 维度   │ 中文效果  │ 费用     │ 部署方式  │
    ├─────────────────────┼────────┼──────────┼──────────┼──────────┤
    │ ZhipuAI embedding-3 │ 1024   │ ★★★★★   │ 有免费额度│ 云端 API  │
    │ BAAI/bge-large-zh   │ 1024   │ ★★★★★   │ 开源免费  │ 本地/云端 │
    │ text-embedding-3-s  │ 1536   │ ★★★☆☆   │ OpenAI收费│ 云端 API  │
    │ nomic-embed-text    │ 768    │ ★★★★☆   │ 开源免费  │ 本地运行  │
    └─────────────────────┴────────┴──────────┴──────────┴──────────┘
    """
    print('\n' + '=' * 70)
    print('Demo 3: Embeddings - 嵌入模型')
    print('=' * 70)
    print('\n[1] 嵌入模型信息')
    print(f'    模型名称: {embeddings.model}')
    print(f'    向量维度: {embeddings.dimensions}')
    print(f'    提供商:   智谱 AI')
    print('\n[2] 单条文本向量化 (embed_query)')
    print('    适用场景: 用户提问时，将问题转为向量用于搜索')
    text = 'LangChain 是一个用于开发 LLM 应用的开源框架'
    vector = embeddings.embed_query(text)
    print(f'    文本: "{text}"')
    print(f'    向量长度: {len(vector)} 维')
    print(f'    前 10 维: {vector[:10]}')
    print(f'    💡 这是一个 1024 维的向量，每个维度代表文本的某个语义特征')
    print('\n[3] 批量文本向量化 (embed_documents)')
    print('    适用场景: 知识库预处理时，批量转换所有文档')
    texts = ['LangChain 支持多种大模型接入', 'RAG 是检索增强生成的缩写', '向量数据库用于存储嵌入向量', 'Prompt Template 管理提示词模板']
    vectors = embeddings.embed_documents(texts)
    print(f'    处理了 {len(vectors)} 条文本')
    print(f'    每条向量维度: {len(vectors[0])}')
    print(f'    💡 批量调用比循环调用更高效，减少了 API 往返次数')
    print('\n[4] 向量相似度计算')
    print('    原理: 语义相近的文本，其向量之间的余弦相似度更高')
    import numpy as np

    def cosine_similarity(v1, v2):
        """计算两个向量的余弦相似度"""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    query_vector = embeddings.embed_query('AI 框架')
    print(f'\n    查询向量: "AI 框架"')
    print(f'    与各文本的相似度得分:\n')
    for text, vector in zip(texts, vectors):
        sim = cosine_similarity(query_vector, vector)
        bar = '█' * int(sim * 20)
        print(f'      {sim:.4f}  {bar}  {text[:30]}...')
    print('\n    💡 结论: 向量化的「LangChain 支持多种大模型」与查询最相似')
    print('    这证明了 Embedding 确实捕捉到了语义关系！')
    print('\n[NOTE] Embedding 选型建议:')
    print('   • 中文场景首选: ZhipuAI embedding-3 或 BGE')
    print('   • 英文场景可选: text-embedding-3-small (OpenAI)')
    print('   • 私有化部署: nomic-embed-text 或 BGE (开源可本地跑)')
    print('   • 维度选择: 768-1536 维之间，维度越高通常语义表达能力越强')

def demo_vector_stores():
    """
    演示如何使用 Chroma 向量数据库存储和检索嵌入向量。
    
    【什么是向量数据库？】
    普通数据库擅长存储和查询结构化数据（数字、字符串）
    向量数据库专门用于存储和检索向量，支持高效的相似度搜索
    
    【为什么需要专用向量数据库？】
    暴力搜索（逐个比较所有向量）在大规模数据下太慢：
    • 1 万条向量 × 1024 维 → 仍需毫秒级响应
    • 100 万条向量 × 暴力搜索 → 秒级甚至分钟级，不可接受
    • 向量数据库使用近似最近邻（ANN）算法，将搜索优化到微秒级
    
    【主流向量数据库对比】
    ┌──────────┬───────────┬───────────┬──────────────────────────┐
    │  名称    │  部署方式 │  规模     │        特点               │
    ├──────────┼───────────┼───────────┼──────────────────────────┤
    │ Chroma   │ 本地文件  │ <100万    │ 极简 API，自动持久化      │
    │ FAISS    │ 本地内存  │ <1000万   │ Facebook 开源，速度最快   │
    │ Qdrant   │ 本地/云   │ 大规模    │ 丰富过滤功能，rust 实现   │
    │ Milvus   │ 分布式    │ 超大规模  │ 云原生，亿级向量支持      │
    │ Pinecone │ 纯云端    │ 任意规模  │ 免运维，按量计费          │
    └──────────┴───────────┴───────────┴──────────────────────────┘
    
    【数据存储结构示意】
    ┌─────────────────────────────────────────────┐
    │  Chroma 向量数据库                          │
    │  ┌──────┬──────────────────┬──────────────┐ │
    │  │ ID   │ 向量(1024维)      │ 元数据        │ │
    │  ├──────┼──────────────────┼──────────────┤ │
    │  │  1   │ [0.12, -0.45...] │ source:a.txt │ │
    │  │  2   │ [0.33, 0.67...]  │ source:b.pdf │ │
    │  │  3   │ [-0.11, 0.88...] │ source:c.md  │ │
    │  └──────┴──────────────────┴──────────────┘ │
    │                                               │
    │  搜索时：给定查询向量 → 找最近的 K 条记录      │
    └─────────────────────────────────────────────┘
    """
    print('\n' + '=' * 70)
    print('Demo 4: Vector Stores - 向量数据库 (Chroma)')
    print('=' * 70)
    print('\n[1] 准备示例文档')
    documents = [Document(page_content='LangChain 是一个用于开发 LLM 应用的开源框架，提供丰富的组件', metadata={'source': 'doc1.txt', 'chapter': '简介'}), Document(page_content='RAG（检索增强生成）通过检索相关文档来增强 LLM 的回答质量', metadata={'source': 'doc2.txt', 'chapter': 'RAG'}), Document(page_content='向量数据库如 Chroma 支持持久化存储和高效的相似度搜索', metadata={'source': 'doc3.txt', 'chapter': '向量库'}), Document(page_content='Prompt Template 用于管理提示词模板，支持变量注入和格式化', metadata={'source': 'doc4.txt', 'chapter': '提示词'})]
    for i, doc in enumerate(documents, 1):
        print(f"    [{i}] {doc.page_content[:40]}... | 来源: {doc.metadata['source']}")
    print('\n[2] 创建 Chroma 向量数据库')
    persist_dir = os.path.join(tempfile.gettempdir(), 'chroma_rag')
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    print(f'    ✅ 向量库已创建')
    print(f'    持久化路径: {persist_dir}')
    print('\n[3] 添加文档到向量库')
    vectorstore.add_documents(documents)
    count = vectorstore._collection.count()
    print(f'    ✅ 已添加 {len(documents)} 个文档')
    print(f'    当前记录数: {count}')
    print('\n[4] 执行相似度搜索')
    query = '什么是 LangChain？'
    results = vectorstore.similarity_search(query, k=2)
    print(f'    查询: "{query}"')
    print(f'    返回 {len(results)} 个结果:\n')
    for i, doc in enumerate(results, 1):
        print(f'    [{i}] {doc.page_content}')
        print(f"        来源: {doc.metadata['source']}")
        print(f"        章节: {doc.metadata['chapter']}")
    print('\n[5] 带过滤条件的搜索')
    filtered_results = vectorstore.similarity_search(query='RAG', k=2, filter={'chapter': 'RAG'})
    print(f"    查询: 'RAG', 过滤: chapter='RAG'")
    for doc in filtered_results:
        print(f"      ✓ {doc.page_content[:50]}... | 章节: {doc.metadata['chapter']}")
    print('\n[6] 验证持久化（重新加载）')
    loaded_vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    loaded_count = loaded_vectorstore._collection.count()
    print(f'    重新加载后记录数: {loaded_count}')
    print(f'    ✅ 数据持久化成功，无需重新 Embedding！')
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)
    print('\n[NOTE] Chroma 使用指南:')
    print('   • 创建: Chroma(embedding_function=..., persist_directory=path)')
    print('   • 添加: vectorstore.add_documents(documents)')
    print('   • 搜索: vectorstore.similarity_search(query, k=N, filter={})')
    print('   • 加载: 用相同 persist_directory 创建新实例即可自动加载')
    print('   • 优点: API 简洁，自动持久化，适合开发和测试')

def demo_retriever():
    """
    演示如何使用 Retriever 进行高级检索。
    
    【Retriever 是什么？】
    Retriever 是 LangChain 提供的检索抽象层，它封装了从向量库查找相关文档的逻辑。
    
    相比直接使用 vector_store.similarity_search()，Retriever 有以下优势：
    
    ┌─────────────────┬────────────────────────────────────────────┐
    │  直接调用 VS     │                                             │
    │  Retriever       │                                             │
    ├─────────────────┼────────────────────────────────────────────┤
    │  接口统一        │ 不管底层是 FAISS、Chroma 还是 Pinecone      │
    │                 │ 都用 retriever.invoke(query)               │
    ├─────────────────┼────────────────────────────────────────────┤
    │  LCEL 可组合    │ 可以直接放入 LangChain 管道中               │
    │                 │ chain = retriever | parser | llm           │
    ├─────────────────┼────────────────────────────────────────────┤
    │  策略切换       │ 轻松切换 similarity/MMR/BM25 等不同策略    │
    └─────────────────┴────────────────────────────────────────────┘
    
    【常用检索策略】
    ┌───────────────────┬──────────────────────────────────────┐
    │  策略             │  适用场景                            │
    ├───────────────────┼──────────────────────────────────────┤
    │ similarity        │ 通用语义搜索（最常用）               │
    │ mmr               │ 需要结果多样化、避免重复             │
    │ similarity_score  │ 只返回高置信度结果                   │
    │                   │ threshold                            │
    │ BM25              │ 关键词精确匹配（专业术语、人名等）   │
    │ Ensemble          │ 融合多种策略，提高召回率             │
    │ MultiQuery        │ 一个问题生成多个查询变体             │
    └───────────────────┴──────────────────────────────────────┘
    
    【在 RAG 中的位置】
    
    用户问题 ──▶ ┌──────────┐ ──▶ 相关文档 ──▶ Prompt ──▶ LLM ──▶ 回答
                  │ Retriever│
                  └──────────┘
                    ↑
              「怎么搜」「搜什么」
              由检索策略决定
    """
    print('\n' + '=' * 70)
    print('Demo 5: Retrievers - 高级检索策略')
    print('=' * 70)
    documents = [Document(page_content='LangChain 是一个用于开发 LLM 应用的框架', metadata={'source': 'a'}), Document(page_content='RAG 通过检索增强生成来提高回答质量', metadata={'source': 'b'}), Document(page_content='向量数据库存储文本的嵌入向量', metadata={'source': 'c'}), Document(page_content='Prompt Template 管理提示词模板', metadata={'source': 'd'}), Document(page_content='Agent 可以让模型自主调用工具', metadata={'source': 'e'})]
    persist_dir = os.path.join(tempfile.gettempdir(), 'chroma_retriever')
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    vectorstore.add_documents(documents)
    print('\n[1] 基本相似度搜索 (similarity_search)')
    print('    最基础的检索方式，返回最相似的 K 个文档')
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})
    results = retriever.invoke('什么是 LangChain？')
    print(f"    查询: '什么是 LangChain？'")
    print(f'    返回 {len(results)} 个结果:\n')
    for i, doc in enumerate(results, 1):
        print(f'      [{i}] {doc.page_content}')
    print('\n[2] MMR 检索 (最大边际相关性)')
    print('    在保持相关性的同时，增加结果的多样性')
    print('    适用: 不想返回太多相似的内容，希望覆盖不同角度')
    mmr_retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 5})
    mmr_results = mmr_retriever.invoke('AI 应用开发')
    print(f"    查询: 'AI 应用开发'")
    print(f'    MMR 返回 {len(mmr_results)} 个结果:\n')
    for i, doc in enumerate(mmr_results, 1):
        print(f'      [{i}] {doc.page_content}')
    print('    💡 MMR 会尽量让返回的结果互不相同，避免重复')
    print('\n[3] 自定义检索器（添加元数据过滤）')
    print('    在实际生产中，我们经常需要按条件过滤搜索结果')

    def custom_retriever(query: str, top_k: int=2):
        """自定义检索函数，带元数据过滤"""
        return vectorstore.similarity_search(query, k=top_k, filter={'source': 'a'})
    custom_results = custom_retriever('LangChain')
    print(f"    自定义检索返回 {len(custom_results)} 个结果（已过滤 source='a'）")
    for doc in custom_results:
        print(f'      ✓ {doc.page_content}')
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)
    print('\n[NOTE] 检索策略选择指南:')
    print('   • 通用场景        → similarity（最简单，最常用）')
    print('   • 需要多样性      → MMR（避免结果重复）')
    print('   • 要求高质量      → similarity_score_threshold（过滤低质量）')
    print('   • 专有名词多      → BM25（精确关键词匹配）')
    print('   • 追求高召回      → Ensemble（混合策略）')

def demo_full_rag_pipeline():
    """
    演示如何使用 LCEL 搭建完整的 RAG Pipeline。
    
    【RAG 是什么？】
    RAG = Retrieval Augmented Generation（检索增强生成）
    
    核心理念：让大模型「开卷考试」
    • 传统方式：直接问 LLM，它只能凭训练时的记忆回答（可能过时、有幻觉）
    • RAG 方式：先检索相关知识，再把知识作为上下文一起交给 LLM
    
    类比：
    ┌──────────────────────────────────────────────────────────┐
    │  没有 RAG：考试闭卷，全靠记忆                              │
    │  有 RAG：考试开卷，可以查资料                              │
    │                                                          │
    │  RAG 带来的好处：                                          │
    │  ✅ 回答更准确（基于真实知识）                              │
    │  ✅ 可以减少幻觉（有依据可查）                              │
    │  ✅ 知识可以随时更新（改知识库就行，不用重训练）             │
    │  ✅ 可以追溯答案来源（知道参考了哪篇文档）                  │
    └──────────────────────────────────────────────────────────┘
    
    【RAG 完整流程图】
    
    用户提问
        │
        ▼
    ┌─────────────────────────────────────────────────────┐
    │  Step 1: Retriever（检索）                           │
    │  将问题转为向量，在知识库中搜索最相关的文档片段       │
    └──────────────────┬──────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────┐
    │  Step 2: Prompt Template（拼接）                     │
    │  将问题和检索到的文档组合成结构化的提示词             │
    └──────────────────┬──────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────┐
    │  Step 3: LLM（生成）                                │
    │  大模型基于检索到的知识，生成最终回答                 │
    └──────────────────┬──────────────────────────────────┘
                       │
                       ▼
                  最终答案
    
    【LCEL 管道构建】
    LangChain Expression Language (LCEL) 是一种声明式的管道构建方式：
    
    rag_chain = (
        {
            "context": retriever | format_docs,  # 检索并格式化文档
            "question": RunnablePassthrough()    # 透传用户问题
        }
        | rag_prompt                            # 拼接提示词
        | get_model()                           # 调用 LLM
        | StrOutputParser()                     # 提取纯文本
    )
    
    这个管道可以像函数一样调用：
    result = rag_chain.invoke("什么是 RAG？")
    """
    print('\n' + '=' * 70)
    print('Demo 6: Full RAG Pipeline - 端到端管道')
    print('=' * 70)
    print('\n[Step 1] 准备示例文档')
    print('        模拟一个小型技术知识库')
    documents = [Document(page_content='LangChain 是一个用于开发由大语言模型驱动的应用程序的开源框架。它提供了丰富的组件，包括提示模板、记忆管理、检索增强生成等。', metadata={'source': 'langchain_intro.txt', 'chapter': '介绍'}), Document(page_content='RAG（Retrieval Augmented Generation，检索增强生成）是一种结合检索系统和生成模型的技术。通过检索相关文档，LLM 可以生成更准确、更有依据的回答。', metadata={'source': 'rag_concept.txt', 'chapter': 'RAG'}), Document(page_content='向量数据库如 Chroma 是一个轻量级的向量数据库，支持持久化存储和高效的相似度搜索。它非常适合中小型项目的 RAG 应用。', metadata={'source': 'chroma_doc.txt', 'chapter': '向量库'}), Document(page_content='Prompt Template 用于管理提示词模板。通过使用 {variable} 占位符，可以动态注入不同的内容，实现灵活的提示词管理。', metadata={'source': 'prompt_template.txt', 'chapter': '提示词'})]
    for i, doc in enumerate(documents, 1):
        print(f"    [{i}] {doc.page_content[:40]}... | 来源: {doc.metadata['source']}")
    print('\n[Step 2] 创建向量库并添加文档')
    persist_dir = os.path.join(tempfile.gettempdir(), 'chroma_rag_pipeline')
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    vectorstore.add_documents(documents)
    print(f'    ✅ 已添加 {len(documents)} 个文档到向量库')
    print('\n[Step 3] 创建检索器')
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    print(f'    ✅ 检索器已创建 (k=2，返回最相关的 2 条)')
    print('\n[Step 4] 定义 RAG 提示词模板')
    print('        好的提示词模板是 RAG 效果的关键')
    rag_prompt = ChatPromptTemplate.from_messages([('system', '你是一个智能助手，擅长基于给定的上下文信息回答用户问题。\n\n回答要求：\n1. 基于提供的上下文信息回答，不要编造信息\n2. 如果上下文中没有相关信息，请诚实地告知用户\n3. 回答要简洁明了，不要太啰嗦\n4. 如果有多个来源，请综合各来源的信息\n\n【上下文信息】\n{context}\n\n【用户问题】\n{question}'), ('human', '{question}')])
    print('    ✅ 提示词模板已创建')
    print('    包含 system（角色设定+指令）和 human（用户问题）')
    print('\n[Step 5] 构建 LCEL RAG 管道')

    def format_docs(docs):
        return '\n\n'.join([f"[来源: {doc.metadata['source']}] {doc.page_content}" for doc in docs])
    rag_chain = {'context': retriever | format_docs, 'question': RunnablePassthrough()} | rag_prompt | get_model() | StrOutputParser()
    print('    ✅ RAG 管道已构建')
    print('    管道结构:')
    print('    {context: retriever|format, question: passthrough}')
    print('         ↓')
    print('    rag_prompt')
    print('         ↓')
    print('    ChatZhipuAI (智谱 AI LLM)')
    print('         ↓')
    print('    StrOutputParser → 最终文本')
    print('\n[Step 6] 执行 RAG 查询')
    print('        体验端到端的 RAG 回答流程\n')
    questions = ['什么是 LangChain？', 'RAG 技术是什么？', 'Chroma 向量数据库有什么特点？', '猪八戒娶媳妇讲的是什么']
    for question in questions:
        print(f"{'─' * 60}")
        print(f'👤 用户: {question}\n')
        result = rag_chain.invoke(question)
        print(f'🤖 助手: {result}\n')
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)
    print('\n' + '=' * 70)
    print('【RAG Pipeline 完整总结】')
    print('=' * 70)
    print('\n    完整 RAG Pipeline 的核心组件：\n\n    ┌────────────────────────────────────────────────────────────┐\n    │  Phase    │  组件              │  作用                    │\n    ├────────────────────────────────────────────────────────────┤\n    │  2-1      │ Document Loaders   │  从各种来源加载文档       │\n    │  2-2      │ Text Splitters     │  将长文档切分为合适小块    │\n    │  2-3      │ Embeddings         │  将文本转换为语义向量      │\n    │  2-4      │ Vector Stores      │  存储向量，支持相似度搜索  │\n    │  2-5      │ Retrievers         │  封装检索逻辑，提供统一接口│\n    │  2-6      │ LCEL Pipeline      │  组装完整问答链路          │\n    └────────────────────────────────────────────────────────────┘\n\n    LCEL 管道简写形式：\n      rag_chain = (\n          {"context": retriever | format_docs, "question": RunnablePassthrough()}\n          | rag_prompt\n          | llm\n          | StrOutputParser()\n      )\n\n    调用方式：\n      answer = rag_chain.invoke("用户问题")\n    ')
if __name__ == '__main__':
    print_lesson(__file__)
    print('\n    +----------------------------------------------------------+\n    |                                                          |\n    |       LangChain RAG Pipeline 完整教程                     |\n    |       (Phase 2-1 至 2-6 端到端演示)                       |\n    |                                                          |\n    |  覆盖内容:                                               |\n    |  Document Loaders / Text Splitters /                     |\n    |  Embeddings / Vector Stores / Retrievers /               |\n    |  Full RAG Pipeline                                       |\n    |                                                          |\n    +----------------------------------------------------------+\n    ')
    print_run_plan([
        ('端到端 RAG', 'Loader→Split→Embed→检索→生成。拆开的各环节函数仍在本文件，可单独调用'),
    ])
    check_api_key()
    try:
        demo_full_rag_pipeline()
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        print('\n提示: 检查网络连接和 ZHIPUAI_API_KEY 环境变量。')
        raise
    print('\n[SUCCESS] RAG Pipeline 教程全部完成！')
    print('\n📚 下一步学习建议:')
    print('   1. 动手修改代码，尝试不同的参数组合')
    print('   2. 用你自己的文档替换示例文档')
    print('   3. 探索更多的检索策略和调优技巧')
    print('   4. 学习如何部署到生产环境')