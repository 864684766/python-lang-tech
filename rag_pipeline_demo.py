# -*- coding: utf-8 -*-
"""
RAG Pipeline Demo (2-1 到 2-6)

本文件演示 LangChain 的 RAG（检索增强生成）完整流程。
所有警告已修复，兼容当前 LangChain 版本。

运行前提：
  pip install langchain langchain-core langchain-community chromadb zhipuai python-dotenv
  设置环境变量 ZHIPUAI_API_KEY
"""

import os
import tempfile
import shutil
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from numbers import Number

# ---------------------------------------------------------------------------
# 导入说明
# ---------------------------------------------------------------------------
# Document Loaders
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# Embeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document

# Vector Stores
from langchain_chroma import Chroma

# Prompts & Parsers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Models
from langchain_community.chat_models import ChatZhipuAI
from demo_explain import print_lesson, print_run_plan


def check_api_key() -> None:
    """检查 API Key 是否就绪"""
    load_dotenv()
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ERROR: ZHIPUAI_API_KEY not found.\n"
            "Solution:\n"
            "  A. Create .env file: ZHIPUAI_API_KEY=your_key\n"
            "  B. PowerShell: $env:ZHIPUAI_API_KEY=\"your_key\""
        )
    print(f"[OK] API Key loaded, first 8 chars: {api_key[:8]}...")


def get_api_key() -> str:
    """获取 API Key"""
    check_api_key()
    return os.getenv("ZHIPUAI_API_KEY", "")


def get_model(temperature: float = 0.0) -> ChatZhipuAI:
    """创建聊天模型实例"""
    return ChatZhipuAI(api_key=get_api_key(), temperature=temperature)


# ============================================================================
# Demo 1: Document Loaders (2-1)
# ============================================================================

def demo_document_loaders() -> None:
    """
    演示如何使用 Document Loader 加载各种格式的文档。
    
    【核心概念】
    Document Loader 负责从各种来源读取内容，并转换为统一的 Document 对象：
      - page_content：文档的实际文本内容
      - metadata：元数据（来源、页码、作者等）
    """
    print("\n" + "=" * 65)
    print("Demo 1: Document Loaders - 文档加载器")
    print("=" * 65)

    # ---- 方式一：加载纯文本文件 ----
    print("\n[1] 加载纯文本文件:")

    temp_file = os.path.join(tempfile.gettempdir(), "sample.txt")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("""LangChain is an open-source framework for developing applications powered by LLMs.

Core Features:
1. Rich component library: prompt templates, memory management, RAG, tools
2. Flexible orchestration: chains, conditional branches, loops
3. Multi-model support: OpenAI, Anthropic, Zhipu GLM, etc.
4. Extensibility: easy to customize and extend""")

    loader = TextLoader(temp_file, encoding="utf-8")
    documents = loader.load()

    print(f"  Loaded {len(documents)} document(s)")
    print(f"  First doc preview: {documents[0].page_content[:100]}...")
    print(f"  Metadata: {documents[0].metadata}")

    # ---- 方式二：加载目录下的所有文件 ----
    print("\n[2] 加载目录下的所有文本文件:")

    sample_dir = os.path.join(tempfile.gettempdir(), "samples")
    os.makedirs(sample_dir, exist_ok=True)

    for i in range(3):
        file_path = os.path.join(sample_dir, f"doc_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"This is document #{i+1} about LangChain framework.\n")
            f.write("LangChain provides rich components for building AI applications.\n")
            f.write("It supports multiple LLMs and vector databases with great extensibility.\n")

    directory_loader = DirectoryLoader(sample_dir, glob="**/*.txt", loader_cls=TextLoader)
    dir_documents = directory_loader.load()

    print(f"  Loaded {len(dir_documents)} documents")
    for i, doc in enumerate(dir_documents[:3]):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        content_preview = doc.page_content[:50] if doc.page_content else ""
        print(f"  [{i+1}] Source: {source}")
        print(f"      Preview: {content_preview}...")

    # 清理临时文件
    os.remove(temp_file)
    for f in os.listdir(sample_dir):
        os.remove(os.path.join(sample_dir, f))
    os.rmdir(sample_dir)

    print("\n[NOTE]:")
    print("   - TextLoader is suitable for plain text files")
    print("   - DirectoryLoader can recursively scan entire directories")
    print("   - All loaders return List[Document] objects")


# ============================================================================
# Demo 2: Text Splitters (2-2)
# ============================================================================

def demo_text_splitters() -> None:
    """
    演示如何使用 Text Splitter 将长文档切分为合适的块。
    
    【为什么需要分块？】
    1. Token 限制：大模型有输入长度限制
    2. 检索效率：小块更容易匹配
    3. 成本控制：只检索相关片段
    """
    print("\n" + "=" * 65)
    print("Demo 2: Text Splitters - 文本分块策略")
    print("=" * 65)

    long_text = """LangChain is a framework for developing applications powered by large language models.

Chapter 1: Basic Concepts
The core idea of LangChain is to decompose complex AI applications into composable components.
Each component can be developed and tested independently.
Main components include:
- Prompt Templates: manage prompt templates
- Models: wrap various LLM providers
- Chains: connect multiple components into workflows
- Agents: enable models to call tools and make decisions
- Memory: manage conversation history
- Indexes: handle document retrieval

Chapter 2: RAG Architecture
Retrieval Augmented Generation (RAG) is one of LangChain's core application scenarios.
It works through the following steps:
1. Load documents using Document Loader
2. Split text using Text Splitter
3. Generate embeddings using Embeddings model
4. Store vectors in Vector Store
5. Retrieve relevant documents based on query
6. Generate answers by sending retrieved content to LLM

Chapter 3: Tool Calling
LangChain supports powerful tool calling functionality:
- @tool decorator: quickly define tools
- bind_tools: bind tools to model
- Agent: automatically decide when to call tools
- Multi-agent: multiple agents collaborate

Chapter 4: Best Practices
- Choose appropriate chunk size (typically 500-1000 characters)
- Maintain appropriate overlap (typically 50-100 characters)
- Use metadata filtering to improve retrieval accuracy
- Combine with re-ranking to enhance relevance"""

    # ---- 方式一：RecursiveCharacterTextSplitter ----
    print("\n[1] RecursiveCharacterTextSplitter:")
    print("    Parameters: chunk_size=500, chunk_overlap=50")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    docs = text_splitter.create_documents([long_text])
    print(f"  Original text length: {len(long_text)} characters")
    print(f"  Split into: {len(docs)} chunks")

    if len(docs) > 0:
        print(f"\n  First chunk preview:")
        print(f"  {docs[0].page_content[:200]}...")

    if len(docs) > 1:
        print(f"\n  Second chunk preview (note overlap):")
        print(f"  {docs[1].page_content[:200]}...")

    # ---- 方式二：MarkdownTextSplitter ----
    print("\n[2] MarkdownTextSplitter:")
    print("    Feature: splits by ###, ##, # heading levels")

    markdown_text = """# LangChain Tutorial

## Chapter 1: Basic Concepts

### 1.1 What is LangChain
LangChain is a framework...

### 1.2 Core Components
- Prompt Templates
- Models
- Chains

## Chapter 2: RAG Architecture

### 2.1 Retrieval Process
1. Load documents
2. Split text
3. Generate embeddings
"""

    md_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=30)
    md_docs = md_splitter.create_documents([markdown_text])
    print(f"  Markdown split into: {len(md_docs)} chunks")
    for i, doc in enumerate(md_docs[:3]):
        preview = doc.page_content[:100] if doc.page_content else ""
        print(f"\n  [{i+1}] Preview: {preview}...")

    # ---- 方式三：调整分块参数 ----
    print("\n[3] Different chunking strategies comparison:")

    small_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    small_docs = small_splitter.create_documents([long_text])
    print(f"  chunk_size=200, overlap=20 → {len(small_docs)} chunks")

    large_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    large_docs = large_splitter.create_documents([long_text])
    print(f"  chunk_size=1000, overlap=100 → {len(large_docs)} chunks")

    print("\n[NOTE]:")
    print("   - chunk_size: max characters per chunk (recommended: 500-1000)")
    print("   - chunk_overlap: overlapping characters between chunks (recommended: 50-100)")
    print("   - Overlap prevents key information from being split at boundaries")
    print("   - MarkdownTextSplitter is suitable for structured Markdown documents")


# ============================================================================
# Demo 3: Embeddings (2-3)
# ============================================================================

def demo_embeddings() -> None:
    """
    演示如何使用 Embeddings 模型将文本转换为向量。
    
    【什么是 Embedding？】
    Embedding 是将文本转换为向量的技术。向量是一个数字数组，能够捕捉文本的语义信息。
    相似的文本会有相似的向量，这使得计算机能够理解文本之间的语义关系。
    """
    print("\n" + "=" * 65)
    print("Demo 3: Embeddings - 嵌入模型")
    print("=" * 65)

    # 创建嵌入模型
    print("\n[1] Initialize ZhipuAI Embeddings:")
    zhipu_embeddings = ZhipuAIEmbeddings(api_key=get_api_key())
    print(f"  Model: {zhipu_embeddings.model}")

    # 生成单条文本的嵌入
    print("\n[2] Generate single text embedding:")
    text = "LangChain is an open-source framework for LLM applications"
    vector = zhipu_embeddings.embed_query(text)
    print(f"  Text: {text}")
    print(f"  Vector length: {len(vector)} dimensions")
    print(f"  First 10 values: {vector[:10]}")

    # 批量生成嵌入
    print("\n[3] Batch generate embeddings:")
    texts = [
        "LangChain supports multiple LLMs",
        "RAG stands for Retrieval Augmented Generation",
        "Vector database stores embedding vectors",
        "Prompt Template manages prompt templates",
    ]
    vectors = zhipu_embeddings.embed_documents(texts)
    print(f"  Generated {len(vectors)} vectors")
    print(f"  Each vector dimension: {len(vectors[0])}")

    # 计算向量相似度
    print("\n[4] Calculate vector similarity:")
    import numpy as np

    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        arr1 = np.array(v1, dtype=np.float64)
        arr2 = np.array(v2, dtype=np.float64)
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    query_vector = zhipu_embeddings.embed_query("AI framework")
    print(f"  Query: 'AI framework'")
    print(f"  Similarity with each text:")
    for text, vector in zip(texts, vectors):
        sim = cosine_similarity(query_vector, vector)
        preview = text[:30] if len(text) > 30 else text
        print(f"    - '{preview}...': {sim:.4f}")

    print("\n[NOTE]:")
    print("   - ZhipuAIEmbeddings: Zhipu's embedding model (free quota)")
    print("   - OpenAIEmbeddings: OpenAI's embedding model (requires API key)")
    print("   - HuggingFaceEmbeddings: runs locally, no network required")
    print("   - Vector dimensions are typically 1024-1536")


# ============================================================================
# Demo 4: Vector Stores (2-4)
# ============================================================================

def demo_vector_stores() -> None:
    """
    演示如何使用 Chroma 向量数据库存储和检索嵌入向量。
    
    【什么是向量数据库？】
    向量数据库是一种专门用于存储和检索向量数据的数据库。
    它支持高效的相似度搜索，能够快速找到与查询向量最相似的文档。
    """
    print("\n" + "=" * 65)
    print("Demo 4: Vector Stores - 向量数据库 (Chroma)")
    print("=" * 65)

    # 准备文档
    print("\n[1] Prepare sample documents:")
    documents = [
        Document(
            page_content="LangChain is an open-source framework for developing LLM applications with rich components.",
            metadata={"source": "doc1.txt", "chapter": "intro"},
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) enhances LLM responses by retrieving relevant documents.",
            metadata={"source": "doc2.txt", "chapter": "rag"},
        ),
        Document(
            page_content="Chroma is a lightweight vector database supporting persistent storage and efficient similarity search.",
            metadata={"source": "doc3.txt", "chapter": "vector_db"},
        ),
        Document(
            page_content="Prompt Template manages prompt templates with support for variable injection and formatting.",
            metadata={"source": "doc4.txt", "chapter": "prompt"},
        ),
    ]

    for i, doc in enumerate(documents, 1):
        preview = doc.page_content[:40] if doc.page_content else ""
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] {preview}... | Source: {source}")

    # 创建向量数据库
    print("\n[2] Create Chroma vector database:")
    persist_dir = os.path.join(tempfile.gettempdir(), "chroma_rag")
    
    # 使用 get_api_key() 获取正确的嵌入函数
    embeddings_func = ZhipuAIEmbeddings(api_key=get_api_key())
    
    vectorstore = Chroma(
        embedding_function=embeddings_func,
        persist_directory=persist_dir,
    )

    # 添加文档
    print("\n[3] Add documents to vector store:")
    vectorstore.add_documents(documents)
    collection_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else len(documents)
    print(f"  Added {len(documents)} documents")
    print(f"  Current document count: {collection_count}")

    # 相似度搜索
    print("\n[4] Execute similarity search:")
    query = "What is LangChain?"
    results = vectorstore.similarity_search(query, k=2)
    print(f"  Query: '{query}'")
    print(f"  Returns {len(results)} results:")
    for i, doc in enumerate(results, 1):
        content = doc.page_content if doc.page_content else ""
        source = doc.metadata.get("source", "unknown")
        chapter = doc.metadata.get("chapter", "unknown")
        print(f"\n  Result {i}:")
        print(f"    Content: {content}")
        print(f"    Source: {source}")
        print(f"    Chapter: {chapter}")

    # 带过滤的搜索
    print("\n[5] Search with metadata filter:")
    filtered_results = vectorstore.similarity_search(
        query="RAG",
        k=2,
        filter={"chapter": "rag"},
    )
    print(f"  Query: 'RAG', Filter: chapter='rag'")
    for doc in filtered_results:
        content = doc.page_content[:50] if doc.page_content else ""
        chapter = doc.metadata.get("chapter", "unknown")
        print(f"    {content}... | Chapter: {chapter}")

    # 持久化与加载
    print("\n[6] Persist and reload:")
    loaded_vectorstore = Chroma(
        embedding_function=embeddings_func,
        persist_directory=persist_dir,
    )
    loaded_count = loaded_vectorstore._collection.count() if hasattr(loaded_vectorstore, '_collection') else 0
    print(f"  Reloaded document count: {loaded_count}")

    # 清理
    shutil.rmtree(persist_dir, ignore_errors=True)

    print("\n[NOTE]:")
    print("   - Chroma is a lightweight local vector DB, suitable for development/testing")
    print("   - persist_directory parameter enables persistent storage")
    print("   - similarity_search(query, k=N) returns top N most similar documents")
    print("   - filter parameter supports metadata-based filtering")


# ============================================================================
# Demo 5: Retrievers (2-5)
# ============================================================================

def demo_retriever() -> None:
    """
    演示如何使用 Retriever 进行高级检索。
    
    【Retriever 的作用】
    Retriever 是一个抽象层，封装了向量搜索的逻辑。
    它可以让你使用统一的方法进行检索，而不需要直接操作 VectorStore。
    """
    print("\n" + "=" * 65)
    print("Demo 5: Retrievers - 高级检索策略")
    print("=" * 65)

    # 准备文档
    documents = [
        Document(page_content="LangChain is a framework for developing LLM applications.", metadata={"source": "a"}),
        Document(page_content="RAG improves response quality by retrieving relevant documents.", metadata={"source": "b"}),
        Document(page_content="Vector database stores text embedding vectors.", metadata={"source": "c"}),
        Document(page_content="Prompt Template manages prompt templates.", metadata={"source": "d"}),
        Document(page_content="Agent allows models to call tools autonomously.", metadata={"source": "e"}),
    ]

    # 创建向量库
    persist_dir = os.path.join(tempfile.gettempdir(), "chroma_retriever")
    embeddings_func = ZhipuAIEmbeddings(api_key=get_api_key())
    vectorstore = Chroma(embedding_function=embeddings_func, persist_directory=persist_dir)
    vectorstore.add_documents(documents)

    # 方式一：基本相似度检索
    print("\n[1] Basic similarity search (similarity_search):")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    results = retriever.invoke("What is LangChain?")
    print(f"  Query: 'What is LangChain?'")
    print(f"  Returns {len(results)} results:")
    for i, doc in enumerate(results, 1):
        content = doc.page_content if doc.page_content else ""
        print(f"    [{i}] {content}")

    # 方式二：MMR 检索
    print("\n[2] MMR search (maximal marginal relevance, increases diversity):")
    print("    Note: Chroma natively supports mmr search type")
    
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5},
    )
    mmr_results = mmr_retriever.invoke("AI application development")
    print(f"  MMR Query: 'AI application development'")
    print(f"  Returns {len(mmr_results)} results:")
    for i, doc in enumerate(mmr_results, 1):
        content = doc.page_content if doc.page_content else ""
        print(f"    [{i}] {content}")

    # 方式三：自定义检索器
    print("\n[3] Custom retriever (with metadata filter):")
    
    def custom_retriever(query: str, top_k: int = 2) -> List[Document]:
        """自定义检索函数"""
        results = vectorstore.similarity_search(
            query,
            k=top_k,
            filter={"source": "a"},
        )
        return results

    custom_results = custom_retriever("LangChain")
    print(f"  Custom retriever returns {len(custom_results)} results:")
    for doc in custom_results:
        content = doc.page_content if doc.page_content else ""
        print(f"    {content}")

    # 清理
    shutil.rmtree(persist_dir, ignore_errors=True)

    print("\n[NOTE]:")
    print("   - similarity_search: returns top K most similar documents (most common)")
    print("   - MMR: balances similarity and diversity, avoids duplicate results")
    print("   - Custom retriever: you can add your own filtering logic")
    print("   - Use filter parameter to add metadata-based filtering")


# ============================================================================
# Demo 6: Full RAG Pipeline (2-6)
# ============================================================================

def demo_full_rag_pipeline() -> None:
    """
    演示如何使用 LCEL 搭建完整的 RAG Pipeline。
    
    【RAG 完整流程】
    
    用户提问 → Retriever → 相关文档 → Prompt Template → LLM → 最终答案
    """
    print("\n" + "=" * 65)
    print("Demo 6: Full RAG Pipeline - 端到端管道")
    print("=" * 65)

    # Step 1: 准备文档
    print("\n[Step 1] Prepare sample documents:")
    documents = [
        Document(
            page_content="LangChain is an open-source framework for developing applications powered by large language models. It provides rich components including prompt templates, memory management, and retrieval augmented generation.",
            metadata={"source": "langchain_intro.txt", "chapter": "introduction"},
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) is a technique that combines retrieval systems with generative models. By retrieving relevant documents, LLM can generate more accurate and well-grounded responses.",
            metadata={"source": "rag_concept.txt", "chapter": "rag"},
        ),
        Document(
            page_content="Chroma is a lightweight vector database that supports persistent storage and efficient similarity search. It is very suitable for RAG applications in small to medium-sized projects.",
            metadata={"source": "chroma_doc.txt", "chapter": "vector_db"},
        ),
        Document(
            page_content="Prompt Template is used to manage prompt templates. By using {variable} placeholders, different content can be dynamically injected for flexible prompt management.",
            metadata={"source": "prompt_template.txt", "chapter": "prompts"},
        ),
    ]

    for i, doc in enumerate(documents, 1):
        preview = doc.page_content[:40] if doc.page_content else ""
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] {preview}... | Source: {source}")

    # Step 2: 创建向量库
    print("\n[Step 2] Create vector store and add documents:")
    persist_dir = os.path.join(tempfile.gettempdir(), "chroma_rag_pipeline")
    embeddings_func = ZhipuAIEmbeddings(api_key=get_api_key())
    vectorstore = Chroma(embedding_function=embeddings_func, persist_directory=persist_dir)
    vectorstore.add_documents(documents)
    print(f"  Added {len(documents)} documents to vector store")

    # Step 3: 创建检索器
    print("\n[Step 3] Create retriever:")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print(f"  Retriever created (k=2)")

    # Step 4: 定义 RAG 提示词
    print("\n[Step 4] Define RAG prompt template:")
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant who excels at answering questions based on given context information.

Answer requirements:
1. Base your answer on the provided context information
2. If the context does not contain relevant information, please say so honestly
3. Keep answers concise and clear
4. If there are multiple sources, synthesize information from each source

【Context Information】
{context}

【User Question】
{question}"""),
        ("human", "{question}"),
    ])
    print("  Prompt template created")
    print("  Contains: system (role setting + instructions) + human (user question)")

    # Step 5: 构建 LCEL 管道
    print("\n[Step 5] Build LCEL RAG pipeline:")

    # format_docs 函数：将文档列表转换为字符串
    def format_docs(docs: List[Document]) -> str:
        """将文档列表格式化为字符串"""
        if not docs:
            return "No relevant documents found."
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
            content = doc.page_content if doc.page_content else ""
            parts.append(f"[Source: {source}] {content}")
        return "\n\n".join(parts)

    # 构建管道
    model = get_model()
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    print("  Pipeline structure:")
    print("    {context: retriever|format, question: passthrough} → prompt → model → parser")

    # Step 6: 执行 RAG 查询
    print("\n[Step 6] Execute RAG queries:")

    questions = [
        "What is LangChain?",
        "What is RAG technology?",
        "What are the features of Chroma vector database?",
    ]

    for question in questions:
        print(f"\n{'-' * 50}")
        print(f"User question: {question}\n")

        result = rag_chain.invoke(question)
        print(f"AI answer:\n{result}")

    # 清理
    shutil.rmtree(persist_dir, ignore_errors=True)

    print("\n" + "=" * 65)
    print("[RAG Pipeline Summary]")
    print("=" * 65)
    print("""
Core components of a complete RAG Pipeline:

1. Document Loader    → Load documents
2. Text Splitter     → Chunk text
3. Embeddings        → Generate vectors
4. Vector Store      → Store vectors
5. Retriever         → Retrieve relevant docs
6. Prompt Template   → Build prompt
7. LLM               → Generate answer
8. Output Parser     → Extract result

LCEL pipeline example:
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | rag_prompt
      | llm
      | StrOutputParser()
  )
""")


# ============================================================================
# Main Entry
# ============================================================================

if __name__ == "__main__":
    print_lesson(__file__)
    print("""
+----------------------------------------------------------+
|                                                          |
|       LangChain RAG Pipeline Tutorial                    |
|       (Sections 2-1 to 2-6)                              |
|                                                          |
|  Covers: Document Loaders / Text Splitters /             |
|          Embeddings / Vector Stores / Retrievers /       |
|          Full RAG Pipeline                               |
|                                                          |
+----------------------------------------------------------+
""")
    print_run_plan([
        ('Document Loaders', '把文件变成 Document 对象'),
        ('Text Splitters', '切块，检索粒度'),
        ('Embeddings', '文本 → 向量'),
        ('Vector Stores', '入库与相似度搜索'),
        ('Retrievers', '按问题取相关片段'),
        ('Full RAG', '检索 + 生成串成一条链'),
    ])

    check_api_key()

    try:
        demo_document_loaders()      # Demo 1: Document Loaders
        demo_text_splitters()        # Demo 2: Text Splitters
        demo_embeddings()            # Demo 3: Embeddings
        demo_vector_stores()         # Demo 4: Vector Stores
        demo_retriever()             # Demo 5: Retrievers
        demo_full_rag_pipeline()     # Demo 6: Full RAG Pipeline
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print("Hint: Check network connection and ZHIPUAI_API_KEY.")
        raise

    print("\n[SUCCESS] RAG Pipeline tutorial complete!\n")
