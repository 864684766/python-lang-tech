"""
========================================================

LangChain Phase 2-4

Vector Stores 教材级 Demo
（完整版注释 + 应用场景说明）


学习目标：
1. 理解 Vector Store 是什么、解决什么问题
2. 掌握 FAISS 和 Chroma 两个主流向量库的使用
3. 理解持久化存储的意义和用法
4. 理解 Metadata 过滤的原理
5. 理解 VectorStore → Retriever 的转换机制
6. 为 RAG 和 Agent 打基础


🔗 知识脉络：
┌─────────────────────────────────────────────────────────────┐
│  Embedding 之后发生了什么？                                 │
│                                                             │
│  文本 ──▶ Embedding 模型 ──▶ 向量 [0.1, 0.5, ...]          │
│                                                     │       │
│                              问题来了：                   │
│                              "向量存在哪里？怎么快速搜索？" │
│                                                     │       │
│                              ┌──────────────┐          │
│                              │ Vector Store │          │
│                              │  (向量数据库) │          │
│                              └──────┬───────┘          │
│                                     │                  │
│                          相似度搜索：                    │
│                          "哪个文档最像我的问题？"        │
└─────────────────────────────────────────────────────────────┘


常见 Vector Store 对比：
┌──────────┬───────────┬───────────┬──────────────────┐
│  名称    │  部署方式 │  适合场景 │     特点         │
├──────────┼───────────┼───────────┼──────────────────┤
│ FAISS    │ 本地内存  │ 中小规模  │ Facebook 开源，  │
│          │           │ 开发测试  │ 速度极快         │
├──────────┼───────────┼───────────┼──────────────────┤
│ Chroma   │ 本地文件  │ 快速原型  │ 极简 API，自动   │
│          │           │ 教学演示  │ 持久化，metadata │
│          │           │           │ 过滤很方便       │
├──────────┼───────────┼───────────┼──────────────────┤
│ Pinecone │ 云端托管  │ 生产环境  │ 免运维，按量付费 │
├──────────┼───────────┼───────────┼──────────────────┤
│ Milvus   │ 本地/云端 │ 大规模    │ 分布式，支持亿级 │
└──────────┴───────────┴───────────┴──────────────────┘


========================================================
"""

from typing import List
from pathlib import Path

from langchain_core.documents import Document
from dotenv import load_dotenv  # 从 .env 文件中自动加载环境变量

# ======================================================
# 📖 Demo0 - Vector Store 概念入门
#
# 【前因】为什么需要 Vector Store？
#   - Embedding 产生了向量，但向量不能放着不管
#   - 我们需要一个地方存起来，方便以后搜索
#   - 普通数据库不支持向量相似度搜索（那是 Vector Store 的事）
#
# 【核心职责】
#   1. 保存向量            — 把向量存到磁盘或内存
#   2. 保存原始文本        — 记住向量对应的是哪段文字
#   3. 保存元数据          — 记录来源、分类等附加信息
#   4. 快速相似度搜索      — 给定一个查询向量，找最相似的 N 个
#
# 【实际例子】
#   你的公司知识库有 10000 篇技术文档
#   每篇文档都被转换成向量，存在 FAISS 里
#   当用户问"怎么配置 API Key"时
#   FAISS 快速找到最相关的 3 篇文档返回给你
# ======================================================

def demo_0_vector_store_concept() -> None:
    """
    介绍 Vector Store 的作用。

    数据流示意：

    ┌─────────────┐       ┌──────────────────┐       ┌─────────────┐
    │   原始文档    │ ───▶ │  Embedding 模型    │ ───▶ │   向量数据   │
    │             │       │  (768维浮点数)     │       │             │
    │ "LangChain" │       └──────────────────┘       │  FAISS /    │
    │ "Python..." │                                    │  Chroma /   │
    └─────────────┘                                    │  Milvus ... │
                                                     └──────┬──────┘
                                                            │
                                                    用户提问时：
                                                    "什么是 LangChain？"
                                                            │
                                                    ┌──────▼──────┐
                                                    │ 相似度搜索    │
                                                    │ 返回相关文档  │
                                                    └─────────────┘
    """

    print("\n========== Demo0: Vector Store 概念 ==========")

    print("""
    📌 什么是 Vector Store？
    ─────────────────────────────────────────
    
    Vector Store = 向量数据库
    专门用于存储和检索「向量」的数据库
    
    它做的事情很简单：
    1. 存：把 [文本 + 向量 + 元数据] 一起保存
    2. 搜：给一个查询向量，找出最相似的 K 条记录
    
    ─────────────────────────────────────────
    数据长什么样？
    ─────────────────────────────────────────
    
    文档："LangChain 是什么？"
    
    对应的向量：
    [
      0.123, -0.456, 0.789, ...   ← 几百上千个浮点数
    ]
    
    存储结构（概念）：
    ┌──────┬─────────────────────────┬──────────────────┐
    │ ID   │ 文本内容                 │ 向量              │
    ├──────┼─────────────────────────┼──────────────────┤
    │  1   │ "LangChain 是什么？"    │ [0.12, -0.45...] │
    │  2   │ "Python 编程语言"       │ [0.33, 0.67...]  │
    │  3   │ "RAG 检索增强生成"      │ [-0.11, 0.88...] │
    └──────┴─────────────────────────┴──────────────────┘
    
    💡 关键点：
    • 向量是「语义指纹」——相似内容的向量在空间中距离近
    • 搜索本质上是「找最近的邻居」（Nearest Neighbor Search）
    • FAISS/Chroma 内部用了高效的近似最近邻算法，不用逐个比较
    """)


# ======================================================
# 📖 Demo1 - 创建 Document 对象
#
# 【为什么要先了解 Document？】
#   Vector Store 不是直接存字符串的！
#   它存的是 LangChain 的 Document 对象
#   Document = page_content（正文）+ metadata（元数据）
#
# 【metadata 有什么用？】
#   比如你有员工手册（HR 部门）和技术文档（研发部门）
#   给每篇文档打上 metadata={"department": "HR"} 或 {"department": "R&D"}
#   搜索时就可以过滤："只搜 HR 部门的文档"
#
# 【page_content vs metadata】
#   page_content: 会被 Embedding，变成向量（参与相似度计算）
#   metadata:     不参与向量计算，只用于过滤和展示来源
# ======================================================

def demo_1_create_documents() -> List[Document]:
    """
    创建 LangChain Document 对象。
    
    Document 是 LangChain 的核心数据单元，包含两个关键部分：
    
    ┌─────────────────────────────────────────────┐
    │  page_content: 实际文本内容                  │
    │    → 会被 Embedding 模型转成向量              │
    │    → 参与相似度计算                          │
    │                                             │
    │  metadata:     附加信息（字典格式）           │
    │    → 不参与向量计算                          │
    │    → 可用于过滤、溯源、展示来源               │
    └─────────────────────────────────────────────┘
    
    返回:
        Document 列表，供后续 Demo 使用
    """

    print("\n========== Demo1: 创建 Document 对象 ==========")

    documents: List[Document] = [
        Document(
            page_content="LangChain 是用于开发 LLM 应用的框架",
            metadata={"source": "langchain.txt", "category": "AI"}
        ),
        Document(
            page_content="Python 是一种非常流行的编程语言",
            metadata={"source": "python.txt", "category": "Programming"}
        ),
        Document(
            page_content="RAG 结合知识库增强大模型回答能力",
            metadata={"source": "rag.txt", "category": "AI"}
        ),
    ]

    print(f"📚 创建了 {len(documents)} 个 Document 对象\n")

    for idx, doc in enumerate(documents, 1):
        print(f"【文档 {idx}】")
        print(f"  📝 page_content: {doc.page_content}")
        print(f"  🏷️  metadata:    {doc.metadata}")
        print()

    print("💡 注意：page_content 参与向量计算，metadata 只用于过滤和标识")
    return documents


# ======================================================
# 📖 Demo2 - FAISS 基础用法
#
# 【FAISS 是什么？】
#   FAISS (Facebook AI Similarity Search) 是 Facebook 开源的
#   高效向量搜索引擎，支持 GPU 加速
#
# 【适用场景】
#   - 开发阶段快速验证
#   - 中等规模数据（万到百万级向量）
#   - 需要部署到生产环境但预算有限
#
# 【基本流程】
#   Step 1: 准备 Document 列表
#   Step 2: 调用 FAISS.from_documents(documents, embedding_model)
#           内部完成：Embedding → 建立索引
#   Step 3: 调用 similarity_search(query, k=N)
#           内部完成：Query Embedding → 向量搜索 → 返回结果
#
# 【关键参数】
#   k: 返回最相似的 K 个结果（默认 4）
# ======================================================

def demo_2_faiss_basic(documents: List[Document]) -> None:
    """
    演示 FAISS 的基础使用。
    
    FAISS 是最常用的轻量级向量库之一。
    
    使用流程：
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Document 列表 │──▶│ from_documents │──▶│ similarity_  │
    │  + Embedding │    │  (建索引)     │    │ search (搜)  │
    └──────────────┘    └──────────────┘    └──────────────┘
    
    ⚠️ 教学使用 FakeEmbedding，实际项目请替换为真实模型。
    """

    print("\n========== Demo2: FAISS 基础 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    # 教学使用 FakeEmbedding 模拟向量生成
    # 实际项目替换为：OllamaEmbeddings / ZhipuAIEmbeddings / OpenAIEmbeddings
    embedding_model = FakeEmbeddings(size=8)

    # from_documents 一步完成：Embedding + 建立索引
    # 等价于：
    #   vectors = embedding_model.embed_documents(texts)
    #   faiss = FAISS(vectors, embedding_model, documents)
    vector_store = FAISS.from_documents(documents, embedding_model)

    print("✅ FAISS 索引创建成功")

    # 用户查询
    query: str = "你的爱好是什么？"
    print(f"\n❓ 用户查询: \"{query}\"")

    # 执行相似度搜索，返回最相似的 2 条
    results: List[Document] = vector_store.similarity_search(query, k=2)

    print(f"\n📊 搜索结果 (k=2):")
    for index, document in enumerate(results, start=1):
        print(f"\n  【结果 {index}】")
        print(f"  内容: {document.page_content}")
        print(f"  来源: {document.metadata.get('source', 'N/A')}")


# ======================================================
# 📖 Demo3 - FAISS 相似度分数
#
# 【为什么要看分数？】
#   光知道"哪个文档最相关"还不够
#   我们需要知道"有多相关"
#   这有助于：
#   1. 设置阈值：分数太低就认为不相关，不返回给用户
#   2. 调试优化：观察搜索质量，调整 chunk_size 等参数
#   3. 用户体验：告诉用户"这是根据您的搜索找到最相关的 3 条结果"
#
# 【分数含义】
#   similarity_search_with_score 返回 (Document, score)
#   score 越小 = 距离越近 = 越相似（FAISS 默认返回欧氏距离的平方）
#   如果用余弦相似度则是越大越好（0~1）
#
# 【实际应用】
#   客服机器人：只有相似度 > 0.8 的结果才当作有效答案
#   低于阈值的直接回答"抱歉，我在知识库中没找到相关内容"
# ======================================================

def demo_3_faiss_similarity_score(documents: List[Document]) -> None:
    """
    演示查看搜索结果的相似度分数。
    
    相比 similarity_search，这个方法额外返回每个结果的评分。
    对于判断"搜索结果是否靠谱"非常有用。
    """

    print("\n========== Demo3: FAISS 相似度分数 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)
    vector_store = FAISS.from_documents(documents, embedding_model)

    # similarity_search_with_score 返回 (document, score) 对
    results = vector_store.similarity_search_with_score("今天星期几", k=3)

    print(f"❓ 查询: \"LangChain 是什么\"\n")

    for idx, (document, score) in enumerate(results, 1):
        print(f"  【结果 {idx}】")
        print(f"  文本: {document.page_content}")
        print(f"  距离分数: {float(score):.6f}  （越小表示越相似）")
        print()

    print("💡 应用建议：设置 score_threshold 过滤低质量结果")


# ======================================================
# 📖 Demo4 - Chroma 基础
#
# 【Chroma vs FAISS 的区别】
#   FAISS:
#     - 侧重纯向量搜索性能
#     - 需要手动管理持久化
#     - 元数据过滤支持稍弱
#   
#   Chroma:
#     - API 更简洁易用
#     - 原生支持持久化（一行代码搞定）
#     - metadata 过滤功能强大（支持 AND/OR/操作符）
#     - 内置 SQLite 做元数据存储
#
# 【Chroma 的使用场景】
#   - 教学演示（代码最少）
#   - 中小型项目（< 100 万向量）
#   - 需要便捷 metadata 过滤的场景
# ======================================================

def demo_4_chroma_basic(documents: List[Document]) -> None:
    """
    演示 Chroma 向量数据库的基础使用。
    
    Chroma 是目前最受欢迎的轻量级向量库，API 极其简洁。
    
    三大优势：
    1. 极简 API — from_documents 一行搞定
    2. 自动持久化 — persist_directory 参数指定即可
    3. 强大的 metadata 过滤 — 支持复杂条件查询
    """

    print("\n========== Demo4: Chroma 基础 ==========")

    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)

    # Chroma 与 FAISS 用法几乎一样，但后续持久化更方便
    chroma_db = Chroma.from_documents(documents, embedding_model)

    print("✅ Chroma 向量库创建成功")

    query = "如何增强大模型能力？"
    print(f"\n❓ 查询: \"{query}\"")

    results = chroma_db.similarity_search(query, k=2)

    print(f"\n📊 搜索结果:")
    for document in results:
        print(f"  • {document.page_content}")
        print(f"    来源: {document.metadata.get('source', 'N/A')}")


# ======================================================
# 📖 Demo5 - FAISS 本地持久化保存
#
# 【为什么需要持久化？】
#   想象这个场景：
#   
#   你有一个 10 万条的技术文档知识库
#   每次程序启动都要重新 Embedding → 这太慢了！
#   而且 Embedding API 调用也花钱 😢
#   
#   解决办法：把索引保存到磁盘，下次直接加载
#
# 【保存什么？】
#   - 向量数据（最重要）
#   - Document 的 page_content
#   - Document 的 metadata
#   - 索引结构（加速搜索）
#
# 【保存格式】
#   FAISS 保存为 .bin 索引文件 + pickle 序列化的 Document 数据
# ======================================================

def demo_5_faiss_save(documents: List[Document]) -> None:
    """
    演示将 FAISS 索引保存到本地磁盘。
    
    为什么要保存？这是一个很重要的工程实践！
    
    ┌─────────────────────────────────────────────────────┐
    │  没有持久化的问题：                                   │
    │  每次启动程序 → 重新 Embedding 所有文档 → 耗时耗钱   │
    │                                                     │
    │  有了持久化：                                        │
    │  首次：文档 → Embedding → 保存索引                   │
    │  之后：直接加载索引 → 立刻能搜索                      │
    └─────────────────────────────────────────────────────┘
    """

    print("\n========== Demo5: FAISS 本地保存 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)

    # 创建并保存
    vector_store = FAISS.from_documents(documents, embedding_model)

    save_path = Path("./faiss_db")
    vector_store.save_local(str(save_path))

    print(f"✅ FAISS 索引已保存到: {save_path.absolute()}")
    print(f"   包含 {vector_store.index.ntotal} 条向量记录")
    print(f"\n💡 下次启动程序时，直接用 load_local 加载，无需重新 Embedding！")


# ======================================================
# 📖 Demo6 - FAISS 本地加载
#
# 【从磁盘恢复索引】
#   加载后可以直接用，和刚创建的 vector_store 完全一样
#
# 【allow_dangerous_deserialization=True 的含义】
#   FAISS 使用 pickle 来序列化 Document 数据
#   pickle 可以执行任意代码，所以有风险
#   教学环境可以放心开启
#   生产环境中，确保只加载自己保存的文件
#
# 【生产环境最佳实践】
#   1. 自己保存的文件 → 可以放心加载
#   2. 用户上传的索引文件 → 不要直接加载（安全风险）
#   3. 替代方案：用 Chroma/Pinecone 等更安全的存储后端
# ======================================================

def demo_6_faiss_load() -> None:
    """
    演示从本地磁盘加载 FAISS 索引。
    
    这是 Demo5 的配套操作：保存 → 重启程序 → 加载 → 继续搜索
    
    ⚠️ 安全提示：
    allow_dangerous_deserialization=True
    原因：FAISS 内部使用 pickle 反序列化 Document 数据
    教学环境可以开启；生产环境只加载自己信任的文件
    """

    print("\n========== Demo6: FAISS 加载 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)

    # 从磁盘加载之前保存的索引
    # 不需要重新 Embedding！直接可用
    vector_store = FAISS.load_local(
        "./faiss_db",
        embedding_model,
        allow_dangerous_deserialization=True  # 允许加载 pickle 格式的 Document
    )

    print("✅ FAISS 索引加载成功")

    # 加载后直接使用，和新建的一样
    result = vector_store.similarity_search("LangChain", k=1)

    print(f"\n📊 搜索结果:")
    print(f"  {result[0].page_content}")
    print(f"\n💡 加载后无需重新 Embedding，搜索速度极快")


# ======================================================
# 📖 Demo7 - Chroma 持久化
#
# 【Chroma 的持久化更简单】
#   只需在建库时传入 persist_directory 参数
#   Chroma 会自动处理保存和加载
#   不用像 FAISS 那样分两步（save_local + load_local）
#
# 【Chroma 的数据存储结构】
#   persist_directory/
#   ├── chroma.sqlite3      ← 元数据和向量索引
#   └── *.bin               ← FAISS 索引文件
#
# 【何时选择 FAISS vs Chroma？】
#   选 FAISS：追求极致搜索性能，或已有 FAISS 基础设施
#   选 Chroma：想要更简单的 API，需要强大的 metadata 过滤
# ======================================================

def demo_7_chroma_persist(documents: List[Document]) -> None:
    """
    演示 Chroma 的持久化存储。
    
    Chroma 的持久化比 FAISS 更优雅：
    只需在创建时指定 persist_directory，Chroma 自动管理
    
    FAISS 的做法：
      store.save_local(path)    ← 显式保存
      store.load_local(path)    ← 显式加载
    
    Chroma 的做法：
      Chroma(..., persist_directory=path)  ← 创建时指定，自动保存
      Chroma(..., persist_directory=path)  ← 再创建同一个路径，自动加载
    """

    print("\n========== Demo7: Chroma 持久化 ==========")

    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)

    chroma_path = "./chroma_db"

    # 创建并指定持久化目录
    # Chroma 会自动将向量和元数据保存到该目录
    chroma = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=chroma_path
    )

    print(f"✅ Chroma 向量库创建并保存到: {chroma_path}")
    print(f"   当前记录数: {chroma._collection.count()}")
    print(f"\n💡 下次创建相同路径的 Chroma 实例，会自动加载已有数据")
    print(f"   无需重新 Embedding，也无需手动 save/load")


# ======================================================
# 📖 Demo8 - Metadata 过滤
#
# 【什么是 Metadata 过滤？】
#   你的知识库可能包含多种类型的文档：
#   - 技术文档、产品手册、人力资源政策
#   - 不同部门、不同日期、不同作者
#   
#   用户问："技术部门有什么文档？"
#   我们只想搜 technical_docs 分类下的，不想混入 HR 文档
#
# 【过滤语法】
#   filter={"category": "AI"}           ← 精确匹配
#   filter={"category": {"$in": ["A","B"]}}  ← 多选一
#   filter={"category": {"$ne": "HR"}}  ← 排除
#   filter={"year": {"$gte": 2023}}     ← 范围过滤
#
# 【实际场景】
#   多租户系统：每个公司的数据互相隔离
#   权限控制：普通员工只能搜"公开文档"，管理员能搜全部
# ======================================================

def demo_8_metadata_filter(documents: List[Document]) -> None:
    """
    演示基于 Metadata 的过滤搜索。
    
    这是生产环境最常用的功能之一！
    
    实际例子：
    ─────────────────────────────────────────
    公司知识库包含：
      - AI 技术文档（category=AI）
      - Python 教程（category=Programming）
      - 其他杂项
    
    用户只想看 AI 相关的：
      filter={"category": "AI"}
    
    或者只看某个来源的：
      filter={"source": "rag.txt"}
    """

    print("\n========== Demo8: Metadata 过滤搜索 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)
    vector_store = FAISS.from_documents(documents, embedding_model)

    # 只搜索 category="AI" 的文档
    results = vector_store.similarity_search(
        "知识库",
        k=3,
        filter={"category": "AI"}  # ← 过滤条件：只看 AI 分类
    )

    print("❓ 查询: \"知识库\"，过滤: category='AI'\n")
    print("📊 AI 分类结果:")
    for document in results:
        print(f"  • {document.page_content}")
        print(f"    分类: {document.metadata.get('category', 'N/A')}")

    print("\n💡 过滤作用：缩小搜索范围，提高结果精准度")
    print("   适用于：多分类知识库、多租户系统、权限控制等场景")


# ======================================================
# 📖 Demo9 - VectorStore 转换为 Retriever
#
# 【为什么要用 Retriever？】
#   VectorStore 本身是个「仓库」，Retriever 是「取货员」
#   Retriever 封装了搜索逻辑，提供了统一接口
#   
#   好处：
#   1. 统一 API — 不管是 FAISS、Chroma 还是 Pinecone，都用 .invoke()
#   2. 可组合 — 可以和 LCEL 管道无缝衔接
#   3. 可扩展 — 可以链式组合多个检索策略
#
# 【实际 RAG 流程】
#   用户问题 → Retriever.invoke(question) → 相关文档 → Prompt → LLM → 回答
#
# 【search_kwargs 参数】
#   k=2         → 返回最相似的 2 条
#   score_threshold=0.5 → 只返回相似度超过 0.5 的
#   filter={}   → 加上元数据过滤
# ======================================================

def demo_9_vectorstore_retriever(documents: List[Document]) -> None:
    """
    演示将 VectorStore 转换为 Retriever。
    
    Retriever 是 RAG 的核心组件！
    
    为什么不用 vector_store.similarity_search() 而要用 retriever？
    
    对比：
    ┌────────────────────────┬──────────────────────────┐
    │ vector_store.search()  │     retriever.invoke()   │
    ├────────────────────────┼──────────────────────────┤
    │ 耦合具体向量库          │ 统一的 LangChain 接口     │
    │ 不能直接进 LCEL 管道    │ 可以链式组合其他组件       │
    └────────────────────────┴──────────────────────────┘
    
    LCEL 管道示例：
      chain = (retriever | format_docs) | prompt | llm
                                    ↑
                         这里就是 Retriever
    """

    print("\n========== Demo9: VectorStore → Retriever ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)
    vector_store = FAISS.from_documents(documents, embedding_model)

    # 将 VectorStore 转为 Retriever
    # search_kwargs 配置搜索参数
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 调用检索
    docs = retriever.invoke("什么是 RAG")

    print("❓ 查询: \"什么是 RAG\"\n")
    print("📊 检索结果:")
    for document in docs:
        print(f"  • {document.page_content}")
        print(f"    来源: {document.metadata.get('source', 'N/A')}")

    print("\n💡 Retriever 输出可直接喂给 LLM 作为上下文")
    print("   这就是 RAG 的「R」（Retrieval）部分")


# ======================================================
# 📖 Demo10 - 智谱 AI 真实 Embedding
#
# 【为什么用真实模型？】
#   前面的 Demo 都用的 FakeEmbedding（随机数）
#   FakeEmbedding 不能真正区分语义：
#   "猫" 和 "猫咪" 的距离 可能比 "猫" 和 "狗" 更远
#   因为向量是随机生成的，没有任何语义信息
#
#   真实模型（ZhipuAI/BGE/OpenAI）才会让语义相近的文本
#   在向量空间中距离更近
#
# 【前置准备】
#   1. 确保已配置环境变量 ZHIPUAI_API_KEY
#   2. 安装依赖: pip install langchain-community zhipuai
#
# 【常见 Embedding 模型对比】
#   模型                    维度    中文效果    费用
#   ──────────────────────────────────────────
#   ZhipuAI embedding-3    1024    ★★★★★    有免费额度
#   BAAI/bge-large-zh      1024    ★★★★★    开源可本地跑
#   text-embedding-3-small 1536    ★★★☆☆    OpenAI 收费
#   nomic-embed-text       768     ★★★★☆    开源可本地跑
# ======================================================

def demo_10_zhipu_embedding(documents: List[Document]) -> None:
    """
    使用智谱 AI 的真实 Embedding 模型。
    
    这是从「教学演示」走向「真实应用」的关键一步！
    
    与 FakeEmbedding 的区别：
    ┌────────────────────────────────────────────────────┐
    │  FakeEmbedding: 向量是随机的，无语义意义            │
    │  ZhipuAIEmbedding: 向量反映真实语义，相似文本距离近  │
    └────────────────────────────────────────────────────┘
    
    前置条件：
    1. 设置环境变量 ZHIPUAI_API_KEY
    2. pip install langchain-community
    """

    print("\n========== Demo10: 智谱 AI Embedding ==========")

    import os
    load_dotenv()
    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("\n⚠️  未配置智谱 API Key")
        print("   请设置: export ZHIPUAI_API_KEY='your-key'")
        print("   或添加 .env 文件: ZHIPUAI_API_KEY=your-key")
        print("   获取 API Key: https://open.bigmodel.cn")
        return

    try:
        from langchain_community.embeddings import ZhipuAIEmbeddings
        from langchain_community.vectorstores import FAISS

        # 使用智谱 AI 的 embedding-3 模型
        embedding_model = ZhipuAIEmbeddings(model="embedding-3", api_key=api_key)

        print(f"✅ 嵌入模型: {embedding_model.model}")
        print(f"   向量维度: {embedding_model.dimensions}")

        # 构建向量库
        vector_store = FAISS.from_documents(documents, embedding_model)

        # 执行搜索
        result = vector_store.similarity_search("什么是 LangChain", k=2)

        print("\n📊 搜索结果（语义搜索）:")
        for i, document in enumerate(result, 1):
            print(f"  [{i}] {document.page_content}")
            print(f"      来源: {document.metadata.get('source', 'N/A')}")

        print("\n✅ 真实模型搜索完成！语义相似度更有意义。")

    except ImportError:
        print("\n❌ 缺少依赖包")
        print("   请运行: pip install langchain-community")
    except Exception as error:
        print(f"\n❌ 运行失败: {error}")


# ======================================================
# 📖 Demo11 - 完整向量索引 Pipeline
#
# 【Pipeline 是什么？】
#   把多个步骤串起来，形成一个完整的数据处理流水线
#
# 【完整流程】
#   原始文本
#     │
#     ▼
#   Text Splitter（文本切分）
#     │  把长文档切成小块，每块独立向量化
#     ▼
#   Document 列表
#     │
#     ▼
#   Embedding（向量化）
#     │  每块文本 → 一个向量
#     ▼
#   Vector Store（存储）
#     │  存入 FAISS/Chroma，建立索引
#     ▼
#   Retriever（检索器）
#     │  封装搜索逻辑，对外提供统一接口
#     ▼
#   用户查询 → 返回相关文档片段
#
# 【Chunking 策略的重要性】
#   chunk_size 太小：丢失上下文，检索结果断章取义
#   chunk_size 太大：噪音多，检索精度下降
#   chunk_overlap：相邻块重叠一部分，避免边界截断丢失关键信息
#
# 【推荐参数】
#   chunk_size: 500~1000 字符（中文约 250~500 字）
#   chunk_overlap: 50~100 字符
# ======================================================

def demo_11_complete_vector_pipeline() -> None:
    """
    演示完整的知识库构建 Pipeline。
    
    这是从原始文本到可用检索的完整流程！
    
    ┌──────────────────────────────────────────────────────────┐
    │                    完整 RAG 数据流                        │
    │                                                          │
    │  原始文本                                                │
    │      │                                                   │
    │      ▼                                                   │
    │  ┌─────────────┐    按固定大小切分，保留重叠              │
    │  │ TextSplitter│ ──────────────────────────────────▶ 小块 │
    │  └─────────────┘                                       └────┘
    │                                                       │
    │      ▲                                               │
    │      │ 用户查询                                        │
    │  ┌─────────────┐                                       │
    │  │  Retriever  │ ◀─────────────────────────────────────┘
    │  │ (向量搜索)  │    返回最相关的 K 个文档片段
    │  └─────────────┘
    │      ▲
    │      │ 向量化 + 建立索引
    │  ┌─────────────┐
    │  │ VectorStore │
    │  │  (FAISS)    │
    │  └─────────────┘
    │      ▲
    │      │ Embedding
    │  ┌─────────────┐
    │  │ Embedding   │
    │  └─────────────┘
    └──────────────────────────────────────────────────────────┘
    """

    print("\n========== Demo11: 完整向量索引 Pipeline ==========")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    # ─── Step 1: 原始文本 ───
    text = """LangChain 是一个用于开发大模型应用的开源框架，提供了丰富的组件库。
    RAG（检索增强生成）通过引入外部知识库，显著提升大模型的准确性和可信度。
    Vector Store 负责高效地存储和检索文本向量，是现代 AI 应用的基础设施。
    智能客服系统是 RAG 的典型应用场景，可以快速回答用户的常见问题。"""

    print("📝 Step 1: 原始文本")
    print(f"   长度: {len(text)} 字符")

    # ─── Step 2: 文本切分 ───
    # chunk_size: 每个块的最大字符数
    # chunk_overlap: 相邻块的重叠字符数（避免边界截断重要信息）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,    # 每块最多 50 字符
        chunk_overlap=10  # 相邻块重叠 10 字符
    )

    chunks = splitter.split_text(text)

    print(f"\n🔪 Step 2: 文本切分")
    print(f"   切分成 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"   [{i}] {chunk}")

    # ─── Step 3: 包装为 Document ───
    documents = [
        Document(page_content=chunk, metadata={"source": "demo.txt"})
        for chunk in chunks
    ]

    print(f"\n📦 Step 3: 包装为 Document 对象")
    print(f"   共 {len(documents)} 个 Document")

    # ─── Step 4: 向量化 + 建立索引 ───
    embedding_model = FakeEmbeddings(size=8)
    vector_store = FAISS.from_documents(documents, embedding_model)

    print(f"\n🧮 Step 4: 向量化并建立索引")
    print(f"   向量维度: {embedding_model.size}")
    print(f"   索引条目: {vector_store.index.ntotal}")

    # ─── Step 5: 转换为 Retriever ───
    retriever = vector_store.as_retriever()

    print(f"\n🔍 Step 5: 创建 Retriever")

    # ─── Step 6: 执行检索 ───
    results = retriever.invoke("什么是增强模型回答？")

    print(f"\n❓ 用户查询: \"什么是增强模型回答？\"")
    print(f"\n📊 检索结果:")
    for i, document in enumerate(results, 1):
        print(f"  [{i}] {document.page_content}")
        print(f"      来源: {document.metadata.get('source', 'N/A')}")

    print("\n" + "=" * 50)
    print("🎯 完整流程总结:")
    print("   原始文本 → 切分 → 包装 → 向量化 → 建索引 → 检索")
    print("   这就是 RAG 的「检索」部分，后续再加上 LLM 就能生成回答了")
    print("=" * 50)


# ======================================================
# 🏃 Main 入口
# ======================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║      LangChain Phase 2-4: Vector Stores 教程          ║
    ║          FAISS + Chroma + Metadata + Pipeline         ║
    ╚══════════════════════════════════════════════════════╝
    """)

    print("\n📖 教程导航:")
    print("   Demo0  → 概念理解（Vector Store 是什么？）")
    print("   Demo1  → 创建 Document 对象（数据结构基础）")
    print("   Demo2  → FAISS 基础搜索")
    print("   Demo3  → 查看相似度分数（判断结果质量）")
    print("   Demo4  → Chroma 基础（更简洁的 API）")
    print("   Demo5  → FAISS 本地保存（持久化索引）")
    print("   Demo6  → FAISS 加载（从磁盘恢复）")
    print("   Demo7  → Chroma 持久化（自动保存）")
    print("   Demo8  → Metadata 过滤（精准定位）")
    print("   Demo9  → VectorStore → Retriever（RAG 核心组件）")
    print("   Demo10 → 智谱 AI 真实 Embedding（告别 Fake）")
    print("   Demo11 → 完整索引 Pipeline（端到端流程）")

    # Demo0 是概念讲解，不需要参数
    demo_0_vector_store_concept()

    # Demo1 创建 Document，后续 Demo 复用
    documents = demo_1_create_documents()

    # 依次演示
    # demo_2_faiss_basic(documents)        # FAISS 基础
    # demo_3_faiss_similarity_score(documents)  # 相似度分数
    # demo_4_chroma_basic(documents)       # Chroma 基础
    # demo_5_faiss_save(documents)         # FAISS 保存
    # demo_6_faiss_load()                  # FAISS 加载
    demo_7_chroma_persist(documents)     # Chroma 持久化
    demo_8_metadata_filter(documents)    # Metadata 过滤
    demo_9_vectorstore_retriever(documents)  # Retriever 转换
    demo_10_zhipu_embedding(documents)   # 智谱真实 Embedding
    demo_11_complete_vector_pipeline()   # 完整 Pipeline

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║         Phase 2-4 Vector Store 全部完成 ✅            ║
    ║         下一阶段: Phase 2-5 Retriever 高级用法        ║
    ╚══════════════════════════════════════════════════════╝
    """)
