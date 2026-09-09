"""
========================================================

LangChain Phase 2-5

Retriever 教材级 Demo
（完整版注释 + 应用场景说明）


学习目标：
1. 理解 Retriever 在 RAG 中的作用和地位
2. 掌握 6 种检索策略：similarity / MMR / 阈值 / BM25 / Ensemble / MultiQuery
3. 理解每种策略的适用场景和优缺点
4. 学会根据业务需求选择合适的检索方式


🔗 知识脉络：
┌─────────────────────────────────────────────────────────────┐
│  检索 (Retrieval) 是 RAG 最核心的环节                       │
│                                                             │
│  用户问题 ──▶ 转化为向量 ──▶ 在向量库中搜索 ──▶ 返回相关文档  │
│                ↑                                            │
│           Retriever 决定「怎么搜」                            │
│                                                             │
│  搜索策略选择：                                              │
│  • similarity          → 最基础的语义搜索                    │
│  • MMR                 → 在相似基础上增加多样性               │
│  • score_threshold     → 只返回高质量结果                    │
│  • BM25                → 关键词匹配（适合精确术语）           │
│  • Ensemble            → 语义 + 关键词融合                  │
│  • MultiQuery          → 一个问题拆成多个，扩大召回          │
└─────────────────────────────────────────────────────────────┘


========================================================
"""

from typing import List
from langchain_core.documents import Document


# ======================================================
# 📋 数据准备
# ======================================================

def create_demo_documents() -> List[Document]:
    """
    创建测试知识库文档。
    
    这些文档模拟了一个小型技术知识库，涵盖：
    - LangChain 框架介绍
    - RAG 技术原理
    - Python 编程基础
    - FAISS 向量搜索
    - Ollama 本地模型
    
    返回:
        Document 列表，包含 page_content 和 metadata
    """
    documents = [
        Document(
            page_content="LangChain 是一个用于开发大语言模型应用的框架，支持 RAG 和 Agent 开发",
            metadata={"source": "langchain"}
        ),
        Document(
            page_content="RAG 通过外部知识库增强大模型回答能力，可以减少幻觉问题",
            metadata={"source": "rag"}
        ),
        Document(
            page_content="Python 是一种简单易学的高级编程语言，广泛应用于人工智能领域",
            metadata={"source": "python"}
        ),
        Document(
            page_content="FAISS 是 Facebook 开发的向量搜索引擎，可以快速完成相似度搜索",
            metadata={"source": "faiss"}
        ),
        Document(
            page_content="智谱 AI 提供高质量的中文 Embedding 模型和 LLM API 服务",
            metadata={"source": "zhipu"}
        ),
    ]
    return documents


# ======================================================
# 📖 Demo0 - Retriever 概念入门
#
# 【VectorStore vs Retriever 的区别】
#   VectorStore 就像一个「图书馆」：
#     - 负责存储所有书籍（文档+向量）
#     - 知道每本书放在哪个书架
#
#   Retriever 就像一个「图书管理员」：
#     - 负责决定怎么帮你找书
#     - 可以用不同的策略（最近、最新、最相关...）
#     - 对外提供统一的借书接口
#
# 【为什么需要 Retriever？】
#   1. 抽象层：隐藏底层向量库的差异
#      （不管用的是 FAISS、Chroma 还是 Pinecone，接口都一样）
#   2. 可组合性：可以和其他组件拼接成管道
#      （Retriever → Prompt → LLM）
#   3. 策略灵活：轻松切换不同的搜索策略
# ======================================================

def demo_0_retriever_concept() -> None:
    """
    介绍 Retriever 的基本概念。
    
    Retriever = 检索器
    它是连接「用户问题」和「向量数据库」的桥梁
    
    常见检索策略对比：
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │  策略           │  原理                │  适用场景            │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ similarity      │ 余弦相似度，找最近邻  │ 通用语义搜索          │
    │ MMR             │ 相似度+多样性         │ 需要去重、丰富视角    │
    │ score_threshold │ 设定最低相似度阈值    │ 只返回高置信度结果    │
    │ BM25            │ 关键词频率匹配        │ 专有名词、精确匹配    │
    │ Ensemble        │ 多种策略融合          │ 追求最高召回率        │
    │ MultiQuery      │ 一个问题生成多个查询  │ 模糊问题、扩大召回    │
    └─────────────────┴──────────────────────┴──────────────────────┘
    """

    print("\n========== Demo0: Retriever 概念 ==========")

    print("""
    📌 什么是 Retriever？
    ─────────────────────────────────────────
    
    Retriever（检索器）是 LangChain 中的一个抽象层，
    它封装了从向量数据库检索相关文档的逻辑。
    
    ┌─────────────────────────────────────────────────────┐
    │  VectorStore 负责：                                 │
    │    ✅ 存储向量                                       │
    │    ✅ 建立索引                                       │
    │    ❌ 不决定「怎么搜」                               │
    │                                                     │
    │  Retriever 负责：                                    │
    │    ✅ 决定搜索策略（similarity / MMR / BM25 ...）   │
    │    ✅ 封装统一接口（.invoke()）                      │
    │    ✅ 可以组合进 LCEL 管道                           │
    └─────────────────────────────────────────────────────┘
    
    常用检索策略：
    • similarity      → 最基础的语义相似度搜索
    • MMR             → 相关性好 + 结果多样化
    • BM25            → 关键词匹配（传统搜索）
    • Ensemble        → 混合多种策略的结果
    • MultiQuery      → 一个查询自动生成多个变体
    """)


# ======================================================
# 📖 Demo1 - （跳过，见 2_4 Demo9）
#
# ======================================================
# 📖 Demo2 - VectorStore.as_retriever()
#
# 【as_retriever() 做了什么？】
#   它把 VectorStore 包装成了一个 Retriever 对象
#   之后你就可以用统一的 .invoke() 方法来检索
#
# 【search_kwargs 是什么？】
#   传给底层向量库搜索方法的额外参数
#   常见的有：
#     k=3         → 返回 3 条最相似结果
#     score_threshold=0.7  → 只返回相似度 >= 0.7 的
#     filter={}   → 加上元数据过滤条件
#
# 【生产建议】
#   通常 k 设置在 3~5 之间
#   太小可能遗漏相关信息
#   太大引入过多噪音，增加 LLM 的处理负担和成本
# ======================================================

def demo_2_vectorstore_retriever() -> None:
    """
    演示将 VectorStore 转换为 Retriever。
    
    这是最基础也是最常用的方式。
    几乎所有 RAG 系统的起点都是这一步。
    """

    print("\n========== Demo2: VectorStore → Retriever ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    documents = create_demo_documents()
    embeddings = FakeEmbeddings(size=8)

    # 第一步：创建向量库
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"✅ 向量库已创建，共 {len(documents)} 条文档")

    # 第二步：转换为 Retriever
    # search_kwargs 控制搜索行为
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    print(f"✅ Retriever 已创建 (k=2)")

    # 第三步：执行检索
    results = retriever.invoke("什么是 RAG？")

    print(f"\n❓ 查询: \"什么是 RAG？\"")
    print(f"📊 返回 {len(results)} 条结果:\n")

    for idx, document in enumerate(results, 1):
        print(f"  [{idx}] {document.page_content}")
        print(f"      来源: {document.metadata.get('source', 'N/A')}")
        print()

    print("💡 实际 RAG 中，这些文档会作为上下文喂给 LLM 生成最终回答")


# ======================================================
# 📖 Demo3 - search_kwargs 参数调节
#
# 【k 参数的影响】
#   k=1: 只取最相关的那一条（最精确，但可能遗漏）
#   k=5: 取最相关的 5 条（更完整，但可能有噪音）
#
#   经验值：k=3~5 是较好的平衡点
#
# 【score_threshold 参数的影响】
#   score_threshold=0.5: 相似度低于 0.5 的结果被丢弃
#   适合对回答质量要求高的场景（如医疗、法律）
#   不适合创意类问答（可能过于严格）
#
# 【多条件组合】
#   search_kwargs={"k": 5, "score_threshold": 0.6, "filter": {"category": "AI"}}
#   同时满足：返回 5 条、分数 >= 0.6、只搜 AI 分类
# ======================================================

def demo_3_search_kwargs() -> None:
    """
    演示调节 Retriever 的搜索参数。
    
    search_kwargs 是最常用的调优手段。
    通过调整参数，可以控制「返回多少」「质量门槛多高」
    """

    print("\n========== Demo3: search_kwargs 参数调节 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    documents = create_demo_documents()
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))

    # 调整 k=3：返回 3 条最相关的
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("人工智能")

    print(f"❓ 查询: \"人工智能\"")
    print(f"   参数: k=3")
    print(f"   返回数量: {len(docs)}\n")

    for doc in docs:
        print(f"  • {doc.page_content}")
        print(f"    来源: {doc.metadata.get('source', 'N/A')}")

    print("\n💡 调参技巧：")
    print("   k 偏小 → 更精确，可能遗漏")
    print("   k 偏大 → 更全面，可能引入噪音")
    print("   建议从 k=3 开始，根据实际效果调整")


# ======================================================
# 📖 Demo4 - MMR 检索
#
# 【MMR 是什么？】
#   MMR = Maximum Marginal Relevance（最大边际相关性）
#
#   标准相似度搜索的问题：
#   假设库里有很多关于"猫"的文档
#   用户搜"猫"，标准搜索可能返回 5 条都讲"猫"的内容
#   信息重复，对用户没有帮助
#
#   MMR 的解决方案：
#   在保持一定相关性的同时，尽量让结果多样化
#   返回：关于"猫"的 A 方面 + B 方面 + C 方面...
#
# 【MMR 参数详解】
#   k: 最终返回几条
#   fetch_k: 先取多少条候选（默认 2*k）
#   lambda_mult: 多样性权重
#     lambda=0 → 只看多样性（完全不相关也返回）
#     lambda=1 → 只看相关性（标准相似度搜索）
#     lambda=0.5 → 兼顾两者（推荐起始值）
#
# 【适用场景】
#   - 用户想全面了解某个主题
#   - 知识库中有大量同质内容
#   - 需要减少重复信息的场景
# ======================================================

def demo_4_mmr_retriever() -> None:
    """
    演示 MMR（最大边际相关性）检索。
    
    MMR 的核心思想：既要相关，又要多样
    
    标准 similarity 搜索：
      问题: "猫"
      结果: [猫A, 猫B, 猫C, 猫D, 猫E]  ← 全部高度相似
    
    MMR 搜索（lambda=0.5）：
      结果: [猫A, 猫科动物习性, 猫的品种, 养猫注意事项, 猫的进化史]
           ← 相关但各不相同
    
    参数说明：
      k        → 最终返回数量
      fetch_k  → 候选池大小（先取更多，再筛选）
      lambda_mult → 相关性 vs 多样性的权衡（0=全多样，1=全相关）
    """

    print("\n========== Demo4: MMR 检索 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    documents = create_demo_documents()
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))

    # 使用 MMR 策略
    # search_type="mmr" 切换到最大边际相关性模式
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5}
    )

    results = retriever.invoke("人工智能开发")

    print("❓ 查询: \"人工智能开发\"")
    print("   策略: MMR (k=3, fetch_k=5, lambda=0.5)")
    print("   特点: 在保持相关的同时增加结果多样性\n")

    for idx, document in enumerate(results, 1):
        print(f"  [{idx}] {document.page_content}")
        print(f"      来源: {document.metadata.get('source', 'N/A')}")

    print("\n💡 适用场景：")
    print("   • 希望结果覆盖不同角度，而非重复同一信息")
    print("   • 知识库中存在大量相似内容时")
    print("   • 用户需要了解一个主题的全貌")


# ======================================================
# 📖 Demo5 - 相似度阈值过滤
#
# 【为什么需要阈值？】
#   有些搜索结果质量很差，返回给用户反而造成误导
#   比如用户问了一个知识库完全没有的内容
#   这时强行返回一个低质量的片段，不如直接说「不知道」
#
# 【阈值怎么设？】
#   需要结合实际业务来调：
#   • 通用问答: threshold=0.3~0.5（宽松）
#   • 专业领域: threshold=0.6~0.8（严格）
#   • 医疗/法律: threshold=0.8+（非常严格）
#
# 【注意】
#   阈值是「距离分数」不是「相似度分数」
#   FAISS 默认返回的是 distance（越小越好）
#   所以 score_threshold 实际上是距离上限
# ======================================================

def demo_5_score_threshold() -> None:
    """
    演示相似度阈值检索。
    
    设置最小相似度门槛，低于该分数结果被丢弃
    
    实际价值：
    ┌──────────────────────────────────────────────────────┐
    │  用户问了一个知识库没有的问题                          │
    │                                                      │
    │  不设阈值：                                           │
    │    返回最相关的 3 条 → 但每条都很勉强                 │
    │    LLM 基于错误上下文瞎编 → 产生幻觉                  │
    │                                                      │
    │  设阈值后：                                           │
    │    所有结果都低于阈值 → 返回空 → 告诉用户「不知道」   │
    │    LLM 诚实地说 "抱歉，知识库中没有相关内容"           │
    └──────────────────────────────────────────────────────┘
    """

    print("\n========== Demo5: 相似度阈值检索 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    documents = create_demo_documents()
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))

    # 设置相似度阈值
    # score_threshold: 只有分数 >= 此值的结果才会返回
    # 注意：具体含义取决于向量库实现，FAISS 这里是距离上限
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )

    results = retriever.invoke("Python")

    print("❓ 查询: \"Python\"")
    print("   策略: similarity_score_threshold")
    print("   阈值: 0.5, 最多返回: 5 条\n")
    print(f"   实际返回: {len(results)} 条")

    for document in results:
        print(f"  • {document.page_content}")

    print("\n💡 阈值调优建议：")
    print("   • 阈值过高 → 结果太少，可能漏掉相关内容")
    print("   • 阈值过低 → 结果太多，引入无关噪音")
    print("   • 建议先在开发环境用真实数据测试不同阈值的效果")


# ======================================================
# 📖 Demo6 - BM25 关键词检索
#
# 【BM25 是什么？】
#   BM25 是传统搜索引擎（如 Google）使用的关键词匹配算法
#   它不看语义，只看词频：
#   • 查询词在文档中出现次数越多 → 得分越高
#   • 文档越短（包含查询词的密度越高）→ 得分越高
#   • 稀有词比常见词权重更高
#
#   和语义搜索的区别：
#   语义搜索：「苹果手机」能匹配到「iPhone」
#   BM25：「苹果手机」只能匹配包含「苹果」「手机」这两个词的文档
#
# 【BM25 的优势】
#   1. 对专有名词、人名、品牌名特别准确
#   2. 不受 Embedding 模型质量影响
#   3. 不依赖向量库，纯文本匹配
#   4. 搜索速度极快
#
# 【BM25 的劣势】
#   1. 无法理解语义（同义词、近义词不行）
#   2. 需要精确匹配关键词
#
# 【实际例子】
#   用户问："LangChain 的 API 是什么？"
#   BM25 能精确命中包含 "LangChain" 这个词的文档
#   即使用户的措辞和文档不完全一致也能命中
# ======================================================

def demo_6_bm25_retriever() -> None:
    """
    演示 BM25 关键词检索。
    
    BM25 是传统搜索引擎的核心算法，不做语义理解，只做关键词匹配。
    
    什么时候用 BM25？
    • 文档中包含大量专有名词（产品名、人名、术语）
    • 用户问题中有关键词，但不确定语义表达
    • 作为语义搜索的补充，提高召回率
    
    注意：BM25 不依赖 Embedding 模型，也不需要向量数据库
    它直接对文本进行词频统计和排名
    """

    print("\n========== Demo6: BM25 关键词检索 ==========")

    from langchain_community.retrievers import BM25Retriever

    documents = create_demo_documents()

    # BM25 直接从 Document 列表创建，不需要 Embedding
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 2  # 返回 Top 2

    print("❓ 查询: \"LangChain框架\"")
    print("   策略: BM25 关键词匹配\n")

    results = retriever.invoke("LangChain 框架")

    for document in results:
        print(f"  • {document.page_content}")
        print(f"    来源: {document.metadata.get('source', 'N/A')}")

    print("\n💡 BM25 特点：")
    print("   ✓ 对包含专有名词的查询特别准确")
    print("   ✓ 不需要 Embedding 模型，无需 API 调用")
    print('   ✗ 无法理解语义（"LangChain" ≠ "LLM框架"）')
    print('   ✗ 用户换种说法可能找不到结果')


# ======================================================
# 📖 Demo7 - Ensemble 混合检索
#
# 【为什么要混合？】
#   单一检索策略总有短板：
#   • 语义搜索（向量）: 懂语义但可能错过精确关键词
#   • BM25（关键词）: 精确但不懂语义
#
#   混合两者 = 取两者的长处，补两者的短处
#
#   举例：
#   用户问："LLM 开发框架"
#   • 向量搜索 → 可能找到"LangChain"（因为语义相近）
#   • BM25 → 找不到"LLM"在文档中的匹配（因为文档写的是"大语言模型"）
#   • 混合 → 两条路都走，召回率翻倍
#
# 【权重是什么意思？】
#   weights=[0.5, 0.5] 表示两种策略各占一半
#   weights=[0.7, 0.3] 表示语义搜索更重要
#   根据实际效果调整
# ======================================================

def demo_7_ensemble_retriever() -> None:
    """
    演示 Ensemble（混合）检索。
    
    将多种检索策略的结果融合在一起，提升召回率
    
    混合策略的价值：
    ┌────────────────────────────────────────────────────┐
    │  用户查询："如何使用大语言模型开发应用？"            │
    │                                                    │
    │  纯向量搜索：可能找到 LangChain 相关内容 ✅          │
    │  纯 BM25：找不到 "大语言模型"（文档写的是 LLM）❌   │
    │  混合检索：两者结合，确保不遗漏 ✅                  │
    └────────────────────────────────────────────────────┘
    """

    print("\n========== Demo7: Ensemble 混合检索 ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_community.retrievers import BM25Retriever

    documents = create_demo_documents()

    # 向量检索器
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # BM25 检索器
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    # 混合检索器：各占 50% 权重
    from langchain_classic.retrievers import EnsembleRetriever

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # BM25 和 向量搜索 各占一半
    )

    print("❓ 查询: \"向量数据库\"")
    print("   策略: BM25(50%) + 向量搜索(50%)\n")

    results = ensemble.invoke("向量数据库")

    print(f"📊 混合检索返回 {len(results)} 条结果:\n")

    for idx, document in enumerate(results, 1):
        print(f"  [{idx}] {document.page_content}")
        print(f"      来源: {document.metadata.get('source', 'N/A')}")

    print("\n💡 混合检索适用场景：")
    print("   • 用户查询方式不固定（有的说术语，有的说日常语言）")
    print("   • 知识库内容多样，单一策略容易遗漏")
    print("   • 追求高召回率，愿意牺牲一点精确度")


# ======================================================
# 📖 Demo8 - MultiQuery 检索
#
# 【MultiQuery 是什么？】
#   用一个 LLM 把用户的一个问题，改写/扩展成多个不同的问法
#   然后用每个问法分别去搜索
#   最后合并所有结果，去重后返回
#
#   为什么有效？
#   用户的问题可能有歧义、表述不清、或用了非标准术语
#   例如用户问："咋整那个聊天机器人框架"
#   MultiQuery 会改写为：
#     "如何开发聊天机器人？"
#     "有哪些 LLM 对话框架？"
#     "Chatbot 开发工具推荐"
#   然后用三种问法分别搜索，大大提高命中率
#
# 【需要什么模型？】
#   需要一个 Chat Model 来生成改写后的查询
#   这里使用智谱 AI 的 ChatGLM 模型（参考 Demo6 的方式）
#
# 【适用场景】
#   • 用户问题表述模糊或不规范
#   • 知识库中有大量专业术语，用户可能不知道正确叫法
#   • 搜索结果召回率低，需要扩大搜索范围
# ======================================================

def demo_8_multi_query_retriever() -> None:
    """
    演示 MultiQuery 检索。
    
    MultiQueryRetriever 的核心思想：
    「一个好问题可能找不到答案，但三个好问题可以」
    
    工作流程：
    用户问题 → LLM 改写为 3 个不同问法 → 各自检索 → 合并去重 → 返回
    
    适合场景：
    • 用户问题模糊、口语化、或用了不标准的术语
    • 需要扩大搜索召回率
    • 知识库覆盖面广，单一路径容易遗漏
    """

    print("\n========== Demo8: MultiQuery 检索 ==========")

    import os
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from dotenv import load_dotenv  # 从 .env 文件中自动加载环境变量

    documents = create_demo_documents()
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))
    base_retriever = vector_store.as_retriever()
    load_dotenv()

    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("\n⚠️  未配置智谱 API Key，演示简化版")
        print("   MultiQuery 需要 LLM 来改写查询，请使用真实模型或配置 API Key")
        print("\n   配置方式:")
        print("   export ZHIPUAI_API_KEY='your-key'")
        return

    try:
        from langchain_community.chat_models import ChatZhipuAI

        llm = ChatZhipuAI(api_key=api_key, temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

        results = retriever.invoke("什么是RAG")

        print("❓ 原始查询: \"什么是RAG\"")
        print("   LLM 会自动将其改写为多个不同问法进行搜索\n")
        print("📊 合并去重后返回结果:\n")

        for idx, document in enumerate(results, 1):
            print(f"  [{idx}] {document.page_content}")
            print(f"      来源: {document.metadata.get('source', 'N/A')}")

        print("\n💡 MultiQuery 的作用：")
        print("   一个模糊问题 → 多个精准问法 → 更高的召回率")

    except Exception as error:
        print(f"\n❌ MultiQuery 需要智谱 AI 模型: {error}")
        print("   请确保已配置 ZHIPUAI_API_KEY 环境变量")


# ======================================================
# 📖 Demo9 - Contextual Compression 检索
#
# 【为什么要压缩？】
#   向量搜索返回的结果可能很长，包含大量无关内容
#   如果直接把整个文档片段喂给 LLM：
#   • Token 消耗多 → 成本高
#   • 噪音多 → LLM 可能被带偏
#   • 上下文窗口有限 → 塞不下太多
#
#   Contextual Compression 的做法：
#   先用 LLM 快速扫描每个文档片段
#   只保留和用户问题相关的内容，删掉无关部分
#   然后把这些精简后的内容喂给最终的 LLM
#
#   效果类比：
#   原始检索：找到 5 篇论文，每篇 2000 字
#   压缩后：从每篇论文中只摘出和「RAG 架构」相关的 300 字
#
# 【适用场景】
#   • 文档片段较长，包含大量冗余信息
#   • 对 LLM 的输入成本敏感
#   • 需要精确的上下文，不希望无关信息干扰
# ======================================================

def demo_9_contextual_compression() -> None:
    """
    演示上下文压缩检索。
    
    背景：向量搜索返回的文档往往很长，其中只有一部分和问题相关
    压缩器的作用：用 LLM 快速审阅每个文档，提取相关部分，删除无关内容
    
    类比：
    ┌──────────────────────────────────────────────────────┐
    │  向量检索 → 找到 5 篇相关文件（每篇 500 字）          │
    │       ↓                                               │
    │  压缩器 → 从每篇中提取最相关的 100 字                 │
    │       ↓                                               │
    │  最终给 LLM 的上下文 → 500 字精粹（而非 2500 字原文）│
    └──────────────────────────────────────────────────────┘
    
    优势：节省 Token、减少噪音、降低幻觉风险
    """

    print("\n========== Demo9: Contextual Compression 检索 ==========")

    import os
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings
    from dotenv import load_dotenv  # 从 .env 文件中自动加载环境变量

    load_dotenv()
    documents = create_demo_documents()
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))
    base_retriever = vector_store.as_retriever()

    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("\n⚠️  未配置智谱 API Key，跳过压缩检索演示")
        print("   Contextual Compression 需要 LLM 来审阅和精简内容")
        return

    try:
        from langchain_community.chat_models import ChatZhipuAI
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor

        llm = ChatZhipuAI(api_key=api_key, temperature=0)

        # LLMChainExtractor：用 LLM 提取与查询相关的文档片段
        compressor = LLMChainExtractor.from_llm(llm)

        # 组合压缩器和基础检索器
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        results = retriever.invoke("什么是LangChain")

        print("❓ 查询: \"什么是LangChain\"")
        print("   策略: 向量检索 + LLM 上下文压缩\n")
        print("📊 压缩后的结果:\n")

        for idx, document in enumerate(results, 1):
            original_len = len(document.page_content)
            print(f"  [{idx}] {document.page_content[:100]}...")
            print(f"      原始长度: {original_len} 字符")

        print("\n💡 压缩效果：去除无关内容，只保留与查询相关的部分")
        print("   减少 Token 消耗，提升 LLM 回答质量")

    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        print("   请安装: pip install langchain")
    except Exception as error:
        print(f"\n❌ 运行失败: {error}")


# ======================================================
# 📖 Demo10 - 完整检索 Pipeline
#
# 【从检索到回答的完整链路】
#   用户提问 → Retriever 检索 → 获取相关文档 → 组合 Prompt → LLM 回答
#
# 【这个 Demo 做什么】
#   模拟完整 RAG 的检索部分
#   展示从问题输入到拿到上下文文档的完整流程
#
# 【下一步是什么？】
#   拿到 context 后，还需要：
#   1. 构建 Prompt Template（把问题和上下文拼成提示词）
#   2. 调用 LLM（智谱/通义/OpenAI）
#   3. 解析输出
#   这些内容在 Phase 2-6 中演示
# ======================================================

def demo_10_complete_retriever_pipeline() -> None:
    """
    演示完整的 RAG 检索流程。
    
    这是整个 RAG 系统的核心：检索阶段
    
    完整链路：
    ┌──────────────────────────────────────────────────────────┐
    │  用户: "什么是 RAG？"                                     │
    │        ↓                                                 │
    │  ┌─────────────────┐                                     │
    │  │  Retriever      │  ← 从向量库中找出最相关的文档        │
    │  │  (k=2)          │                                     │
    │  └────────┬────────┘                                     │
    │           ↓                                              │
    │  ┌─────────────────┐                                     │
    │  │  相关文档       │  ← 2 条最相关的知识库条目            │
    │  └────────┬────────┘                                     │
    │           ↓                                              │
    │  ┌─────────────────┐                                     │
    │  │  Prompt 模板    │  ← 把问题 + 文档拼成提示词           │
    │  └────────┬────────┘                                     │
    │           ↓                                              │
    │  ┌─────────────────┐                                     │
    │  │  LLM 生成回答   │  ← 基于检索到的知识回答用户          │
    │  └─────────────────┘                                     │
    └──────────────────────────────────────────────────────────┘
    
    本 Demo 演示到「拿到相关文档」这一步。
    """

    print("\n========== Demo10: 完整检索 Pipeline ==========")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    documents = create_demo_documents()

    # 构建向量库
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=8))

    # 创建检索器（取最相关的 2 条）
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    question = "什么是 RAG？"
    context_documents = retriever.invoke(question)

    print(f"❓ 用户问题: {question}\n")
    print(f"📊 检索到的上下文文档（{len(context_documents)} 条）:\n")

    for idx, document in enumerate(context_documents, 1):
        print(f"  ┌{'─' * 50}┐")
        print(f"  │ [{idx}] {document.page_content}")
        print(f"  │ 来源: {document.metadata.get('source', 'N/A')}")
        print(f"  └{'─' * 50}┘")

    print(f"\n💡 下一步：将这 {len(context_documents)} 条文档作为上下文，")
    print(f"   配合 Prompt 模板交给 LLM，生成最终回答")
    print(f"   （见 Phase 2-6 完整 RAG Pipeline）")


# ======================================================
# 🏃 Main 入口
# ======================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║      LangChain Phase 2-5: Retriever 教程             ║
    ║      6种检索策略：similarity / MMR / BM25 / ...      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    print("\n📖 教程导航:")
    print("   Demo0  → 概念理解（Retriever 是什么？）")
    print("   Demo2  → 基础检索（VectorStore.as_retriever）")
    print("   Demo3  → 参数调节（k值、阈值等）")
    print("   Demo4  → MMR 检索（相关+多样）")
    print("   Demo5  → 阈值过滤（只返回高质量结果）")
    print("   Demo6  → BM25 关键词检索（精确匹配）")
    print("   Demo7  → Ensemble 混合检索（语义+关键词）")
    print("   Demo8  → MultiQuery 检索（一问多搜）")
    print("   Demo9  → 上下文压缩（精简无关内容）")
    print("   Demo10 → 完整检索 Pipeline")

    # demo_0_retriever_concept()
    # demo_2_vectorstore_retriever()
    # demo_3_search_kwargs()
    # demo_4_mmr_retriever()
    # demo_5_score_threshold()
    # demo_6_bm25_retriever()
    # demo_7_ensemble_retriever()
    # demo_8_multi_query_retriever()
    demo_9_contextual_compression()
    demo_10_complete_retriever_pipeline()

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║         Phase 2-5 Retriever 全部完成 ✅               ║
    ║         下一阶段: Phase 2-6 完整 RAG Pipeline        ║
    ╚══════════════════════════════════════════════════════╝
    """)
