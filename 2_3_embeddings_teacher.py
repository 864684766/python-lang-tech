"""
========================================================

LangChain Phase 2-3: Embeddings 教材级 Demo
（完整版注释 + 应用场景说明）


📚 学习目标：
1. 理解 Embedding 是什么、为什么需要它
2. 掌握 LangChain Embedding 接口（Fake / Ollama / ZhipuAI）
3. 理解语义相似度计算（余弦相似度）
4. 为 RAG（检索增强生成）打基础


🔗 知识脉络：
┌─────────────────────────────────────────────────────────────┐
│  传统搜索：关键词匹配（"Python" ≠ "编程语言"）               │
│                                                             │
│  向量搜索：语义匹配（"Python" ≈ "编程语言"）                │
│           ↓                                                 │
│        Embedding：把文本变成向量                             │
│           ↓                                                 │
│      RAG 检索：找最相似的文档片段                            │
└─────────────────────────────────────────────────────────────┘


========================================================
"""

import os
from typing import List

import numpy as np

from dotenv import load_dotenv  # 从 .env 文件中自动加载环境变量

# ======================================================
# 📖 Demo0 - Embedding 概念入门
# 
# 【前因】为什么需要 Embedding？
#   - 计算机不认识文字，只认识数字
#   - 我们需要把"语言"翻译成"数学向量"
#   - 这样就能计算"两个句子有多相似"
#
# 【应用场景】
#   - 搜索引擎：用户搜"苹果手机"，能找到"iPhone"相关结果
#   - 推荐系统：根据你喜欢的电影，推荐相似风格
#   - 聊天机器人：找到最相关的知识库内容回答用户
#
# 【核心思想】
#   语义相近的文本 → 向量在空间中距离很近
#   例如："cat" 和 "kitty" 的向量比 "car" 更接近
# ======================================================

def demo_0_embedding_concept() -> None:
    """
    介绍 Embedding 基本概念。
    
    想象过程：
    ┌──────────┐     ┌─────────────────┐     ┌──────────────┐
    │  文本     │ ──▶ │   Embedding 模型 │ ──▶ │  数字向量     │
    │ "你好"   │     │  (神经网络)       │     │ [0.1, 0.5...] │
    └──────────┘     └─────────────────┘     └──────────────┘
    """

    print("\n========== Demo0: Embedding 概念 ==========")

    print("""
    📌 什么是 Embedding？
    ─────────────────────────────────────────
    
    简单说：把「文字」变成「一串数字」。
    
    示例：
    
    输入文本："LangChain 是一个 LLM 开发框架"
    
    经过 Embedding 模型处理后：
    
    输出向量（以 3 维为例，实际通常是 768~4096 维）：
    [
      0.123,   ← 第一个维度
      0.532,   ← 第二个维度  
      0.876,   ← 第三个维度
      ...
    ]
    
    💡 关键特性：
    • 相似的文本 → 向量距离近（比如"猫"和"猫咪"）
    • 不相关的文本 → 向量距离远（比如"猫"和"汽车"）
    • 可以用数学公式计算相似度（后面 Demo4 会演示）
    """)


# ======================================================
# 📖 Demo1 - FakeEmbedding（测试专用）
# 
# 【为什么要有这个？】
#   - 正式使用 Embedding 需要先调 API / 下载模型
#   - 学习初期我们只想理解"接口怎么用"
#   - FakeEmbedding 随机生成向量，帮我们练习接口调用
#
# 【实际项目中的选择】
#   1. OllamaEmbeddings  — 本地运行，免费，适合教学/私有数据
#   2. OpenAIEmbeddings  — 收费，效果好，需联网
#   3. BGE / GTE         — 开源模型，中文效果好
#   4. 智谱AI / 通义千问  — 国内 API，延迟低
#
# 【接口说明】
#   - embed_query(text)   → 把单个文本转成向量
#   - embed_documents(list) → 批量转换多个文本
# ======================================================

def demo_1_fake_embedding() -> None:
    """
    使用 FakeEmbedding 熟悉 LangChain 接口。
    
    ⚠️ 注意：这只是教学工具！
    生成的向量是随机的，没有任何语义信息。
    真实场景中必须使用专业模型。
    """

    print("\n========== Demo1: FakeEmbedding 接口演示 ==========")

    # 导入测试用的 Mock 模型
    from langchain_community.embeddings import FakeEmbeddings

    # 创建模型实例，size=5 表示生成 5 维向量
    embedding_model = FakeEmbeddings(size=5)

    text: str = "LangChain 是什么？"

    # 调用 embed_query 方法：单条文本 → 向量
    vector: List[float] = embedding_model.embed_query(text)

    print(f"📝 输入文本: {text}")
    print(f"\n📐 生成向量 (共 {len(vector)} 维):")
    print(f"   {vector}")

    # 展示两种方法的差异
    print("\n--- 对比：单条 vs 批量 ---")
    
    texts_batch = ["苹果", "香蕉", "橘子"]
    vectors_batch = embedding_model.embed_documents(texts_batch)
    
    print(f"📝 输入文本: {texts_batch}")
    print(f"\n📐 批量输出:")
    for t, v in zip(texts_batch, vectors_batch):
        print(f"   '{t}' → {v}")


# ======================================================
# 📖 Demo2 - Query Embedding（用户提问向量化）
# 
# 【RAG 流程第一步】
#   用户提问 → 问题向量化 → 在数据库中找相似内容 → 返回答案
#
# 【这里做什么】
#   演示如何把用户的自然语言问题转换成向量
#   这个向量将用于后续的向量搜索
#
# 【实际例子】
#   用户在客服系统问："怎么退款？"
#   系统把这个问句转成向量，去知识库找最相似的 FAQ
# ======================================================

def demo_2_query_embedding() -> None:
    """
    演示如何将用户查询（Query）转换为向量。
    
    这是 RAG 检索的第一步！
    后续会用这个向量去向量数据库里搜索最相似的内容。
    """

    print("\n========== Demo2: 用户查询（Query）向量化 ==========")

    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=8)

    # 模拟用户输入的自然语言问题
    query: str = "如何学习 LangChain？"

    # 将问题转换为向量
    query_vector: List[float] = embedding_model.embed_query(query)

    print(f"❓ 用户提问: \"{query}\"")
    print(f"\n📊 问题已转换为向量 (8维):")
    print(f"   {query_vector}")
    print(f"\n💡 下一步：用这个向量去向量数据库搜索相似内容")


# ======================================================
# 📖 Demo3 - Document Embedding（文档知识库向量化）
# 
# 【RAG 流程第二步】
#   把知识库中的所有文档也转成向量，存入向量数据库
#
# 【为什么要批量处理？】
#   - 节省时间：一次性调用比循环调用快得多
#   - 降低延迟：API 调用有网络开销，批量更划算
#   - 减少成本：某些 API 按请求次数计费
#
# 【实际例子】
#   公司知识库有 1000 篇文档
#   系统启动时批量向量化，存入向量数据库（如 Chroma、Milvus）
#   之后每次用户提问，只需求那个问题的向量（Demo2），然后搜索
# ======================================================

def demo_3_document_embedding() -> None:
    """
    演示批量将文档转换为向量。
    
    这是构建 RAG 知识库的关键步骤！
    所有文档先向量化 → 存入向量数据库 → 供后续检索使用
    """

    print("\n========== Demo3: 文档批量向量化 ==========")

    from langchain_community.embeddings import FakeEmbeddings

    embedding_model = FakeEmbeddings(size=4)

    # 假设这是从某处加载的知识库文档列表
    documents: List[str] = [
        "LangChain 是用于构建 LLM 应用的框架",
        "Python 是最流行的编程语言之一",
        "机器学习是人工智能的一个分支",
        "向量数据库用于存储和搜索嵌入向量"
    ]

    # 批量向量化（注意：返回的是二维列表）
    vectors: List[List[float]] = embedding_model.embed_documents(documents)

    print(f"📚 共处理 {len(documents)} 条文档\n")

    for idx, (doc, vec) in enumerate(zip(documents, vectors), 1):
        print(f"【文档 {idx}】")
        print(f"  内容: {doc}")
        print(f"  向量: {vec}")
        print()


# ======================================================
# 📖 Demo4 - 余弦相似度（向量搜索的核心算法）
# 
# 【为什么要算相似度？】
#   当我们有了查询向量和文档向量后，
#   需要知道"哪个文档最符合用户的问题"
#   这就是向量搜索的本质：找角度最接近的向量
#
# 【余弦相似度的直观理解】
#   - 两个向量方向完全相同 → 相似度 = 1.0（最相似）
#   - 两个向量互相垂直     → 相似度 = 0.0（无关）
#   - 两个向量方向相反     → 相似度 = -1.0（完全相反）
#
# 【实际应用】
#   用户提问 → 生成查询向量 → 与库中所有文档向量计算相似度
#   → 取相似度最高的前 K 个文档 → 作为回答的依据
# ======================================================

def demo_4_cosine_similarity() -> None:
    """
    手写字母相似度计算，理解向量搜索的底层原理。
    
    虽然实际项目直接用模型提供的 similarity_score 方法，
    但理解原理有助于调试和优化检索效果。
    """

    print("\n========== Demo4: 余弦相似度计算 ==========")

    def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        """
        计算两个向量的余弦相似度。
        
        公式：
        similarity = (A · B) / (||A|| × ||B||)
        
        参数:
            vector_a: 第一个向量
            vector_b: 第二个向量
        
        返回:
            float: 相似度分数（-1.0 到 1.0）
                  越接近 1.0 越相似，越接近 -1.0 越不相似
        """
        array_a = np.array(vector_a, dtype=float)
        array_b = np.array(vector_b, dtype=float)
        
        # 点积（分子）
        dot_product = np.dot(array_a, array_b)
        
        # 模长（分母）
        norm_a = np.linalg.norm(array_a)
        norm_b = np.linalg.norm(array_b)
        
        # 余弦相似度
        similarity_value = dot_product / (norm_a * norm_b)
        
        return float(similarity_value)

    # 示例 1：完全相同的向量 → 应该得到 1.0
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    score1 = cosine_similarity(vector1, vector2)
    print(f"📍 相同向量比较:")
    print(f"   [{vector1}] vs [{vector2}]")
    print(f"   相似度: {score1:.4f}  ✓ 完全一致")

    # 示例 2：不同的向量 → 应该有差异
    vector3 = [1, 0, 0]
    score2 = cosine_similarity(vector1, vector3)
    print(f"\n📍 不同向量比较:")
    print(f"   [{vector1}] vs [{vector3}]")
    print(f"   相似度: {score2:.4f}  ~ 完全不同方向")

    # 示例 3：半相似 → 介于 0 和 1 之间
    vector4 = [1, 1, 1]
    score3 = cosine_similarity(vector1, vector4)
    print(f"\n📍 部分相似:")
    print(f"   [{vector1}] vs [{vector4}]")
    print(f"   相似度: {score3:.4f}  ~ 有部分共同方向")


# ======================================================
# 📖 Demo5 - Ollama Embedding（本地免费模型）
# 
# 【为什么用本地模型？】
#   1. 隐私敏感数据不能上传云端
#   2. 避免 API 费用积累
#   3. 离线环境也能使用
#   4. 可以 fine-tune 适合自己领域的模型
#
# 【前置准备】
#   1. 安装 Ollama: https://ollama.com
#   2. 拉取模型: ollama pull nomic-embed-text
#      （也可以用 mxbai-embed-large, bge-m3 等）
#
# 【常见模型对比】
#   - nomic-embed-text: 轻量，速度快，中文够用
#   - bge-m3:          多语言，效果好，稍慢
#   - mxbai-embed-large: 英文效果好
# ======================================================

def demo_5_ollama_embedding() -> None:
    """
    使用本地 Ollama 服务进行真实的 Embedding。
    
    这是一个完整的实战示例，展示了如何在生产环境中使用
    本地部署的 Embedding 模型。
    """

    print("\n========== Demo5: Ollama 本地 Embedding ==========")
    print("⚠️  确保已安装 Ollama 并运行了以下命令:")
    print("   $ ollama pull nomic-embed-text")

    try:
        from langchain_ollama import OllamaEmbeddings

        # 连接到本地的 Ollama 服务
        # model 参数指定使用哪个模型
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        text: str = "LangChain RAG 学习指南"

        # 调用真实模型获取向量
        vector: List[float] = embedding_model.embed_query(text)

        print(f"\n📝 输入文本: {text}")
        print(f"📐 向量维度: {len(vector)} 维")
        print(f"📊 前 10 维值: {vector[:10]}")
        print(f"\n✅ 成功！这是真实的语义向量（不是随机数）")
        
        # 保存完整向量到文件（可选）
        print("\n💾 向量特征（可用于后续分析）:")
        print(f"   最小值: {min(vector):.6f}")
        print(f"   最大值: {max(vector):.6f}")
        print(f"   平均值: {sum(vector)/len(vector):.6f}")

    except ImportError:
        print("\n❌ 错误: 未安装 langchain_ollama")
        print("   请运行: pip install langchain_ollama")
    except Exception as error:
        print(f"\n❌ Ollama 连接失败: {error}")
        print("   可能原因:")
        print("   1. Ollama 服务未启动（运行: ollama serve）")
        print("   2. 模型未下载（运行: ollama pull nomic-embed-text）")


# ======================================================
# 📖 Demo6 - 智谱 AI Embedding（国产云端 API）
# 
# 【为什么用国内 API？】
#   1. 国内访问速度快，延迟低
#   2. 支持中文语义理解（比 OpenAI 更好）
#   3. 按量付费，无需购买服务器
#   4. 适合快速原型验证
#
# 【支持的模型】
#   - embedding-2:   基础版，性价比高
#   - embedding-3:   增强版，效果更好
#
# 【环境变量配置】
#   export ZHIPUAI_API_KEY="你的key"
#   或在 .env 文件中设置
# ======================================================

def demo_6_zhipu_embedding() -> None:
    """
    使用智谱 AI 的 Embedding API。
    
    适用于需要在线服务、对中文语义理解要求高的场景。
    """
    load_dotenv()
    print("\n========== Demo6: 智谱 AI Embedding ==========")

    # 从环境变量读取 API Key
    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("\n⚠️  未配置智谱 API Key")
        print("   请设置环境变量: export ZHIPUAI_API_KEY='your-key'")
        print("   或者在 .env 文件中添加: ZHIPUAI_API_KEY=your-key")
        print("\n   获取 API Key: https://open.bigmodel.cn")
        return

    try:
        from langchain_community.embeddings import ZhipuAIEmbeddings

        # 初始化模型，指定使用 embedding-3（效果更好）
        embedding_model = ZhipuAIEmbeddings(
            model="embedding-3",
            api_key=api_key
        )

        # 测试中文理解能力
        test_cases = [
            "人工智能的未来发展",
            "机器学习与深度学习的区别",
            "如何构建 RAG 系统"
        ]

        print("\n📝 测试中文语义理解:")
        for text in test_cases:
            vector = embedding_model.embed_query(text)
            print(f"\n   文本: \"{text}\"")
            print(f"   向量长度: {len(vector)} 维")
            print(f"   前 5 维: {vector[:5]}")

        print("\n✅ 智谱 AI Embedding 工作正常！")

    except ImportError:
        print("\n❌ 未安装 langchain_community")
        print("   请运行: pip install langchain-community")
    except Exception as error:
        print(f"\n❌ 调用失败: {error}")


# ======================================================
# 📖 Demo7 - 完整 RAG 流程演示
# 
# 【RAG 是什么？】
#   RAG = Retrieval Augmented Generation（检索增强生成）
#   
#   简单说：让大模型"开卷考试"
#   先检索相关知识 → 再基于知识回答
#
# 【完整流程】
#   Step 1: 文档切分（Document Splitter）
#           ↓ 把长文档切成小块，便于精确检索
#   Step 2: 向量化（Embedding）
#           ↓ 每块内容都转换成向量
#   Step 3: 存入向量数据库（Vector Store）
#           ↓ 如 Chroma、Milvus、Pinecone
#   Step 4: 用户提问
#           ↓ 问题也转成向量
#   Step 5: 相似度搜索
#           ↓ 找出最相关的文档块
#   Step 6: 组合上下文生成回答
#           ↓ LLM 根据检索到的内容回答问题
#
# 【这个 Demo 演示了】
#   步骤 2 和 3 的简化版（用 FakeEmbedding + 内存存储）
# ======================================================

def demo_7_rag_pipeline() -> None:
    """
    完整演示 RAG 流程中的数据预处理阶段。
    
    这是生产环境的标准做法：
    - 文档先切片（Chunking）避免超出模型上下文限制
    - 再批量向量化存入数据库
    - 查询时只对问题向量化，然后搜索
    """

    print("\n========== Demo7: 完整 RAG 流程演示 ==========")

    from langchain_core.documents import Document
    from langchain_community.embeddings import FakeEmbeddings

    # ─── Step 1: 准备原始文档 ───
    # 模拟从 PDF/网页/数据库加载的内容
    raw_documents = [
        Document(
            page_content="LangChain 提供了开箱即用的组件，可以快速搭建 LLM 应用。",
            metadata={"source": "tech_blog.txt", "section": "intro"}
        ),
        Document(
            page_content="RAG（检索增强生成）通过引入外部知识库来提升大模型的准确性。",
            metadata={"source": "tech_blog.txt", "section": "rag"}
        ),
        Document(
            page_content="向量数据库（如 Chroma、Milvus）专门用于存储和检索 Embedding 向量。",
            metadata={"source": "tech_blog.txt", "section": "database"}
        ),
    ]

    print("📚 Step 1: 加载原始文档")
    for i, doc in enumerate(raw_documents, 1):
        print(f"   [{i}] {doc.page_content[:30]}...")
        print(f"      来源: {doc.metadata.get('source')}")

    # ─── Step 2: 提取文本内容 ───
    # 实际项目中，这一步可能涉及文本清洗、格式转换等
    texts = [doc.page_content for doc in raw_documents]

    # ─── Step 3: 批量向量化 ───
    # 使用 FakeEmbedding 模拟（实际用 Ollama/Zhipu/OpenAI）
    embedding_model = FakeEmbeddings(size=6)
    vectors = embedding_model.embed_documents(texts)

    print(f"\n🔄 Step 2: 批量向量化 ({len(texts)} 条文档)")
    print(f"   向量维度: {len(vectors[0])} 维")

    # ─── Step 4: 展示结果 ───
    # 实际项目中，这里会把 (text, vector, metadata) 存入向量数据库
    print(f"\n💾 Step 3: 存储到向量数据库")
    
    for idx, (doc, vector) in enumerate(zip(raw_documents, vectors), 1):
        print(f"\n   【记录 {idx}】")
        print(f"   内容: {doc.page_content}")
        print(f"   元数据: {doc.metadata}")
        print(f"   向量: {vector}")
        
        # 模拟存入数据库的操作（实际用 db.add_documents）
        print(f"   ✓ 已索引")

    print("\n" + "="*50)
    print("🎯 完整流程总结:")
    print("   原始文档 → 切片 → 向量化 → 存入向量库")
    print("   用户问题 → 向量化 → 相似度搜索 → 返回相关文档")
    print("   LLM 根据相关文档生成最终回答")
    print("="*50)


# ======================================================
# 🏃 Main 入口
# ======================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        LangChain Embeddings 完整教程                 ║
    ║        Phase 2-3: 从概念到 RAG 实战                   ║
    ╚══════════════════════════════════════════════════════╝
    """)

    print("\n📖 教程导航:")
    print("   Demo0  → 概念理解（这是什么？）")
    print("   Demo1  → 接口熟悉（怎么调用？）")
    print("   Demo2  → 查询向量化（RAG 第一步）")
    print("   Demo3  → 文档批量向量化（RAG 第二步）")
    print("   Demo4  → 相似度计算（搜索原理）")
    print("   Demo5  → Ollama 本地模型（生产环境）")
    print("   Demo6  → 智谱 AI API（云端服务）")
    print("   Demo7  → 完整 RAG 流程（实战总结）")

    # 依次运行所有演示
    demo_0_embedding_concept()
    demo_1_fake_embedding()
    demo_2_query_embedding()
    demo_3_document_embedding()
    demo_4_cosine_similarity()
    demo_5_ollama_embedding()
    demo_6_zhipu_embedding()
    demo_7_rag_pipeline()

    print("\n✅ 全部 Demo 运行完成！")
    print("\n💡 下一步学习建议:")
    print("   1. 理解向量数据库（Chroma、Milvus）的工作原理")
    print("   2. 学习 Text Splitter（如何智能切分文档）")
    print("   3. 实践完整的 RAG 应用（结合 LLM 回答）")
