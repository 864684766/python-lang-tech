"""
========================================================

LangChain Phase 2-2
Text Splitters 教材级 Demo


学习目标:

1. 理解为什么需要文本切分
2. 掌握 LangChain Text Splitter体系
3. 理解:
        Document
            |
            v
        Splitter
            |
            v
        List[Document]

4. 为后续 RAG:
        Embedding
        VectorStore
        Retriever

    做准备。


--------------------------------------------------------

知识路线:

Demo0:
    为什么需要Text Splitter


Demo1:
    CharacterTextSplitter

    最简单字符切分


Demo2:
    RecursiveCharacterTextSplitter

    企业RAG最常用


Demo3:
    MarkdownHeaderTextSplitter

    保留标题结构


Demo4:
    HTMLHeaderTextSplitter

    网页知识库


Demo5:
    Document完整切分流程


Demo6:
    SemanticChunker

    高级语义切分


========================================================
"""

import os

# ======================================================
# 工具函数
# ======================================================


def load_text(file_path: str | None, default_text: str) -> str:
    """
    读取文本辅助函数


    参数:
        file_path:
            本地文件路径

        default_text:
            如果文件不存在，使用默认文本


    返回:

        str:
            文本内容

    """

    if file_path and os.path.exists(file_path):

        print(f"读取文件:{file_path}")

        with open(file_path, "r", encoding="utf-8") as f:

            return f.read()

    if file_path:

        print("文件不存在，使用默认文本")

    return default_text


# ======================================================
# Demo0
# 为什么需要Text Splitter
# ======================================================


def demo_0_why_splitter():
    """
    学习目标:
    理解为什么大模型应用需要文本切分。
    问题:
    一个10万字PDF:
        |
        v
    直接发送LLM?
    会导致:
    1. Token过多
    2. 成本增加
    3. 检索不准确
    所以需要:
        Document
            |
        Text Splitter
            |
        Chunk
    """

    print("\n========== Demo0 为什么需要切分 ==========")

    text = """
    LangChain 是大模型应用开发框架。
    RAG系统需要先读取知识库。
    知识库可能包含大量PDF、网页、数据库数据。
    如果一次发送全部内容给LLM，
    会造成上下文过长。
    因此需要文本切分。
    """

    print("原始文本长度:", len(text))

    print("""
        切分流程:
        Document
           |
           v
        Text Splitter
           |
           v
        Chunk1
        Chunk2
        Chunk3
    """)


# ======================================================
# Demo1
# CharacterTextSplitter
# ======================================================


def demo_1_character_splitter(file_path: str | None = None):
    """
    最基础切分器。
    学习目标:
        理解:
        chunk_size
        chunk_overlap
    参数:
        chunk_size:
            每个块最大长度
        chunk_overlap:
            相邻块重复内容
    """

    print("\n========== Demo1 CharacterTextSplitter ==========")

    from langchain_text_splitters import CharacterTextSplitter

    text = load_text(
        file_path,
        """

        LangChain是一套构建大模型应用的框架。

        它支持RAG和Agent开发。

        文档切分是RAG的重要步骤。

        """,
    )

    splitter = CharacterTextSplitter(
        # 每个chunk最大字符
        chunk_size=50,
        # 重叠字符数量
        chunk_overlap=10,
        # 按换行切
        separator="\n",
    )

    chunks = splitter.split_text(text)

    print("Chunk数量:", len(chunks))

    for index, chunk in enumerate(chunks, 1):

        print(f"\nChunk {index}:")

        print(chunk)


# ======================================================
# Demo2
# RecursiveCharacterTextSplitter
# ======================================================


def demo_2_recursive_splitter(file_path=None):
    """
    企业RAG最常使用。
    切分优先级:
    1. 段落
    2. 换行
    3. 空格
    4. 字符
    优点:
    尽量保持语义完整。
    """

    print("\n========== Demo2 RecursiveCharacterTextSplitter ==========")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = load_text(
        file_path,
        """
        第一章：LangChain介绍
        LangChain是一个LLM应用开发框架。
        第二章 RAG系统
        RAG通过检索外部知识增强模型回答能力。
        第三章 Agent
        Agent可以调用工具完成复杂任务。
        """,
    )

    splitter = RecursiveCharacterTextSplitter(
        # chunk大小
        chunk_size=5,
        # 保留上下文
        chunk_overlap=2,
        # 中文环境增加标点
        separators=["\n\n", "\n", "。", "！", "？", " "],
    )

    chunks = splitter.split_text(text)

    print("生成Chunk:", len(chunks))

    for i, c in enumerate(chunks, 1):

        print(f"\n--- Chunk {i} ---")

        print(c)


# ======================================================
# Demo3
# MarkdownHeaderTextSplitter
# ======================================================


def demo_3_markdown_header_splitter(file_path=None):
    """
    Markdown知识库切分。

    特点:
    保留标题信息。
    适合:
        技术文档
        README
        API文档
    """

    print("\n========== Demo3 MarkdownHeader ==========")

    from langchain_text_splitters import MarkdownHeaderTextSplitter

    markdown = load_text(
        file_path,
        """
        # LangChain教程
        ## Document Loader
        负责加载数据。
        ## Text Splitter
        负责文本切分。
        ## Retriever
        负责搜索相关内容。
        """,
    )

    headers = [("#", "一级标题"), ("##", "二级标题")]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

    docs = splitter.split_text(markdown)

    for doc in docs:

        print("\n内容:")

        print(doc.page_content)

        print("Metadata:")

        print(doc.metadata)


# ======================================================
# Demo4
# HTMLHeaderTextSplitter
# ======================================================


def demo_4_html_header_splitter():
    """
    HTML网页知识库切分。
    常用于:
        网站
        帮助中心
        产品文档
    """

    print("\n========== Demo4 HTML Header ==========")

    from langchain_text_splitters import HTMLHeaderTextSplitter

    html = """
    <h1>Python教程</h1>
    <p>Python是一种编程语言。</p>
    <h2>数据类型</h2>
    <p>列表和字典是常见结构。</p>
    <h2>Examples</h2>
    <p>Create a camera.</p>
    <pre><code class="language-python">camera_id = workspace.create_entity(...)</code></pre>
    """

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[("h1", "一级标题"), ("h2", "二级标题")],
        return_each_element=True,
    )

    docs = splitter.split_text(html)

    for doc in docs:

        print("\n内容:")

        print(doc.page_content)

        print("Metadata:", doc.metadata)

# ======================================================
# Main
# ======================================================


if __name__ == "__main__":

    print("""
        =================================
        
        LangChain Phase2-2
        
        Text Splitters 教材Demo
        
        =================================
        """)

    demo_0_why_splitter()

    demo_1_character_splitter(file_path="./testfiles/大气层介绍.txt")

    demo_2_recursive_splitter()

    demo_3_markdown_header_splitter()

    demo_4_html_header_splitter()

    print("\n全部Demo运行结束")
