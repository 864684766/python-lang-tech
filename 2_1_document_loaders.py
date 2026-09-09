"""
=====================================================
LangChain Document Loader 教材级 Demo

章节:
    Phase 2-1 Document Loaders

学习目标:
    1. 理解 LangChain Document 对象
    2. 掌握不同数据源加载方式
    3. 理解 Loader -> Document 流程
    4. 为 RAG 学习做准备

LangChain核心流程:

    数据源
      |
      |
    Document Loader
      |
      |
    List[Document]
      |
      |
    Text Splitter
      |
      |
    Embedding
      |
      |
    VectorStore


安装:

pip install langchain
pip install langchain-community
pip install pypdf
pip install beautifulsoup4
pip install python-docx
pip install unstructured
pip install pdfminer.six

=====================================================
"""

import os
import tempfile
import shutil
from pathlib import Path

# =====================================================
# Demo0: Document对象基础
# =====================================================


def demo_0_document_basic():
    """
    学习 LangChain 最核心的数据结构: Document

    Document包含两个核心字段:
    1. page_content - 文档正文
    2. metadata     - 文档附加信息

    后续所有 Loader: CSV / PDF / Web / Word / 自定义
    最终都会生成: List[Document]
    """
    print("\n========== Demo0 Document基础 ==========")

    from langchain_core.documents import Document

    doc = Document(
        page_content="LangChain 是一个构建LLM应用的开发框架",
        metadata={"source": "manual", "page": 1},
    )

    print("正文:", doc.page_content)
    print("Metadata:", doc.metadata)


# =====================================================
# Demo1: TextLoader
# =====================================================


def demo_1_text_loader(file_path=None):
    """
    TextLoader: 最简单的文本加载器
    适合 .txt 文件
    """
    print("\n========== Demo1 TextLoader ==========")

    from langchain_community.document_loaders import TextLoader

    temp_file = None
    try:
        if file_path is None:
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", encoding="utf-8", delete=False
            )
            f.write("""LangChain学习笔记

Document Loader负责加载数据。
TextLoader是最简单的加载器，适用于纯文本文件。
""")
            f.close()
            temp_file = f.name
        else:
            temp_file = file_path

        loader = TextLoader(temp_file, encoding="utf-8")
        documents = loader.load()

        print(f"Document数量: {len(documents)}")
        for doc in documents:
            print(f"\n内容预览: {doc.page_content[:60]}...")
            print(f"Metadata: {doc.metadata}")

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


# =====================================================
# Demo2: CSVLoader
# =====================================================


def demo_2_csv_loader(csv_path=None):
    """
    CSVLoader: 读取结构化CSV文件
    每行转换为一个 Document
    """
    print("\n========== Demo2 CSVLoader ==========")

    from langchain_community.document_loaders import CSVLoader

    temp_file = None
    try:
        if csv_path is None:
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", encoding="utf-8", delete=False
            )
            f.write("name,role\nTom,Developer\nJack,Manager\n")
            f.close()
            temp_file = f.name
        else:
            temp_file = csv_path

        loader = CSVLoader(temp_file, encoding="utf-8")
        documents = loader.load()

        print(f"加载数量: {len(documents)}")
        for doc in documents:
            print(f"\n---\n内容: {doc.page_content}\nMetadata: {doc.metadata}")

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


# =====================================================
# Demo3: DirectoryLoader
# =====================================================


def demo_3_directory_loader(directory=None):
    """
    DirectoryLoader: 批量加载目录下的文件
    支持 glob 模式匹配
    """
    print("\n========== Demo3 DirectoryLoader ==========")

    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    temp_dir = None
    try:
        if directory is None:
            temp_dir = tempfile.mkdtemp()
            for i in range(3):
                path = os.path.join(temp_dir, f"doc{i}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"这是第{i}个知识文档\n包含重要的AI知识。")
            directory = temp_dir

        loader = DirectoryLoader(
            directory,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()

        print(f"目录文档数量: {len(documents)}")
        for doc in documents:
            print(f"\n---\n内容: {doc.page_content}\n来源: {doc.metadata.get('source')}")

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# =====================================================
# Demo4: PyPDFLoader (PDF文档)
# =====================================================


def demo_4_pypdf_loader(pdf_path=None):
    """
    PyPDFLoader: 加载PDF文件
    每个页面转换为一个 Document

    依赖: pip install pypdf
    """
    print("\n========== Demo4 PyPDFLoader ==========")

    from langchain_community.document_loaders import PyPDFLoader

    # 尝试使用测试文件，如果没有则跳过
    
    if not os.path.exists(pdf_path):
        print("[跳过] 暂无 sample.pdf 测试文件")
        print("提示: 可以将任意PDF文件复制到 ./testfiles/ 目录下测试")
        return

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"PDF页数: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"\n--- 第{i+1}页 ---")
        print(f"内容预览: {doc.page_content[:80]}...")
        print(f"Metadata: {doc.metadata}")


# =====================================================
# Demo5: UnstructuredLoader (Word/PPT/Excel/HTML)
# =====================================================


def demo_5_unstructured_loader(file_path=None):
    """
    UnstructuredLoader: 加载多种格式`文件
    支持: .docx, .pptx, .xlsx, .html, .md 等

    依赖: pip install "unstructured[all-docs]"
    """
    print("\n========== Demo5 UnstructuredLoader ==========")

    # 尝试加载各种格式的示例
    test_files = [
        ("./testfiles/独孤九剑专题：01三点估算法.pptx", "PPTX"),
        ("./testfiles/第四版十大知识点、五大过程组0730.xlsx", "XLSX"),
        ("./testfiles/index.html", "HTML"),
    ]

    found = False
    for path, fmt in test_files:
        if os.path.exists(path):
            found = True
            try:
                from langchain_community.document_loaders import UnstructuredFileLoader
                loader = UnstructuredFileLoader(path, mode="elements")
                documents = loader.load()

                print(f"\n[{fmt}] 加载了 {len(documents)} 个元素")
                for i, doc in enumerate(documents):  # 只显示前3个
                    print(f"  [{i}] {doc.page_content[:60]}...")
                    print(f"       类型: {doc.metadata.get('category', 'N/A')}")
            except Exception as e:
                print(f"[{fmt}] 加载失败: {e}")

    if not found:
        print("[跳过] 暂无示例文件")
        print("提示: 可将 .docx / .html 文件放到 ./testfiles/ 目录下测试")


# =====================================================
# Demo6: WebBaseLoader (网页抓取)
# =====================================================


def demo_6_web_base_loader():
    """
    WebBaseLoader: 从网页抓取内容
    使用 BeautifulSoup 解析 HTML

    依赖: pip install beautifulsoup4 lxml
    """
    print("\n========== Demo6 WebBaseLoader ==========")

    from langchain_community.document_loaders import WebBaseLoader

    # 示例：加载一个技术博客页面
    urls = [
        "https://news.ycwb.com/ikimvkctjo/content_54268594.htm",
    ]

    try:
        loader = WebBaseLoader(urls)
        documents = loader.load()

        print(f"抓取页面数: {len(documents)}")
        for i, doc in enumerate(documents):
            print(f"\n--- 页面{i+1} ---")
            print(f"标题: {doc.metadata.get('title', 'N/A')}")
            print(f"内容预览: {doc.page_content[:150]}...")
            print(f"来源: {doc.metadata.get('source', 'N/A')}")
    except Exception as e:
        print(f"[跳过] 网页加载失败（可能需要网络）: {e}")


# =====================================================
# Demo7: 自定义Loader（继承BaseLoader）
# =====================================================


def demo_7_custom_loader():
    """
    自定义Loader: 继承 BaseLoader
    适用于特殊数据源（数据库、API、企业系统）

    关键步骤:
    1. 继承 BaseLoader
    2. 实现 _load_documents() 方法
    3. 返回 List[Document]
    """
    print("\n========== Demo7 自定义Loader ==========")

    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    from typing import List, Optional

    class JSONLoader(BaseLoader):
        """
        自定义JSON文件加载器
        将JSON数组中的每个对象转换为一个Document
        """

        def __init__(self, file_path: str, content_key: str = "text"):
            super().__init__()
            self.file_path = file_path
            self.content_key = content_key

        def load(self) -> List[Document]:
            import json

            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            for i, item in enumerate(data):
                # 提取内容
                content = item.get(self.content_key, str(item))

                # 构建元数据
                metadata = {
                    "source": self.file_path,
                    "index": i,
                    "total": len(data),
                }
                # 添加其他字段到metadata
                for key, value in item.items():
                    if key != self.content_key:
                        metadata[key] = value

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            return documents

    class APIDataLoader(BaseLoader):
        def __init__(self, url, method="GET", headers=None, params=None, json_body=None):
            self.url = url
            self.method = method
            self.headers = headers or {}
            self.params = params
            self.json = json_body

        def load(self):

            import requests

            response = requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                params=self.params,
                json=self.json,
            )

            response.raise_for_status()

            data = response.json()

            docs = []

            for item in data["data"]:

                docs.append(
                    Document(
                        page_content=item["content"],
                        metadata={"id": item["id"], "title": item["title"]},
                    )
                )

            return docs

    # 演示1：JSON加载器
    print("\n【示例1】JSONLoader")
    temp_json = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    )
    temp_json.write('[{"text": "LangChain是优秀的框架", "author": "开发者A"}, {"text": "RAG提升回答质量", "author": "研究者B"}]')
    temp_json.close()

    json_loader = JSONLoader(temp_json.name)
    docs = json_loader.load()
    print(f"加载了 {len(docs)} 条记录")
    for doc in docs:
        print(f"  - {doc.page_content[:30]}... (作者: {doc.metadata.get('author')})")

    os.remove(temp_json.name)

    # 演示2：API数据加载器
    print("\n【示例2】APIDataLoader")
    api_loader = APIDataLoader(
        "https://api.xxx.com/knowledge/search",
        method="POST",
        headers={
            "Authorization":"Bearer xxx"
        },
        json_body={
            "keyword":"LangChain",
            "page":1,
            "size":20
        }
    )
    docs = api_loader.load()
    print(f"加载了 {len(docs)} 条知识记录")
    for doc in docs:
        print(f"  - [{doc.metadata['title']}] {doc.page_content[:30]}...")

    print("\n💡 要点:")
    print("   · 继承 BaseLoader 并实现 load() 方法")
    print("   · 返回 List[Document]，每个Document包含 page_content 和 metadata")
    print("   · metadata 可用于追踪数据来源、分页信息等")


# =====================================================
# 主程序
# =====================================================


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║      LangChain Document Loader 教材Demo              ║
    ║         (Phase 2-1)                                  ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # Demo0: 基础概念
    demo_0_document_basic()

    # Demo1: TextLoader
    demo_1_text_loader()

    # Demo2: CSVLoader
    demo_2_csv_loader()

    # Demo3: DirectoryLoader
    demo_3_directory_loader()

    # Demo4: PyPDFLoader
    demo_4_pypdf_loader(pdf_path="./testfiles/独孤九剑专题：03投资回收期.pdf")

    # Demo5: UnstructuredLoader
    demo_5_unstructured_loader()

    # Demo6: WebBaseLoader
    demo_6_web_base_loader()

    # Demo7: 自定义Loader
    demo_7_custom_loader()

    print("\n✅ 所有Loader示例执行完毕！")
    print("\n📚 下一步: 学习 Text Splitters (2_2_text_splitters.py)")
