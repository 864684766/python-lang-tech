"""
Python 类型教材 ⑫ NumPy ndarray
第三方（pip）里的多维数字数组。RAG / 向量检索会碰到；不是 list，也不是标准库 array。
运行: python python_basics/pb_12_numpy.py
需要: pip install numpy（本仓库 RAG demo 一般已经装过）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_counter,
    print_lesson,
    print_ok,
    print_section,
    print_tip,
    print_trap,
)


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑫ NumPy ndarray：向量计算用的数组（第三方）")
    print("=" * 65)

    try:
        import numpy as np
    except ImportError:
        print_trap(
            "没有安装 numpy，本节跳过",
            "pip install numpy    后面 Embedding / RAG 也要用。",
        )
        return

    print_section("三种「array」不要混")
    print("    list              内置，任意值，Agent 消息/工具用这个")
    print("    array.array       标准库，一维同类数字，几乎不用")
    print("    numpy.ndarray     pip，可多维，算 embedding 相似度用这个")
    print_ok(f"np.array([0.1, 0.5, 0.2]) type={type(np.array([0.1, 0.5, 0.2]))}")

    print_section("长什么样：shape / dtype / 下标")
    vec = np.array([0.1, 0.5, 0.2], dtype=np.float64)
    print_ok(f"{vec}    shape={vec.shape}  dtype={vec.dtype}  vec[0]={vec[0]}")
    docs = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    print_ok(f"一批文档向量 shape={docs.shape}  （篇数, 维度）")
    print_ok(f"第 0 篇 docs[0] → {docs[0]}")

    print_section("Agent / RAG：API 给的是 list，算相似度再转 ndarray")
    # 和 rag_pipeline_demo / 2_3_embeddings_teacher 同一套公式，这里用 3 维假向量
    query = [1.0, 0.0, 0.0]  # 接口/模型返回的通常是 list[float]
    weather = [0.9, 0.1, 0.0]
    code = [0.0, 0.1, 0.9]

    def cosine(a: list[float], b: list[float]) -> float:
        x = np.array(a, dtype=np.float64)
        y = np.array(b, dtype=np.float64)
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom == 0:
            return 0.0
        return float(np.dot(x, y) / denom)

    print_ok(f"query vs 天气文档 → {cosine(query, weather):.3f}    更接近")
    print_ok(f"query vs 代码文档 → {cosine(query, code):.3f}    更远")
    print_ok("JSON 往外送用 .tolist()，不要直接 json.dumps(ndarray)")

    print_section("常见报错 / 能跑但是错")
    print_trap(
        "list 的 + 是拼接，ndarray 的 + 是逐个相加",
        f"[1,2]+[3,4]={[1, 2] + [3, 4]}    "
        f"np.array([1,2])+np.array([3,4])={np.array([1, 2]) + np.array([3, 4])}",
    )

    def as_bool() -> None:
        if np.array([0.1, 0.5]):
            pass

    print_counter("if ndarray 不能当 list 那样判断空不空", as_bool, ValueError)
    print_counter(
        "维度对不上不能相加",
        lambda: np.array([1.0, 0.2]) + np.array([1.0, 0.2, 0.3]),
        ValueError,
    )
    print_ok("工具名、messages 继续用 list，不要 np.array(['search', ...])")

    print_tip(
        "Embedding 接口给你 list[float]；检索时 np.array 再 np.dot。"
        " 完整 RAG 见 2_3_embeddings_teacher.py / rag_pipeline_demo.py。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
