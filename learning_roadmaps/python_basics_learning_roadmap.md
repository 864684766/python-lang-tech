# Python 类型学习路线

给完全新手。每种类型一个文件，带【正例】和会报错的【反例】。对照 JS 的地方写在注释里。⑩～⑫ 补 `enumerate`、标准库 `array`、第三方 NumPy（RAG 向量）。

**不需要 API Key。**

## 怎么跑

在**仓库根目录**执行（不要在 `learning_roadmaps/` 里跑）：

```bash
# 全部
python python_basics/pb_run_all.py

# 单节
python python_basics/pb_0_types_overview.py
```

## 顺序

| 节 | 文件 | 学什么 | 盯住的反例 |
|----|------|--------|------------|
| ⓪ | `python_basics/pb_0_types_overview.py` | 有哪些常用类型、`type()`、可变/不可变 | `'3' + 1` 不会像 JS 变成 `'31'` |
| ① | `pb_1_none.py` | `None`；`is None` | `if not name` 会把 `''` 也当成没有 |
| ② | `pb_2_bool.py` | `True`/`False`；什么算假 | 写成 `true`；空 `[]` 在 Python 是假 |
| ③ | `pb_3_int_float.py` | `int` / `float`；`/` 与 `//` | `int('3.9')`；`0.1+0.2 == 0.3` |
| ④ | `pb_4_str.py` | f-string、`strip`、`split`/`join` | 只调用 `strip()` 不赋值，原串不变 |
| ⑤ | `pb_5_list.py` | 可变列表、遍历、复制 | `b = a` 不是复制；没有 `.find()` |
| ⑥ | `pb_6_tuple.py` | 不可变、解包 | `('hello')` 不是元组；不能改下标 |
| ⑦ | `pb_7_dict.py` | `.get`、`in` 查键、`**` 展开 | 没有的键用 `[]` 会 `KeyError` |
| ⑧ | `pb_8_set.py` | 去重、并交差 | `{}` 是空字典，空集合写 `set()` |
| ⑨ | `pb_9_range.py` | `for i in range(n)` | `range(3)` 不含 `3` |
| ⑩ | `pb_10_enumerate.py` | 给 list 编号；Agent 遍历工具 / `tool_calls` / 消息 | `enumerate(xs)[0]` 会 TypeError |
| ⑪ | `pb_11_array.py` | `array.array`；内置 vs 标准库 vs pip | JS Array ≠ `array`；`array` 不能塞 dict |
| ⑫ | `pb_12_numpy.py` | `ndarray`；假 embedding 余弦相似度 | `+` 不是拼接；`if 数组` 会 ValueError |

⑫ 需要 `numpy`（`pip install numpy`）。没装时该节会跳过，不挡后面的课。

## 学完能干什么

读 LangChain / MCP 代码时，配置和 `call_tool` 参数就是 `dict`，工具列表是 `list`，找不到用 `None`。给工具或消息编号用 `enumerate`。Embedding 接口给你 `list[float]`，检索时再 `np.array`。标准库 `array` 不是日常容器。

下一步：[Python 语言](./python_lang_learning_roadmap.md)（类 / 标注 / with / 异步），再 [标准库](./python_stdlib_learning_roadmap.md)，然后 [FastAPI](./fastapi_learning_roadmap.md)。
