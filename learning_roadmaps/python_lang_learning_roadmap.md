# Python 语言学习路线

给会 C# / JS、刚跑完类型课的人。类、标注、`with`、异步。程序员基础，不绑死 LangChain。

**不需要 API Key。** 前置：[Python 类型](./python_basics_learning_roadmap.md)。

## 怎么跑

在**仓库根目录**执行：

```bash
python python_lang/pl_run_all.py

python python_lang/pl_0_oop.py
python python_lang/pl_1_typing.py
python python_lang/pl_2_contextlib.py
python python_lang/pl_3_asyncio.py
```

## 顺序

每课文件头和运行开头都有「是什么 / 用在 / 不用」。先看总览再跑。

| 节 | 文件 | 是什么 | 什么时候用 | 不用 | 盯住的反例 |
|----|------|--------|------------|------|------------|
| ⓪ | `pl_0_oop.py` | 用 class 绑数据和操作 | 写 Tool / 读 `.invoke()`；分清 dataclass 和 Enum | 只传一份 dict 就够；校验用 Pydantic | 忘了 `self`；把 `_foo` 当 C# `private` |
| ① | `pl_1_typing.py` | 给参数写说明书（像 TS） | 给人看；给 `@tool` / FastAPI / LangGraph 读 | 别指望 `: str` 挡住 `echo(123)` | 以为写了标注就会校验；`Annotated` 只当注释 |
| ② | `pl_2_contextlib.py` | 进入开、离开关 | 文件、MCP session、个数不定的资源 | 没有开关这对动作 | 出了 `with` 还读文件对象 |
| ③ | `pl_3_asyncio.py` | 单线程交错等 I/O | `gather`、FastAPI 中间件、MCP、`ainvoke` | 纯 CPU；普通脚本的 `sleep` | 只调用协程函数不 `run`；`async` 里 `time.sleep` |

FastAPI `fb_2` 的 `async def`、LangChain `5_3` 的 `ainvoke`、MCP `7_2` 都建立在 ③ 上，那些文件**不是**异步语法课。

下一步：[标准库](./python_stdlib_learning_roadmap.md)，然后 [FastAPI](./fastapi_learning_roadmap.md)。
