"""
Python 语言教材 ⓪ 面向对象（给会 C# 的人）

是什么:
  用 class 把数据和操作绑在一起。封装 / 继承 / 多态都在。
  和 C# 同构，细节不同：可多继承、没有强制 private、多态常靠鸭子类型。

用在:
  1) 写 Tool / Agent 节点这种「有状态、有方法」的对象
  2) 读 LangChain 源码里的 class Prompt / Model —— 都靠 .invoke()
  3) 分清 @dataclass 模型 和 Enum 固定选项

不用:
  只传一份 dict 就够的配置，不必先上 class
  校验对外数据用 Pydantic，不是 dataclass

运行: python python_lang/pl_0_oop.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_counter,
    print_lesson,
    print_ok,
    print_overview,
    print_section,
    print_tip,
    print_trap,
)


class Tool:
    """普通 class：数据在实例上，方法第一个参数是 self。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._calls = 0  # 下划线 = 约定「别从外面改」，不是 C# private

    def call(self) -> str:
        self._calls += 1
        return f"{self.name} 第 {self._calls} 次"

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


class SearchTool(Tool):
    """单继承：复用 Tool 的 __init__ / call，再加自己的方法。"""

    def search(self, q: str) -> str:
        return f"{self.name} 搜 {q!r}"


class TimestampMixin:
    """混入：只提供能力，不单独当模型用。"""

    def stamp(self) -> str:
        return f"{type(self).__name__} ok"


class LoggedSearch(SearchTool, TimestampMixin):
    """多继承：C# 一个父类 + 多个接口；Python 可以多个父类。"""

    pass


class Prompt:
    def invoke(self, text: str) -> str:
        return f"prompt:{text}"


class FakeModel:
    def invoke(self, text: str) -> str:
        return f"model:{text}"


def run_invoke(x) -> str:
    """不声明接口。有 .invoke 就能跑 —— 鸭子类型。"""
    return x.invoke("hi")


@dataclass
class Skill:
    name: str
    description: str


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"


def demo() -> None:
    print("\n" + "=" * 65)
    print("⓪ 面向对象：和 C# 同构，细节不同")
    print("=" * 65)

    print_overview(
        "用 class 把数据和操作绑在一起。封装 / 继承 / 多态都在。",
        [
            "写 Tool / Agent 节点这种「有状态、有方法」的对象",
            "读 LangChain 源码：Prompt / Model 都靠 .invoke()，不问是不是同一个父类",
            "分清 @dataclass 模型 和 Enum 固定选项",
        ],
        "只传一份 dict 就够的配置不必先上 class。校验对外数据用 Pydantic，不是 dataclass。",
    )

    print_section("封装：数据和方法在实例上")
    t = Tool("lookup")
    print_ok(f"{t}  call → {t.call()}  name={t.name}")
    print_trap(
        "当成 C# 的 private",
        f"t._calls={t._calls} 仍然读得到。下划线只是约定，解释器不拦。",
    )

    print_section("继承：class 子(父)")
    s = SearchTool("search")
    print_ok(f"SearchTool 是 Tool 的子类: {isinstance(s, Tool)}")
    print_ok(s.search("LangGraph"))

    print_section("多继承（C# 没有「两个 class 当父类」）")
    logged = LoggedSearch("search")
    print_ok(f"{logged.search('mcp')}  | mixin: {logged.stamp()}")
    print_ok(f"MRO（同名方法从左到右找）: {[c.__name__ for c in LoggedSearch.__mro__]}")

    print_section("多态 / 鸭子类型：不问出身，问会不会")
    print_ok(f"Prompt.invoke → {run_invoke(Prompt())}")
    print_ok(f"FakeModel.invoke → {run_invoke(FakeModel())}")
    print_trap(
        "C# 通常要 IRunnable；Python 默认只看有没有方法",
        "LangChain 里 Prompt / Model / Parser 都能 .invoke()，不必先继承同一棵树。",
    )

    print_section("模型类 ≠ 枚举")
    skill = Skill(name="weather", description="查天气")
    print_ok(f"@dataclass 模型: {skill}")
    print_ok(f"Enum 是固定选项: {Role.USER}  value={Role.USER.value}")
    print_counter("枚举没有的取值", lambda: Role("admin"), ValueError)

    def forgot_self():
        class Bad:
            def call():  # noqa: N805
                return "x"

        Bad().call()

    print_counter("实例方法忘了写 self", forgot_self, TypeError)
    print_ok("方法第一个参数必须是 self（C# 的 this 是隐式的）")

    print_tip(
        "三大特征都在。差别：无强制 private、默认可多继承、多态常靠鸭子类型。"
        " 对外校验用 Pydantic（FastAPI fb_1），不要把 dataclass 当 DTO 防火墙。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
