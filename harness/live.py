"""各课共用的模型和道具。

加法、回声、笔记都是道具，用来把 Harness 跑起来。课要看的是谁建议、谁执行。
"""

from __future__ import annotations

from langchain_core.tools import tool

from model_provider import check_api_key, get_model


def live_model():
    """取出本课用的聊天模型。

    先确认 .env 里有 Key。temperature 固定为 0，同一句问法尽量得到同一类回复。
    返回: model_provider.get_model() 的结果，通常是 GLM。
    """
    check_api_key()
    return get_model(temperature=0)


# 道具。docstring 会交给模型当工具说明，所以只写「何时该调用」。
@tool
def add(a: int, b: int) -> str:
    """把两个整数相加。只要做加法，就必须调用这个工具，不要心算。"""
    return str(int(a) + int(b))


@tool
def echo(text: str) -> str:
    """原样返回一段文字。需要复述、确认原文时使用。"""
    return text


@tool
def notes_write(text: str) -> str:
    """把一句话写入笔记。这会改动记录，只有用户明确要求记录时才调用。"""
    return "已记下: " + text


# 系统提示道具：逼模型走 add，这样终端里才看得出「建议」和「执行」是两行。
MUST_USE_ADD = (
    "你是一个会用工具的助手。"
    "任何加减运算都必须调用 add 工具，禁止心算。"
    "得到工具结果后再用中文简短回答。"
)


def print_harness_job(doing: str, *, prop: str) -> None:
    """每课开头印两行，把道具和要学的那层分开。

    doing: 这一课 Harness 新加上的那一块，印在「Harness:」后面。
    prop: 用来把课跑起来的例子，印在「道具:」后面。例如「3+5 / add 工具」。
    """
    print("\n  · 盯这里（不然会看成普通例子）")
    print(f"    道具:     {prop}")
    print(f"    Harness:  {doing}")
    print("    不是:     又学一遍「怎么调 GLM」——那是 1-1 / 3_2 已经做过的")

