"""
Python 标准库教材 ⓪ json（内存里的 dict ↔ 一段可传递的文本）

是什么:
  JSON 是纯文本。Python 的 dict 只活在内存里，别的语言、磁盘、网络都看不懂。
  四个名字: dumps / loads 对字符串；dump / load 对文件。差一个 s，对象完全不同。
  对照: JS 的 JSON.stringify / parse；C# 的 JsonSerializer。

用在:
  1) HTTP body、工具参数、模型 structured output
  2) 写/读配置文件（.json）
  3) 任何「要把结构交给另一个程序」的地方

不用:
  对象只在本进程里用，直接传 dict，不必 dumps 再 loads

运行: python python_stdlib/ps_0_json.py
"""
from __future__ import annotations

import json  # 标准库：不用 pip。专门做「Python 对象 ↔ JSON 文本」
import sys
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


def demo() -> None:
    print("\n" + "=" * 65)
    print("⓪ json：为什么要转、四个函数各干什么")
    print("=" * 65)

    print_overview(
        "把内存里的 dict/list 变成一段文本，或反过来。程序之间靠这段文本传结构。",
        [
            "HTTP body、工具参数、模型 structured output",
            "写/读 .json 配置文件",
            "对照 JS JSON.stringify/parse、C# JsonSerializer",
        ],
        "对象只在本进程里用，直接传 dict，不必 dumps 再 loads。",
    )

    print_section("JSON 是什么（和 Python dict 不是同一种东西）")
    print("    Python 的 dict 只活在内存里，别的语言看不懂。")
    print("    JSON 是一段字符串，HTTP、配置文件、模型输出都用它传结构。")
    print("    对照: JS 的 JSON.stringify / JSON.parse")
    print("          C# 的 JsonSerializer.Serialize / Deserialize")
    # 这是 Python 对象（dict）。还不是 JSON。
    payload = {"name": "search", "args": {"q": "LangGraph"}}
    print_ok(f"Python dict: {payload}    type={type(payload)}")

    print_section("dumps：dump + s，s = string（变成字符串）")
    print("    读法: dump-s，把对象「倒出来」成 str。")
    print("    参数 1 = 要转的 dict/list；返回值 = str。")
    # ensure_ascii=False：允许输出中文。默认 True 会把中文变成 \\uXXXX。
    text = json.dumps(payload, ensure_ascii=False)
    print_ok(f"json.dumps(payload) → {text!r}")
    print_ok(f"这时 type={type(text)}    已经是字符串，可以放进 HTTP body / 文件")
    print_ok("ensure_ascii=False  让中文保持「查天气」，不要变成 \\u67e5\\u5929\\u6c14")

    print_section("loads：load + s，从字符串装回来")
    print("    读法: load-s，把 JSON 字符串「装回」Python 对象。")
    print("    参数 1 = 一段 JSON 文本；返回值 = dict 或 list。")
    back = json.loads(text)
    print_ok(f"json.loads(text) → {back}    type={type(back)}")
    print_ok(f"来回一趟相等? {back == payload}")

    print_section("dump / load：没有 s = 和文件打交道")
    print("    dumps/loads 的 s 是 string。去掉 s，第二个参数换成「已打开的文件」。")
    print("    dump(对象, 文件)   写入磁盘，返回 None（不是字符串）")
    print("    load(文件)         从磁盘读回来，返回 dict/list")
    path = Path(__file__).with_name("_demo.json")
    # "w" = 写文本。encoding 必须写 utf-8，否则中文在 Windows 上会乱。
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print_ok(f"json.dump(...) 写到 {path.name}    返回值是 None，内容在文件里")
    with path.open(encoding="utf-8") as f:
        from_file = json.load(f)
    print_ok(f"json.load(f) → {from_file}")
    path.unlink()  # 演示完删掉临时文件

    print_trap(
        "dumps 和 dump 搞反",
        "json.dumps(obj) 得到 str，再自己 write。"
        " json.dump(obj, f) 直接写文件，得到 None。"
        " 写成 x = json.dump(obj, f) 再当字符串用，x 是 None。",
    )

    print_section("JSON 文本比 Python 字面量更严")
    print("    合法: {\"q\": \"hi\"}     键必须双引号；true/false/null 全小写")
    print("    非法: {'q': 'hi'}      Python dict 允许单引号，JSON 不允许")
    print("    非法: {\"q\": \"hi\",}    最后一个逗号，模型经常多写")
    print_counter(
        "把 Python 字典字面量（单引号）交给 loads",
        lambda: json.loads("{'name': 'search'}"),
        json.JSONDecodeError,
    )
    print_counter(
        "模型输出多了一个尾逗号",
        lambda: json.loads('{"q": "hi",}'),
        json.JSONDecodeError,
    )
    print_ok("报错类型是 JSONDecodeError：文本不是合法 JSON，不是 dumps 写错了")

    print_section("不是所有 Python 值都能 dumps")
    print("    能转: dict / list / str / int / float / bool / None")
    print("    不能: set、datetime、numpy 数组、普通 class 实例")
    print_counter("set 直接 dumps", lambda: json.dumps({1, 2}), TypeError)
    print_ok("先转成能表达的: list(a_set)、datetime 用 .isoformat()、对象用 __dict__ 或 dataclass")

    print_tip(
        "记四个名字: dumps=对象→字符串, loads=字符串→对象, dump=写文件, load=读文件。"
        " 过网络、给模型看、写配置，都先 dumps；读回来 loads。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
