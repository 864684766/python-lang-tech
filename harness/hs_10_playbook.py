"""
Harness 教材 ⑩ 改一条，留下其余

上一节把进度放进文件，说明不必把全部历史塞进提示词。系统提示本身也会越改越差。

这一节看怎么改一本已经在用的手册。

先看一个现象。手册原来是三条：

    [
        "1|做加法必须调用 add",
        "2|记笔记前先读进度文件",
        "3|失败了先看 progress.txt",
    ]

让模型「整理得更短」，它交回一整段：

    "要多用工具。"

旧的第 1、2、3 条都不在里面了。

这里有两个问题。用这一段替换整个列表，add 和进度文件的规则就没了。若每次都要求「再短一点」，列表会越改越空。

以前把整本手册发回去，让模型输出一个「更好的版本」，再整份替换。读起来更顺，规则少了。

这一节模型可以提议，Harness 只接受单独的一行「编号|说明」：

    apply_delta("3|失败了先看 progress.txt")   # 新编号，追加
    apply_delta("1|做加法必须调用 add")         # 同一编号，只改这一条
    apply_delta("要多用工具。")                 # 没有「编号|」，整段退回，1、2、3 还在

合并按这个规则做，不再让模型重写全文。

反面教材：手册原来是两条：「1|做加法必须调用 add」和「2|记笔记前先读进度文件」。模型交回一句「要多用工具」。若用这句话替换整本手册，编号变成只有一条 blob，add 和进度文件这两条规则都没了。下一次做加法，模型不再被要求调用 add。正确的合并是只接受单独一行，例如「3|失败了先看 progress.txt」，1 和 2 原样留下。

影响：手册可以变长、可以改某一条，不会因为一次整理把别的条删掉。

运行: python harness/hs_10_playbook.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_trap
from harness.live import live_model, print_harness_job
from harness.mini import text_of

Entry = dict[str, str]
_DELTA = re.compile(r"(\w+)\|(\S.*)")


def render(book: list[Entry]) -> str:
    return "\n".join(f"{item['id']}|{item['text']}" for item in book)


def rewrite_all(blob: str) -> list[Entry]:
    """整篇替换。旧编号全部作废。"""
    return [{"id": "blob", "text": blob.strip()}]


def apply_delta(book: list[Entry], proposal: str) -> tuple[list[Entry], str]:
    """只接受单独一条「编号|说明」。其余原样退回。"""
    line = proposal.strip()
    matched = _DELTA.fullmatch(line)
    if matched is None:
        return book, "rejected_rewrite"
    entry_id, text = matched.group(1), matched.group(2).strip()
    if any(item["text"] == text for item in book):
        return book, "deduped"
    merged: list[Entry] = []
    replaced = False
    for item in book:
        if item["id"] == entry_id:
            merged.append({"id": entry_id, "text": text})
            replaced = True
        else:
            merged.append(dict(item))
    if not replaced:
        merged.append({"id": entry_id, "text": text})
    return merged, "merged" if replaced else "appended"


def _ids(book: list[Entry]) -> list[str]:
    return [item["id"] for item in book]


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑩ 改一条，留下其余")
    print("=" * 65)

    print_harness_job(
        "Curator 是确定性合并，不是再叫一次模型重写全文",
        prop="两行操作手册",
    )

    print_idea(
        phenomenon="手册越改越短，上次写过的「必须用 add」下次就不见了。",
        gap="上下文是一组可增删的条目。模型可以提议一条，Harness 负责合并。",
        before="把整段系统提示发回去，让模型输出一个「更好的版本」。",
        impact="同一编号是修订，新编号是追加，说明文字重复就丢掉。",
        counter="接受一整段自由文本当新手册。这就是 context collapse，也是 brevity bias。",
    )

    book: list[Entry] = [
        {"id": "1", "text": "做加法必须调用 add"},
        {"id": "2", "text": "记笔记前先读进度文件"},
    ]
    original_chars = len(render(book))

    print_section("整篇重写会掏空")
    collapsed = rewrite_all("要多用工具")
    print_trap(
        "更短的一句话换成了整本手册",
        f"原来 {_ids(book)} 共 {original_chars} 字；现在 {_ids(collapsed)}，正文是 {collapsed[0]['text']!r}",
    )
    assert _ids(collapsed) == ["blob"]
    assert len(collapsed[0]["text"]) < original_chars

    print_section("增量：追加、修订、去重")
    added, how = apply_delta(book, "3|失败了先看 progress.txt")
    assert how == "appended", how
    assert _ids(added) == ["1", "2", "3"]
    revised, how = apply_delta(added, "2|记笔记前先读 progress.txt，不要凭记忆")
    assert how == "merged", how
    assert _ids(revised) == ["1", "2", "3"]
    assert revised[0]["text"] == book[0]["text"]
    again, how = apply_delta(revised, "9|失败了先看 progress.txt")
    assert how == "deduped", how
    assert again is revised
    print_ok(render(revised))

    print_section("模型想压缩全文，Harness 不收")
    ai = live_model().invoke(
        [
            SystemMessage(
                content=(
                    "把用户给的手册收成一段更短的话。"
                    "不要使用「编号|说明」这种单行格式，不要列清单。"
                )
            ),
            HumanMessage(content="手册如下：\n" + render(revised)),
        ]
    )
    proposal = text_of(ai).strip()
    kept, how = apply_delta(revised, proposal)
    print_ok(f"GLM 提议: {proposal!r}")
    print_ok(f"Harness 判定: {how}")
    assert _ids(book)[0] in _ids(kept)
    assert "2" in _ids(kept)
    assert "3" in _ids(kept)
    if how == "rejected_rewrite":
        assert kept is revised
        print_ok("整段提议被退回，三条都还在")
    else:
        print_ok("即便格式碰巧合法，也只动了一条，其余编号还在")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
