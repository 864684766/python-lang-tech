"""
Harness 教材 ⑨ 下一轮只读文件，不读上一轮聊天

上一节（hs_5）把窗口切开了：全部记录留着，模型只看最近几条。那些句子仍在这一次进程的内存里。

这一节看进程关掉之后。

先看一个现象。会话 A 在内存里做过这些事：

    聊天记录 = ["记下充电器", "天气不错", "f1 已完成"]
    feature_list.json = [{"id": "f1", "status": "passing"}, {"id": "f2", "status": "failing"}]
    progress.txt 末尾多了一行 "f1 已完成"

进程关掉之后，聊天记录那三句没了。新开的会话 B 如果只靠聊天，不知道下一件是 f2。

这里的问题是：做到哪一步如果只写在聊天里，下一轮接不上。把「天气不错」也塞进下一轮，提示词又变胖。进度若是一整篇 Markdown，模型重写时漏掉一行，那一行就没了。

以前把整个聊天存下来，下次原样塞回去。能接上，越跑越长。

这一节会话 B 的聊天从空列表开始，只读那两个文件：

    聊天记录 = []
    读 feature_list.json  → f1 是 passing，下一件是 f2
    读 progress.txt       → 最后一行是 "f1 已完成"

会话 B 不挂工具，所以它不能把 f2 改成 passing。它只能根据文件说出下一件是什么。

反面教材：只在旧会话的聊天里写「f1 已完成」，却没有保存任务状态。新会话从空历史开始时，就无法恢复进度。如何避免增量更新误删旧条目，统一在 hs_10 展开。

影响：仓库里的文件是下一轮唯一的事实。上一轮聊天里说过、却没落盘的，对新会话不存在。

运行: python harness/hs_9_handoff.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool

from demo_explain import print_idea, print_lesson, print_ok, print_section, print_tip, print_trap
from harness.live import live_model, print_harness_job
from harness.mini import Runner, called_tools


FEATURES = [
    {"id": "f1", "title": "记下：会议在周四", "status": "failing"},
    {"id": "f2", "title": "记下：带充电器", "status": "failing"},
    {"id": "f3", "title": "记下：提前十分钟", "status": "failing"},
]


def _write_features(root: Path, features: list[dict[str, str]]) -> None:
    (root / "feature_list.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_features(root: Path) -> list[dict[str, str]]:
    return json.loads((root / "feature_list.json").read_text(encoding="utf-8"))


def init_desk(root: Path) -> None:
    _write_features(root, [dict(item) for item in FEATURES])
    (root / "progress.txt").write_text("尚未开始\n", encoding="utf-8")


def brief(root: Path) -> str:
    """新会话允许看见的全部。上一轮聊天不在这里。"""
    features = _read_features(root)
    progress = (root / "progress.txt").read_text(encoding="utf-8").rstrip()
    nxt = next(item for item in features if item["status"] == "failing")
    done = [item["id"] for item in features if item["status"] == "passing"]
    return (
        "你是新开的会话，看不到上一轮聊天。\n"
        f"进度文件:\n{progress}\n"
        f"已完成: {done or '无'}\n"
        f"下一个未完成: {nxt['id']} {nxt['title']}"
    )


def finish_feature_at(root: Path, feature_id: str) -> str:
    features = _read_features(root)
    hit = False
    for item in features:
        if item["id"] == feature_id:
            item["status"] = "passing"
            hit = True
    if not hit:
        return f"没有这个功能: {feature_id}"
    _write_features(root, features)
    with (root / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"完成 {feature_id}\n")
    return f"已落盘 {feature_id}"


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑨ 下一轮只读文件，不读上一轮聊天")
    print("=" * 65)

    print_harness_job(
        "Harness 把进度写成结构化文件。新进程的上下文从文件重建，不从内存里的 Thread 重建",
        prop="三行待办 + progress.txt",
    )

    print_idea(
        phenomenon="hs_5 会把窗口切短，但 Thread 还在这次进程的内存里。进程退出，交接棒就断了。",
        gap="耐久状态放在仓库文件里。聊天只是这一轮的草稿纸。",
        before="把全部历史存进数据库，下次原样灌回提示词。越跑越满。",
        impact="功能清单用 JSON，一次只改一条的 status。进度文件只追加一行。",
        counter="进度只写在内存聊天里，进程退出后新会话无法恢复。",
    )

    desk: dict[str, Path] = {}

    @tool
    def finish_feature(feature_id: str) -> str:
        """把功能清单里某一个 id 标成 passing，并在进度文件追加一行。"""
        return finish_feature_at(desk["root"], feature_id)

    with tempfile.TemporaryDirectory(prefix="hs9-") as raw:
        root = Path(raw)
        desk["root"] = root
        init_desk(root)
        chatter = "刚才还聊了天气，这段只存在本轮内存里"

        print_section("会话 A：做完一条，然后丢掉聊天")
        system_a = (
            brief(root)
            + "\n本轮只做 f1。必须调用 finish_feature，feature_id 填 f1，不要口头答应。"
        )
        turn_a = Runner(live_model(), [finish_feature], system=system_a).run(
            "完成功能 f1"
        )
        assert "finish_feature" in called_tools(turn_a), called_tools(turn_a)
        features = _read_features(root)
        by_id = {item["id"]: item["status"] for item in features}
        assert by_id == {"f1": "passing", "f2": "failing", "f3": "failing"}, by_id
        print_ok(f"清单状态 {by_id}")
        print_ok((root / "progress.txt").read_text(encoding="utf-8").rstrip())
        print_ok(f"丢掉的聊天（文件里找不到）: {chatter}")
        assert chatter not in (root / "progress.txt").read_text(encoding="utf-8")
        assert chatter not in (root / "feature_list.json").read_text(encoding="utf-8")

        print_section("会话 B：空历史，只读文件")
        system_b = brief(root)
        assert "天气" not in system_b
        assert "f2" in system_b
        assert "f1" in system_b
        print_ok("发给新会话的全文:")
        for line in system_b.splitlines():
            print_ok(f"  {line}")
        turn_b = Runner(live_model(), [], system=system_b).run(
            "下一个未完成功能的标题是什么？只回答标题原文。"
        )
        assert turn_b.status == "completed"
        print_ok(f"GLM 说: {turn_b.final_text!r}")
        print_ok("判断依据是文件摘要里有没有 f2，不是这句话措辞是否一字不差")

    print_tip("仓库里的文件是下一轮唯一的事实。Slack 和上一轮聊天，对新会话都不存在。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
