"""⑭ 两项检查：补丁格式正确吗？执行任务真的变好了吗？

【现象】模型有时能写出一条很像手册的规则，例如：

    2|收到记笔记请求后，先调用 read_progress，再调用 write_note

【问题】这条文字“看起来正确”，仍然只是提案；它没有证明模型执行任务时
真的会先读进度。若只看文字就合并，可能把一条没有效果的规则加入手册。

【解决办法】把验收拆成两关：
  1. updating：提案是否符合「编号|说明」格式，能否按约定追加到手册？
  2. benefit：用同一个模型、同一组任务，分别运行旧手册和候选手册，检查真实的
     工具调用顺序和写入内容。只有格式合格且成绩确实提升，才采用候选手册。

【反例】如果只检查 updating，模型写出「2|多使用工具」也可能被接受；这句话
格式合法，却没有告诉模型先调用哪个工具。另一个反例是只看最终回答中的“已完成”，
模型可能没有调用 read_progress，甚至没有真的写入笔记。

【本例任务】把「开会」和「带线」分别写入笔记。每项任务必须先调用
read_progress，再调用 write_note，并且写入内容必须保持原文。

【运行流程】旧手册执行并记录失败 → 模型根据记录提出补丁 → 检查补丁格式
→ 候选手册重新执行相同任务 → 比较成绩并决定是否采用。

运行：python harness/hs_14_two_axes.py
本例调用真实模型；成绩由真实工具调用记录决定，不保证每次一定提升。
所有笔记只保存在内存中，不修改磁盘文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from harness.live import live_model
from harness.mini import Runner, text_of


TASKS = ["开会", "带线"]
# 故意放入一条不完整的旧规则：它要求写笔记，却漏掉了“先读进度”。
# 模型若遵循这条旧规则，第一次执行就会漏读进度，产生可供分析的失败记录。
# 但模型行为不保证固定：若旧手册已经全部通过，本例不会把同分提案算成提升。
OLD_MANUAL = "1|收到记笔记请求后，直接调用 write_note 写入原文。"


def valid_patch(text: str) -> bool:
    """第一关：只验证补丁协议，不判断补丁是否有用。

    这里限制为单行「2|说明」，是为了演示增量修改：模型只能追加第 2 条，
    不能返回整本手册覆盖旧内容。这个检查不能替代后面的行为测试。
    """
    parts = text.split("|", 1)
    if len(parts) != 2:
        return False
    number, instruction = parts
    return number == "2" and bool(instruction.strip()) and "\n" not in text


def run_tasks(model, manual: str) -> tuple[int, str]:
    """用同一个模型执行两项任务，根据真实工具记录打分。

    重要的是这里没有传入“强执行器/弱执行器”的人为开关。模型真的要调用
    read_progress 和 write_note，Harness 才能从 actions、notes 判断结果。
    因此候选手册是否有效，会真实影响后面的 benefit。
    """
    passed = 0
    reports = []

    for task in TASKS:
        # 每个任务从空记录开始，避免上个任务的调用或笔记影响这次评分。
        actions = []
        notes = []

        @tool
        def read_progress() -> str:
            """读取当前项目的进度。"""
            # 返回的是固定的模拟进度，不读取磁盘；调用和记录动作是真实发生的。
            actions.append("read_progress")
            return "项目进度：会议待召开，连接线待准备。"

        @tool
        def write_note(text: str) -> str:
            """把原文写入笔记。"""
            # 工具说明会交给模型，因此前后评测保持相同说明，只改变手册。
            actions.append("write_note")
            notes.append(text)
            return "已写入：" + text

        Runner(
            model,
            [read_progress, write_note],
            system=(
                "请按照操作手册完成任务，实际操作必须调用工具。"
                "有冲突时，以后面的规则为准。工具报告写入成功后，结束任务。\n"
                "操作手册：\n" + manual
            ),
            max_steps=4,
            context_budget=None,
            show_trace=False,
        ).run(f"请把「{task}」写入笔记，保持原文。")

        # 严格检查：一次读取、一次写入，顺序正确，而且没有写错或重复写入。
        # 如果只检查模型最后说了什么，这里的顺序和真实副作用就会被漏掉。
        ok = actions == ["read_progress", "write_note"] and notes == [task]
        if ok:
            passed += 1
        status = "通过" if ok else "失败"
        report = f"任务：{task}；调用顺序：{actions}；笔记：{notes}；{status}"
        reports.append(report)
        print(report)

    return passed, "\n".join(reports)


def demo() -> None:
    print("⑭ 格式正确，不代表执行有效")
    print("验收：每个任务必须先读进度，再写入原文。共两项任务。")

    # 整次实验复用同一个模型配置，前后只有手册发生变化。
    # 这样成绩差异更可能来自手册，而不是换了模型或换了任务。
    model = live_model()

    # 先测基线：不知道旧手册是否有效，就无法判断候选手册有没有带来提升。
    print("\n第一步：使用旧手册，记录真实失败情况")
    print(OLD_MANUAL)
    before, report = run_tasks(model, OLD_MANUAL)
    print(f"旧手册成绩：{before}/{len(TASKS)}")

    # 模型现在能看到旧手册和真实执行记录，提案不再是脱离上下文的随口生成。
    print("\n第二步：模型根据旧手册和执行结果，提出一条补丁")
    ai = model.invoke([
        SystemMessage(content=(
            "你负责改进操作手册。只输出一行：2|具体操作规则。"
            "这条规则会追加到旧手册后，有冲突时以新规则为准。"
            "不要加代码围栏或解释。"
        )),
        HumanMessage(content=(
            f"旧手册：\n{OLD_MANUAL}\n"
            f"执行记录：\n{report}\n"
            "验收要求：每项任务都只调用一次 read_progress，然后调用一次 write_note，"
            "并正确写入用户原文。请根据记录提出改进规则。"
        )),
    ])
    proposal = text_of(ai).strip()
    print(f"模型的完整提案：{proposal!r}")

    # updating 只回答“能不能按协议合并”，不能回答“合并后是否有效”。
    print("\n第三步：检查格式（updating）")
    if not valid_patch(proposal):
        print("格式不合格：必须是单行 2|说明。停止评测，旧手册保持不变。")
        return
    print("格式合格。但还不能说明任务做得更好了。")

    # 只有到了这一步，才真正测 benefit：相同模型、相同任务、不同手册。
    print("\n第四步：追加补丁，用相同模型重新执行相同任务（benefit）")
    candidate_manual = OLD_MANUAL + "\n" + proposal
    print("候选手册：\n" + candidate_manual)
    after, _ = run_tasks(model, candidate_manual)

    print(f"\n成绩对比：旧手册 {before}/{len(TASKS)} → 候选手册 {after}/{len(TASKS)}")
    active_manual = OLD_MANUAL
    # 同时要求“全部通过”和“比旧手册更好”，不接受只修好部分任务的提案。
    # 这仍然是一次小实验：即使全部通过，也可能存在随机波动或未覆盖的场景。
    if after == len(TASKS) and after > before:
        active_manual = candidate_manual
        print("采用：格式合格，全部任务通过，而且通过数量确实增加。")
    else:
        print("不采用：没有同时达到全部通过和成绩提升，保留旧手册。")
    print("当前采用的手册：\n" + active_manual)
    print("\n注意：这是两项任务的一次实验；实际成绩可能变化，不能代表所有场景。")


if __name__ == "__main__":
    demo()
