"""最小 Harness：真正执行 GLM，而不是按剧本演戏。

账本只有三格，都是这份代码自己的名字，不是模型的新语法，也不是操作系统线程：

- Thread：整段对话，就是按时间排好的一串 Item。本文件没有单独的 Thread 类。
- Turn：一次 run() 或 resume()，从用户这句话到本轮停下来。
- Item：Thread 里的一条。用户的话、模型的回复、工具结果各算一条。

模型只返回文字，或「我想调用某个工具」。跑函数、写回结果、决定停不停，都是 Runner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool


@dataclass
class Item:
    """Thread 里的一条记录。

    kind: 这条是谁留下的。
        "user" 用户说的话。
        "model" 模型的回复。可能只是文字，也可能在 raw 里藏着工具建议。
        "tool_result" Harness 执行工具之后写回来的结果。
    text: 这条的正文。工具结果的字符串也放这里。
    tool: 工具名。kind 为 tool_result 时才有意义。
    args: 调用工具时用的参数。kind 为 tool_result 时才有意义。
    tool_call_id: 和模型那次工具建议配对的编号。写回 ToolMessage 时要带上，否则下一轮接口会报错。
    raw: 模型原始回复（AIMessage）。kind 为 model、且这轮真的问过模型时才有。里面有 tool_calls。
    """

    kind: str
    text: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    raw: Any = None


@dataclass
class Turn:
    """一次 run() 或 resume() 的结果。调用方拿这个对象看本轮停在哪。

    status: 本轮怎么停的。
        "completed" 模型给出了终答，或用户拒绝了工具。
        "waiting_approval" 工具在人审名单里，还没执行，等 resume()。
        "failed" 步数用完仍在要工具，被保险丝停掉。
    items: 这一轮结束时的整本 Thread（含更早传入的 history）。
    events: 本轮发生过的事，按顺序记。例如 "model"、"tool_request:add"、"tool_done:add"。
    final_text: 模型最后说的那句。还在等人批准时是空字符串。
    pending: 被拦住、还没执行的工具建议。每项是 name / args / id。
    pending_ai: 提出这些工具建议的那条原始 AIMessage。resume 时不再用它重放，只留着对照。
    """

    status: str
    items: list[Item]
    events: list[str] = field(default_factory=list)
    final_text: str = ""
    pending: list[dict[str, Any]] = field(default_factory=list)
    pending_ai: AIMessage | None = None


def tool_calls_of(ai: AIMessage) -> list[dict[str, Any]]:
    """取出模型这条回复里的工具建议。没有建议时返回空列表。

    ai: 模型的一条 AIMessage。
    返回: 每项大致是 {"name", "args", "id"}。这只是建议，函数还没跑。
    """
    return list(getattr(ai, "tool_calls", None) or [])


def text_of(ai: AIMessage) -> str:
    """把模型回复收成一段普通字符串。

    ai: 模型的一条 AIMessage。content 有时是字符串，有时是一块一块的列表。
    返回: 拼好的文字。只有工具建议、没有正文时，可能是空字符串。
    """
    content = ai.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def items_to_messages(items: list[Item], system: str | None = None) -> list[BaseMessage]:
    """把账本翻译成 GLM 能读的消息列表。

    items: 要送出去的 Item。通常已经是投影（窗口内的最近几条），不是整本 Thread。
    system: 系统提示。有内容就放在最前面；None 或空字符串则不加。
    返回: SystemMessage / HumanMessage / AIMessage / ToolMessage 的列表。
    """
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    for item in items:
        if item.kind == "user":
            messages.append(HumanMessage(content=item.text))
        elif item.kind == "model":
            if item.raw is not None:
                messages.append(item.raw)
            else:
                messages.append(AIMessage(content=item.text))
        elif item.kind == "tool_result":
            messages.append(
                ToolMessage(content=item.text, tool_call_id=item.tool_call_id or "missing")
            )
    return messages


def trace(who: str, msg: str) -> None:
    """在终端打一行，标明这句话是谁说的。

    who: 标签。课里用 "Host"、"Harness"、"模型"。
    msg: 这一行要给人看的说明。
    """
    print(f"  [{who:<8}] {msg}")


class Runner:
    """Harness 本体。

    它做四件事：把 Thread 整理后问模型、按建议执行工具、把结果写回 Thread、决定这一轮停不停。
    模型碰不到这里的 Python 函数。

    构造之后身上的属性：
    tools: 工具名到工具对象的字典，执行时按名字查找。
    bound: 已经 bind_tools 的模型。问它才会在回复里带上工具建议。
    max_steps / context_budget / system / show_trace: 与构造参数相同。
    approval_tools: 人审名单，一律存成 set。传入 frozenset 也会转成 set。
    calls: 这台 Runner 已经问过模型的次数，从 0 开始，每问一次加 1。
    """

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool],
        *,
        max_steps: int = 8,
        approval_tools: set[str] | frozenset[str] = frozenset(),
        context_budget: int | None = 4,
        system: str | None = None,
        show_trace: bool = True,
    ) -> None:
        """组装一台 Runner。星号后面的参数必须写名字来传。

        model: 聊天模型，本课是 GLM。只负责回答，不负责执行。
        tools: 这台 Harness 允许执行的工具列表。模型只能建议其中的名字。
        max_steps: 一轮里最多问模型几次。到了还在要工具，就停。这是保险丝。
        approval_tools: 执行前必须等人点头的工具名。
            类型是一组字符串，set 或 frozenset 都行。
            默认 frozenset() 是空集合：谁都不用审批。
            用不可变的空 frozenset 当默认值，避免多次构造共用同一个可变 set。
        context_budget: 发给模型的滑动窗口，单位是 Item 条数。
            None 表示整本 Thread 都送。
            设成 4 就只送最近 4 条。更早的仍留在 Thread 里，不会压成摘要。
        system: 每次提问都放在最前面的系统提示。None 表示不放。
        show_trace: True 时打印 [Host] / [Harness] / [模型]。False 时静默。
        """
        self.tools = {t.name: t for t in tools}
        self.bound = model.bind_tools(tools)
        self.max_steps = max_steps
        self.approval_tools = set(approval_tools)
        self.context_budget = context_budget
        self.system = system
        self.show_trace = show_trace
        self.calls = 0

    def _say(self, who: str, msg: str) -> None:
        """show_trace 为真时才打印。who、msg 的含义和 trace() 相同。"""
        if self.show_trace:
            trace(who, msg)

    def _visible(self, items: list[Item]) -> list[Item]:
        """做投影：从整本 Thread 里切出发给模型的那一段。

        items: 当前完整 Thread。
        返回: context_budget 为 None 时原样返回；否则只返回最后 context_budget 条。
        被切掉的旧 Item 仍在传入的 items 里，这里不删除、也不摘要。
        """
        if self.context_budget is None:
            return items
        return items[-self.context_budget :]

    def _invoke(self, items: list[Item]) -> AIMessage:
        """问一次模型。

        items: 问之前的完整 Thread。真正送出的是 _visible 切出来的投影。
        返回: 模型的 AIMessage。calls 加 1，表示这台 Runner 问过模型的次数。
        """
        self.calls += 1
        visible = self._visible(items)
        if self.context_budget is not None:
            self._say(
                "Harness",
                f"管窗口：Thread {len(items)} 条，只把最近 {len(visible)} 条给 GLM",
            )
        messages = items_to_messages(visible, self.system)
        ai = self.bound.invoke(messages)
        if not isinstance(ai, AIMessage):
            ai = AIMessage(content=str(ai))
        return ai

    def run(self, user_text: str, history: list[Item] | None = None) -> Turn:
        """开始一轮新的 Turn。

        user_text: 用户这一句。会先追加成一条 kind 为 user 的 Item。
        history: 更早的 Thread。None 表示从空白对话开始。不会改掉调用方手里的那个列表。
        返回: 这一轮的 Turn。可能已完成，也可能停在 waiting_approval。
        """
        items = list(history or [])
        items.append(Item(kind="user", text=user_text))
        self._say("Host", f"用户说：{user_text}")
        self._say("Harness", "开始一轮 Turn：组上下文、问模型、必要时自己执行工具")
        return self._loop(items)

    def resume(self, turn: Turn, *, approve: bool) -> Turn:
        """人审之后继续。只能用于 status 为 waiting_approval 的 Turn。

        turn: run() 停下来时交回来的那一轮。
        approve: True 才执行刚才被拦住的工具，然后继续问模型。
            False 不执行，给每个建议写一条「用户拒绝了」，这一轮就此结束。
        返回: 继续之后的新 Turn。
        """
        if turn.status != "waiting_approval" or not turn.pending:
            raise ValueError("没有等待批准的工具")
        items = list(turn.items)
        events = list(turn.events)
        if not approve:
            for call in turn.pending:
                items.append(
                    Item(
                        kind="tool_result",
                        tool=str(call.get("name") or ""),
                        text="用户拒绝了",
                        tool_call_id=str(call.get("id") or ""),
                    )
                )
            events.append("denied")
            self._say("Harness", "用户拒绝，工具没有执行")
            return Turn("completed", items, events, final_text="已拒绝")
        self._say("Harness", "用户批准，现在才执行刚才被拦住的工具")
        events.extend(self._run_tools(items, turn.pending))
        return self._loop(items, events)

    def _loop(self, items: list[Item], events: list[str] | None = None) -> Turn:
        """问模型、看建议、执行或停住。run() 和批准后的 resume() 都进这里。

        items: 这一轮已经记下的 Thread，含最新的用户句或刚写回的工具结果。
        events: 已经发生的事件。None 时从 "turn_start" 记起。resume 会把之前的事件传进来。
        返回: 完成、等待批准、或步数用尽的 Turn。
        """
        events = list(events or ["turn_start"])
        for step in range(self.max_steps):
            self._say("Harness", f"第 {step + 1} 步：把 Thread 投影发给 GLM（模型看不到你的 Python 函数）")
            ai = self._invoke(items)
            events.append("model")
            items.append(Item(kind="model", text=text_of(ai), raw=ai))
            calls = tool_calls_of(ai)
            if not calls:
                said = text_of(ai)
                self._say("模型", f"决定开口，不再调工具：{said!r}")
                self._say("Harness", "收到终答，结束 Turn。模型全程没有执行过任何函数")
                events.append("turn_end")
                return Turn("completed", items, events, final_text=said)
            for call in calls:
                self._say(
                    "模型",
                    f"只是建议调用 {call.get('name')}({call.get('args')})  ← 到这里还没算、没写文件",
                )
            names = [str(c.get("name") or "") for c in calls]
            events.append("tool_request:" + ",".join(names))
            blocked = [c for c in calls if str(c.get("name") or "") in self.approval_tools]
            if blocked:
                self._say("Harness", f"拦住 { [c.get('name') for c in blocked] }，等人批准。这就是人审")
                events.append("waiting_approval")
                return Turn(
                    "waiting_approval",
                    items,
                    events,
                    pending=calls,
                    pending_ai=ai,
                )
            events.extend(self._run_tools(items, calls))
        self._say("Harness", f"已达 max_steps={self.max_steps}，强制停。保险丝在 Harness，不在模型")
        events.append("step_limit")
        return Turn("failed", items, events, final_text="超过步数上限")

    def _run_tools(self, items: list[Item], calls: list[dict[str, Any]]) -> list[str]:
        """真正执行工具，并把每条结果追加进 items。

        items: 正在记的 Thread。本方法会就地追加 kind 为 tool_result 的 Item。
        calls: 模型给出的工具建议，每项含 name、args、id。
        返回: 事件名列表，形如 "tool_done:add"。未知工具不会抛错，结果写成「没有这个工具」。
        """
        events: list[str] = []
        for call in calls:
            name = str(call.get("name") or "")
            args = dict(call.get("args") or {})
            call_id = str(call.get("id") or "")
            fn = self.tools.get(name)
            if fn is None:
                result = f"没有这个工具: {name}"
            else:
                result = str(fn.invoke(args))
            self._say("Harness", f"真正执行 {name}{args} → {result!r}  ← 这一行模型做不到")
            items.append(
                Item(
                    kind="tool_result",
                    tool=name,
                    args=args,
                    text=result,
                    tool_call_id=call_id,
                )
            )
            events.append("tool_done:" + name)
        return events


def called_tools(turn: Turn) -> list[str]:
    """按发生顺序列出这轮 Thread 里真正执行过的工具名。

    turn: 一轮的结果。只统计 kind 为 tool_result 的 Item。
    返回: 工具名列表。用户拒绝时，对应条目的 tool 名也在里面，正文是「用户拒绝了」。
    """
    return [i.tool for i in turn.items if i.kind == "tool_result"]
