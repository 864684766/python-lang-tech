"""
========================================================
LangChain Phase 7-1
Agent Skills（Anthropic / agentskills 规范）
========================================================

规范：https://agentskills.io  （由 Anthropic 提出，开放标准；不是 OpenAI）

渐进披露三层（核心思想：别把所有技能文件一次塞进 prompt）：
  1. Metadata     启动时只把 name + description 放进上下文
  2. Instructions 命中后再加载该技能 SKILL.md 正文
  3. Resources    scripts/ references/ assets/ 按正文需要再读或执行
                  跑脚本时只把 stdout 送回模型，不要把源码塞进上下文
                  （stdout = 进程标准输出，即 print 打到终端的那部分）

本 Demo 的工具对应规范里的 filesystem + bash：
  read_skill_file ≈ 读技能目录文件
  run_python / run_shell ≈ 执行 bundled scripts / 说明书中的命令

运行: python 7_1_skill_registry_demo.py
"""
from __future__ import annotations  # 让类型注解里的 | 在旧 Python 也能写（如 str | None）

import re  # 正则：校验技能名、切 YAML、改写 shell 里的 python
import shlex  # 把「带空格的命令行字符串」拆成参数列表，避免自己手写 split
import subprocess  # 真正去操作系统里跑命令（Level 3 执行脚本）
import sys  # 取当前解释器路径 sys.executable，避免 PATH 上那个 python 不是本 venv
from dataclasses import dataclass  # 轻量数据类：一份技能的内存表示
from pathlib import Path  # 路径对象，比字符串拼接更安全（resolve / relative_to）

import yaml  # 解析 SKILL.md 开头的 YAML frontmatter（name / description）
from dotenv import load_dotenv
from langchain.agents import create_agent  # 模型 ↔ 工具循环的工厂函数
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool  # 把普通函数变成 LLM 可调用的 Tool
from langgraph.errors import GraphRecursionError  # 图步数用尽；生产里要捕获，不要甩 traceback
from pydantic import BaseModel, Field  # 约束「选技能」的结构化输出

from demo_explain import print_lesson, print_run_plan, print_step, print_tip
from model_provider import check_api_key, get_model

load_dotenv()  # 从 .env 读 API Key，不要把密钥写进源码

# 技能根目录：本文件同级的 skills/。里面任意深度的 SKILL.md 都会被扫描
SKILLS_DIR = Path(__file__).parent / "skills"
# 规范要求 name：小写字母、数字、连字符，最长 64（和文件夹名可以不同）
_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
# 读文件时截断，防止一份巨大 reference.md 撑爆上下文
_MAX_READ_CHARS = 80_000
# 子进程超时（秒）。脚本卡死时不要一直占着 Agent 循环
_TOOL_TIMEOUT = 60
# Agent 图最多走多少步（model / tools 各算 1 步）。宁可变步数报错，也不让 find 空转
_AGENT_RECURSION_LIMIT = 16


@dataclass(frozen=True)  # frozen=True：解析后当不可变快照，避免运行中被改掉
class AgentSkill:
    """内存里的一份 Agent Skill。磁盘上的实体仍是 SKILL.md + 旁边的 scripts/ 等。"""

    name: str  # frontmatter 的 name，catalog 和 pick_skill 都用这个字符串
    description: str  # 给模型看的「什么时候用我」；Level 1 只注入这一句 + name
    body: str  # 第二段 --- 之后的 Markdown 正文；激活后才进 system prompt
    path: Path  # SKILL.md 的绝对路径，用来定位技能根目录
    disable_model_invocation: bool = False  # 规范字段：True 则不进 catalog，模型选不到

    @property
    def directory(self) -> Path:
        """技能根目录 = SKILL.md 所在文件夹。相对路径都相对这里解析。"""
        return self.path.parent


def parse_skill_md(path: Path) -> AgentSkill:
    """解析一份标准 SKILL.md：YAML frontmatter + Markdown 正文。

    文件长这样：
        ---
        name: weather
        description: ...
        ---
        # 下面才是给模型看的操作步骤
    """
    # utf-8-sig：有的编辑器会写 BOM（文件开头那个不可见 \\ufeff），去掉以免 --- 对不上
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("---"):
        raise ValueError("缺少 YAML frontmatter（文件必须以 --- 开头）")
    # 跳过开头的 --- 和紧跟的换行，rest 现在从 YAML 第一行开始
    rest = text[3:].lstrip("\r\n")
    # 找「第二个 ---」：YAML 结束、正文开始。必须带换行，避免 YAML 里的 --- 误切
    end = re.search(r"\n---\s*\n", rest)
    if not end:
        raise ValueError("frontmatter 没有第二段 ---")
    # 只把 frontmatter 当 YAML 解析，不要把整份 Markdown 丢进 yaml.safe_load
    meta = yaml.safe_load(rest[: end.start()]) or {}
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"name 不合法: {name!r}")
    if not description:
        raise ValueError("缺少 description（AI 靠它决定用不用）")
    return AgentSkill(
        name=name,
        description=description,
        body=rest[end.end() :].strip(),  # 第二段 --- 之后：操作说明书
        path=path,
        # YAML 里是 kebab-case：disable-model-invocation
        disable_model_invocation=bool(meta.get("disable-model-invocation")),
    )


class SkillRegistry:
    """技能登记处：只负责「磁盘上有哪些 Skill」。

    选哪个、怎么做，都不在这里写 if/else，全部交给模型（pick_skill + SKILL.md）。
    """

    def __init__(self, root: Path) -> None:
        self.root = root  # 一般是 ./skills
        # name → AgentSkill。dict 同时当索引和去重（同名只留先扫到的）
        self._skills: dict[str, AgentSkill] = {}

    def discover(self) -> list[AgentSkill]:
        """递归扫描 root 下所有 SKILL.md 并解析。坏文件跳过，不中断整次扫描。"""
        self._skills.clear()
        if not self.root.is_dir():
            return []
        # rglob：任意深度。允许 skills/weather/weather/SKILL.md 这种套一层的布局
        for skill_md in sorted(self.root.rglob("SKILL.md")):
            try:
                skill = parse_skill_md(skill_md)
            except ValueError as exc:
                print(f"  ⚠️  跳过 {skill_md.relative_to(self.root)}: {exc}")
                continue
            if skill.name in self._skills:
                print(f"  ⚠️  重名 {skill.name}，忽略 {skill_md}")
                continue
            self._skills[skill.name] = skill
        return self.list_skills()

    def list_skills(self) -> list[AgentSkill]:
        """全部已解析技能（含 disable-model-invocation 的，扫描展示时仍列出）。"""
        return list(self._skills.values())

    def get(self, name: str) -> AgentSkill | None:
        """按 catalog 里的 name 取完整技能；没有则 None。"""
        return self._skills.get(name)

    def catalog(self) -> list[AgentSkill]:
        """Level 1 真正给模型看的子集：关掉「禁止模型调用」的那些。"""
        return [s for s in self.list_skills() if not s.disable_model_invocation]

    def catalog_text(self) -> str:
        """规范 Level 1：启动时注入的 metadata 格式为 `name - description`。

        一行一个技能。模型在 pick_skill 里只看这个字符串，看不到 SKILL.md 正文。
        """
        lines = [f"{s.name} - {s.description}" for s in self.catalog()]
        return "\n".join(lines) or "(no skills)"


class SkillPick(BaseModel):
    """pick_skill 的结构化输出。with_structured_output 会逼模型填这两个字段。"""

    name: str | None = Field(description="catalog 里的 skill 名；都不适用则 null")
    reason: str = Field(description="一句话：为什么选它，或为什么都不选")


def pick_skill(question: str, registry: SkillRegistry) -> SkillPick:
    """只看 name+description，不看正文（渐进披露第一层）。

    这一步是一次普通 LLM 调用，还没有 create_agent，也没有工具。
    """
    names = {s.name for s in registry.catalog()}  # 合法名字集合，用来校验模型别瞎编
    # 模型输出必须能解析成 SkillPick，而不是一段「我选了 weather」的散文
    picker = get_model().with_structured_output(SkillPick)
    pick = picker.invoke(
        [
            SystemMessage(
                content=(
                    "Agent Skills are available. Per the Agent Skills spec, only "
                    "name and description (level 1 metadata) are loaded until a skill is activated.\n"
                    "If the user request matches a skill's description, return that name. "
                    "If none match, name=null. Do not pick a skill that is merely 'maybe useful'.\n\n"
                    f"{registry.catalog_text()}"  # 这里才是 Level 1 真正注入的内容
                )
            ),
            HumanMessage(content=question),
        ]
    )
    # 有的模型/封装可能返回 dict 而不是 Pydantic 实例，两头都兼容
    raw = (getattr(pick, "name", None) or "").strip() or None
    reason = getattr(pick, "reason", "") if not isinstance(pick, dict) else pick.get("reason", "")
    # 模型编了个 catalog 里没有的名字 → 当成没选，避免后面 get() 拿到 None 还假装激活了
    if raw not in names:
        return SkillPick(name=None, reason=reason)
    return SkillPick(name=raw, reason=reason)


def _inside_skill(skill_dir: Path, relative_path: str) -> Path:
    """把「相对技能根的路径」解析成绝对路径，并挡住目录穿越。

    模型填的是 scripts/foo.py 这种字符串。若不 resolve + relative_to，
    可能被 ../../Windows/System32 之类逃出技能目录。
    """
    if not relative_path.strip() or Path(relative_path).is_absolute():
        raise ValueError("必须是技能目录内的相对路径，例如 scripts/office/soffice.py")
    root = skill_dir.resolve()  # 规范化：去掉 . / ..，得到真实绝对路径
    target = (root / relative_path).resolve()
    try:
        # target 必须落在 root 下面；否则 relative_to 会抛 ValueError
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("禁止访问技能目录以外的路径") from exc
    if not target.exists():
        raise ValueError(f"不存在: {relative_path}")
    return target


def _run_cmd(cmd: list[str], cwd: Path) -> str:
    """用参数列表跑一条命令（不走 shell），把 stdout 收成字符串还给模型。

    capture_output=True：stdout/stderr 进管道，不直接打到你的终端。
    返回给模型的是这段输出，不是脚本源码（Level 3 的关键）。
    """
    try:
        completed = subprocess.run(
            cmd,  # 例: [python.exe, "E:/.../test.py", "--city", "Beijing"]
            cwd=str(cwd),  # 工作目录钉在技能根，相对路径按 SKILL.md 写就行
            capture_output=True,  # 收集 stdout / stderr，不要泄漏到父进程终端
            text=True,  # 返回 str 而不是 bytes
            encoding="utf-8",
            errors="replace",  # 个别乱码用替换字符，避免整次工具调用崩掉
            timeout=_TOOL_TIMEOUT,
            check=False,  # 非 0 退出码不抛异常，我们自己把 stderr 拼进返回值
        )
    except subprocess.TimeoutExpired:
        return f"命令超过 {_TOOL_TIMEOUT} 秒未结束，已终止"
    except FileNotFoundError as exc:
        return f"找不到可执行文件: {exc.filename or cmd[0]}"
    out = (completed.stdout or "").strip()  # 标准输出：print / 程序正常结果
    err = (completed.stderr or "").strip()  # 标准错误：报错、警告
    if completed.returncode != 0:
        # 失败时两路都给模型，方便它决定是改参数还是停下来报告
        return f"exit={completed.returncode}\nstdout:\n{out}\nstderr:\n{err}"
    return out or "(没有打印任何内容)"


def make_skill_tools(skill: AgentSkill) -> list:
    """对应规范里的 filesystem + bash。

    三个 @tool 定义在函数里面：闭包钉死 skill_dir，模型填参时不必（也不能）
    再传技能根路径，降低「跑到别的技能目录」的风险。
    """
    skill_dir = skill.directory.resolve()

    @tool
    def read_skill_file(relative_path: str) -> str:
        """读取当前技能目录内的文件（说明书里链接的脚本、模板、reference.md）。

        这是 Level 3「读资源」。大脚本优先用 run_python 执行，而不是先把源码读进来。

        Args:
            relative_path: 相对技能根目录，如 scripts/comment.py 或 scripts/office/soffice.py
        """
        try:
            target = _inside_skill(skill_dir, relative_path)
        except ValueError as exc:
            return f"拒绝读取: {exc}"
        if not target.is_file():
            return f"不是文件: {relative_path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_READ_CHARS:
            return text[:_MAX_READ_CHARS] + f"\n…(截断，共 {len(text)} 字符)"
        print(f"  ▶ read_skill_file: {relative_path}")  # 给你看的日志，不是给模型的
        return text  # 这个 return 才会进入 ToolMessage，模型下一轮能看见

    @tool
    def run_python(relative_path: str, argv: str = "") -> str:
        """运行技能目录内任意深度的 Python 脚本。路径以 SKILL.md 为准。

        Args:
            relative_path: 如 test.py、scripts/merge_runs.py、scripts/office/soffice.py
            argv: 命令行参数，空格分隔
        """
        try:
            script = _inside_skill(skill_dir, relative_path)
        except ValueError as exc:
            return f"拒绝执行: {exc}"
        if script.suffix.lower() != ".py" or not script.is_file():
            return f"只能执行已存在的 .py: {relative_path}"
        print(f"  ▶ run_python: {relative_path} {argv}".rstrip())
        # sys.executable：当前正在跑 demo 的那个 Python，保证和依赖在同一环境
        # shlex.split(..., posix=False)：按 Windows 规则拆 argv，路径里的反斜杠不会被吃掉
        return _run_cmd(
            [sys.executable, str(script), *shlex.split(argv, posix=False)],
            skill_dir,
        )

    @tool
    def run_shell(command: str) -> str:
        """在技能根目录执行 SKILL.md 中的 shell 命令（unzip、pandoc、curl、zip 等）。

        若命令以 python / python3 开头，会改用当前解释器。
        helpers/ 等内部模块不要当 CLI 跑，除非说明书明确要求。

        Args:
            command: 一整条命令，工作目录已是技能根目录
        """
        raw = command.strip()
        if not raw:
            return "命令为空"
        print(f"  ▶ run_shell: {raw}")
        # 拦住「find /e/test ...」这类全盘搜索。缺用户文件应停，而不是扫磁盘
        if re.search(r"\bfind\s+[/\\]", raw, re.I) or re.search(
            r"\bfind\s+[A-Za-z]:", raw
        ):
            return (
                "禁止在技能目录外搜索磁盘。用户未提供文件路径时，"
                "按 SKILL.md 说明步骤并停止，不要重试 find。"
            )
        try:
            parts = shlex.split(raw, posix=False)  # "curl -s url" → ["curl", "-s", "url"]
        except ValueError as exc:
            return f"命令解析失败: {exc}"
        if not parts:
            return "命令为空"
        if parts[0].lower() in {"python", "python3", "py"}:
            parts[0] = sys.executable  # 单条命令：直接换成当前解释器路径
        # 带管道/串联时必须走 shell，否则 && 只是普通参数
        needs_shell = any(tok in raw for tok in ("&&", "||", "|", ";", "`"))
        if needs_shell:
            return _run_cmd_shell(raw, skill_dir)
        return _run_cmd(parts, skill_dir)

    return [read_skill_file, run_python, run_shell]


def _python_for_shell() -> str:
    """给 shell=True 的命令串用的解释器路径。

    Windows 路径常带空格（如 C:\\Program Files\\...），必须加引号，
    但不能用 POSIX 的 shlex.quote（那会加单引号，cmd 不认）。
    """
    exe = str(sys.executable)
    return f'"{exe}"' if " " in exe else exe


def _run_cmd_shell(command: str, cwd: Path) -> str:
    """说明书里带 && 的复合命令。python 换成当前解释器（Windows 不用 posix 引号）。

    例：python test.py && echo done
    正则只替换「命令位置」上的 python/python3/py，避免误伤参数里的单词。
    """
    tweaked = re.sub(
        r"(^|[;&|]\s*)(python3?|py)\b",
        lambda m: m.group(1) + _python_for_shell(),
        command,
        flags=re.I,
    )
    try:
        completed = subprocess.run(
            tweaked,  # 一整串字符串，由系统 shell 解析
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TOOL_TIMEOUT,
            check=False,
            shell=True,  # 允许 && | ；代价是比参数列表模式更危险，所以前面拦了 find
        )
    except subprocess.TimeoutExpired:
        return f"命令超过 {_TOOL_TIMEOUT} 秒未结束，已终止"
    out = (completed.stdout or "").strip()
    err = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return f"exit={completed.returncode}\nstdout:\n{out}\nstderr:\n{err}"
    return out or "(没有打印任何内容)"


def activated_skill_prompt(skill: AgentSkill) -> str:
    """规范 Level 2：激活后注入 SKILL.md 正文。宿主只补目录与渐进披露，不写产品口号。

    create_agent 的 system_prompt 就是这段。模型从这里知道：
      - 技能名叫什么、根目录在哪
      - 相对路径相对谁
      - 该读文件还是该跑脚本
    """
    return (
        "An Agent Skill has been activated. Follow its SKILL.md instructions below.\n"
        f"Skill name: {skill.name}\n"
        f"Skill directory: {skill.directory}\n"
        "Paths in the skill are relative to that directory.\n"
        "Progressive disclosure (level 3): read referenced files only when the instructions "
        "require them (read_skill_file). Prefer executing bundled scripts and keep their "
        "source out of context; use run_python or run_shell and rely on stdout. "
        "Do not treat helpers/ or schemas/ as CLI entrypoints unless the instructions say so.\n"
        "If this question can be answered from the instructions without tools, answer directly; "
        "do not call tools just to look busy.\n"
        "If a user file required by the instructions is missing, stop and say what path is needed. "
        "Do not search the host filesystem outside the skill directory or retry the same failed command.\n"
        "If a tool errors, report the error; do not claim success.\n\n"
        f"{skill.body}"  # 真正的操作步骤：来自 SKILL.md 正文
    )


def _as_text(content: object) -> str:
    """AIMessage.content 一般是 str，偶发 list/结构化块时压成字符串。"""
    return content if isinstance(content, str) else str(content)


def _direct_answer(question: str, why: str) -> str:
    """工具或技能用不上（以及选技能失败时的降级）：用模型自身能力回答，不抛给用户。"""
    result = get_model().invoke(
        [SystemMessage(content=why), HumanMessage(content=question)]
    )
    return _as_text(result.content)


def execute_with_skill(question: str, skill: AgentSkill | None) -> str:
    """按 SKILL.md 执行；无技能则普通回答。

    市面常见分层：
      · 不需要工具 → 模型直接答（不是异常）
      · 工具失败 → 说明原因，不许装成功（工具已把 stderr 送回模型）
      · 图空转超步数 → 捕获 GraphRecursionError，友好收尾，不甩 traceback
    """
    if skill is None:
        return _direct_answer(
            question,
            "没有适用的 Agent Skill。用你自己的能力直接简短回答，"
            "不要假装调用了某个 skill 或工具。",
        )

    # 有技能：工具挂在身上，但模型可以不调（例如术语表已在 SKILL.md 里）
    agent = create_agent(
        get_model(),
        make_skill_tools(skill),
        system_prompt=activated_skill_prompt(skill),
    )
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"recursion_limit": _AGENT_RECURSION_LIMIT},
        )
    except GraphRecursionError:
        # 对用户：说明卡住了。对开发者：终端打一行，demo 继续下一题
        print_tip(
            f"步数用尽（{_AGENT_RECURSION_LIMIT}）。"
            "产品里捕获 GraphRecursionError，不把异常栈交给用户。"
        )
        return "这次操作步骤过多，已经停止，没有完整做完。请把问题说得更具体一些，或稍后再试。"
    except Exception as exc:
        print(f"  ⚠️  执行中断: {type(exc).__name__}: {exc}")
        return (
            f"刚才执行技能时出错，没有完成操作（{type(exc).__name__}）。"
            "请稍后重试，或换一种问法。"
        )

    # 正常结束：最后一条是自然语言。中间可能调了工具，也可能一轮都没调
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  工具返回: {str(msg.content)[:240]}")
    return _as_text(result["messages"][-1].content)


def handle_question(question: str, registry: SkillRegistry) -> tuple[SkillPick, str]:
    """对外入口：一个问题 → 选技能（Level 1）→ 按 SKILL.md 做（Level 2/3）。

    选技能失败时降级为普通问答，避免整场对话崩掉。
    """
    try:
        pick = pick_skill(question, registry)
    except Exception as exc:
        print(f"  ⚠️  选技能失败，改用模型直接回答: {type(exc).__name__}: {exc}")
        pick = SkillPick(name=None, reason=f"选技能失败，降级直答（{type(exc).__name__}）")
        answer = _direct_answer(
            question,
            "选技能这一步失败了。用你自己的能力直接回答用户问题，不要提内部错误。",
        )
        return pick, answer

    skill = registry.get(pick.name) if pick.name else None
    answer = execute_with_skill(question, skill)
    return pick, answer


def demo_1_scan(registry: SkillRegistry) -> None:
    """演示扫描：把磁盘上的 Skill 列出来，还不调用模型。"""
    print("\n" + "=" * 65)
    print("① 扫描 skills/ —— 丢进来的标准 Skill 都会出现")
    print("=" * 65)
    skills = registry.discover()
    print(f"  共 {len(skills)} 份\n")
    for skill in skills:
        rel = skill.path.relative_to(SKILLS_DIR)
        has_scripts = (skill.directory / "scripts").is_dir()
        print(f"  · {skill.name}")
        print(f"      {rel}")
        print(f"      {skill.description[:90]}...")
        print(f"      scripts/: {'有（按 SKILL.md 调用，含深层入口）' if has_scripts else '无'}")
    print_tip("程序不维护脚本白名单。选中后模型读 SKILL.md，按正文去调工具。")


def demo_2_follow_playbook(registry: SkillRegistry) -> None:
    """端到端：四个问题走同一条 handle_question。选技能发生在 pick_skill（只看 catalog）。"""
    print("\n" + "=" * 65)
    print("② 用户提问 → 选技能 → 按 SKILL.md 执行")
    print("=" * 65)
    questions = [
        "What's the weather in Beijing tomorrow",  # 应命中 weather
        "把「你好世界」翻译成英文",  # 应命中 translate
        "已有一份合同.docx（文件尚未放到工作区）。按 docx 技能说明书："
        "合并碎字和加批注应调用哪些脚本、顺序如何？不要搜索磁盘，缺文件就直接说明缺路径。",
        "1+1 等于几？",  # 不应选任何 skill
    ]
    for q in questions:
        pick, text = handle_question(q, registry)
        print_step(pick.name or "无 skill", q[:56], pick.reason)
        preview = text[:360].replace("\n", "\n    ")
        print(f"  回答:\n    {preview}")


if __name__ == "__main__":
    print_lesson(__file__)
    print(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║     Phase 7-1: 提问 → 选 Skill → 按 SKILL.md 做事         ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )
    print_run_plan(
        [
            ("扫描", "rglob SKILL.md"),
            ("提问", "pick_skill(catalog) → 注入正文 → 工具"),
        ]
    )
    check_api_key()
    registry = SkillRegistry(SKILLS_DIR)
    demo_1_scan(registry)
    demo_2_follow_playbook(registry)
    print("\n🎉 对外只需 handle_question(用户问题)。下一步 7-2 才是 MCP。\n")
