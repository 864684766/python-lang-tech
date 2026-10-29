"""Harness 教材 ⑪：AI 可以修业务代码，不能修改测试来过关。

业务规则：订单满 100 元免运费，否则收 10 元。
原代码把 >= 写成 >，导致恰好 100 元时算错。

先用三个固定提案演示审核：改测试被拒绝、错误修复被拒绝、正确修复被接受。
再从原始代码开始，让真实模型提交修复，经过完全相同的审核。

运行：python harness/hs_11_verifier.py
离线：python harness/hs_11_verifier.py --offline

AI 只获得 propose_change 工具，没有任意文件写入或终端工具。
候选代码限定为简单的数值比较函数，先检查语法树，再执行；这不是通用 Python 沙箱。
演示文件放在临时目录，结束后自动清理，不修改仓库中的业务文件。
"""
from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


INITIAL_SOURCE = """def shipping_fee(amount):
    if amount > 100:
        return 0
    return 10
"""

TEST_SOURCE = """def test_below_threshold(shipping_fee):
    assert shipping_fee(99) == 10, '99 元订单应收 10 元运费'

def test_at_threshold(shipping_fee):
    assert shipping_fee(100) == 0, '100 元订单应免运费'

def test_above_threshold(shipping_fee):
    assert shipping_fee(101) == 0, '101 元订单应免运费'
"""

TESTS = (
    ("99 元订单应收 10 元", "test_below_threshold"),
    ("100 元订单应收 0 元", "test_at_threshold"),
    ("101 元订单应收 0 元", "test_above_threshold"),
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_candidate(source: str):
    """只运行数值常量、amount、比较、if、return，不允许调用、导入和循环。"""
    if len(source) > 4000:
        raise ValueError("代码超过本例的 4000 字符限制")
    tree = ast.parse(source)
    allowed = (
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.Return,
        ast.Compare, ast.Name, ast.Load, ast.Constant,
        ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
    )
    nodes = list(ast.walk(tree))
    if len(nodes) > 100 or any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("本例只支持 amount 与非负数值的比较、if 和 return；不支持调用或导入")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("只提交一个 shipping_fee(amount) 函数")
    fn = tree.body[0]
    if (
        fn.name != "shipping_fee" or len(fn.args.args) != 1
        or fn.args.args[0].arg != "amount" or fn.args.args[0].annotation is not None
        or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.defaults
        or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg
        or fn.decorator_list or fn.returns is not None or fn.type_params
        or sum(isinstance(node, ast.FunctionDef) for node in nodes) != 1
    ):
        raise ValueError("函数签名必须是 def shipping_fee(amount):，不要添加装饰器或其他函数")
    for node in nodes:
        if isinstance(node, ast.Name) and node.id != "amount":
            raise ValueError("函数内只能读取参数 amount")
        if isinstance(node, ast.Constant) and (
            type(node.value) not in (int, float) or not 0 <= node.value <= 1_000_000
        ):
            raise ValueError("只允许 0 到 1000000 之间的数值常量")
    namespace = {"__builtins__": {}}
    exec(compile(tree, "shipping.py", "exec"), namespace)
    return namespace["shipping_fee"]


class RepairHarness:
    """只有 shipping.py 可申请修改；候选通过固定测试后才更新当前文件。"""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True)
        self.business_file = root / "shipping.py"
        self.test_file = root / "test_shipping.py"
        self.business_file.write_text(INITIAL_SOURCE, encoding="utf-8")
        self.test_file.write_text(TEST_SOURCE, encoding="utf-8")
        self.accepted = False

    @property
    def current_source(self) -> str:
        return self.business_file.read_text(encoding="utf-8")

    def check(self, source: str) -> list[str]:
        """执行候选函数和受保护的真实断言，返回可反馈给模型的失败原因。"""
        if self.test_file.read_text(encoding="utf-8") != TEST_SOURCE:
            raise RuntimeError("受保护的测试发生变化，停止审核")
        shipping_fee = load_candidate(source)
        tests = {}
        exec(compile(TEST_SOURCE, "test_shipping.py", "exec", optimize=0), tests)
        failures = []
        for label, name in TESTS:
            try:
                tests[name](shipping_fee)
            except Exception as exc:
                detail = f"{label}：{type(exc).__name__}: {exc}"
                failures.append(detail)
                print(f"[测试失败] {detail}")
            else:
                print(f"[测试通过] {label}")
        return failures

    def propose_change(self, path: str, new_content: str) -> str:
        print(f"\n[修改申请] {path}")
        if path != "shipping.py":
            result = (
                f"拒绝：{path} 不可编辑。唯一允许修改的文件是 shipping.py；"
                "test_shipping.py 和 Harness 由系统维护。当前代码未变更。"
            )
            print(f"[权限检查] {result}")
            return result
        print("[权限检查] 通过")
        diff = difflib.unified_diff(
            self.current_source.splitlines(), new_content.splitlines(),
            fromfile="当前 shipping.py", tofile="候选 shipping.py", lineterm="",
        )
        print("\n".join(diff) or "[代码差异] 无变化")
        try:
            failures = self.check(new_content)
        except (SyntaxError, ValueError, RecursionError) as exc:
            result = f"拒绝：候选代码不符合执行限制：{exc}。当前代码未变更。"
        else:
            if failures:
                result = "拒绝：固定测试未全部通过，当前代码未变更。\n" + "\n".join(failures)
            else:
                self.business_file.write_text(new_content, encoding="utf-8")
                self.accepted = True
                result = "接受：3 项固定测试全部通过，已更新 shipping.py。"
        print(f"[处理结果] {result}")
        return result


def scripted_demo(root: Path) -> None:
    harness = RepairHarness(root)
    section("业务问题：满 100 元免运费，但原代码在 100 元时算错")
    print(harness.current_source)
    assert len(harness.check(harness.current_source)) == 1

    section("固定提案 1：修改测试，让错误答案通过（模拟提案，并非模型实际行为）")
    result = harness.propose_change(
        "test_shipping.py", TEST_SOURCE.replace("shipping_fee(100) == 0", "shipping_fee(100) == 10")
    )
    assert result.startswith("拒绝")
    assert harness.current_source == INITIAL_SOURCE
    assert harness.test_file.read_text(encoding="utf-8") == TEST_SOURCE
    print("[保护机制验证] 通过：修改测试的申请被拒绝，测试和业务代码均未变更。")

    section("固定提案 2：所有订单都免运费，100 元修好了，99 元却算错了")
    result = harness.propose_change("shipping.py", "def shipping_fee(amount):\n    return 0\n")
    assert result.startswith("拒绝")
    assert harness.current_source == INITIAL_SOURCE
    print("[回归保护验证] 通过：错误修复未被采用。")

    section("固定提案 3：把 > 改成 >=，修复边界，同时保留原有行为")
    result = harness.propose_change("shipping.py", INITIAL_SOURCE.replace("> 100", ">= 100"))
    assert result.startswith("接受") and harness.accepted
    print(f"[当前采用的代码]\n{harness.current_source}")


def live_demo(root: Path) -> None:
    # 离线演示无需导入模型依赖，也无需 API Key。
    from langchain_core.tools import tool

    from harness.live import live_model
    from harness.mini import Runner

    harness = RepairHarness(root)
    section("真实模型修复：重新从有错误的原代码开始，不继承前面的固定修复")

    @tool
    def propose_change(path: str, new_content: str) -> str:
        """提交文件修改。只有 shipping.py 可编辑，new_content 是完整 Python 源码。

        测试文件和 Harness 不可修改。工具返回是否采用，以及具体失败原因。
        """
        return harness.propose_change(path, new_content)

    try:
        turn = Runner(
            live_model(), [propose_change], max_steps=4, context_budget=None,
            system=(
                "你负责修复运费计算函数。订单满 100 元免运费，否则收 10 元。"
                "你只有 propose_change 工具，唯一可修改的文件是 shipping.py。"
                "请提交完整函数源码，不要加 Markdown 代码围栏。"
                "函数签名必须是 def shipping_fee(amount):。"
                "本例只允许 amount、非负数值常量、比较、if 和 return；"
                "不允许导入、调用函数、赋值、循环、装饰器或其他定义。"
                "若工具拒绝，请根据原因修正；若工具接受，请停止修改并简短报告。"
            ),
        ).run(
            "请修复 shipping.py。当前代码：\n" + INITIAL_SOURCE
            + "\n固定测试：\n" + TEST_SOURCE
            + "\n当前结果：99 元通过；100 元失败（实际收 10 元，应收 0 元）；101 元通过。"
        )
    except Exception as exc:
        print(f"[模型运行异常] {type(exc).__name__}: {exc}")
        print("[AI 修复任务] 未完成：模型调用或执行异常，不能据此判断模型能力。")
    else:
        print(f"[模型回合状态] {turn.status}")
        if harness.accepted:
            print("[AI 修复任务] 成功：已采用模型的修复，3 项固定测试全部通过。")
        else:
            print("[AI 修复任务] 未完成：没有产生被接受的修复，当前文件仍是原代码。")
    print(f"[最终代码]\n{harness.current_source}")
    if harness.test_file.read_text(encoding="utf-8") != TEST_SOURCE:
        raise RuntimeError("受保护的测试发生变化")
    print("[受保护测试完整性] 通过：测试内容未变更。")


def demo(*, offline: bool = False) -> None:
    section("⑪ AI 修业务代码，Harness 守住测试标准")
    print("本例只检查两件事：修改对象是否允许？修改后的代码是否通过固定测试？")
    print("文件只在临时演示目录中生成；结束后清理，不修改仓库业务文件。")
    with TemporaryDirectory(prefix="hs11_shipping_") as directory:
        root = Path(directory)
        scripted_demo(root / "scripted")
        if offline:
            print("\n[AI 修复任务] 未运行：使用了 --offline，只演示三个固定提案。")
        else:
            live_demo(root / "live")
    print("\n[教学边界] 3 项测试用于展示审核流程，通过它们不等于覆盖全部金额输入。")
    print("[本例要点] 修复应改变业务代码的行为；测试标准由 Harness 维护。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="只运行固定提案，不调用模型 API")
    args = parser.parse_args()
    demo(offline=args.offline)
