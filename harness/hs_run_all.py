"""
Harness + Deep Agents 总入口。其中的在线课程需要 GLM Key；概念地图不调用模型。

全部: python harness/hs_run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import print_lesson, print_run_plan, print_step

DEMOS = [
    ("hs_0_map.py", "三层和三个词"),
    ("hs_1_cannot_act.py", "会说不会做"),
    ("hs_2_tool_loop.py", "工具循环"),
    ("hs_3_step_limit.py", "步数上限"),
    ("hs_4_approval.py", "人审"),
    ("hs_5_context.py", "上下文有界"),
    ("hs_7_when_not.py", "什么时候不用"),
    ("hs_8_vendors.py", "厂家图纸"),
    ("da_0_create.py", "Deep Agents 接 GLM"),
    ("hs_9_handoff.py", "进度落在文件上"),
    ("hs_10_playbook.py", "手册按条改"),
    ("hs_11_verifier.py", "验收器在环外"),
    ("hs_12_lint.py", "报错里写上怎么修"),
    ("hs_13_disclosure.py", "入口是地图"),
    ("hs_14_two_axes.py", "两轴分开打分"),
    ("hs_15_ladder.py", "优化对象的梯子"),
]


def main() -> None:
    print_lesson(__file__)
    print_run_plan([(name, why) for name, why in DEMOS])
    here = Path(__file__).parent
    for filename, why in DEMOS:
        print_step(filename, why)
        runpy.run_path(str(here / filename), run_name="__main__")
    print("\n跑完了。记住：思想是建议 vs 执行；厂家差在房子图纸；进化专题里进度在文件上，验收器在环外。\n")


if __name__ == "__main__":
    main()
