"""
Python 标准库教材 ⑥ logging（按级别打日志，线上能关掉 DEBUG）

是什么:
  logging 按级别和名字输出。print 没有级别，服务上关不掉。
  对照: C# 的 ILogger；Node 的 pino / winston transport。

用在:
  1) FastAPI / Agent 服务的关键路径：请求进来、工具失败、模型变慢
  2) 线上把级别调到 WARNING，DEBUG 自动消失
  3) 出口可换：终端、文件、由 Docker/systemd 收集 stdout

不用:
  教材课堂演示用 print_ok 更清楚
  一次性脚本打两行也不必上 logging

运行: python python_stdlib/ps_6_logging.py
"""
from __future__ import annotations

import logging  # 模块 logging。常用：getLogger、debug/info/warning/error
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_lesson,
    print_ok,
    print_overview,
    print_section,
    print_tip,
    print_trap,
)


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑥ logging：logger、handler、级别")
    print("=" * 65)

    print_overview(
        "按级别和名字打日志。print 没有级别，线上关不掉。",
        [
            "FastAPI / Agent 服务的关键路径：请求进来、工具失败、模型变慢",
            "线上把级别调到 WARNING，DEBUG 自动消失",
            "出口可换：终端、文件；容器里通常打 stdout 让平台收集",
        ],
        "教材课堂演示用 print_ok 更清楚。一次性脚本打两行也不必上 logging。",
    )

    print_section("先搭一条管道：logger 是源头，handler 是出口")
    print("    对照: C# 的 ILogger（怎么记）+ Provider（打到哪）；")
    print("          winston 的 logger + transport。")
    print("    下面 7 行各干什么：")
    print("    getLogger(名字)     拿到记录器。同名永远是内存里同一个对象（配置，不是日志仓库）。")
    print("    handlers.clear()    教材防重复：这一课跑两次时，不要叠两个终端出口。")
    print("    StreamHandler(流)   出口之一：写到一个「能 .write 的流」。这里是 sys.stdout = 终端。")
    print("    Formatter('...')    每行长什么样。%(levelname)s 级别，%(name)s logger 名。")
    print("    addHandler          把这个出口接到 logger 上。一个 logger 可以挂多个出口。")
    print("    setLevel            低于这个级别的，这个 logger 直接丢掉，不送给 handler。")
    print("    propagate=False     不要再传给父 logger（根 logger 常自带一个出口，会打两遍）。")
    log = logging.getLogger("langtest.demo")
    log.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    print_section("五个级别（从松到严）")
    print("    DEBUG < INFO < WARNING < ERROR < CRITICAL")
    print("    log.debug('...%s', 值)  占位用 %s，不要先 f-string 再传入。")
    print("    级别被关掉时，%s 那套不会白拼字符串。")
    log.debug("调试：向量维度=%s", 3)
    log.info("正常：工具 lookup 完成")
    log.warning("警告：模型慢了")
    log.error("错误：JSON 解析失败")
    print_ok("上面四行是 logging 打的，不是 print")

    print_trap(
        "只 print 当日志",
        "print 没有级别。服务上用 getLogger(__name__)，__name__ 会自动变成模块路径。",
    )

    print_section("setLevel 过滤")
    print("    调成 WARNING 之后，DEBUG 和 INFO 被吞，WARNING 及以上还在。")
    log.setLevel(logging.WARNING)
    log.info("这条 INFO 不会出现")
    log.warning("只剩 WARNING 及以上")
    print_ok("setLevel(WARNING) 之后 DEBUG/INFO 被吞")

    print_section("打到文件：FileHandler（可以，和终端是同一种「出口」）")
    print("    换出口 = 换 Handler，logger.debug/info 那几行不用改。")
    print("    FileHandler(路径)           追加写入一个文件。")
    print("    RotatingFileHandler         按大小切开，避免单个 log 无限涨。")
    print("    TimedRotatingFileHandler    按天切开。")
    print("    encoding='utf-8'            Windows 上不写这个，中文会乱。")
    path = Path(__file__).with_name("_demo.log")
    file_log = logging.getLogger("langtest.demo.file")
    file_log.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    file_log.addHandler(file_handler)
    file_log.setLevel(logging.INFO)
    file_log.propagate = False
    file_log.info("这一条在文件里，不在上面的终端 logger")
    file_handler.close()
    file_log.handlers.clear()
    text = path.read_text(encoding="utf-8").strip()
    print_ok(f"读回 {path.name}: {text}")
    path.unlink()

    print_section("还能打到哪；服务挂到机器上通常怎么接")
    print("    标准库还有: StreamHandler(stderr)、SysLogHandler（Linux syslog）、")
    print("    SMTPHandler（发邮件，ERROR 才用）、QueueHandler（丢给别的线程再写）。")
    print("    日常 Python 服务（FastAPI 挂 systemd / Docker / K8s）常见三种载体：")
    print("    1) stdout/stderr   现在最常见。容器/K8s/journald 自己收集。应用只打终端。")
    print("    2) 本地文件        传统虚拟机：FileHandler + 轮转。要自己管磁盘和权限。")
    print("    3) 日志平台        Loki / ELK / CloudWatch / 阿里云 SLS。")
    print("       应用仍多打 stdout，由采集器收走；或接厂商 SDK。不要在业务里自己 HTTP 打日志。")
    print_ok("对照 Koa：pino 打 stdout 给 pm2/docker；不是把日志 list 存在进程里。")
    print_ok("先会 stdout；要落盘就加 FileHandler。平台收集是运维侧，不是换一套 logging API。")

    print_section("名字是层级")
    print("    'langtest.demo' 是 'langtest' 的孩子。可以只把业务 logger 调到 DEBUG。")
    print("    第三方库各用各的名字，互不干扰。")
    print_ok("getLogger('langtest.demo') 挂在 getLogger('langtest') 下面")

    print_tip(
        "logger 负责级别和名字；handler 负责打到哪。"
        " 容器里打 stdout；自己管机器再写文件。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
