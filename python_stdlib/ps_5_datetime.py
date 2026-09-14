"""
Python 标准库教材 ⑤ datetime / time（墙上几点、过期、秒表）

是什么:
  两个独立的标准库，不是一套 API 的前后缀。
  datetime 管「墙上几点、过期、时区、格式化」。
  time 管「睡一会儿、按秒表、Unix 时间戳」。
  唯一能对上的地方：time.time() 的浮点数 ↔ datetime.fromtimestamp / .timestamp()。

用在:
  1) 存库 / 比过期：datetime.now(timezone.utc) + timedelta
  2) 给用户看上海/北京时间：ZoneInfo("Asia/Shanghai")
  3) 测模型耗时：time.perf_counter 相减（和「现在几点」无关）

不用:
  async 函数里不要 time.sleep（会卡住事件循环，见 pl_3）
  naive（没标时区）和带时区的不能直接比
  不要用 perf_counter 当墙上时钟

运行: python python_stdlib/ps_5_datetime.py
"""
from __future__ import annotations

import sys
import time  # 模块 time：sleep、perf_counter、time()。不是「日期」。
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo  # 城市时区。上海/北京都用 Asia/Shanghai（固定 UTC+8）

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_explain import (
    print_lesson,
    print_ok,
    print_overview,
    print_section,
    print_tip,
    print_trap,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def demo() -> None:
    print("\n" + "=" * 65)
    print("⑤ datetime / time：每个名字干什么")
    print("=" * 65)

    print_overview(
        "datetime 管墙上几点；time 管秒表和休眠。两套库，不是联动前缀。",
        [
            "存库 / 比过期：datetime.now(timezone.utc) + timedelta",
            "给用户看上海时间：ZoneInfo('Asia/Shanghai')",
            "测耗时：time.perf_counter 相减（不要当「几点」）",
        ],
        "async 里不要 time.sleep。naive 和带时区的不要直接比。",
    )

    print_section("两个库先分清（不是一套东西）")
    print("    datetime  某一个瞬间、加减小时、标时区、strftime。日历用这个。")
    print("    time      sleep / perf_counter / time()。秒表和休眠用这个。")
    print("    对照: C# 的 DateTimeOffset 对 datetime；Stopwatch / Thread.Sleep 对 time。")
    print("    它们不会自动联在一起。唯一握手: Unix 时间戳（下面单独跑）。")

    print_section("datetime.now(时区)")
    print("    datetime 是「某一个瞬间」。now() = 现在。")
    print("    timezone.utc  世界标准时。存库、比过期用这个。")
    print("    不传时区得到 naive（天真）时间：看起来像本地，但对象上没标时区。")
    print("    .isoformat()  转成 '2026-09-11T02:45:22+00:00' 这种字符串，方便 JSON。")
    print("    对照: JS 的 new Date() / toISOString()；C# 的 DateTime.UtcNow")
    now_utc = datetime.now(timezone.utc)
    now_naive = datetime.now()
    print_ok(f"带时区 UTC: {now_utc.isoformat()}")
    print_ok(f"不带时区 naive: {now_naive.isoformat()}    ← 不知道是哪条时区")

    print_section("上海 / 北京时间：ZoneInfo('Asia/Shanghai')")
    print("    上海和北京是同一条时区：中国标准时间，固定 UTC+8，没有夏令时。")
    print("    IANA 名字写 Asia/Shanghai。一般没有 Asia/Beijing。")
    print("    datetime.now(SHANGHAI)  现在的上海墙上钟点，带 +08:00。")
    now_sh = datetime.now(SHANGHAI)
    print_ok(f"现在上海: {now_sh.isoformat()}")
    print_ok(f"同一个瞬间转上海: {now_utc.astimezone(SHANGHAI).isoformat()}")
    print("    astimezone = 换算钟点。UTC 04:00 → 上海 12:00。数字会变，瞬间不变。")

    print_section("naive 补上「这是上海时间」：replace，不是 astimezone")
    print("    用户填了 2026-09-14 12:00，没写时区，但你们约定就是上海。")
    print("    replace(tzinfo=...)  数字不动，只贴标签 +08:00。")
    naive_clock = datetime(2026, 9, 14, 12, 0, 0)
    labeled = naive_clock.replace(tzinfo=SHANGHAI)
    print_ok(f"naive {naive_clock.isoformat()}  → replace → {labeled.isoformat()}")
    utc_4am = datetime(2026, 9, 14, 4, 0, 0, tzinfo=timezone.utc)
    print_ok(f"UTC 04:00 astimezone → {utc_4am.astimezone(SHANGHAI).isoformat()}    钟点变成 12:00")
    print_trap(
        "把一段「其实是 UTC 的数字」naive 直接 replace 成上海",
        "4 点贴上 +08:00，还是 4 点，并没有加 8 小时。"
        " 你知道数字已经是上海钟 → replace；你手里是 UTC → astimezone。",
    )
    wrong = datetime(2026, 9, 14, 4, 0, 0).replace(tzinfo=SHANGHAI)
    print_ok(f"反例长这样: {wrong.isoformat()}    不是上海中午")

    print_section("timedelta：一段时间（不是某一个瞬间）")
    print("    timedelta(hours=1)  表示「一小时这么长」，可以和 datetime 相加。")
    print("    过期时间 = 现在 + timedelta。TTL / 会话过期用这个。")
    print("    两个 datetime 相减得到 timedelta；.total_seconds() 换成秒。")
    expires = now_utc + timedelta(hours=1)
    print_ok(f"一小时后: {expires.isoformat()}")
    print_ok(f"已经过期? {datetime.now(timezone.utc) >= expires}")
    print_ok(f"差多少秒: {(expires - now_utc).total_seconds()}")

    print_section("strftime：datetime 自己的格式化，不经过 time 模块")
    print("    strftime = string format time。按你给的格子把时间变成字符串。")
    print("    %Y 年 %m 月 %d 日 %H 时 %M 分 %S 秒")
    print_ok(f"UTC:     {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print_ok(f"上海:    {now_sh.strftime('%Y-%m-%d %H:%M:%S')} CST")

    print_section("唯一握手：Unix 时间戳（time.time ↔ datetime）")
    print("    time.time()  从 1970-01-01 UTC 起的秒数（float）。给别的系统比大小。")
    print("    datetime.fromtimestamp(秒, 时区)  把这个秒数变成墙上几点。")
    print("    某个 datetime.timestamp()         变回秒数。")
    print("    除此之外两套 API 各用各的，不要混着点。")
    ts = time.time()
    from_ts = datetime.fromtimestamp(ts, SHANGHAI)
    print_ok(f"time.time() → {ts:.0f}")
    print_ok(f"fromtimestamp(..., 上海) → {from_ts.isoformat()}")
    print_ok(f"再 .timestamp() → {from_ts.timestamp():.0f}    和刚才同一秒")

    print_section("秒表：time.perf_counter / time.sleep（和 datetime 无关）")
    print("    perf_counter()  高精度单调时钟，只适合相减测耗时，不要当「几点」。")
    print("    它对不上 fromtimestamp，也没有时区。")
    print("    sleep(秒)       堵住当前线程等这么久。async 函数里不要用它。")
    t0 = time.perf_counter()
    time.sleep(0.05)
    print_ok(f"sleep 0.05s 实际 {time.perf_counter() - t0:.3f}s    model_provider 打耗时就是这个")

    print_trap(
        "naive 和 UTC 直接比",
        "带时区和不带时区不能用 >。存库统一 datetime.now(timezone.utc)。",
    )
    try:
        _ = now_utc > now_naive
    except TypeError as e:
        print_ok(f"正是这个错: {type(e).__name__}: {e}")

    print_trap(
        "在 async 函数里 time.sleep",
        "会卡住整条事件循环。异步等待用 await asyncio.sleep，见 pl_3_asyncio.py。",
    )

    print_tip(
        "墙上几点 / 时区 / 过期 → datetime；睡一会儿 / 打秒表 → time；"
        " 只有时间戳能互相转。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
