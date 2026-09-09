"""
FastAPI 教材 ⓪ 最小接口
对照 Koa：router.get + ctx.body。这里没有 ctx，返回 dict 就会变成 JSON。
本文件用 TestClient，不用自己起服务器。
运行: python fastapi_basics/fb_0_hello.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from demo_explain import print_lesson, print_ok, print_section, print_tip, print_trap

app = FastAPI()


# 装饰器 = 注册路由。等价 Koa：router.get('/hello', handler)
@app.get("/hello")
def hello():
    # 返回 dict → FastAPI 自动 JSON。Koa 是 ctx.body = { ... }
    return {"msg": "hi"}


@app.post("/hello")
def hello_post():
    return {"msg": "created"}


def demo() -> None:
    print("\n" + "=" * 65)
    print("⓪ FastAPI 最小接口（对照 Koa）")
    print("=" * 65)

    # TestClient 假装发 HTTP，这样教材可以一条命令跑完
    client = TestClient(app)

    print_section("正例：GET /hello")
    res = client.get("/hello")
    print_ok(f"HTTP {res.status_code}  body={res.json()}")

    print_section("反例：用错 HTTP 方法")
    res_wrong = client.delete("/hello")
    print_trap(
        "DELETE 一个只声明了 GET/POST 的路径",
        f"HTTP {res_wrong.status_code}（405 Method Not Allowed）。"
        "Koa 里没挂上的 method 同样是 404/405，取决于你怎么写路由。",
    )

    print_section("反例：路径写错")
    res_404 = client.get("/helo")
    print_trap(
        "拼错 URL",
        f"HTTP {res_404.status_code}。FastAPI 不会猜你想访问 /hello。",
    )

    print_section("和 Koa 的差别（先记住这一句）")
    print("    Koa：一切从 ctx 上拿。FastAPI：参数写在函数签名里（下一节校验）。")
    print("    返回值：Koa 赋值 ctx.body；FastAPI 直接 return。")

    print_tip("先会 GET 返回 JSON，再学校验和中间件。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
