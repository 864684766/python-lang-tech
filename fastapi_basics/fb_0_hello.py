"""
FastAPI 教材 ⓪ 最小接口

是什么:
  用装饰器把函数挂到 URL 上，return dict 就是 JSON。没有 Koa 的 ctx。
  本文件用 TestClient 假装发请求，不用自己起服务器。

用在:
  1) 先确认「这就是一个 HTTP 接口」：GET /hello → {"msg": "hi"}
  2) 对照 Koa：router.get + ctx.body = {...}

不用:
  参数校验、登录、日志 —— 分别是后面两节，本节不管

运行: python fastapi_basics/fb_0_hello.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# TestClient 内部用 httpx，会打一条无害警告，教材里关掉
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from demo_explain import print_lesson, print_ok, print_overview, print_section, print_tip, print_trap

# FastAPI() 造一个应用。后面 @app.get 都注册到它身上。
app = FastAPI()


# @app.get("/hello") 读法：当有人 GET /hello 时，调用下面这个函数。
# 等价 Koa：router.get('/hello', handler)
@app.get("/hello")
def hello():
    # 返回 dict → FastAPI 自动变成 JSON 响应。Koa 是 ctx.body = { ... }
    return {"msg": "hi"}


@app.post("/hello")
def hello_post():
    return {"msg": "created"}


def demo() -> None:
    print("\n" + "=" * 65)
    print("⓪ FastAPI 最小接口（对照 Koa）")
    print("=" * 65)

    print_overview(
        "用装饰器把函数挂到 URL 上，return dict 就是 JSON。没有 Koa 的 ctx。",
        [
            "先确认「这就是一个 HTTP 接口」：GET /hello → {\"msg\": \"hi\"}",
            "对照 Koa：router.get + ctx.body = {...}",
        ],
        "参数校验、登录、日志分别是后面两节，本节不管。",
    )

    print_section("TestClient：教材里不真的起端口")
    print("    TestClient(app)  假装发 HTTP。client.get / .post / .delete 对应方法。")
    print("    返回的 res.status_code  状态码  200 成功 / 404 没有 / 405 方法不对")
    print("    res.json()              把响应 body 解析成 dict")
    client = TestClient(app)

    print_section("正例：GET /hello")
    res = client.get("/hello")
    print_ok(f"HTTP {res.status_code}  body={res.json()}")

    print_section("反例：用错 HTTP 方法")
    print("    /hello 只声明了 GET 和 POST。DELETE 没有处理函数。")
    res_wrong = client.delete("/hello")
    print_trap(
        "DELETE 一个只声明了 GET/POST 的路径",
        f"HTTP {res_wrong.status_code}（405 Method Not Allowed）。"
        "Koa 里没挂上的 method 同样是 404/405。",
    )

    print_section("反例：路径写错")
    res_404 = client.get("/helo")
    print_trap(
        "拼错 URL",
        f"HTTP {res_404.status_code}。FastAPI 不会猜你想访问 /hello。",
    )

    print_section("和 Koa 的差别（先记住这一句）")
    print("    Koa：一切从 ctx 上拿。FastAPI：参数写在函数签名里（下一节校验）。")
    print("    返回值：Koa 赋值 ctx.body；FastAPI 直接 return dict。")

    print_tip("先会 GET 返回 JSON，再学校验和中间件。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
