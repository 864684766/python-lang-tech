"""
FastAPI 教材 ② 中间件 和 Depends

是什么:
  HTTP 中间件包住所有请求（洋葱，先进后出）。
  Depends 只挂在某条路由上（鉴权 / 注入）。
  call_next = Koa 的 next()：不调用，后面的路由不会跑。

用在:
  1) 全局日志、request_id、CORS → 中间件
  2) 登录、当前用户、注入依赖 → Depends
  3) 对照 Koa：app.use / router.use

不用:
  参数长什么样对不对 → 上一节类型 / Pydantic，不要手写在中间件里

运行: python fastapi_basics/fb_2_middleware.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from demo_explain import print_lesson, print_ok, print_overview, print_section, print_tip, print_trap

app = FastAPI()
LOG: list[str] = []  # 用来把洋葱进出顺序打印出来，不是业务状态


@app.middleware("http")
async def outer(request: Request, call_next):
    """
    @app.middleware("http")  每个 HTTP 请求都会进这里。
    request     当前请求。request.state 可挂自己的字段（类似 Koa ctx.state）。
    call_next   一个异步函数：await 它才会去下一层中间件/路由。
    必须 async def：FastAPI 规定中间件是协程。语法见 pl_3_asyncio.py。
    """
    LOG.append("outer 进")  # await 之前 = 进
    response = await call_next(request)
    LOG.append("outer 出")  # await 之后 = 出
    response.headers["X-Outer"] = "1"
    return response


@app.middleware("http")
async def inner(request: Request, call_next):
    LOG.append("inner 进")
    request.state.request_id = "req-demo"
    response = await call_next(request)
    LOG.append("inner 出")
    response.headers["X-Request-Id"] = request.state.request_id
    return response


def require_token(authorization: str | None = Header(default=None)) -> str:
    """
    Depends 用的函数：返回值会注入到路由参数里。
    Header(default=None)  从请求头 Authorization 取值；没有也不先 422，好让我们自己 raise 401。
    HTTPException(401)    主动返回「没登录」。和校验失败的 422 不是一回事。
    removeprefix("Bearer ")  去掉前缀，留下 token 本身。
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="需要 Authorization: Bearer ...")
    return authorization.removeprefix("Bearer ")


@app.get("/ping")
def ping(request: Request):
    # 中间件写在 request.state 上的字段，路由里这么读
    return {"pong": True, "request_id": request.state.request_id}


@app.get("/secret")
def secret(token: str = Depends(require_token)):
    # Depends(require_token)：先跑 require_token，成功才进 secret；失败 401。
    return {"ok": True, "token": token}


@app.get("/secret-strict")
def secret_strict(authorization: str = Header()):
    """Header() 没有 default：缺这个头是 422（校验失败），不是 401。"""
    return {"authorization": authorization}


broken = FastAPI()


@broken.middleware("http")
async def forgot_next(request: Request, call_next):
    # 没有 await call_next → /never 的处理函数根本不会跑
    return JSONResponse({"error": "中间件直接返回了，/never 没机会执行"})


@broken.get("/never")
def never():
    return {"should": "not see this"}


def demo() -> None:
    print("\n" + "=" * 65)
    print("② 中间件（洋葱）和 Depends")
    print("=" * 65)

    print_overview(
        "横切：中间件包住所有请求（洋葱）；Depends 只挂在某条路由上（鉴权/注入）。",
        [
            "全局日志、request_id、CORS → 中间件",
            "登录、当前用户、注入依赖 → Depends",
            "对照 Koa：app.use / router.use",
        ],
        "参数长什么样对不对 → 上一节类型 / Pydantic，不要手写在中间件里。",
    )

    print_section("正例：两层 HTTP 中间件的进出顺序")
    print("    后注册的中间件包在更外层（和直觉相反，但和 Starlette 一致）。")
    print("    先进后出：inner 进 → outer 进 → 路由 → outer 出 → inner 出")
    LOG.clear()
    client = TestClient(app)
    res = client.get("/ping")
    print_ok(f"GET /ping → HTTP {res.status_code} {res.json()}")
    print(f"    响应头 X-Request-Id={res.headers.get('x-request-id')}")
    print(f"    洋葱日志: {LOG}")

    print_section("正例：Depends 当「这条路由的中间件」")
    print("    headers={...}  模拟请求头。Bearer 后面是 token。")
    res = client.get("/secret", headers={"Authorization": "Bearer abc123"})
    print_ok(f"带 token → HTTP {res.status_code} {res.json()}")

    print_section("反例：Depends 缺 header → 401；裸 Header() 缺头 → 422")
    res = client.get("/secret")
    print_trap(
        "不传 Authorization（我们自己 raise 401）",
        f"HTTP {res.status_code} {res.json()}。函数 secret() 根本没执行。",
    )
    res = client.get("/secret-strict")
    print_trap(
        "Header() 没写 default，缺头变成校验失败",
        f"HTTP {res.status_code} {res.json()}。想表达「没登录」要在 Depends 里 raise 401。",
    )

    print_section("反例：中间件忘了 call_next")
    print("    对照 Koa 忘了 await next()。JSONResponse 手动造一个 JSON 响应。")
    bad = TestClient(broken)
    res = bad.get("/never")
    print_trap(
        "忘了 await call_next，路由不跑",
        f"HTTP {res.status_code} {res.json()}。/never 的返回值你看不到。",
    )

    print_section("怎么选")
    print("    全局日志、CORS、request_id → HTTP 中间件（所有路径）。")
    print("    登录、取当前用户、注入数据库 → Depends（按路由挂）。")
    print("    参数长什么样对不对 → 上一节的类型 / Pydantic，不是中间件。")

    print_tip("中间件必须 await call_next 并 return 它的响应。校验不要手写在中间件里，用类型。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
