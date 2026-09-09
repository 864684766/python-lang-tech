"""
FastAPI 教材 ② 中间件 和 Depends
对照 Koa：
  - HTTP 中间件 ≈ app.use，仍然是洋葱：先进后出
  - Depends ≈ 某个路由上的中间件，结果有点像 ctx.state
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

from demo_explain import print_lesson, print_ok, print_section, print_tip, print_trap

# ---------- 正常的那份 app：洋葱中间件 + 鉴权 Depends ----------
app = FastAPI()
LOG: list[str] = []


@app.middleware("http")
async def outer(request: Request, call_next):
    # 先执行（进）：像 Koa 里 await next() 之前
    LOG.append("outer 进")
    response = await call_next(request)
    # 后执行（出）：像 Koa 里 await next() 之后
    LOG.append("outer 出")
    response.headers["X-Outer"] = "1"
    return response


@app.middleware("http")
async def inner(request: Request, call_next):
    LOG.append("inner 进")
    request.state.request_id = "req-demo"  # 类似 Koa 的 ctx.state
    response = await call_next(request)
    LOG.append("inner 出")
    response.headers["X-Request-Id"] = request.state.request_id
    return response


def require_token(authorization: str | None = Header(default=None)) -> str:
    """
    Depends：只对用到它的路由生效，不像 HTTP 中间件那样全球拦截。
    有点像 Koa 里某个 router.use(auth)。
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="需要 Authorization: Bearer ...")
    return authorization.removeprefix("Bearer ")


@app.get("/ping")
def ping(request: Request):
    return {"pong": True, "request_id": request.state.request_id}


@app.get("/secret")
def secret(token: str = Depends(require_token)):
    return {"ok": True, "token": token}


@app.get("/secret-strict")
def secret_strict(authorization: str = Header()):
    """反例：Header 没有 default 时，缺头是 422（校验失败），不是 401。"""
    return {"authorization": authorization}


# ---------- 反例 app：中间件忘了 call_next ----------
broken = FastAPI()


@broken.middleware("http")
async def forgot_next(request: Request, call_next):
    # 没有 await call_next，后面的路由函数根本不会跑
    return JSONResponse({"error": "中间件直接返回了，/never 没机会执行"})


@broken.get("/never")
def never():
    return {"should": "not see this"}


def demo() -> None:
    print("\n" + "=" * 65)
    print("② 中间件（洋葱）和 Depends")
    print("=" * 65)

    print_section("正例：两层 HTTP 中间件的进出顺序")
    LOG.clear()
    client = TestClient(app)
    res = client.get("/ping")
    print_ok(f"GET /ping → HTTP {res.status_code} {res.json()}")
    print(f"    响应头 X-Request-Id={res.headers.get('x-request-id')}")
    print(f"    洋葱日志: {LOG}")
    print("    后注册的 inner 更靠外：inner 进 → outer 进 → 路由 → outer 出 → inner 出")
    print("    （和 Koa 一样先进后出；FastAPI 后写的 middleware 包在更外层。）")

    print_section("正例：Depends 当「这条路由的中间件」")
    res = client.get("/secret", headers={"Authorization": "Bearer abc123"})
    print_ok(f"带 token → HTTP {res.status_code} {res.json()}")

    print_section("反例：Depends 缺 header")
    res = client.get("/secret")
    print_trap(
        "不传 Authorization（我们自己 raise 401）",
        f"HTTP {res.status_code} {res.json()}。函数 secret() 根本没执行。",
    )
    res = client.get("/secret-strict")
    print_trap(
        "Header() 没写 default，缺头变成校验失败",
        f"HTTP {res.status_code} {res.json()}。想表达「没登录」要用 Depends 里 raise 401，不要指望裸 Header。",
    )

    print_section("反例：中间件忘了 call_next")
    bad = TestClient(broken)
    res = bad.get("/never")
    print_trap(
        "Koa 里忘了 await next()，后面的中间件/路由也不会跑",
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
