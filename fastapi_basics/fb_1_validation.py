"""
FastAPI 教材 ① 校验
Koa 默认不帮你检查参数类型；FastAPI 看函数签名和 Pydantic 模型，错了直接 422。
运行: python fastapi_basics/fb_1_validation.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from demo_explain import print_lesson, print_ok, print_section, print_tip, print_trap

app = FastAPI()


class UserIn(BaseModel):
    """请求 JSON 的形状。字段少了、类型错了，进不了下面的函数。"""

    name: str
    age: int = Field(ge=0, le=150)  # 0～150，超出也 422


@app.get("/users/{user_id}")
def get_user(user_id: int, verbose: bool = False):
    """
    user_id 来自路径 /users/1
    verbose 来自查询 ?verbose=true
    类型写了 int / bool，框架会帮你转、也会帮你拒。
    """
    return {"user_id": user_id, "verbose": verbose}


@app.get("/unsafe/{user_id}")
def unsafe(user_id):
    """反例：参数没有类型注解，FastAPI 当字符串，不会校验。"""
    return {"user_id": user_id, "type": type(user_id).__name__}


@app.post("/users")
def create_user(body: UserIn):
    # body 已经是 UserIn 对象，不是裸 dict
    return {"name": body.name, "age": body.age}


def demo() -> None:
    print("\n" + "=" * 65)
    print("① 校验：类型写在参数上")
    print("=" * 65)
    client = TestClient(app)

    print_section("正例：路径参数 + 查询参数")
    res = client.get("/users/7?verbose=true")
    print_ok(f"GET /users/7?verbose=true → HTTP {res.status_code} {res.json()}")

    print_section("反例：路径里不是整数")
    res = client.get("/users/abc")
    print_trap(
        "Koa 里 ctx.params.id 往往是字符串 'abc'，你自己 Number() 才发现错",
        f"FastAPI 直接 HTTP {res.status_code}，body={res.json()}",
    )

    print_section("反例：没写类型 = 不校验")
    res = client.get("/unsafe/abc")
    print_trap(
        "unsafe(user_id) 没有 : int",
        f"HTTP {res.status_code} {res.json()}。abc 被原样收下，这是故意写坏的路由。",
    )

    print_section("正例：JSON body + Pydantic")
    res = client.post("/users", json={"name": "alice", "age": 20})
    print_ok(f"POST /users → HTTP {res.status_code} {res.json()}")

    print_section("反例：缺字段")
    res = client.post("/users", json={"name": "alice"})
    print_trap("body 没有 age", f"HTTP {res.status_code}  {res.json()}")

    print_section("反例：age 超出范围")
    res = client.post("/users", json={"name": "alice", "age": -1})
    print_trap("age=-1 不满足 Field(ge=0)", f"HTTP {res.status_code}  {res.json()}")

    print_section("反例：age 类型不对")
    res = client.post("/users", json={"name": "alice", "age": "二十"})
    print_trap("age 是中文，不是 int", f"HTTP {res.status_code}  {res.json()}")

    print_tip("要校验就写类型。不写类型 FastAPI 当 str，等于退回 Koa 的「自己检查」。")


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
