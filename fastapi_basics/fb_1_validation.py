"""
FastAPI 教材 ① 校验

是什么:
  路径 / 查询 / Body 靠类型和 Pydantic 检查，错了直接 422，函数不执行。
  Koa 默认不拦，你自己 Number()；FastAPI 看签名就拦。
  Pydantic = 用类型标注在运行时校验数据的库（FastAPI 自带依赖）。

用在:
  1) /users/7 的 7 必须是 int；不是就 422
  2) POST body 必须有 name/age，age 还要在 Field(ge=0, le=150) 里

不用:
  参数不写类型 = 不校验，等于退回 Koa
  普通函数的 : str 也不会拦，那是 pl_1；这里是框架在拦

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

from demo_explain import print_lesson, print_ok, print_overview, print_section, print_tip, print_trap

app = FastAPI()


class UserIn(BaseModel):
    """
    BaseModel = Pydantic 的模型基类。继承它 = 这份 JSON 要按字段校验。
    Field(ge=0, le=150)  ge=greater or equal 最小值；le=less or equal 最大值。
    超出范围也 422，不只是「是不是 int」。
    """

    name: str
    age: int = Field(ge=0, le=150)


@app.get("/users/{user_id}")
def get_user(user_id: int, verbose: bool = False):
    """
    {user_id} 出现在路径里 → 路径参数。/users/7 的 7 会传进来。
    verbose 没有在路径里 → 查询参数。?verbose=true
    : int / : bool 框架会帮你从字符串转类型；转失败 → 422。
    """
    return {"user_id": user_id, "verbose": verbose}


@app.get("/unsafe/{user_id}")
def unsafe(user_id):
    """反例：参数没有类型注解，FastAPI 当字符串，不会校验。"""
    return {"user_id": user_id, "type": type(user_id).__name__}


@app.post("/users")
def create_user(body: UserIn):
    # 参数类型是 BaseModel → 从 JSON body 里填。进来时已经是 UserIn，不是裸 dict。
    # 所以用 body.name 点属性，不是 body['name']。
    return {"name": body.name, "age": body.age}


def demo() -> None:
    print("\n" + "=" * 65)
    print("① 校验：类型写在参数上")
    print("=" * 65)

    print_overview(
        "路径 / 查询 / Body 靠类型和 Pydantic 检查，错了直接 422，函数不执行。",
        [
            "/users/7 的 7 必须是 int；不是就 422",
            "POST body 必须有 name/age，age 还要满足 Field(ge=0)",
        ],
        "参数不写类型 = 不校验。普通函数的 : str 也不会拦，那是 pl_1。",
    )

    client = TestClient(app)

    print_section("正例：路径参数 + 查询参数")
    print("    /users/{user_id} 的 user_id 来自 URL。verbose 来自 ?verbose=")
    print("    client.get 的返回值和上一节一样：status_code + .json()")
    res = client.get("/users/7?verbose=true")
    print_ok(f"GET /users/7?verbose=true → HTTP {res.status_code} {res.json()}")

    print_section("反例：路径里不是整数")
    print("    422 Unprocessable Entity = 请求形状不对，被校验拦住，函数没执行。")
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
    print("    POST 的 json={...} 会变成请求体。参数类型是 UserIn → 按模型填字段。")
    res = client.post("/users", json={"name": "alice", "age": 20})
    print_ok(f"POST /users → HTTP {res.status_code} {res.json()}")

    print_section("反例：缺字段 / 超出 Field / 类型不对")
    res = client.post("/users", json={"name": "alice"})
    print_trap("body 没有 age", f"HTTP {res.status_code}  {res.json()}")

    res = client.post("/users", json={"name": "alice", "age": -1})
    print_trap("age=-1 不满足 Field(ge=0)", f"HTTP {res.status_code}  {res.json()}")

    res = client.post("/users", json={"name": "alice", "age": "二十"})
    print_trap("age 是中文，不是 int", f"HTTP {res.status_code}  {res.json()}")

    print_tip(
        "要校验就写类型。不写类型 FastAPI 当 str，等于退回 Koa 的「自己检查」。"
        " 这和 pl_1_typing 不同：那里普通函数的 : str 运行时不拦；这里框架会拦。"
    )


if __name__ == "__main__":
    print_lesson(__file__)
    demo()
