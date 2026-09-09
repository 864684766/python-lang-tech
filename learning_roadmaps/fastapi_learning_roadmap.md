# FastAPI 学习路线

给会 Koa 的人：HTTP 思路能迁移，写法差在「没有 `ctx`」和「类型即校验」。

**不需要 API Key。** 用 `TestClient`，不用自己起服务器。

前置：[Python 类型](./python_basics_learning_roadmap.md)（至少 `str` / `dict` / `None`）。

## 怎么跑

在**仓库根目录**执行：

```bash
python fastapi_basics/fb_run_all.py

python fastapi_basics/fb_0_hello.py
python fastapi_basics/fb_1_validation.py
python fastapi_basics/fb_2_middleware.py
```

## 顺序

| 节 | 文件 | 学什么 | 对照 Koa | 盯住的反例 |
|----|------|--------|----------|------------|
| ⓪ | `fb_0_hello.py` | `@app.get`，`return dict` 即 JSON | `ctx.body = {...}` | 错方法 **405**；错路径 **404** |
| ① | `fb_1_validation.py` | 路径/Query/Body；Pydantic | Koa 默认不校验 | `/users/abc` → **422**；参数不写类型就不校验 |
| ② | `fb_2_middleware.py` | HTTP 洋葱中间件；`Depends` | `app.use` / `router.use` | 忘了 `call_next` 路由不跑；缺 Header 是 **401** 还是 **422** |

## 怎么选（② 末尾也有）

- 全局日志、`request_id`、CORS → HTTP 中间件
- 登录、当前用户、注入依赖 → `Depends`
- 参数长什么样 → 写在函数签名上，不要塞进中间件

下一步：要接大模型走 [LangChain](./langchain_learning_roadmap.md)；只要 HTTP 服务，本路线够用。
