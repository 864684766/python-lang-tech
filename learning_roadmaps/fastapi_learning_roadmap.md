# FastAPI 学习路线

给会 Koa 的人：HTTP 思路能迁移，写法差在「没有 `ctx`」和「类型即校验」。

**不需要 API Key。** 用 `TestClient`，不用自己起服务器。

前置：[Python 类型](./python_basics_learning_roadmap.md)（至少 `str` / `dict` / `None`）。建议已跑 [语言 ① typing](./python_lang_learning_roadmap.md) 和 [标准库 ⓪ json](./python_stdlib_learning_roadmap.md)。

## 怎么跑

在**仓库根目录**执行：

```bash
python fastapi_basics/fb_run_all.py

python fastapi_basics/fb_0_hello.py
python fastapi_basics/fb_1_validation.py
python fastapi_basics/fb_2_middleware.py
```

## 顺序

每课文件头和运行开头都有「是什么 / 用在 / 不用」。先看总览再跑。

| 节 | 文件 | 是什么 | 什么时候用 | 不用 | 对照 Koa | 盯住的反例 |
|----|------|--------|------------|------|----------|------------|
| ⓪ | `fb_0_hello.py` | 装饰器挂路由，`return dict` 即 JSON | 先确认「这就是一个 HTTP 接口」 | 校验 / 登录 / 日志是后面两节 | `ctx.body = {...}` | 错方法 **405**；错路径 **404** |
| ① | `fb_1_validation.py` | 类型和 Pydantic 拦脏数据，错了 **422** | `/users/7` 必须是 int；没默认值 = 必填，有默认值 = 选填 | 参数不写类型 = 不校验 | Koa 默认不校验 | `/users/abc` → **422**；`Field(ge=0)` 不是选填 |
| ② | `fb_2_middleware.py` | 中间件包全部请求；`Depends` 挂单条路由 | 全局日志 / `request_id`；登录注入 | 参数长什么样（那是上一节） | `app.use` / `router.use` | 忘了 `call_next`；缺 Header 是 **401** 还是 **422** |

## 怎么选（② 末尾也有）

- 全局日志、`request_id`、CORS → HTTP 中间件
- 登录、当前用户、注入依赖 → `Depends`
- 参数长什么样 → 写在函数签名上，不要塞进中间件

下一步：要接大模型走 [LangChain](./langchain_learning_roadmap.md)；只要 HTTP 服务，本路线够用。
