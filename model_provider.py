# -*- coding: utf-8 -*-
"""
全项目统一的模型获取模块。

支持两种「主模型」接入方式（自动识别）:
  A. 智谱官方 API     → ChatZhipuAI（open.bigmodel.cn）
  B. 千帆 Token Plan  → ChatOpenAI 兼容模式（代理商 bce-v3 Key）
     Base URL: https://qianfan.baidubce.com/v2/tokenplan/personal

备用:
  OPENAI_API_KEY + OPENAI_API_BASE（可选，代理商网关）

.env 示例（千帆 Token Plan + GLM）:
  ZHIPUAI_API_KEY=bce-v3/ALTAK-xxx
  ZHIPUAI_API_BASE=https://qianfan.baidubce.com/v2/tokenplan/personal
  ZHIPUAI_MODEL=glm-5.2
  OPENAI_API_KEY=sk-xxx
  OPENAI_API_BASE=https://你的代理商地址/v1
  MODEL_PROVIDER_LOG=1
"""

from __future__ import annotations

import os
import time
from numbers import Number
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableSerializable
from pydantic import ConfigDict

load_dotenv()

_fallback_notice_printed = False
_LOG_PREFIX = "[model]"

# 千帆 Token Plan 个人版（OpenAI 兼容）默认地址
QIANFAN_TOKENPLAN_BASE = "https://qianfan.baidubce.com/v2/tokenplan/personal"

# Token Plan 文档支持的 model 参数示例（glm-5.2 为智谱 GLM 系列）
TOKENPLAN_DEFAULT_MODEL = "glm-5.2"
NATIVE_ZHIPU_DEFAULT_MODEL = "glm-4-flash"


def _log_enabled() -> bool:
    return os.getenv("MODEL_PROVIDER_LOG", "1").lower() not in ("0", "false", "no", "off")


def _log(msg: str) -> None:
    if _log_enabled():
        print(f"{_LOG_PREFIX} {msg}")


def _zhipu_key() -> str | None:
    return os.getenv("ZHIPUAI_API_KEY")


def _openai_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def _zhipu_api_base() -> str | None:
    return os.getenv("ZHIPUAI_API_BASE") or os.getenv("QIANFAN_API_BASE")


def _openai_api_base() -> str | None:
    return os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")


def _is_qianfan_tokenplan() -> bool:
    """
    识别千帆 Token Plan 个人版 Key（bce-v3/...）或专属 Base URL。

    此类 Key 只能走 https://qianfan.baidubce.com/v2/tokenplan/personal ，
    不能用于智谱官方 open.bigmodel.cn 或千帆常规 /v2 接口。
    """
    key = _zhipu_key() or ""
    base = _zhipu_api_base() or ""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider in ("qianfan_tokenplan", "tokenplan", "qianfan"):
        return True
    if key.startswith("bce-v3/"):
        return True
    return "tokenplan/personal" in base


def _primary_provider_label() -> str:
    return "千帆TokenPlan" if _is_qianfan_tokenplan() else "Zhipu"


def _normalize_openai_base_url(base: str) -> str:
    """去掉末尾 /chat/completions，ChatOpenAI 会自动拼接。"""
    base = base.rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def _default_primary_model(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_model = os.getenv("ZHIPUAI_MODEL")
    if env_model:
        return env_model
    return TOKENPLAN_DEFAULT_MODEL if _is_qianfan_tokenplan() else NATIVE_ZHIPU_DEFAULT_MODEL


def _unwrap_runnable(runnable: Any, *, max_depth: int = 5) -> Any:
    """剥开 RunnableBinding / with_retry 等包装，找到底层 ChatModel。"""
    current = runnable
    for _ in range(max_depth):
        bound = getattr(current, "bound", None)
        if bound is not None and bound is not current:
            current = bound
            continue
        break
    return current


def _describe_runnable(runnable: Any) -> str:
    """人类可读的模型描述，如 ChatZhipuAI(glm-4-flash)。"""
    target = _unwrap_runnable(runnable)
    cls = type(target).__name__
    model_id = (
        getattr(target, "model_name", None)
        or getattr(target, "model", None)
        or getattr(target, "model_id", None)
    )
    temp = getattr(target, "temperature", None)
    base = getattr(target, "openai_api_base", None)
    if not base:
        client = getattr(target, "client", None)
        client_base = getattr(client, "base_url", None) if client else None
        if client_base and str(client_base).startswith("http"):
            base = str(client_base)
    if base and not str(base).startswith("http"):
        base = None
    parts = [cls]
    if model_id:
        parts.append(f"model={model_id}")
    if base:
        parts.append(f"base={base}")
    if temp is not None:
        parts.append(f"temp={temp}")
    return f"{parts[0]}({', '.join(parts[1:])})" if len(parts) > 1 else cls


def _summarize_input(input: Any) -> str:
    """摘要 invoke 入参（消息数、最后一条用户内容预览）。"""
    if isinstance(input, list):
        messages = input
    elif isinstance(input, dict) and "messages" in input:
        messages = input["messages"]
    else:
        text = str(input)
        return f"input={text[:60]}{'...' if len(text) > 60 else ''}"

    human_preview = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            human_preview = content[:50] + ("..." if len(content) > 50 else "")
            break

    tool_msgs = sum(1 for m in messages if isinstance(m, ToolMessage))
    return f"messages={len(messages)}, tool_msgs={tool_msgs}, last_user={human_preview!r}"


def _summarize_output(output: Any) -> str:
    """摘要 invoke 返回值。"""
    msg = output
    if isinstance(output, ChatResult) and output.generations:
        msg = output.generations[0].message
    if isinstance(msg, AIMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        preview = content[:40] + ("..." if len(content) > 40 else "")
        tc = len(msg.tool_calls) if msg.tool_calls else 0
        usage = (msg.response_metadata or {}).get("token_usage", {})
        tokens = usage.get("total_tokens")
        token_part = f", tokens={tokens}" if tokens else ""
        return f"reply_len={len(content)}, tool_calls={tc}{token_part}, preview={preview!r}"
    text = str(output)
    return f"output={text[:60]}{'...' if len(text) > 60 else ''}"


def _format_error(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{type(exc).__name__} status={exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


def check_api_key(*, require_zhipu: bool = False) -> str:
    """
    检查 API Key 是否就绪。

    Args:
        require_zhipu: True 时必须有智谱 Key（Embedding / 仅 Zhipu 场景）
    """
    zhipu = _zhipu_key()
    openai = _openai_key()

    if require_zhipu and not zhipu:
        raise RuntimeError(
            "❌ 未找到 ZHIPUAI_API_KEY\n"
            "   Embedding 等能力仅支持智谱，请在 .env 中配置 ZHIPUAI_API_KEY"
        )
    if not zhipu and not openai:
        raise RuntimeError(
            "❌ 未找到 ZHIPUAI_API_KEY 或 OPENAI_API_KEY\n"
            "   请在 .env 中至少配置其中一个"
        )

    if zhipu:
        print(f"✅ Zhipu API Key 已加载，前缀: {zhipu[:8]}...")
    if openai:
        print(f"✅ OpenAI API Key 已加载，前缀: {openai[:8]}...")

    _log(f"Key 状态 | zhipu={'有' if zhipu else '无'} | openai={'有' if openai else '无'}")

    return zhipu or openai  # type: ignore[return-value]


def get_api_key() -> str:
    """
    获取智谱 Embedding 等场景用的官方 API Key。

    若当前是千帆 Token Plan Key，请单独配置 ZHIPUAI_EMBEDDING_API_KEY。
    """
    embed_key = os.getenv("ZHIPUAI_EMBEDDING_API_KEY")
    if embed_key:
        return embed_key
    if _is_qianfan_tokenplan():
        raise RuntimeError(
            "❌ 当前 ZHIPUAI_API_KEY 是千帆 Token Plan Key，不能用于 Embedding/RAG 向量。\n"
            "   请在 .env 额外配置 ZHIPUAI_EMBEDDING_API_KEY=智谱官方Key"
        )
    check_api_key(require_zhipu=True)
    return _zhipu_key()  # type: ignore[return-value]


def _should_fallback(exc: BaseException) -> bool:
    """判断主模型失败后是否应切换备用模型。"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (401, 403, 429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, ConnectionError))


def _switch_log(
    exc: BaseException,
    *,
    primary_label: str,
    fallback_label: str,
    primary_desc: str,
    fallback_desc: str,
) -> None:
    _log(
        f"fallback | {primary_label} → {fallback_label} | "
        f"原因={_format_error(exc)} | 备用模型={fallback_desc}"
    )
    if not _log_enabled():
        print(
            f"  ⚠️  {primary_label} 调用失败 ({type(exc).__name__})，"
            f"切换到 {fallback_label}..."
        )


def _invoke_with_logs(
    runnable: Any,
    input: Any,
    config: Any,
    kwargs: dict[str, Any],
    *,
    provider_label: str,
) -> Any:
    """带开始/成功/失败日志的 invoke 封装。"""
    desc = _describe_runnable(runnable)
    input_summary = _summarize_input(input)
    _log(f"invoke → | 提供商={provider_label} | {desc} | {input_summary}")

    t0 = time.perf_counter()
    try:
        result = runnable.invoke(input, config, **kwargs)
        elapsed = time.perf_counter() - t0
        _log(
            f"invoke ← | 状态=成功 | 提供商={provider_label} | "
            f"耗时={elapsed:.2f}s | {_summarize_output(result)}"
        )
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _log(
            f"invoke ← | 状态=失败 | 提供商={provider_label} | "
            f"耗时={elapsed:.2f}s | {_format_error(exc)}"
        )
        raise


async def _ainvoke_with_logs(
    runnable: Any,
    input: Any,
    config: Any,
    kwargs: dict[str, Any],
    *,
    provider_label: str,
) -> Any:
    desc = _describe_runnable(runnable)
    input_summary = _summarize_input(input)
    _log(f"ainvoke → | 提供商={provider_label} | {desc} | {input_summary}")

    t0 = time.perf_counter()
    try:
        result = await runnable.ainvoke(input, config, **kwargs)
        elapsed = time.perf_counter() - t0
        _log(
            f"ainvoke ← | 状态=成功 | 提供商={provider_label} | "
            f"耗时={elapsed:.2f}s | {_summarize_output(result)}"
        )
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _log(
            f"ainvoke ← | 状态=失败 | 提供商={provider_label} | "
            f"耗时={elapsed:.2f}s | {_format_error(exc)}"
        )
        raise


class LoggedFallbackRunnable(RunnableSerializable[Any, Any]):
    """
    Runnable 层 fallback 包装。

    bind_tools() 返回的是 _ChatModelBinding（不是 BaseChatModel），
    不能再包进 ProviderFallbackChatModel，因此在 invoke 层做降级。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: Any
    fallback: Any
    primary_label: str = "Zhipu"
    fallback_label: str = "OpenAI"

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return _invoke_with_logs(
                self.primary,
                input,
                config,
                kwargs,
                provider_label=self.primary_label,
            )
        except Exception as exc:
            if not _should_fallback(exc):
                raise
            _switch_log(
                exc,
                primary_label=self.primary_label,
                fallback_label=self.fallback_label,
                primary_desc=_describe_runnable(self.primary),
                fallback_desc=_describe_runnable(self.fallback),
            )
            return _invoke_with_logs(
                self.fallback,
                input,
                config,
                kwargs,
                provider_label=self.fallback_label,
            )

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return await _ainvoke_with_logs(
                self.primary,
                input,
                config,
                kwargs,
                provider_label=self.primary_label,
            )
        except Exception as exc:
            if not _should_fallback(exc):
                raise
            _switch_log(
                exc,
                primary_label=self.primary_label,
                fallback_label=self.fallback_label,
                primary_desc=_describe_runnable(self.primary),
                fallback_desc=_describe_runnable(self.fallback),
            )
            return await _ainvoke_with_logs(
                self.fallback,
                input,
                config,
                kwargs,
                provider_label=self.fallback_label,
            )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "LoggedFallbackRunnable":
        tool_count = len(tools) if hasattr(tools, "__len__") else "?"
        _log(
            f"bind_tools | 工具数={tool_count} | "
            f"主={_describe_runnable(self.primary)} | "
            f"备={_describe_runnable(self.fallback)}"
        )
        return LoggedFallbackRunnable(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs),
            primary_label=self.primary_label,
            fallback_label=self.fallback_label,
        )

    def with_retry(self, **kwargs: Any) -> "LoggedFallbackRunnable":
        max_attempts = kwargs.get("stop_after_attempt", 3)
        _log(f"with_retry | max_attempts={max_attempts} | 主备模型均启用重试")
        return LoggedFallbackRunnable(
            primary=self.primary.with_retry(**kwargs),
            fallback=self.fallback.with_retry(**kwargs),
            primary_label=self.primary_label,
            fallback_label=self.fallback_label,
        )


class ProviderFallbackChatModel(BaseChatModel):
    """
    双提供商 ChatModel：主模型失败时自动切换备用，并打印切换日志。

    支持 bind_tools / invoke / stream 等常用路径的 fallback 委托。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: Any
    fallback: Any | None = None
    primary_label: str = "Zhipu"
    fallback_label: str = "OpenAI"

    def _generate_with_provider_logs(
        self,
        messages: list[BaseMessage],
        *,
        provider_label: str,
        runnable: Any,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        input_summary = _summarize_input(messages)
        desc = _describe_runnable(runnable)
        _log(f"_generate → | 提供商={provider_label} | {desc} | {input_summary}")

        t0 = time.perf_counter()
        try:
            result = runnable._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            elapsed = time.perf_counter() - t0
            _log(
                f"_generate ← | 状态=成功 | 提供商={provider_label} | "
                f"耗时={elapsed:.2f}s | {_summarize_output(result)}"
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            _log(
                f"_generate ← | 状态=失败 | 提供商={provider_label} | "
                f"耗时={elapsed:.2f}s | {_format_error(exc)}"
            )
            raise

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return self._generate_with_provider_logs(
                messages,
                provider_label=self.primary_label,
                runnable=self.primary,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback is None or not _should_fallback(exc):
                raise
            _switch_log(
                exc,
                primary_label=self.primary_label,
                fallback_label=self.fallback_label,
                primary_desc=_describe_runnable(self.primary),
                fallback_desc=_describe_runnable(self.fallback),
            )
            return self._generate_with_provider_logs(
                messages,
                provider_label=self.fallback_label,
                runnable=self.fallback,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    async def _agenerate_with_provider_logs(
        self,
        messages: list[BaseMessage],
        *,
        provider_label: str,
        runnable: Any,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        input_summary = _summarize_input(messages)
        desc = _describe_runnable(runnable)
        _log(f"_agenerate → | 提供商={provider_label} | {desc} | {input_summary}")

        t0 = time.perf_counter()
        try:
            result = await runnable._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            elapsed = time.perf_counter() - t0
            _log(
                f"_agenerate ← | 状态=成功 | 提供商={provider_label} | "
                f"耗时={elapsed:.2f}s | {_summarize_output(result)}"
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            _log(
                f"_agenerate ← | 状态=失败 | 提供商={provider_label} | "
                f"耗时={elapsed:.2f}s | {_format_error(exc)}"
            )
            raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await self._agenerate_with_provider_logs(
                messages,
                provider_label=self.primary_label,
                runnable=self.primary,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback is None or not _should_fallback(exc):
                raise
            _switch_log(
                exc,
                primary_label=self.primary_label,
                fallback_label=self.fallback_label,
                primary_desc=_describe_runnable(self.primary),
                fallback_desc=_describe_runnable(self.fallback),
            )
            return await self._agenerate_with_provider_logs(
                messages,
                provider_label=self.fallback_label,
                runnable=self.fallback,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        tool_count = len(tools) if hasattr(tools, "__len__") else "?"
        _log(f"bind_tools | 工具数={tool_count} | 降级层=Runnable（LoggedFallbackRunnable）")
        bound_primary = self.primary.bind_tools(tools, **kwargs)
        if self.fallback is None:
            return bound_primary
        bound_fallback = self.fallback.bind_tools(tools, **kwargs)
        return LoggedFallbackRunnable(
            primary=bound_primary,
            fallback=bound_fallback,
            primary_label=self.primary_label,
            fallback_label=self.fallback_label,
        )

    @property
    def _llm_type(self) -> str:
        return "provider-fallback"


def _create_qianfan_tokenplan_model(temperature: Number, model: str | None) -> BaseChatModel:
    """千帆 Token Plan 个人版 —— OpenAI 兼容协议 + bce-v3 Key。"""
    from langchain_openai import ChatOpenAI

    base_url = _normalize_openai_base_url(
        _zhipu_api_base() or QIANFAN_TOKENPLAN_BASE
    )
    model_name = _default_primary_model(model) or TOKENPLAN_DEFAULT_MODEL
    _log(f"create | 千帆TokenPlan | base_url={base_url} | model={model_name}")
    return ChatOpenAI(
        api_key=_zhipu_key(),
        base_url=base_url,
        model=model_name,
        temperature=temperature,
    )


def _create_zhipu_model(temperature: Number, model: str | None) -> BaseChatModel:
    if _is_qianfan_tokenplan():
        return _create_qianfan_tokenplan_model(temperature, model)

    from langchain_community.chat_models import ChatZhipuAI

    kwargs: dict = {"api_key": _zhipu_key(), "temperature": temperature}
    model_name = _default_primary_model(model)
    if model_name:
        kwargs["model"] = model_name
    api_base = _zhipu_api_base()
    if api_base:
        kwargs["api_base"] = _normalize_openai_base_url(api_base)
    return ChatZhipuAI(**kwargs)


def _create_openai_model(temperature: Number, model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "api_key": _openai_key(),
        "temperature": temperature,
        "model": model,
    }
    base = _openai_api_base()
    if base:
        kwargs["base_url"] = _normalize_openai_base_url(base)
    return ChatOpenAI(**kwargs)


def get_model(
    temperature: Number = 0,
    *,
    zhipu_model: str | None = None,
    openai_model: str | None = None,
) -> Any:
    """
    创建 Chat 模型实例。

    - 仅有 Zhipu Key  → ChatZhipuAI
    - 仅有 OpenAI Key → ChatOpenAI
    - 两者都有        → ProviderFallbackChatModel（Zhipu 失败 → OpenAI）

    Args:
        temperature: 采样温度，RAG/Agent 建议 0~0.3
        zhipu_model: 智谱模型名，默认 glm-4-flash
        openai_model: OpenAI 模型名，默认读 OPENAI_MODEL 或 gpt-4o-mini
    """
    global _fallback_notice_printed

    zhipu = _zhipu_key()
    openai = _openai_key()
    if not zhipu and not openai:
        raise RuntimeError("需要 ZHIPUAI_API_KEY 或 OPENAI_API_KEY，请在 .env 中配置")

    openai_model_name = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    primary_model_name = _default_primary_model(zhipu_model)
    primary_label = _primary_provider_label()

    if zhipu and not openai:
        model = _create_zhipu_model(temperature, zhipu_model)
        _log(
            f"get_model | 策略=仅{primary_label} | {_describe_runnable(model)} | temp={temperature}"
        )
        return model

    if openai and not zhipu:
        if not _fallback_notice_printed:
            print("ℹ️  未配置主模型 Key，直接使用 OpenAI")
            _fallback_notice_printed = True
        model = _create_openai_model(temperature, openai_model_name)
        _log(
            f"get_model | 策略=仅OpenAI | {_describe_runnable(model)} | temp={temperature}"
        )
        return model

    if not _fallback_notice_printed:
        print(f"ℹ️  已启用 {primary_label} 主模型 + OpenAI 备用（主模型失败时自动切换）")
        _fallback_notice_printed = True

    primary = _create_zhipu_model(temperature, zhipu_model)
    fallback = _create_openai_model(temperature, openai_model_name)
    _log(
        f"get_model | 策略={primary_label}主+OpenAI备 | "
        f"主={_describe_runnable(primary)} | 备={_describe_runnable(fallback)} | "
        f"temp={temperature}"
    )
    return ProviderFallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_label=primary_label,
    )
