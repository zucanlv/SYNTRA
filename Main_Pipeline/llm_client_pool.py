"""Infrastructure hooks for reusing OpenAI-compatible HTTP clients.

The pipeline issues many small OpenAI-compatible requests to local SGLang.
qwen-agent's OpenAI adapter constructs a new ``openai.OpenAI`` client inside
every request, which defeats HTTP connection pooling under high concurrency.
This module patches that adapter to reuse clients keyed by base URL/API key.
It does not change prompts, batching, retry behavior, or pipeline semantics.
"""

from __future__ import annotations

import copy
import os
import threading
from typing import Any, Dict, Tuple


_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_CLIENT_LOCK = threading.Lock()
_CLIENTS: Dict[Tuple[Tuple[str, str], ...], Any] = {}


def configure_qwen_agent_openai_client_pool(config=None, logger=None) -> None:
    """Apply YAML config and install the qwen-agent OpenAI client pool hook."""

    config = config or {}
    if not isinstance(config, dict):
        config = {}

    enabled = bool(config.get("enabled", True))
    max_connections = config.get("max_connections", None)
    max_keepalive = config.get("max_keepalive_connections", None)
    force_nonstream = bool(config.get("force_nonstream", True))

    if max_connections is not None:
        os.environ["SYN_OPENAI_MAX_CONNECTIONS"] = str(int(max_connections))
    if max_keepalive is not None:
        os.environ["SYN_OPENAI_MAX_KEEPALIVE"] = str(int(max_keepalive))

    if not enabled:
        if logger:
            logger.info("qwen-agent OpenAI client pool hook disabled by config")
        return

    install_qwen_agent_openai_client_pool(logger)
    if force_nonstream:
        install_qwen_agent_nonstream_hook(logger)


def install_qwen_agent_openai_client_pool(logger=None) -> None:
    """Patch qwen-agent's OpenAI adapter to reuse OpenAI clients."""

    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            if logger:
                logger.info("qwen-agent OpenAI client pool hook already active")
            return

        try:
            import openai
            from qwen_agent.llm import oai as qwen_oai
        except Exception as exc:  # pragma: no cover - optional dependency guard
            if logger:
                logger.warning("OpenAI client pool hook not installed: %s", exc)
            return

        if openai.__version__.startswith("0."):
            if logger:
                logger.info("OpenAI client pool hook skipped for openai<1.0")
            _INSTALLED = True
            return

        original_init = qwen_oai.TextChatAtOAI.__init__

        def _normalize_api_kwargs(api_kwargs: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
            return tuple(sorted((str(k), str(v)) for k, v in api_kwargs.items()))

        def _get_client(api_kwargs: Dict[str, Any]):
            key = _normalize_api_kwargs(api_kwargs)
            with _CLIENT_LOCK:
                client = _CLIENTS.get(key)
                if client is not None:
                    return client

                client_kwargs = dict(api_kwargs)
                try:
                    import httpx

                    max_connections = int(os.getenv("SYN_OPENAI_MAX_CONNECTIONS", "1024"))
                    max_keepalive = int(os.getenv("SYN_OPENAI_MAX_KEEPALIVE", "256"))
                    client_kwargs["http_client"] = httpx.Client(
                        limits=httpx.Limits(
                            max_connections=max_connections,
                            max_keepalive_connections=max_keepalive,
                        )
                    )
                except Exception:
                    pass

                client = openai.OpenAI(**client_kwargs)
                _CLIENTS[key] = client
                return client

        def patched_init(self, cfg=None):
            original_init(self, cfg)
            cfg = cfg or {}

            api_base = cfg.get("api_base") or cfg.get("base_url") or cfg.get("model_server")
            api_base = (api_base or "").strip()
            api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
            api_key = (api_key or "EMPTY").strip()

            api_kwargs: Dict[str, Any] = {}
            if api_base:
                api_kwargs["base_url"] = api_base
            if api_key:
                api_kwargs["api_key"] = api_key

            def _prepare_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
                kwargs = dict(kwargs)
                extra_params = ["top_k", "repetition_penalty"]
                if any(k in kwargs for k in extra_params):
                    kwargs["extra_body"] = copy.deepcopy(kwargs.get("extra_body", {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs["extra_body"][k] = kwargs.pop(k)
                if "request_timeout" in kwargs:
                    kwargs["timeout"] = kwargs.pop("request_timeout")
                return kwargs

            def _chat_complete_create(*args, **kwargs):
                client = _get_client(api_kwargs)
                return client.chat.completions.create(*args, **_prepare_kwargs(kwargs))

            def _complete_create(*args, **kwargs):
                client = _get_client(api_kwargs)
                return client.completions.create(*args, **_prepare_kwargs(kwargs))

            self._chat_complete_create = _chat_complete_create
            self._complete_create = _complete_create

        qwen_oai.TextChatAtOAI.__init__ = patched_init
        _INSTALLED = True
        if logger:
            logger.info(
                "Installed qwen-agent OpenAI client pool hook "
                "(SYN_OPENAI_MAX_CONNECTIONS=%s, SYN_OPENAI_MAX_KEEPALIVE=%s)",
                os.getenv("SYN_OPENAI_MAX_CONNECTIONS", "1024"),
                os.getenv("SYN_OPENAI_MAX_KEEPALIVE", "256"),
            )


def install_qwen_agent_nonstream_hook(logger=None) -> None:
    """Default qwen-agent LLM calls to non-streaming mode.

    The pipeline only consumes the final accumulated response from
    ``Assistant.run``. For high-concurrency local OpenAI-compatible services,
    non-streaming avoids holding many streaming responses open and reduces
    CLOSE-WAIT/connection churn without changing prompt or output parsing.
    """

    try:
        from qwen_agent import agent as qwen_agent_base
    except Exception as exc:  # pragma: no cover - optional dependency guard
        if logger:
            logger.warning("qwen-agent non-stream hook not installed: %s", exc)
        return

    if getattr(qwen_agent_base.Agent._call_llm, "_synthetic_nonstream_hook", False):
        if logger:
            logger.info("qwen-agent non-stream hook already active")
        return

    original_call_llm = qwen_agent_base.Agent._call_llm

    def patched_call_llm(self, messages, functions=None, stream=None, extra_generate_cfg=None):
        if stream is not None:
            output = original_call_llm(
                self,
                messages=messages,
                functions=functions,
                stream=stream,
                extra_generate_cfg=extra_generate_cfg,
            )
            if stream:
                yield from output
            else:
                yield output
            return

        output = self.llm.chat(
            messages=messages,
            functions=functions,
            stream=False,
            extra_generate_cfg=qwen_agent_base.merge_generate_cfgs(
                base_generate_cfg=self.extra_generate_cfg,
                new_generate_cfg=extra_generate_cfg,
            ),
        )
        yield output

    patched_call_llm._synthetic_nonstream_hook = True
    qwen_agent_base.Agent._call_llm = patched_call_llm
    if logger:
        logger.info("Installed qwen-agent non-stream hook for default Assistant.run calls")
