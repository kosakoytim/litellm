"""
DeepSeek V4 reasoning_content multi-turn hook for litellm proxy.

Problem: DeepSeek V4 (flash/pro) requires reasoning_content on every assistant
message when thinking mode is active. Frontends like n8n drop this field when
storing conversation history, so by turn 2 there is nothing to preserve.

Fix: capture real reasoning_content from each response, keyed by the first
tool_call_id of that assistant turn (the one field n8n does preserve).
On the next pre-call, re-inject it before the request goes to DeepSeek.

Wire up in your proxy config's litellm_settings:
  callbacks: ["litellm.proxy.custom_hooks.deepseek_reasoning_hook.deepseek_reasoning_hook"]

Reference: https://github.com/BerriAI/litellm/issues/26395
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, List, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

logger = logging.getLogger(__name__)

_DEEPSEEK_V4_PREFIXES = ("deepseek-v4", "deepseek/deepseek-v4")


def _is_deepseek_v4(model: str) -> bool:
    return any(model.startswith(p) for p in _DEEPSEEK_V4_PREFIXES)


def _first_tool_call_id(msg: dict) -> Optional[str]:
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None
    first = tcs[0]
    return first.get("id") if isinstance(first, dict) else getattr(first, "id", None)


class _ReasoningCache:
    """Process-local LRU: first tool_call_id of a turn -> reasoning_content."""

    def __init__(self, maxsize: int = 512) -> None:
        self._d: OrderedDict[str, str] = OrderedDict()
        self._max = maxsize

    def put(self, key: str, value: str) -> None:
        if not key or not value:
            return
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def get(self, key: str) -> Optional[str]:
        if not key or key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]


_cache = _ReasoningCache()


class DeepSeekReasoningHook(CustomLogger):
    """
    Proxy CustomLogger that fixes DeepSeek V4 multi-turn reasoning_content.

    - async_pre_call_hook: re-injects reasoning_content into outgoing messages
    - async_log_success_event: captures reasoning_content from responses
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        model: str = data.get("model", "")
        if not _is_deepseek_v4(model):
            return data

        messages: List[Any] = data.get("messages") or []
        if not messages:
            return data

        # First pass: restore cached reasoning_content by tool_call_id
        has_thinking = False
        for msg in messages:
            if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
                continue
            if msg.get("reasoning_content") is not None:
                has_thinking = True
                continue
            tc_id = _first_tool_call_id(msg)
            if tc_id:
                cached = _cache.get(tc_id)
                if cached is not None:
                    msg["reasoning_content"] = cached
                    has_thinking = True
                    logger.debug(
                        f"[DeepSeekReasoningHook] Restored reasoning_content "
                        f"for tool_call_id={tc_id} ({len(cached)} chars)"
                    )

        # Second pass: backfill empty string on remaining assistant messages
        # so DeepSeek doesn't 400 on turns that had no tool calls
        if has_thinking:
            for msg in messages:
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and msg.get("reasoning_content") is None
                ):
                    msg["reasoning_content"] = ""

        return data

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        try:
            model: str = kwargs.get("model", "")
            if not _is_deepseek_v4(model):
                return

            choices = getattr(response_obj, "choices", None)
            if not choices:
                return

            message = getattr(choices[0], "message", None)
            if message is None:
                return

            reasoning: str = getattr(message, "reasoning_content", None) or ""
            if not reasoning:
                return

            # Key by the first tool_call_id on this assistant message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return

            first = tool_calls[0]
            tc_id = (
                first.get("id")
                if isinstance(first, dict)
                else getattr(first, "id", None)
            )
            if tc_id:
                _cache.put(tc_id, reasoning)
                logger.debug(
                    f"[DeepSeekReasoningHook] Cached reasoning_content "
                    f"for tool_call_id={tc_id} ({len(reasoning)} chars)"
                )
        except Exception as e:
            logger.debug(f"[DeepSeekReasoningHook] capture skipped: {e}")


deepseek_reasoning_hook = DeepSeekReasoningHook()
