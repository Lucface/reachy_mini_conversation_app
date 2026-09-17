"""Tests for the OpenAI GPT-Live-1 conversation backend."""

import json
import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.openai_gpt_live as live_mod
from reachy_mini_conversation_app.config import OPENAI_LIVE_MODEL, OPENAI_LIVE_DEFAULT_VOICE
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.openai_gpt_live import OpenAIGPTLiveHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _plain_handler() -> OpenAIGPTLiveHandler:
    return OpenAIGPTLiveHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


class _FakeLiveSocket:
    """Minimal Live websocket that records sent events and yields server events."""

    def __init__(self, events: tuple[dict[str, Any], ...] = ()) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events = list(events)
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> "_FakeLiveSocket":
        return self

    async def __anext__(self) -> str:
        if not self._events:
            raise StopAsyncIteration
        return json.dumps(self._events.pop(0))


class _FakeLiveConnect:
    def __init__(self, socket: _FakeLiveSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> _FakeLiveSocket:
        return self.socket

    async def __aexit__(self, *_args: Any) -> bool:
        return False


def _session_handler(
    monkeypatch: Any, events: tuple[dict[str, Any], ...]
) -> tuple[OpenAIGPTLiveHandler, _FakeLiveSocket]:
    monkeypatch.setattr(live_mod, "get_live_conversation_instructions", lambda _instance_path=None: "talk")
    monkeypatch.setattr(live_mod, "get_live_delegation_instructions", lambda _instance_path=None: "tools")
    monkeypatch.setattr(live_mod, "get_session_voice", lambda default=OPENAI_LIVE_DEFAULT_VOICE: default)
    monkeypatch.setattr(live_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(live_mod, "get_tool_specs", lambda: [])
    handler = _plain_handler()
    socket = _FakeLiveSocket(events)
    monkeypatch.setattr(handler, "_live_connect", lambda: _FakeLiveConnect(socket))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    return handler, socket


def _drain(handler: OpenAIGPTLiveHandler) -> list[Any]:
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def test_session_config_uses_gpt_live_and_delegation() -> None:
    """session.start should pin gpt-live-1 and register robot tools on Responses."""
    handler = _plain_handler()
    session = handler._build_session_config(
        [
            {
                "type": "function",
                "name": "move_head",
                "description": "Move the head",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    )

    assert session["model"] == OPENAI_LIVE_MODEL
    assert session["audio"]["format"] == {"type": "audio/pcm", "rate": 16000}
    assert session["audio"]["output"]["voice"] == OPENAI_LIVE_DEFAULT_VOICE
    assert session["delegation"]["type"] == "responses"
    assert session["delegation"]["responses"]["tools"][0]["name"] == "move_head"
    assert session["delegation"]["responses"]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_run_session_sends_start_and_plays_audio(monkeypatch: Any) -> None:
    """A mocked Live session should start, then enqueue decoded output audio."""
    pcm = np.array([1, 2, 3, 4], dtype=np.int16)
    events = (
        {"type": "session.started", "session": {"id": "sess_1"}},
        {"type": "session.output_audio.delta", "delta": base64.b64encode(pcm.tobytes()).decode("ascii")},
    )
    handler, socket = _session_handler(monkeypatch, events)

    await handler._run_live_session()

    assert socket.sent[0]["type"] == "session.start"
    assert socket.sent[0]["session"]["model"] == OPENAI_LIVE_MODEL
    items = _drain(handler)
    audio = next(item for item in items if isinstance(item, tuple))
    assert audio[0] == 16000
    assert audio[1].tolist() == [[1, 2, 3, 4]]
    handler.deps.movement_manager.set_speaking.assert_called_with(True)


@pytest.mark.asyncio
async def test_nested_function_call_starts_background_tool(monkeypatch: Any) -> None:
    """Delegated Responses function calls should run through the shared tool manager."""
    events = (
        {"type": "session.started", "session": {"id": "sess_1"}},
        {
            "type": "response.event",
            "delegation_id": "item_1",
            "event": {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "dance",
                    "arguments": '{"dance_name":"happy"}',
                    "call_id": "call_dance",
                },
            },
        },
    )
    handler, _socket = _session_handler(monkeypatch, events)
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="tool-1"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_live_session()

    start_tool.assert_awaited_once()
    assert start_tool.await_args.kwargs["call_id"] == "call_dance"
    assert start_tool.await_args.kwargs["tool_call_routine"].tool_name == "dance"
    messages = [item.args[0] for item in _drain(handler) if isinstance(item, AdditionalOutputs)]
    assert any("Used tool dance" in msg["content"] for msg in messages)


@pytest.mark.asyncio
async def test_receive_appends_input_audio_after_session_start(monkeypatch: Any) -> None:
    """Microphone frames should be ignored until session.started, then appended."""
    handler, socket = _session_handler(monkeypatch, ())
    frame = (16000, np.array([9, 8], dtype=np.int16))

    await handler.receive(frame)
    assert socket.sent == []

    handler.connection = socket
    handler._session_started = True
    await handler.receive(frame)

    append = socket.sent[-1]
    assert append["type"] == "session.input_audio.append"
    assert base64.b64decode(append["audio"]) == np.array([9, 8], dtype=np.int16).tobytes()
