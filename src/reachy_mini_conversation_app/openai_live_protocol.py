"""App-independent OpenAI GPT-Live-1 WebSocket protocol helpers.

Copy this module into another Reachy conversation app (for example Lucface/niero)
and wrap it with that app's audio loop and tool runner. It has no Reachy Mini
imports. The published openai SDK still has no ``client.live``, so this speaks
the documented Live JSON events over ``wss://api.openai.com/v1/live/sessions``.
"""

import json
import uuid
import base64
import logging
from typing import Any, Mapping, Sequence

from websockets.asyncio.client import connect


logger = logging.getLogger(__name__)

LIVE_MODEL = "gpt-live-1"
LIVE_WS_URL = "wss://api.openai.com/v1/live/sessions"
LIVE_SAMPLE_RATE = 16000
DEFAULT_DELEGATION_MODEL = "gpt-5.6-luna"
DEFAULT_VOICE = "marin"

# Documented GPT-Live voices (marin is the API default) plus the Live-only catalog.
LIVE_AVAILABLE_VOICES: list[str] = [
    "marin",
    "cedar",
    "quartz",
    "ripple",
    "vesper",
    "willow",
    "stone",
    "gleam",
    "meridian",
    "bossa",
    "tempo",
    "beacon",
    "delta",
    "cinder",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
]


def normalize_live_voice(voice: str | None, *, fallback: str | None = None) -> str | None:
    """Return a documented Live voice, or fallback when the name is unknown."""
    voice_value = (voice or "").strip()
    if not voice_value:
        return fallback
    voice_by_lowercase = {candidate.lower(): candidate for candidate in LIVE_AVAILABLE_VOICES}
    return voice_by_lowercase.get(voice_value.lower(), fallback)


def to_responses_function_tools(tool_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert function-tool specs to Responses ``delegation.responses.tools`` entries."""
    return [
        {
            "type": "function",
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["parameters"],
        }
        for spec in tool_specs
    ]


def build_live_session_config(
    *,
    conversation_instructions: str,
    delegation_instructions: str,
    voice: str,
    tools: Sequence[Mapping[str, Any]],
    delegation_model: str = DEFAULT_DELEGATION_MODEL,
    sample_rate: int = LIVE_SAMPLE_RATE,
) -> dict[str, Any]:
    """Return the ``session`` object for Live ``session.start``."""
    return {
        "model": LIVE_MODEL,
        "instructions": conversation_instructions,
        "audio": {
            "format": {"type": "audio/pcm", "rate": sample_rate},
            "output": {"voice": voice},
        },
        "delegation": {
            "type": "responses",
            "responses": {
                "model": delegation_model,
                "instructions": delegation_instructions,
                "tools": list(tools),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        },
    }


def encode_live_client_event(event: dict[str, Any]) -> str:
    """Serialize one Live client event, assigning ``event_id`` when missing."""
    if "event_id" not in event:
        event["event_id"] = f"event_{uuid.uuid4().hex[:12]}"
    return json.dumps(event)


def parse_live_server_message(raw: str | bytes) -> dict[str, Any] | None:
    """Parse one Live websocket payload into a JSON object, or None if invalid."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid GPT-Live JSON")
        return None
    return event if isinstance(event, dict) else None


def extract_delegated_function_call(envelope: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """Read ``(name, arguments, call_id)`` from a nested ``response.event`` envelope."""
    nested = envelope.get("event")
    if not isinstance(nested, dict) or nested.get("type") != "response.output_item.done":
        return None
    item = nested.get("item")
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None
    tool_name = item.get("name")
    args_json_str = item.get("arguments")
    call_id = item.get("call_id") or str(uuid.uuid4())
    if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
        return None
    return tool_name, args_json_str, str(call_id)


def session_start_event(session: Mapping[str, Any]) -> dict[str, Any]:
    """Build ``session.start``."""
    return {"type": "session.start", "session": dict(session)}


def input_audio_append_event(pcm_bytes: bytes) -> dict[str, Any]:
    """Build ``session.input_audio.append`` from raw PCM16 bytes."""
    return {
        "type": "session.input_audio.append",
        "audio": base64.b64encode(pcm_bytes).decode("utf-8"),
    }


def instructions_append_event(content: str) -> dict[str, Any]:
    """Build a session-wide ``session.instructions.append``."""
    return {
        "type": "session.instructions.append",
        "delegation_id": None,
        "content": content,
    }


def function_call_output_event(call_id: str, output: Mapping[str, Any] | str) -> dict[str, Any]:
    """Build ``response.item.create`` for a function result."""
    payload = output if isinstance(output, str) else json.dumps(output)
    return {
        "type": "response.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": payload,
        },
    }


def input_image_item_event(jpeg_b64: str) -> dict[str, Any]:
    """Build ``response.item.create`` that queues a JPEG for the Responses backend."""
    return {
        "type": "response.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{jpeg_b64}",
                }
            ],
        },
    }


def response_create_event() -> dict[str, Any]:
    """Build ``response.create`` to start or continue delegated Responses work."""
    return {"type": "response.create"}


def session_close_event() -> dict[str, Any]:
    """Build ``session.close``."""
    return {"type": "session.close"}


def open_live_connection(api_key: str, *, user_agent: str = "reachy-mini-conversation-app") -> Any:
    """Return an async context manager for the Live websocket."""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the GPT-Live-1 backend")
    return connect(
        LIVE_WS_URL,
        additional_headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": user_agent,
        },
    )
