"""Tests for the copyable GPT-Live-1 protocol helpers."""

import json
import base64

from reachy_mini_conversation_app.openai_live_protocol import (
    LIVE_MODEL,
    DEFAULT_VOICE,
    encode_live_client_event,
    input_audio_append_event,
    build_live_session_config,
    parse_live_server_message,
    function_call_output_event,
    to_responses_function_tools,
    extract_delegated_function_call,
)


def test_to_responses_function_tools_preserves_schema() -> None:
    """Function-tool specs become Responses delegation tools."""
    spec = {
        "type": "function",
        "name": "dance",
        "description": "Queue a dance",
        "parameters": {"type": "object", "properties": {}},
    }

    assert to_responses_function_tools([spec]) == [
        {
            "type": "function",
            "name": "dance",
            "description": "Queue a dance",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_build_live_session_config_pins_model_and_delegation() -> None:
    """session.start payload should pin gpt-live-1 and Responses tools."""
    session = build_live_session_config(
        conversation_instructions="talk",
        delegation_instructions="tools",
        voice=DEFAULT_VOICE,
        tools=to_responses_function_tools(
            [
                {
                    "name": "move_head",
                    "description": "Move the head",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        ),
    )

    assert session["model"] == LIVE_MODEL
    assert session["audio"]["format"] == {"type": "audio/pcm", "rate": 16000}
    assert session["audio"]["output"]["voice"] == DEFAULT_VOICE
    assert session["delegation"]["type"] == "responses"
    assert session["delegation"]["responses"]["tools"][0]["name"] == "move_head"


def test_extract_delegated_function_call_reads_nested_item() -> None:
    """Responses function calls arrive nested inside response.event."""
    assert extract_delegated_function_call(
        {
            "type": "response.event",
            "event": {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "dance",
                    "arguments": '{"dance_name":"happy"}',
                    "call_id": "call_dance",
                },
            },
        }
    ) == ("dance", '{"dance_name":"happy"}', "call_dance")
    assert extract_delegated_function_call({"type": "response.event", "event": {"type": "response.completed"}}) is None


def test_encode_and_parse_round_trip_audio_append() -> None:
    """Client audio events should be JSON with assigned event_id and decodable PCM."""
    encoded = encode_live_client_event(input_audio_append_event(b"\x01\x00\x02\x00"))
    event = parse_live_server_message(encoded)
    assert event is not None
    assert event["type"] == "session.input_audio.append"
    assert event["event_id"].startswith("event_")
    assert base64.b64decode(event["audio"]) == b"\x01\x00\x02\x00"


def test_function_call_output_event_serializes_json_payload() -> None:
    """Tool results go back as Responses function_call_output items."""
    event = function_call_output_event("call_1", {"status": "ok"})
    assert event["type"] == "response.item.create"
    assert event["item"]["call_id"] == "call_1"
    assert json.loads(event["item"]["output"]) == {"status": "ok"}
