import json
import time
import uuid
import base64
import random
import asyncio
import logging
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from websockets.exceptions import ConnectionClosedError
from websockets.asyncio.client import connect

from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.config import (
    OPENAI_LIVE_MODEL,
    OPENAI_LIVE_WS_URL,
    OPENAI_GPT_LIVE_BACKEND,
    config,
    get_default_voice,
    get_openai_api_key,
    set_custom_profile,
    get_available_voices,
    get_openai_live_delegation_model,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_greeting_prompt,
    get_live_delegation_instructions,
    get_live_conversation_instructions,
)
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.tools.core_tools import (
    ToolSpec,
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

_TRANSCRIPT_FINAL_DELAY_S: Final[float] = 1.2
_SPEAKING_IDLE_S: Final[float] = 1.5
_OPENAI_LIVE_CONNECT_HEADERS: dict[str, str] = {
    "User-Agent": "reachy-mini-conversation-app",
}


def to_responses_function_tools(tool_specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert app tool specs to Responses function tools for Live delegation."""
    return [
        {
            "type": "function",
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["parameters"],
        }
        for spec in tool_specs
    ]


class OpenAIGPTLiveHandler(ConversationHandler):
    """Realtime stream handler for OpenAI GPT-Live-1 over the Live WebSocket API."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ):
        """Initialize the GPT-Live-1 handler."""
        super().__init__()

        self.deps = deps
        self.connection: Any | None = None
        self.output_queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self.instance_path = instance_path
        self._voice_override: str | None = self._normalize_startup_voice(startup_voice)
        self._connected_event: asyncio.Event = asyncio.Event()
        self.tool_manager = BackgroundToolManager()
        self._in_flight_tool_calls: set[str] = set()
        self._tool_batch_needs_response = False
        self._startup_greeting_sent = False
        self._session_started = False
        self._input_transcript_parts: list[str] = []
        self._output_transcript_parts: list[str] = []
        self._input_final_task: asyncio.Task[None] | None = None
        self._output_final_task: asyncio.Task[None] | None = None
        self._last_output_audio_at: float | None = None
        self._speaking_idle_task: asyncio.Task[None] | None = None

    def _normalize_startup_voice(self, voice: str | None) -> str | None:
        """Return a valid persisted startup voice, or None."""
        return self._resolve_backend_voice(voice, source="persisted startup voice")

    def _resolve_backend_voice(
        self,
        voice: str | None,
        *,
        source: str,
        fallback: str | None = None,
    ) -> str | None:
        """Return a GPT-Live voice, optionally falling back when unsupported."""
        available_voices = get_available_voices(OPENAI_GPT_LIVE_BACKEND)
        voice_value = (voice or "").strip()
        if not voice_value:
            return fallback

        voice_by_lowercase = {candidate.lower(): candidate for candidate in available_voices}
        normalized_voice = voice_by_lowercase.get(voice_value.lower())
        if normalized_voice is not None:
            return normalized_voice

        if voice:
            logger.warning(
                "Ignoring unsupported %s %r; expected one of %s",
                source,
                voice,
                available_voices,
            )
        return fallback

    def _build_session_config(self, tool_specs: list[ToolSpec]) -> dict[str, Any]:
        """Return the GPT-Live session.start payload."""
        return {
            "model": OPENAI_LIVE_MODEL,
            "instructions": get_live_conversation_instructions(self.instance_path),
            "audio": {
                "format": {"type": "audio/pcm", "rate": self.SAMPLE_RATE},
                "output": {"voice": self.get_current_voice()},
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": get_openai_live_delegation_model(),
                    "instructions": get_live_delegation_instructions(self.instance_path),
                    "tools": to_responses_function_tools(tool_specs),
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                },
            },
        }

    def _live_connect(self) -> Any:
        """Return an async context manager for the Live websocket."""
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the GPT-Live-1 backend")
        return connect(
            OPENAI_LIVE_WS_URL,
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                **_OPENAI_LIVE_CONNECT_HEADERS,
            },
        )

    def _is_connected(self) -> bool:
        """Return whether the Live websocket is open."""
        return self.connection is not None and self._session_started

    def _idle_behavior_ready(self) -> bool:
        """Hold idle behavior while the model is still producing audio."""
        if self._last_output_audio_at is None:
            return True
        return (time.monotonic() - self._last_output_audio_at) >= _SPEAKING_IDLE_S

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Send one JSON Live client event."""
        if self.connection is None:
            raise RuntimeError("No active GPT-Live session")
        if "event_id" not in event:
            event["event_id"] = f"event_{uuid.uuid4().hex[:12]}"
        await self.connection.send(json.dumps(event))

    async def _cancel_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def change_voice(self, voice: str) -> str:
        """Change the voice by restarting the Live session."""
        default_voice = get_default_voice(OPENAI_GPT_LIVE_BACKEND)
        resolved_voice = (
            self._resolve_backend_voice(voice, source="requested voice", fallback=default_voice) or default_voice
        )
        self._voice_override = resolved_voice
        if self.connection is not None:
            try:
                await self._restart_session()
                return f"Voice changed to {resolved_voice}."
            except Exception as e:
                logger.warning("Failed to restart Live session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        default_voice = get_default_voice(OPENAI_GPT_LIVE_BACKEND)
        voice = self._voice_override or get_session_voice(default=default_voice)
        return self._resolve_backend_voice(voice, source="session voice", fallback=default_voice) or default_voice

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality by restarting the Live session."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            get_live_conversation_instructions(self.instance_path)
            self.get_current_voice()
            core_tools.initialize_tools(force=True)
        except Exception as exc:
            set_custom_profile(previous_profile)
            logger.error("Failed to resolve personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

        if self.connection is not None:
            try:
                await self._restart_session()
                return "Applied personality and restarted realtime session."
            except Exception as exc:
                logger.warning("Failed to restart Live session after apply: %s", exc)
                return "Applied personality. Will take effect on next connection."

        logger.info(
            "Applied personality recorded: %s (no live connection; will apply on next session)",
            profile or "default",
        )
        return "Applied personality. Will take effect on next connection."

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_live_session()
                return
            except ConnectionClosedError as e:
                logger.warning("GPT-Live websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                self.connection = None
                self._session_started = False
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in the background."""
        try:
            if self.connection is not None:
                try:
                    await self._send_event({"type": "session.close"})
                except Exception:
                    pass
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None
                    self._session_started = False

            try:
                self._connected_event.clear()
            except Exception:
                pass
            asyncio.create_task(self._run_live_session(), name="gpt-live-session-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("GPT-Live session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("GPT-Live session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def say(self, text: str) -> None:
        """Ask GPT-Live to speak ``text`` now via a session instruction."""
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self._is_connected():
            raise RuntimeError("say: no active session")
        await self._send_event(
            {
                "type": "session.instructions.append",
                "delegation_id": None,
                "content": f"Speak this to the user now: {text}",
            }
        )
        self._mark_activity("say")

    async def _send_startup_greeting_prompt(self) -> None:
        """Prompt GPT-Live to open the conversation once the session is ready."""
        if self._startup_greeting_sent or not self._is_connected():
            return

        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            self._startup_greeting_sent = True
            return

        try:
            await self._send_event(
                {
                    "type": "session.instructions.append",
                    "delegation_id": None,
                    "content": greeting_prompt,
                }
            )
            self._startup_greeting_sent = True
            self._mark_activity("startup_greeting_prompt")
            logger.info("Queued startup greeting prompt")
        except Exception as e:
            logger.warning("Failed to queue startup greeting prompt: %s", e)

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Return a finished tool result to the Responses backend."""
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed with error: %s",
                completed_tool.tool_name,
                completed_tool.id,
                completed_tool.error,
            )
            tool_result: dict[str, Any] = {"error": completed_tool.error}
            tool_result_for_model: dict[str, Any] = tool_result
        elif completed_tool.result is not None:
            tool_result = completed_tool.result
            tool_result_for_model = (
                self._sanitize_tool_result_for_model(completed_tool.tool_name, tool_result)
                if isinstance(tool_result, dict)
                else {"result": tool_result}
            )
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                completed_tool.tool_name,
                completed_tool.id,
            )
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        if not self.connection:
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                completed_tool.tool_name,
                completed_tool.id,
            )
            return

        try:
            send_result_to_model = not completed_tool.is_idle_tool_call
            if send_result_to_model:
                self._mark_activity("tool_result_ready")
            model_result_submitted = False
            if send_result_to_model and isinstance(completed_tool.id, str):
                await self._send_event(
                    {
                        "type": "response.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": completed_tool.id,
                            "output": json.dumps(tool_result_for_model),
                        },
                    }
                )
                model_result_submitted = True

            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": json.dumps(tool_result_for_model)})
            )

            if model_result_submitted and completed_tool.tool_name == "camera" and "b64_im" in tool_result:
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                await self._send_event(
                    {
                        "type": "response.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{b64_im}",
                                }
                            ],
                        },
                    }
                )
                logger.info("Queued camera image for GPT-Live Responses delegation")

            if isinstance(completed_tool.id, str):
                self._in_flight_tool_calls.discard(completed_tool.id)

            tool = core_tools.get_tools().get(completed_tool.tool_name)
            if model_result_submitted and (completed_tool.error is not None or tool is None or tool.needs_response):
                self._tool_batch_needs_response = True

            if self._tool_batch_needs_response and not self._in_flight_tool_calls:
                self._tool_batch_needs_response = False
                await self._send_event({"type": "response.create"})
        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._session_started = False

    async def _emit_final_input_transcript(self) -> None:
        """Emit a grouped user transcript after Live deltas settle."""
        try:
            await asyncio.sleep(_TRANSCRIPT_FINAL_DELAY_S)
            transcript = "".join(self._input_transcript_parts).strip()
            self._input_transcript_parts = []
            if not transcript:
                return
            self.deps.movement_manager.set_listening(False)
            self._in_flight_tool_calls.clear()
            self._tool_batch_needs_response = False
            self._mark_activity("user_transcription_completed")
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
            self._emit_transcript("user", transcript, True)
        except asyncio.CancelledError:
            raise

    async def _emit_final_output_transcript(self) -> None:
        """Emit a grouped assistant transcript after Live deltas settle."""
        try:
            await asyncio.sleep(_TRANSCRIPT_FINAL_DELAY_S)
            transcript = "".join(self._output_transcript_parts).strip()
            self._output_transcript_parts = []
            if not transcript:
                return
            self._mark_activity("assistant_transcript_done")
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": transcript}))
            self._emit_transcript("assistant", transcript, True)
        except asyncio.CancelledError:
            raise

    async def _clear_speaking_after_idle(self) -> None:
        """Clear the speaking flag after output audio stops arriving."""
        try:
            await asyncio.sleep(_SPEAKING_IDLE_S)
            self.deps.movement_manager.set_speaking(False)
        except asyncio.CancelledError:
            raise

    async def _start_tool_call(self, call_id: str, tool_name: str, args_json_str: str) -> None:
        """Run one delegated function call through the shared tool manager."""
        logger.info(
            "Tool call received — tool_name=%r, call_id=%s, args=%s",
            tool_name,
            call_id,
            args_json_str,
        )
        self._mark_activity("tool_call_received")
        self._in_flight_tool_calls.add(call_id)
        background_tool = await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(
                tool_name=tool_name,
                args_json_str=args_json_str,
                deps=self.deps,
            ),
            is_idle_tool_call=False,
        )
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": (
                        f"🛠️ Used tool {tool_name} with args {args_json_str}. "
                        f"The tool is now running. Tool ID: {background_tool.tool_id}"
                    ),
                }
            )
        )

    async def _handle_nested_response_event(self, envelope: dict[str, Any]) -> None:
        """Dispatch a Responses event nested inside response.event."""
        nested = envelope.get("event")
        if not isinstance(nested, dict):
            return
        nested_type = nested.get("type")
        if nested_type == "response.output_item.done":
            item = nested.get("item")
            if not isinstance(item, dict) or item.get("type") != "function_call":
                return
            tool_name = item.get("name")
            args_json_str = item.get("arguments")
            call_id = item.get("call_id") or str(uuid.uuid4())
            if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                logger.error(
                    "Invalid delegated tool call: tool_name=%s args=%s call_id=%s",
                    tool_name,
                    args_json_str,
                    call_id,
                )
                return
            await self._start_tool_call(str(call_id), tool_name, args_json_str)
        elif nested_type == "error":
            err = nested.get("error") if isinstance(nested.get("error"), dict) else nested
            msg = err.get("message") if isinstance(err, dict) else str(err)
            logger.error("Delegated Responses error: %s (raw=%s)", msg, nested)

    async def _handle_live_event(self, event: dict[str, Any]) -> None:
        """Handle one top-level GPT-Live server event."""
        event_type = event.get("type")
        logger.debug("GPT-Live event: %s", event_type)

        if event_type == "session.started":
            self._session_started = True
            self._connected_event.set()
            session_payload = event.get("session")
            session_id = session_payload.get("id") if isinstance(session_payload, dict) else None
            logger.info("GPT-Live session started %s", session_id or "<unknown>")
            await self._send_startup_greeting_prompt()

        elif event_type == "session.input_transcript.delta":
            delta = event.get("delta") or ""
            if not isinstance(delta, str) or not delta:
                return
            if not self._input_transcript_parts:
                self._mark_activity("user_speech_started")
                self.deps.movement_manager.set_listening(True)
            self._mark_activity("user_transcription_delta")
            self._input_transcript_parts.append(delta)
            await self._cancel_task(self._input_final_task)
            self._input_final_task = asyncio.create_task(self._emit_final_input_transcript())

        elif event_type == "session.output_transcript.delta":
            delta = event.get("delta") or ""
            if not isinstance(delta, str) or not delta:
                return
            self._output_transcript_parts.append(delta)
            await self._cancel_task(self._output_final_task)
            self._output_final_task = asyncio.create_task(self._emit_final_output_transcript())

        elif event_type == "session.output_audio.delta":
            audio_b64 = event.get("delta")
            if not isinstance(audio_b64, str) or not audio_b64:
                return
            decoded_pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16).reshape(1, -1)
            self._last_output_audio_at = time.monotonic()
            self.deps.movement_manager.set_speaking(True)
            self._mark_activity("assistant_audio_delta")
            await self._cancel_task(self._speaking_idle_task)
            self._speaking_idle_task = asyncio.create_task(self._clear_speaking_after_idle())
            await self.output_queue.put((self.SAMPLE_RATE, decoded_pcm))

        elif event_type == "session.delegation.created":
            self._mark_activity("response_created")

        elif event_type == "response.event":
            await self._handle_nested_response_event(event)

        elif event_type == "error":
            err = event.get("error")
            msg = getattr(err, "message", None)
            if msg is None and isinstance(err, dict):
                msg = err.get("message")
            if msg is None:
                msg = str(err) if err else "unknown error"
            logger.error("GPT-Live error: %s (raw=%s)", msg, err)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))

        elif event_type == "session.closed":
            logger.info("GPT-Live session closed")

    async def _run_live_session(self) -> None:
        """Establish and manage a single GPT-Live session."""
        tool_specs = get_tool_specs()
        logger.info("Tools to be used in conversation: %s", [tool["name"] for tool in tool_specs])
        async with self._live_connect() as conn:
            self.connection = conn
            try:
                await self._send_event(
                    {
                        "type": "session.start",
                        "session": self._build_session_config(tool_specs),
                    }
                )
                logger.info(
                    "GPT-Live session.start sent profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
            except Exception:
                logger.exception("GPT-Live session.start failed; aborting startup")
                raise

            self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
            try:
                async for raw in conn:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    if not isinstance(raw, str):
                        logger.warning("Ignoring non-text GPT-Live message: %s", type(raw).__name__)
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Ignoring invalid GPT-Live JSON")
                        continue
                    if not isinstance(event, dict):
                        continue
                    await self._handle_live_event(event)
            finally:
                await self.tool_manager.shutdown()
                await self._cancel_task(self._input_final_task)
                await self._cancel_task(self._output_final_task)
                await self._cancel_task(self._speaking_idle_task)

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Receive a microphone frame and append it to the Live input stream."""
        if not self._is_connected() or self.connection is None:
            return

        _, audio_frame = frame
        if audio_frame.size == 0:
            return

        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        audio_frame = audio_to_int16(audio_frame)
        try:
            await self._send_event(
                {
                    "type": "session.input_audio.append",
                    "audio": base64.b64encode(audio_frame.tobytes()).decode("utf-8"),
                }
            )
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        await self.tool_manager.shutdown()
        await self._cancel_task(self._input_final_task)
        await self._cancel_task(self._output_final_task)
        await self._cancel_task(self._speaking_idle_task)

        if self.connection:
            try:
                await self._send_event({"type": "session.close"})
            except Exception as e:
                logger.debug("session.close ignored: %s", e)
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug("Connection already closed during shutdown: %s", e)
            except Exception as e:
                logger.debug("connection.close() ignored: %s", e)
            finally:
                self.connection = None
                self._session_started = False

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_available_voices(self) -> list[str]:
        """Return the available GPT-Live voices."""
        return get_available_voices(OPENAI_GPT_LIVE_BACKEND)
