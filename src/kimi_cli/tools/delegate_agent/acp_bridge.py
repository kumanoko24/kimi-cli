"""ACP client bridge — spawns an ACP-compatible agent and streams its activity."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import acp
import acp.schema as acps
from kosong.message import TextPart, ThinkPart
from kosong.message import ToolCall as WireToolCall
from kosong.tooling import ToolError, ToolOk, ToolResult, ToolReturnValue

from kimi_cli.constant import VERSION
from kimi_cli.soul import get_wire_or_none
from kimi_cli.soul.approval import Approval
from kimi_cli.soul.toolset import get_current_tool_call_or_none
from kimi_cli.utils.logging import logger
from kimi_cli.wire.types import SubagentEvent


@dataclass
class AgentConfig:
    command: list[str]
    name: str
    is_acp: bool


AGENT_CONFIGS: dict[str, AgentConfig] = {
    "opencode": AgentConfig(["opencode", "acp"], "OpenCode", is_acp=True),
    "qwen": AgentConfig(["qwen", "--acp"], "Qwen Code", is_acp=True),
    "codex": AgentConfig(["codex", "exec"], "Codex", is_acp=False),
}


def _select_option(
    options: list[acps.PermissionOption], kind: str
) -> acps.RequestPermissionResponse:
    for opt in options:
        if opt.kind == kind:
            return acps.RequestPermissionResponse(
                outcome=acps.AllowedOutcome(outcome="selected", option_id=opt.option_id)
            )
    if options:
        return acps.RequestPermissionResponse(
            outcome=acps.AllowedOutcome(outcome="selected", option_id=options[0].option_id)
        )
    raise ValueError("No permission options available")


class _ACPBridgeClient:
    """Minimal ACP Client: relays session/update notifications to kimi's wire."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        approval: Approval,
        parent_tool_call_id: str | None,
    ) -> None:
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._approval = approval
        self._parent_tool_call_id = parent_tool_call_id
        self._text_parts: list[str] = []
        self._usage: dict[str, Any] = {}

    def _emit(self, event: Any) -> None:
        wire = get_wire_or_none()
        if wire is None:
            return
        try:
            msg = SubagentEvent(
                parent_tool_call_id=self._parent_tool_call_id,
                agent_id=self._agent_id,
                subagent_type=None,
                event=event,
            )
            wire.soul_side.send(msg)
        except Exception as exc:
            logger.debug("SubagentEvent send skipped: {error}", error=exc)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        update_type = type(update).__name__
        match update_type:
            case "AgentMessageChunk":
                content = update.content
                if isinstance(content, acps.TextContentBlock) and content.text:
                    self._text_parts.append(content.text)
                    self._emit(TextPart(text=content.text))
            case "AgentThoughtChunk":
                content = update.content
                if isinstance(content, acps.TextContentBlock) and content.text:
                    self._emit(ThinkPart(think=content.text))
            case "ToolCallStart":
                tc_id = f"{self._agent_id}:{update.tool_call_id}"
                self._emit(
                    WireToolCall(
                        id=tc_id,
                        function=WireToolCall.FunctionBody(
                            name=update.title or "tool",
                            arguments="{}",
                        ),
                    )
                )
            case "ToolCallProgress":
                if update.status in ("completed", "failed"):
                    tc_id = f"{self._agent_id}:{update.tool_call_id}"
                    raw = str(update.raw_output or "")
                    rv: ToolReturnValue = (
                        ToolOk(output=raw)
                        if update.status == "completed"
                        else ToolError(message=raw or "tool failed", brief="failed")
                    )
                    self._emit(ToolResult(tool_call_id=tc_id, return_value=rv))
            case "UsageUpdate":
                used = update.used
                if used:
                    self._usage = {
                        k: getattr(used, k, None)
                        for k in ("total_tokens", "input_tokens", "output_tokens")
                        if getattr(used, k, None) is not None
                    }
            case _:
                pass

    async def request_permission(
        self,
        options: list[acps.PermissionOption],
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> acps.RequestPermissionResponse:
        if self._approval.is_auto_approve():
            return _select_option(options, "allow_once")

        title = getattr(tool_call, "title", None) or "tool action"
        result = await self._approval.request(
            sender=self._agent_name,
            action=title,
            description=f"[{self._agent_name}] {title}",
        )
        return _select_option(options, "allow_once" if result.approved else "reject_once")

    # File stubs — not called because we don't advertise fs capability.
    async def read_text_file(
        self, path: str, session_id: str, **kwargs: Any
    ) -> acps.ReadTextFileResponse:
        return acps.ReadTextFileResponse(content=Path(path).read_text(errors="replace"))

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> None:
        Path(path).write_text(content)

    # Terminal stubs — not advertised.
    async def create_terminal(self, *_a: Any, **_k: Any) -> Any:
        raise acp.RequestError.method_not_found("terminal/create")

    async def terminal_output(self, *_a: Any, **_k: Any) -> Any:
        raise acp.RequestError.method_not_found("terminal/output")

    async def wait_for_terminal_exit(self, *_a: Any, **_k: Any) -> Any:
        raise acp.RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, *_a: Any, **_k: Any) -> Any:
        raise acp.RequestError.method_not_found("terminal/kill")

    async def release_terminal(self, *_a: Any, **_k: Any) -> None:
        pass

    async def ext_method(self, method: str, params: Any) -> dict[str, Any]:
        return {}

    async def ext_notification(self, method: str, params: Any) -> None:
        pass

    def on_connect(self, conn: Any) -> None:
        pass

    def collected_text(self) -> str:
        return "".join(self._text_parts)

    def usage_summary(self) -> dict[str, Any]:
        return self._usage


def _make_raw_observer(bridge: _ACPBridgeClient) -> Any:
    """Return a raw-message observer that parses session/update independently.

    The kimi-cli acp Python SDK (0.8.0) uses ``session_update`` as the
    SessionNotification discriminator, while opencode's TypeScript SDK (0.16.x)
    serialises updates with ``type``.  The router's deserialization therefore
    fails silently.  This observer intercepts raw incoming messages before the
    router touches them and calls bridge.session_update directly.
    """
    from acp.connection import StreamDirection

    def observer(evt: Any) -> None:
        if evt.direction != StreamDirection.INCOMING:
            return
        msg = cast(dict[str, Any], evt.message)
        if msg.get("method") != "session/update":
            return
        params_obj: Any = msg.get("params")
        if not isinstance(params_obj, dict):
            return
        params = cast(dict[str, Any], params_obj)
        session_id = str(params.get("sessionId") or params.get("session_id") or "")
        update_obj: Any = params.get("update")
        if not isinstance(update_obj, dict):
            return
        update_raw = cast(dict[str, Any], update_obj)
        update_type = str(update_raw.get("type") or update_raw.get("session_update") or "")

        # Manually construct the typed update object the bridge expects.
        try:
            update: Any
            match update_type:
                case "agent_message_chunk" | "AgentMessageChunk":
                    content_obj: Any = update_raw.get("content")
                    if isinstance(content_obj, dict):
                        content_raw = cast(dict[str, Any], content_obj)
                    else:
                        return
                    if content_raw.get("type") == "text":
                        update = acps.AgentMessageChunk(
                            session_update="agent_message_chunk",
                            content=acps.TextContentBlock(
                                type="text", text=str(content_raw.get("text") or "")
                            ),
                        )
                    else:
                        return
                case "agent_thought_chunk" | "AgentThoughtChunk":
                    content_obj: Any = update_raw.get("content")
                    if isinstance(content_obj, dict):
                        content_raw = cast(dict[str, Any], content_obj)
                        update = acps.AgentThoughtChunk(
                            session_update="agent_thought_chunk",
                            content=acps.TextContentBlock(
                                type="text", text=str(content_raw.get("text") or "")
                            ),
                        )
                    else:
                        return
                case "tool_call" | "ToolCallStart":
                    tc_id = str(
                        update_raw.get("toolCallId") or update_raw.get("tool_call_id") or ""
                    )
                    update = acps.ToolCallStart(
                        session_update="tool_call",
                        tool_call_id=tc_id,
                        title=str(update_raw.get("title") or "tool"),
                        status=cast(Any, update_raw.get("status")),
                        kind=cast(Any, update_raw.get("kind")),
                        content=None,
                        locations=None,
                        raw_input=None,
                        raw_output=None,
                    )
                case "tool_call_update" | "ToolCallProgress":
                    tc_id = str(
                        update_raw.get("toolCallId") or update_raw.get("tool_call_id") or ""
                    )
                    update = acps.ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id=tc_id,
                        title=str(update_raw.get("title") or "tool"),
                        status=cast(Any, update_raw.get("status")),
                        kind=cast(Any, update_raw.get("kind")),
                        content=None,
                        locations=None,
                        raw_input=None,
                        raw_output=cast(
                            Any, update_raw.get("rawOutput") or update_raw.get("raw_output")
                        ),
                    )
                case "usage_update" | "UsageUpdate":
                    # Skip — not critical for text collection
                    return
                case _:
                    return
        except Exception as exc:
            logger.debug(
                "ACP bridge: failed to parse update {type}: {error}",
                type=update_type,
                error=exc,
            )
            return

        asyncio.get_event_loop().create_task(bridge.session_update(session_id, update))

    return observer


async def run_acp(
    agent_key: str,
    config: AgentConfig,
    task: str,
    cwd: str,
    timeout: int,
    approval: Approval,
    model_id: str | None = None,
) -> ToolReturnValue:
    agent_id = f"delegate:{agent_key}:{uuid.uuid4().hex[:8]}"
    current_tc = get_current_tool_call_or_none()
    parent_tc_id = current_tc.id if current_tc else None

    bridge = _ACPBridgeClient(
        agent_id=agent_id,
        agent_name=config.name,
        approval=approval,
        parent_tool_call_id=parent_tc_id,
    )

    try:
        async with acp.spawn_agent_process(
            lambda _conn: cast(Any, bridge),
            *config.command,
            cwd=cwd,
            transport_kwargs={"limit": 100 * 1024 * 1024},  # 100 MB — matches wire server
            observers=[_make_raw_observer(bridge)],
        ) as (conn, _proc):
            await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=acps.ClientCapabilities(fs=None, terminal=False),
                client_info=acps.Implementation(name="kimi-cli", version=VERSION),
            )
            session = await conn.new_session(cwd=cwd)
            if model_id:
                await conn.set_session_model(model_id=model_id, session_id=session.session_id)
            await asyncio.wait_for(
                conn.prompt(
                    session_id=session.session_id,
                    prompt=[acps.TextContentBlock(type="text", text=task)],
                ),
                timeout=float(timeout),
            )
            # Allow pending notification tasks to run before closing connection.
            await asyncio.sleep(0.5)
    except TimeoutError:
        return ToolError(message=f"{config.name} timed out after {timeout}s.", brief="timeout")
    except Exception as exc:
        logger.warning("{agent} bridge error: {error}", agent=config.name, error=exc)
        return ToolError(message=str(exc)[:2000], brief=f"{config.name} error")

    text = bridge.collected_text()
    if not text.strip():
        return ToolError(message=f"{config.name} returned no text output.", brief="empty response")
    return ToolOk(output=text)


async def run_codex_exec(
    task: str,
    cwd: str,
    timeout: int,
    model_id: str | None = None,
    effort: str | None = None,
) -> ToolReturnValue:
    cmd = ["codex", "exec"]
    if model_id:
        cmd += ["-m", model_id]
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    cmd.append(task)
    proc: asyncio.subprocess.Process | None = None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(timeout))
    except TimeoutError:
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
        return ToolError(message=f"codex timed out after {timeout}s.", brief="timeout")
    except FileNotFoundError:
        return ToolError(
            message="codex executable not found. Install with: npm install -g @openai/codex",
            brief="codex not found",
        )
    except Exception as exc:
        return ToolError(message=str(exc), brief="codex exec error")

    if proc.returncode != 0:
        msg = (stderr.decode(errors="replace") or stdout.decode(errors="replace")).strip()
        return ToolError(message=msg[:5000] or "codex exited non-zero", brief="codex failed")

    output = stdout.decode(errors="replace").strip()
    if not output:
        return ToolError(message="codex returned no output.", brief="empty response")
    return ToolOk(output=output)
