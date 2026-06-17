import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from kosong.tooling import ToolOk

from kimi_cli.tools.delegate_agent.acp_bridge import run_codex_exec


@pytest.mark.asyncio
async def test_run_codex_exec_passes_model_and_effort(monkeypatch):
    recorded: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"done", b""

    async def fake_create_subprocess_exec(
        *cmd: str,
        stdin: Any,
        stdout: Any,
        stderr: Any,
        cwd: str,
    ) -> FakeProcess:
        recorded["cmd"] = cmd
        recorded["stdio"] = (stdin, stdout, stderr)
        recorded["cwd"] = cwd
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = await run_codex_exec(
        task="review this",
        cwd="/tmp/project",
        timeout=30,
        model_id="o3",
        effort="high",
    )

    assert isinstance(result, ToolOk)
    assert result.output == "done"
    assert recorded["cmd"] == (
        "codex",
        "exec",
        "-m",
        "o3",
        "-c",
        'model_reasoning_effort="high"',
        "review this",
    )
    assert recorded["cwd"] == "/tmp/project"


@pytest.mark.asyncio
async def test_run_codex_exec_omits_optional_flags(monkeypatch):
    recorded: dict[str, Sequence[str]] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"done", b""

    async def fake_create_subprocess_exec(
        *cmd: str,
        stdin: Any,
        stdout: Any,
        stderr: Any,
        cwd: str,
    ) -> FakeProcess:
        recorded["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await run_codex_exec(task="review this", cwd="/tmp/project", timeout=30)

    assert recorded["cmd"] == ("codex", "exec", "review this")
