"""DelegateAgent tool — bridges kimi to external AI agents via ACP or exec."""

from pathlib import Path
from typing import Literal

from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.delegate_agent.acp_bridge import (
    AGENT_CONFIGS,
    run_acp,
    run_codex_exec,
)
from kimi_cli.tools.utils import load_desc


class Params(BaseModel):
    agent: Literal["opencode", "qwen", "codex"] = Field(
        description=(
            'Which agent to delegate to. "opencode" for TypeScript/JS/complex multi-file tasks; '
            '"qwen" for Qwen Code (Alibaba coding model); '
            '"codex" for OpenAI Codex one-shot tasks.'
        )
    )
    task: str = Field(
        description=(
            "Complete, self-contained task description for the agent. "
            "Include all context the agent needs: file paths, desired outcome, constraints."
        )
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory. Defaults to the current session work directory.",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=30,
        le=3600,
        description="Maximum seconds to wait for the agent. Default 600 (10 min).",
    )
    model_id: str | None = Field(
        default=None,
        description=(
            "Optional model override for opencode/qwen, e.g. 'lmstudio/qwen3.5-4b'. "
            "If omitted, the agent uses its configured default model."
        ),
    )


class DelegateAgent(CallableTool2[Params]):
    """Delegate a task to an external AI agent (opencode, qwen, or codex)."""

    name: str = "DelegateAgent"
    description: str = load_desc(Path(__file__).parent / "description.md", {})
    params: type[Params] = Params

    def __init__(self, runtime: Runtime, approval: Approval) -> None:
        super().__init__()
        self._runtime = runtime
        self._approval = approval

    async def __call__(self, params: Params) -> ToolReturnValue:
        cwd = params.cwd or str(self._runtime.session.work_dir)
        config = AGENT_CONFIGS[params.agent]

        if config.is_acp:
            return await run_acp(
                agent_key=params.agent,
                config=config,
                task=params.task,
                cwd=cwd,
                timeout=params.timeout_seconds,
                approval=self._approval,
                model_id=params.model_id,
            )
        else:
            return await run_codex_exec(
                task=params.task,
                cwd=cwd,
                timeout=params.timeout_seconds,
            )
