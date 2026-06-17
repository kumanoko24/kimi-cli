Delegate a task to an external AI coding agent (opencode, qwen, or codex) and return their response.

Use this tool when:
- A task is better suited for a different AI model or coding agent
- You want a second opinion on a code change
- The task involves a language or framework where a peer agent may excel

## Agents

- **opencode**: OpenCode — strong at TypeScript/JavaScript projects, complex multi-file refactors, and full-repo reasoning. Uses `opencode acp` server mode; supports streaming tool calls back to kimi's UI.
- **qwen**: Qwen Code (Alibaba) — alternative coding model, useful for cross-validation or when a different perspective is valuable. Uses `qwen --acp` server mode.
- **codex**: OpenAI Codex CLI — good for focused, one-shot code tasks. Runs `codex exec` non-interactively. Supports `model_id` (e.g. `o3`, `codex-mini-latest`) and `effort` (`low`/`medium`/`high` reasoning effort).

## Guidelines

- Write the `task` as a complete, self-contained prompt. Include file paths, expected behavior, constraints, and any relevant context that the agent cannot discover on its own.
- All agents operate in the same working directory (`cwd`) and share the same filesystem. File edits made by the delegate agent are immediately visible.
- The `timeout_seconds` default is 600 (10 minutes). Increase for large tasks.
- Permission requests from opencode/qwen inherit kimi's current yolo/afk setting: if kimi is in yolo mode, all agent tool calls are auto-approved; otherwise, each action is presented to the user for confirmation.
- Do not use DelegateAgent recursively or to replace kimi's own capabilities — prefer it only when delegation adds genuine value.
