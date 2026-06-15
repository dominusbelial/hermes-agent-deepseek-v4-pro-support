#!/usr/bin/env python3
"""
Pi Delegate Tools -- Spawn Pi coding sub-agents via RPC and coordinate them.

Three tools:
  pi_delegate  -- synchronous one-shot delegation (blocks until done)
  pi_spawn     -- start a Pi agent in background, return task_id immediately
  pi_join      -- wait for one or more pi_spawn task_ids, return all results

Pi is configured via ~/.pi/agent/models.json.  Override via env vars:
  PI_DELEGATE_PROVIDER  (default: qwen3-local)
  PI_DELEGATE_MODEL     (default: Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf)
"""

import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = os.getenv("PI_DELEGATE_PROVIDER", "qwen3-local")
_DEFAULT_MODEL = os.getenv(
    "PI_DELEGATE_MODEL",
    "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
)
_DEFAULT_TIMEOUT = 1800.0

# Hard cap: llama-server runs --parallel 2 so at most 2 Pi sessions can run
# concurrently before the server starts queuing requests.  Enforced here so
# Gemma cannot accidentally launch a 3rd session that would stall indefinitely.
_PI_SLOT_SEMAPHORE = threading.Semaphore(2)

# ANSI badge matching Hermes's yellow model-name style
_QWEN_BADGE = "\033[43;30m 🤖 Qwen3-Coder \033[0m"

_PLANNING_PREFIXES = (
    "let me ", "now let me ", "i'll ", "i will ", "i see ", "i notice ",
    "i see that", "let's ", "now i ", "i have a ", "i've ", "i need to ",
    "i'm going to", "i should ", "i can ", "i found ", "i understand",
    "based on ", "given the ", "now,", "first,", "next,", "finally,",
)

def _is_planning_line(text: str) -> bool:
    """True for short inter-turn planning sentences, not substantive output."""
    t = text.lower().strip()
    if len(t) > 160:
        return False
    if t.startswith(("#", "-", "*", "|", "```", ">")):
        return False
    return any(t.startswith(p) for p in _PLANNING_PREFIXES)

_PI_BINARY_CANDIDATES = [
    os.path.expanduser("~/.hermes/node/bin/pi"),
    "pi",
]

# Background task registry for pi_spawn / pi_join
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="pi-worker"
)
_tasks: Dict[str, tuple] = {}  # task_id -> (task_name, Future)
_tasks_lock = threading.Lock()

_PROXY_PATH = os.path.expanduser("~/.hermes/llama_proxy.py")
_PROXY_URL = "http://127.0.0.1:8001"
_proxy_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_pi_binary() -> Optional[str]:
    for candidate in _PI_BINARY_CANDIDATES:
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


def check_pi_requirements() -> bool:
    return _find_pi_binary() is not None


def _proxy_is_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{_PROXY_URL}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _ensure_proxy_running() -> None:
    with _proxy_lock:
        if _proxy_is_up():
            return
        if not os.path.isfile(_PROXY_PATH):
            logger.warning("llama_proxy.py not found at %s", _PROXY_PATH)
            return
        logger.info("Starting llama proxy at %s", _PROXY_PATH)
        subprocess.Popen(
            ["python3", _PROXY_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(10):
            time.sleep(0.5)
            if _proxy_is_up():
                logger.info("Proxy up")
                return
        logger.warning("Proxy did not come up in 5s")


def _collect_stderr(proc: subprocess.Popen) -> List[str]:
    lines: List[str] = []
    try:
        for line in proc.stderr:
            stripped = line.rstrip()
            lines.append(stripped)
            logger.debug("pi stderr: %s", stripped)
    except Exception:
        pass
    return lines


def _drain_stderr(proc: subprocess.Popen) -> None:
    _collect_stderr(proc)


def _run_pi_rpc(
    goal: str,
    context: Optional[str],
    cwd: Optional[str],
    parent_agent,
    timeout: float,
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_proxy_running()

    pi_bin = _find_pi_binary()
    if not pi_bin:
        return {
            "error": (
                "pi binary not found. "
                "Install: npm install -g @earendil-works/pi-coding-agent"
            )
        }

    cmd = [
        pi_bin, "--mode", "rpc",
        "--provider", _DEFAULT_PROVIDER,
        "--model", _DEFAULT_MODEL,
        "--no-session",
    ]

    # parent_agent is not forwarded through registry.dispatch — use the
    # thread-local set by tool_executor so we can reach the active spinner.
    try:
        from model_tools import get_current_agent
        _agent = get_current_agent() or parent_agent
    except Exception:
        _agent = parent_agent

    parent_cb = getattr(_agent, "tool_progress_callback", None)
    _vprint = getattr(_agent, "_vprint", None)

    label = task_name or (goal[:40] + ("..." if len(goal) > 40 else ""))

    def _spinner_print(text: str) -> None:
        spinner = (
            getattr(_agent, "_active_tool_spinner", None)
            or getattr(_agent, "_delegate_spinner", None)
        )
        if spinner:
            try:
                spinner.print_above(text)
                return
            except Exception as exc:
                logger.debug("pi spinner.print_above failed: %s", exc)
        if _vprint:
            try:
                _vprint(text, force=True)
            except Exception as exc:
                logger.debug("pi _vprint failed: %s", exc)

    def _relay(event_type: str, tool_name: str = None, preview: str = None,
               args=None, **kwargs) -> None:
        if parent_cb:
            try:
                parent_cb(event_type, tool_name, preview, args, **kwargs)
            except Exception as exc:
                logger.debug("pi parent_cb relay failed: %s", exc)

    _spinner_print(f" ├─ ⏳ [{label}] waiting for Pi slot…")
    acquired = _PI_SLOT_SEMAPHORE.acquire(timeout=timeout)
    if not acquired:
        return {"error": f"No Pi slot available after {timeout:.0f}s — both slots busy"}
    _spinner_print(f" ├─ 🤖 Pi [{label}] starting")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd or os.getcwd(),
        )
    except Exception as exc:
        _PI_SLOT_SEMAPHORE.release()
        return {"error": f"Failed to launch pi: {exc}"}

    stderr_lines: List[str] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_lines.extend(_collect_stderr(proc)), daemon=True
    )
    stderr_thread.start()

    # Auto-inject repo file tree so Pi skips discovery tool calls.
    # Silently skipped if cwd is not a git repo, git is missing, or times out.
    _repo_tree = ""
    try:
        import shutil as _shutil
        if _shutil.which("git"):
            _tree_out = subprocess.run(
                ["git", "ls-files"],
                cwd=cwd or os.getcwd(),
                capture_output=True, text=True, timeout=5,
            )
            if _tree_out.returncode == 0 and _tree_out.stdout.strip():
                _repo_tree = _tree_out.stdout.strip()
    except Exception:
        pass

    _context_parts = []
    if _repo_tree:
        _context_parts.append(f"Repo files (git ls-files):\n{_repo_tree}")
    if context and context.strip():
        _context_parts.append(context.strip())
    _full_context = "\n\n".join(_context_parts)

    prompt_text = goal
    if _full_context:
        prompt_text = f"{goal}\n\nContext:\n{_full_context}"

    try:
        proc.stdin.write(json.dumps({"type": "prompt", "message": prompt_text}) + "\n")
        proc.stdin.flush()
    except Exception as exc:
        proc.kill()
        return {"error": f"Failed to send prompt to pi: {exc}"}

    _spinner_print(f" ├─ 🤖 Pi [{label}]")
    _relay("subagent.start", preview=goal)

    text_chunks: list[str] = []
    _text_line_buf: str = ""   # buffer for line-flushed streaming display
    _in_think_block: bool = False

    def _flush_text_buf(force: bool = False) -> None:
        nonlocal _text_line_buf
        if not _text_line_buf:
            return
        lines = _text_line_buf.split("\n")
        # Keep the last incomplete line in buffer unless forced
        complete, remainder = lines[:-1], lines[-1]
        for ln in complete:
            ln = ln.strip()
            if ln:
                if _is_planning_line(ln):
                    _spinner_print(f" ├─ 💭 [{label}] {ln}")
                else:
                    _spinner_print(f"{_QWEN_BADGE}{ln}")
        _text_line_buf = "" if force else remainder

    deadline = time.monotonic() + timeout

    try:
        for raw_line in proc.stdout:
            if time.monotonic() > deadline:
                proc.kill()
                return {"error": f"Pi agent timed out after {timeout:.0f}s"}

            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("pi RPC non-JSON: %s", line[:200])
                continue

            _pi_debug = os.getenv("PI_DEBUG_EVENTS")
            if _pi_debug:
                with open(os.path.expanduser("~/pi_events.log"), "a") as _f:
                    _f.write(line + "\n")

            etype = event.get("type")

            if etype == "message_update":
                ae = event.get("assistantMessageEvent", {})
                dtype = ae.get("type")

                if dtype == "text_delta":
                    chunk = ae.get("delta", "")

                    # Detect inline <think> blocks from Qwen3
                    if "<think>" in chunk and not _in_think_block:
                        _in_think_block = True
                        _flush_text_buf(force=True)
                        _spinner_print(f" ├─ 💭 [{label}] thinking...")
                    if "</think>" in chunk and _in_think_block:
                        _in_think_block = False

                    if not _in_think_block and "</think>" not in chunk:
                        # Strip any <think>...</think> that arrived complete in one chunk
                        import re as _re
                        visible = _re.sub(r"<think>.*?</think>", "", chunk, flags=_re.DOTALL)
                        if visible:
                            _text_line_buf += visible
                            _flush_text_buf()

                    text_chunks.append(chunk)
                    _relay("subagent_progress", preview=chunk)

                elif dtype == "thinking_start":
                    _flush_text_buf(force=True)
                    _in_think_block = True
                    _spinner_print(f" ├─ 💭 [{label}] thinking...")

                elif dtype == "thinking_end":
                    _in_think_block = False

                elif dtype == "thinking_delta":
                    chunk = ae.get("delta", "")
                    short = chunk[:80].replace("\n", " ")
                    ellipsis = "..." if len(chunk) > 80 else ""
                    _spinner_print(f' ├─ 💭 [{label}] "{short}{ellipsis}"')
                    _relay("_thinking", chunk)

            elif etype == "tool_execution_start":
                _flush_text_buf(force=True)
                tool_name = event.get("toolName", "tool")
                args = event.get("args") or {}
                # Show the most useful arg: file path, command, or first string value
                arg_hint = ""
                for key in ("path", "command", "query", "content"):
                    if key in args:
                        val = str(args[key])[:80]
                        arg_hint = f" {val}" if val else ""
                        break
                if not arg_hint and args:
                    first_val = next(iter(args.values()), "")
                    arg_hint = f" {str(first_val)[:80]}" if first_val else ""
                _spinner_print(f" ├─ 🔧 [{label}] {tool_name}{arg_hint}")
                _relay("tool.started", tool_name=tool_name, args=args)

            elif etype == "tool_execution_end":
                tool_name = event.get("toolName", "tool")
                is_error = event.get("isError", False)
                result_data = event.get("result") or {}
                blocks = result_data.get("content") if isinstance(result_data, dict) else None
                preview = ""
                if isinstance(blocks, list) and blocks:
                    preview = str(blocks[0].get("text") or "")[:120].replace("\n", " ").strip()
                if is_error and preview:
                    _spinner_print(f"{_QWEN_BADGE}✗ {preview}")
                _relay("tool.completed", tool_name=tool_name, preview=preview,
                       is_error=is_error)

            elif etype == "agent_end":
                break

    except Exception as exc:
        logger.exception("Pi RPC stream error: %s", exc)
        proc.kill()
        return {"error": f"Pi RPC stream error: {exc}"}
    finally:
        _flush_text_buf(force=True)
        _PI_SLOT_SEMAPHORE.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    response = "".join(text_chunks).strip()
    _relay("subagent.complete", preview=response[:200])

    if not response:
        stderr_tail = "\n".join(stderr_lines[-10:]) if stderr_lines else "(no stderr)"
        return {"error": f"Pi agent returned no text response.\nstderr:\n{stderr_tail}"}

    return {"result": response}


# ---------------------------------------------------------------------------
# pi_delegate  (synchronous)
# ---------------------------------------------------------------------------

def pi_delegate(args: dict, **kwargs) -> str:
    from tools.registry import tool_error, tool_result

    goal = (args.get("goal") or "").strip()
    if not goal:
        return tool_error("goal is required")

    outcome = _run_pi_rpc(
        goal=goal,
        context=args.get("context"),
        cwd=args.get("cwd"),
        parent_agent=kwargs.get("parent_agent"),
        timeout=float(args.get("timeout") or _DEFAULT_TIMEOUT),
        task_name=args.get("task_name"),
    )

    if "error" in outcome:
        return tool_error(outcome["error"])
    return tool_result(outcome)


_DELEGATE_SCHEMA = {
    "name": "pi_delegate",
    "description": (
        "Delegate a focused coding or analysis task to Pi (synchronous — blocks until done). "
        "Pi runs Qwen3-Coder-Next locally via RPC and has full file and shell access in cwd. "
        "Use for sequential tasks or when the result is needed before the next step. "
        "For independent parallel tasks use pi_spawn + pi_join instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "Precise imperative coding task. Include file paths, function names, "
                    "and expected behavior. Self-contained — Pi has no conversation history."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Everything Pi needs: relevant file contents, exact error messages, "
                    "project conventions, what must NOT change. More context → better results."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for Pi. Defaults to current directory.",
            },
            "task_name": {
                "type": "string",
                "description": "Short label for display (e.g. 'auth-refactor'). Optional.",
            },
        },
        "required": ["goal"],
    },
}


# ---------------------------------------------------------------------------
# pi_spawn  (background, non-blocking)
# ---------------------------------------------------------------------------

def pi_spawn(args: dict, **kwargs) -> str:
    from tools.registry import tool_error, tool_result

    goal = (args.get("goal") or "").strip()
    if not goal:
        return tool_error("goal is required")

    task_name = (args.get("task_name") or goal[:40]).strip()
    task_id = f"pi-{uuid.uuid4().hex[:8]}"

    future = _executor.submit(
        _run_pi_rpc,
        goal=goal,
        context=args.get("context"),
        cwd=args.get("cwd"),
        parent_agent=kwargs.get("parent_agent"),
        timeout=float(args.get("timeout") or _DEFAULT_TIMEOUT),
        task_name=task_name,
    )

    with _tasks_lock:
        _tasks[task_id] = (task_name, future)

    return tool_result({
        "task_id": task_id,
        "task_name": task_name,
        "status": "running",
    })


_SPAWN_SCHEMA = {
    "name": "pi_spawn",
    "description": (
        "Start a Pi coding agent in the background and return immediately with a task_id. "
        "Use this (instead of pi_delegate) when you have multiple independent coding tasks "
        "that can run in parallel. Spawn all tasks first, then call pi_join with all task_ids "
        "to wait for and collect their results. "
        "Pi uses RPC mode over stdin/stdout — one process per spawn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "Precise imperative coding task. Must be fully self-contained — "
                    "include file paths, expected behavior, constraints. "
                    "Pi has no access to other spawned agents or conversation history."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "All background Pi needs: file contents, errors, conventions, "
                    "files it must NOT touch (to avoid conflicts with sibling agents)."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory. Must be the repo root for most tasks.",
            },
            "task_name": {
                "type": "string",
                "description": (
                    "Short label identifying this agent's role, e.g. 'write-tests', "
                    "'refactor-auth', 'update-docs'. Used in display and pi_join results."
                ),
            },
        },
        "required": ["goal"],
    },
}


# ---------------------------------------------------------------------------
# pi_join  (collect background results)
# ---------------------------------------------------------------------------

def pi_join(args: dict, **kwargs) -> str:
    from tools.registry import tool_error, tool_result

    task_ids = args.get("task_ids")
    if not task_ids or not isinstance(task_ids, list):
        return tool_error("task_ids must be a non-empty list of task_id strings")

    timeout = float(args.get("timeout") or _DEFAULT_TIMEOUT)
    results = {}

    for tid in task_ids:
        with _tasks_lock:
            entry = _tasks.get(tid)

        if not entry:
            results[tid] = {"error": f"Unknown task_id '{tid}' — not spawned or already collected"}
            continue

        task_name, future = entry
        try:
            outcome = future.result(timeout=timeout)
            results[tid] = {"task_name": task_name, **outcome}
        except concurrent.futures.TimeoutError:
            results[tid] = {"task_name": task_name, "error": f"Timed out after {timeout:.0f}s"}
        except Exception as exc:
            results[tid] = {"task_name": task_name, "error": str(exc)}
        finally:
            with _tasks_lock:
                _tasks.pop(tid, None)

    return tool_result({"results": results})


_JOIN_SCHEMA = {
    "name": "pi_join",
    "description": (
        "Wait for one or more background Pi agents started with pi_spawn and collect their results. "
        "Blocks until all specified tasks complete (or timeout). "
        "Returns a dict keyed by task_id, each with task_name and either 'result' or 'error'. "
        "Always call pi_join after pi_spawn — never leave tasks unjoined."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of task_id values returned by pi_spawn calls.",
            },
            "timeout": {
                "type": "number",
                "description": "Max seconds to wait for all tasks. Default 600.",
            },
        },
        "required": ["task_ids"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="pi_delegate",
    toolset="delegation",
    schema=_DELEGATE_SCHEMA,
    handler=pi_delegate,
    check_fn=check_pi_requirements,
    emoji="🤖",
)

registry.register(
    name="pi_spawn",
    toolset="delegation",
    schema=_SPAWN_SCHEMA,
    handler=pi_spawn,
    check_fn=check_pi_requirements,
    emoji="🤖",
)

registry.register(
    name="pi_join",
    toolset="delegation",
    schema=_JOIN_SCHEMA,
    handler=pi_join,
    check_fn=check_pi_requirements,
    emoji="🤖",
)
