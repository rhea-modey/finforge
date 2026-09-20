"""File tools + run_python sandbox, jailed to a world's files/ dir (DESIGN 7.1).

All tools return strings (errors included — never raise to the model), and
every result is truncated to 8000 chars with a [truncated] marker.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_RESULT_CHARS = 8000
_TRUNCATION_MARKER = "\n[truncated]"


def truncate_result(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _root(files_dir: str) -> Path:
    return Path(files_dir).resolve()


def _resolve_jailed(files_dir: str, rel_path: str) -> tuple[Path | None, str | None]:
    """Resolve rel_path inside the jail. Returns (path, None) or (None, error)."""
    root = _root(files_dir)
    try:
        target = (root / str(rel_path)).resolve()
    except (OSError, ValueError) as exc:
        return None, f"Error: cannot resolve path {rel_path!r}: {exc}"
    if not target.is_relative_to(root):
        return None, (f"Error: path {rel_path!r} escapes the files directory jail; "
                      f"use paths relative to the working directory.")
    return target, None


# ---------------------------------------------------------------------------
# the four roster tools
# ---------------------------------------------------------------------------

def list_files(files_dir: str) -> str:
    """Recursive listing with sizes."""
    root = _root(files_dir)
    if not root.is_dir():
        return f"Error: files directory not found: {files_dir}"
    lines = []
    for p in sorted(root.rglob("*")):
        try:
            # skip non-files and symlinks pointing outside the jail
            if not p.is_file() or not p.resolve().is_relative_to(root):
                continue
            lines.append(f"{p.relative_to(root).as_posix()} ({p.stat().st_size} bytes)")
        except OSError:
            continue
    return truncate_result("\n".join(lines) or "(no files)")


def read_file(files_dir: str, path: str, start_line: int = 1,
              max_lines: int = 200) -> str:
    """Line-numbered read; refuses paths escaping the jail."""
    target, err = _resolve_jailed(files_dir, path)
    if err:
        return err
    if target is None or not target.is_file():
        return f"Error: no such file: {path}"
    try:
        start_line = max(1, int(start_line))
        max_lines = max(1, int(max_lines))
    except (TypeError, ValueError):
        return "Error: start_line and max_lines must be integers."
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error: cannot read {path!r}: {exc}"
    all_lines = text.splitlines()
    if not all_lines:
        return "(empty file)"
    if start_line > len(all_lines):
        return (f"Error: start_line {start_line} is beyond the end of the file "
                f"({len(all_lines)} lines).")
    chunk = all_lines[start_line - 1: start_line - 1 + max_lines]
    out = [f"{i}: {ln}" for i, ln in enumerate(chunk, start=start_line)]
    remaining = len(all_lines) - (start_line - 1 + len(chunk))
    if remaining > 0:
        out.append(f"... ({remaining} more lines; continue with "
                   f"start_line={start_line + len(chunk)})")
    return truncate_result("\n".join(out))


def grep_files(files_dir: str, pattern: str, glob: str = "**/*",
               max_matches: int = 50) -> str:
    """Regex search over text files matching glob, `path:line: text` output."""
    root = _root(files_dir)
    if not root.is_dir():
        return f"Error: files directory not found: {files_dir}"
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex {pattern!r}: {exc}"
    try:
        max_matches = max(1, int(max_matches))
    except (TypeError, ValueError):
        return "Error: max_matches must be an integer."
    try:
        candidates = sorted(root.glob(glob or "**/*"))
    except (ValueError, NotImplementedError) as exc:
        return f"Error: invalid glob {glob!r}: {exc}"
    matches: list[str] = []
    capped = False
    for p in candidates:
        try:
            if not p.is_file() or not p.resolve().is_relative_to(root):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                matches.append(f"{rel}:{i}: {line.strip()}")
                if len(matches) >= max_matches:
                    capped = True
                    break
        if capped:
            break
    if not matches:
        return f"(no matches for pattern {pattern!r} in {glob!r})"
    if capped:
        matches.append(f"... (stopped at max_matches={max_matches})")
    return truncate_result("\n".join(matches))


def run_python(files_dir: str, source: str, timeout_s: int = 20) -> str:
    """Run source in an isolated interpreter with cwd = the files dir."""
    root = _root(files_dir)
    if not root.is_dir():
        return f"Error: files directory not found: {files_dir}"
    if not isinstance(source, str) or not source.strip():
        return "Error: run_python needs non-empty Python source."
    try:
        timeout_s = max(1, int(timeout_s))
    except (TypeError, ValueError):
        timeout_s = 20
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", source],
            cwd=str(root), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"Error: run_python timed out after {timeout_s}s"
    except OSError as exc:
        return f"Error: could not start python subprocess: {exc}"
    out = proc.stdout or ""
    if proc.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return truncate_result(out.strip() or "(no output)")


# ---------------------------------------------------------------------------
# toolbox factory (functions bound to one world's files dir)
# ---------------------------------------------------------------------------

def make_toolbox(files_dir: str, run_python_timeout_s: int = 20) -> dict:
    """Callables keyed by tool name, jailed to files_dir. A model-supplied
    timeout_s on run_python is capped at the configured timeout."""
    cfg_timeout = max(1, int(run_python_timeout_s))

    def _run_python(source: str, timeout_s: int | None = None) -> str:
        try:
            t = min(cfg_timeout, int(timeout_s)) if timeout_s else cfg_timeout
        except (TypeError, ValueError):
            t = cfg_timeout
        return run_python(files_dir, source, t)

    return {
        "list_files": lambda: list_files(files_dir),
        "read_file": lambda path, start_line=1, max_lines=200:
            read_file(files_dir, path, start_line, max_lines),
        "grep_files": lambda pattern, glob="**/*", max_matches=50:
            grep_files(files_dir, pattern, glob, max_matches),
        "run_python": _run_python,
    }


# ---------------------------------------------------------------------------
# OpenAI function-calling schemas (orchestrator passes the relevant subset)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Recursively list every file in the task working "
                           "directory, one per line, with sizes in bytes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the task working directory, "
                           "line-numbered. Paths are relative to that directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Relative path of the file to read."},
                    "start_line": {"type": "integer", "default": 1,
                                   "description": "1-based first line to return."},
                    "max_lines": {"type": "integer", "default": 200,
                                  "description": "Maximum number of lines to return."},
                },
                "required": ["path"],
            },
        },
    },
    "grep_files": {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Regex-search text files in the task working directory. "
                           "Returns 'path:line: text' matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string",
                                "description": "Python regular expression."},
                    "glob": {"type": "string", "default": "**/*",
                             "description": "Glob restricting which files to search."},
                    "max_matches": {"type": "integer", "default": 50,
                                    "description": "Stop after this many matches."},
                },
                "required": ["pattern"],
            },
        },
    },
    "run_python": {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a short Python script in an isolated interpreter "
                           "whose working directory is the task files directory "
                           "(so it can open the data files). Returns stdout+stderr. "
                           "Use print() to emit results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string",
                               "description": "Python source code to execute."},
                },
                "required": ["source"],
            },
        },
    },
    # orchestrator-only tools
    "call_agent": {
        "type": "function",
        "function": {
            "name": "call_agent",
            "description": "Run the named agent in a FRESH context with your "
                           "message as its task; returns the agent's final reply. "
                           "Agents share no memory — include everything they need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "Agent name from the roster."},
                    "message": {"type": "string",
                                "description": "The full task message for the agent."},
                },
                "required": ["name", "message"],
            },
        },
    },
    "submit_answer": {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer and end the task. answer_json "
                           "must be a JSON string matching the deliverable schema "
                           "in the task exactly; put narrative in notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer_json": {"type": "string",
                                    "description": "The deliverable as a JSON string."},
                    "notes": {"type": "string",
                              "description": "Narrative notes accompanying the answer."},
                },
                "required": ["answer_json"],
            },
        },
    },
}
