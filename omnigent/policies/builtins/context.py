"""Built-in context-management policies.

Helps agents keep their working context lean.  The guiding principle:
the goal is not fewer tokens *used*, but fewer tokens *wasted* —
sprawling context filled with stale tool results from a prior task
degrades quality without adding value.

The recommended response to a denial is to start a fresh session for
the new task rather than compacting or summarising in place.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from omnigent.policies.schema import (
    PolicyCallable,
    PolicyEvent,
    PolicyResponse,
    request_user_text,
)

_log = logging.getLogger(__name__)

_ContextAction = Literal["ASK", "DENY"]


def _normalise_action(action: object, *, policy_name: str) -> _ContextAction:
    candidate = action.upper() if isinstance(action, str) else "ASK"
    if candidate == "DENY":
        return "DENY"
    if candidate != "ASK":
        _log.warning(
            "%s: unknown action %r — defaulting to ASK",
            policy_name,
            action,
        )
    return "ASK"


# ── detect_task_switch ────────────────────────────────────────────────────────

_TASK_SWITCH_HISTORY_KEY = "_task_switch_history"

_DEFAULT_TASK_SWITCH_PROMPT = """\
You are a conversation-continuity classifier for a coding assistant.

You are given the user's recent messages (the "prior context") and their
latest message. Decide whether the latest message is a continuation of the
same task or the start of a clearly different, unrelated task.

Guidelines:
- CONTINUATION: the latest message follows naturally from prior work — a
  follow-up question, a refinement, a related sub-task, or asking about
  something mentioned earlier.
- TASK_SWITCH: the latest message starts a completely different topic or
  codebase concern with no meaningful connection to what came before.
- When in doubt, prefer CONTINUATION — false positives (blocking a legitimate
  continuation) are more harmful than false negatives.

Return strict JSON only:
{"verdict": "CONTINUATION" | "TASK_SWITCH"}
"""

_TASK_SWITCH_SCHEMA: dict[str, object] = {
    "format": {
        "type": "json_schema",
        "name": "task_switch_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["CONTINUATION", "TASK_SWITCH"],
                },
            },
            "required": ["verdict"],
            "additionalProperties": False,
        },
    },
}


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    Even with structured output, some providers wrap JSON in
    triple-backtick fences. This strips the outermost fence
    so ``json.loads`` succeeds.

    :param text: Raw LLM response text.
    :returns: Text with code fences removed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


def _extract_text(response: object) -> str:
    """Pull plain text out of a PolicyLLMClient response."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    output = getattr(response, "output", None)
    if not isinstance(output, list) or not output:
        return ""
    content = getattr(output[0], "content", None)
    if not isinstance(content, list) or not content:
        return ""
    return getattr(content[0], "text", "") or ""


def detect_task_switch(
    *,
    min_turns: int = 1,
    history_window: int = 10,
    action: str = "ASK",
    classification_prompt: str = _DEFAULT_TASK_SWITCH_PROMPT,
) -> PolicyCallable:
    """Factory: detect when the user switches to an unrelated task.

    Fires on ``request`` events.  Maintains a rolling window of recent
    user messages in ``session_state`` and, once ``min_turns`` prior
    messages have accumulated, asks the server-level LLM to classify the
    latest message as ``CONTINUATION`` or ``TASK_SWITCH``.

    On ``TASK_SWITCH`` the policy returns *action* with a message
    recommending a fresh session — not compaction or summarisation.  The
    window is reset to contain only the switching message so the new
    task can accumulate its own history from a clean state (``DENY``
    path only — the ``ASK`` path cannot write state on decline, so the
    window advances only once the user approves and the next request
    arrives).
    On ``CONTINUATION`` the policy records the new message into state
    and abstains, letting the request through.

    Requires the server to have an ``llm:`` config block; abstains
    (fail-open) when no LLM client is available.

    .. note::
        ``action="DENY"`` is not a security control — user messages are
        interpolated into the classifier prompt and a determined user can
        craft a message that forces a ``CONTINUATION`` verdict (prompt
        injection).  Use this policy for context-hygiene guidance, not
        for access control.

    :param min_turns: Number of prior messages to accumulate before the
        classifier starts firing.  With the default of ``1`` the
        classifier fires on the **second** user message (one prior
        message is enough context to detect a switch).  Set to ``0`` to
        classify from the very first message.
    :param history_window: Maximum number of recent user messages kept
        in state as prior context for the classifier.  Defaults to
        ``10``.  Older messages are dropped as the window slides.
    :param action: Response when a task switch is detected.
        ``"ASK"`` (default) escalates to the user; ``"DENY"`` blocks
        the request outright.  Defaults to ``"ASK"`` because
        false-positive task-switch classifications are more harmful than
        false negatives.
    :param classification_prompt: System prompt for the classifier LLM
        call.  Must instruct the model to return
        ``{"verdict": "CONTINUATION"|"TASK_SWITCH"}``; the schema is
        enforced via structured output regardless.
    :returns: An async policy callable that fires on ``request`` events.
    """
    normalised_action = _normalise_action(action, policy_name="detect_task_switch")

    async def evaluate(event: PolicyEvent) -> PolicyResponse | None:
        """Classify the new user message and flag task switches.

        Reads ``session_state[_TASK_SWITCH_HISTORY_KEY]`` for prior
        context and writes the updated window back via
        ``state_updates``.

        :param event: Policy event dict.
        :returns: *action* when a task switch is detected; ``None``
            (abstain) otherwise.
        """
        if event.get("type") != "request":
            return None

        new_message = request_user_text(event.get("data"))
        if not new_message.strip():
            return None

        state = event.get("session_state") or {}
        raw_history = state.get(_TASK_SWITCH_HISTORY_KEY)
        history = (
            [item for item in raw_history if isinstance(item, str)]
            if isinstance(raw_history, list)
            else []
        )

        # Slide the window: append new message, keep last history_window entries.
        updated_history = [*history, new_message[:500]][-history_window:]

        # Not enough prior turns yet — record and pass through.
        if len(history) < min_turns:
            _log.debug(
                "detect_task_switch: history_len=%d < min_turns=%d — accumulating",
                len(history),
                min_turns,
            )
            return {
                "result": "ALLOW",
                "state_updates": [
                    {
                        "key": _TASK_SWITCH_HISTORY_KEY,
                        "action": "set",
                        "value": updated_history,
                    }
                ],
            }

        # ── Classify ────────────────────────────────────────────────────
        llm_client = event.get("llm_client")
        if llm_client is None:
            _log.warning(
                "detect_task_switch: no llm_client — server has no llm: config. Abstaining."
            )
            return None

        prior_context = "\n".join(f"- {msg}" for msg in history[-history_window:])
        user_prompt = f"Prior messages:\n{prior_context}\n\nLatest message:\n{new_message[:500]}"

        try:
            response = await llm_client.create(
                instructions=classification_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    }
                ],
                text=_TASK_SWITCH_SCHEMA,
            )
            raw = _extract_text(response)
            if not raw:
                return None
            raw = _strip_code_fences(raw)
            verdict_obj = json.loads(raw)
        except Exception:  # noqa: BLE001 — fail-open
            _log.exception("detect_task_switch: classifier call failed")
            return None

        verdict = verdict_obj.get("verdict", "") if isinstance(verdict_obj, dict) else ""

        if verdict == "TASK_SWITCH":
            _log.info("detect_task_switch: TASK_SWITCH detected — action=%s", normalised_action)
            # Reset the window to the switching message alone so the new
            # task accumulates fresh history from here.  state_updates on a
            # DENY are applied immediately; state_updates on an ASK are only
            # applied if the user approves — on decline the window stays
            # pinned, so the next message will be re-classified against the
            # pre-switch context (which is the safe / over-prompting direction).
            return {
                "result": normalised_action,
                "reason": (
                    "This message looks like the start of a new, unrelated task. "
                    "The current session carries context from prior work that will "
                    "waste capacity without helping here. "
                    "Start a fresh session for this task to keep context lean."
                ),
                "state_updates": [
                    {
                        "key": _TASK_SWITCH_HISTORY_KEY,
                        "action": "set",
                        "value": [new_message[:500]],
                    }
                ],
            }

        if verdict == "CONTINUATION":
            # Update history and let the request through.
            return {
                "result": "ALLOW",
                "state_updates": [
                    {
                        "key": _TASK_SWITCH_HISTORY_KEY,
                        "action": "set",
                        "value": updated_history,
                    }
                ],
            }

        # Unrecognised verdict — fail open.
        return None

    return evaluate


# ── notes_context ────────────────────────────────────────────────────────────

_NOTE_TERM = re.compile(r"[a-z0-9][a-z0-9_-]{3,}")
_NOTE_STOPWORDS = frozenset(
    [
        "about",
        "after",
        "again",
        "also",
        "been",
        "before",
        "being",
        "could",
        "does",
        "doing",
        "from",
        "have",
        "here",
        "into",
        "just",
        "like",
        "make",
        "more",
        "most",
        "much",
        "need",
        "only",
        "other",
        "over",
        "please",
        "should",
        "some",
        "such",
        "than",
        "that",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "very",
        "want",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "your",
    ]
)
_NOTE_SUMMARY_KEYS = ("question", "description", "title")
_NOTE_SUMMARY_MAX_CHARS = 160
# Notes scoring under this fraction of the best match are left out.
_NOTE_RELATIVE_CUTOFF = 0.5


def _note_query_terms(message: str) -> set[str]:
    """
    Return the distinct search terms in a user message.

    :param message: The user's prompt text.
    :returns: Lower-cased words of four or more characters, minus stopwords.
    """
    return {term for term in _NOTE_TERM.findall(message.lower()) if term not in _NOTE_STOPWORDS}


def _note_summary(note: Path, text: str) -> str:
    """
    Return a one-line description of a note.

    Prefers a frontmatter ``question`` / ``description`` / ``title`` field, then
    the first ``# `` heading, then the file name.

    :param note: Path to the note.
    :param text: The note's contents.
    :returns: A summary of at most :data:`_NOTE_SUMMARY_MAX_CHARS` characters.
    """
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, _, value = line.partition(":")
            if key.strip() in _NOTE_SUMMARY_KEYS and value.strip():
                return value.strip()[:_NOTE_SUMMARY_MAX_CHARS]
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()[:_NOTE_SUMMARY_MAX_CHARS]
    return note.stem


def _visible_notes(root: Path) -> list[Path]:
    """
    Return the ``*.md`` files under *root*, skipping hidden directories and files.

    :param root: Directory to search recursively, e.g. a vault folder.
    :returns: Sorted note paths, excluding any under ``.trash``, ``.obsidian`` etc.
    """
    return sorted(
        note
        for note in root.rglob("*.md")
        if not any(part.startswith(".") for part in note.relative_to(root).parts)
    )


def _search_notes(
    roots: list[Path],
    terms: set[str],
    *,
    max_results: int,
    min_score: float,
    display: Callable[[Path], str] = str,
) -> list[str]:
    """
    Rank markdown notes under *roots* by the rarity-weighted terms they contain.

    Each matched term scores ``log(1 + notes / notes containing it)``, doubled
    when it is in the file name, so a rare word outweighs common ones. Notes
    below *min_score*, or below half the best note's score, are left out.
    Unreadable files are skipped.

    :param roots: Note files or directories searched recursively for ``*.md``.
    :param terms: Search terms from :func:`_note_query_terms`.
    :param max_results: Maximum number of notes to return.
    :param min_score: Score a note must reach to be listed.
    :param display: Turns a note's path into the path shown to the model.
    :returns: ``"- <path> — <summary>"`` lines, best match first.
    """
    notes: list[tuple[Path, str, str, str]] = []
    for root in roots:
        for note in [root] if root.is_file() else _visible_notes(root):
            try:
                text = note.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            notes.append((note, text, note.stem.lower(), text.lower()))
    present = {
        term: [term in name or term in body for _, _, name, body in notes] for term in terms
    }
    weight = {
        term: math.log(1 + len(notes) / hits.count(True))
        for term, hits in present.items()
        if any(hits)
    }
    scored: list[tuple[float, str, str]] = []
    for index, (note, text, name, _) in enumerate(notes):
        score = sum(
            term_weight * (2 if term in name else 1)
            for term, term_weight in weight.items()
            if present[term][index]
        )
        if score >= min_score:
            scored.append((score, display(note), _note_summary(note, text)))
    if not scored:
        return []
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    cutoff = scored[0][0] * _NOTE_RELATIVE_CUTOFF
    return [
        f"- {path} — {summary}" for score, path, summary in scored[:max_results] if score >= cutoff
    ]


def notes_context(
    paths: list[str],
    max_results: int = 5,
    min_score: float = 6.0,
    show_paths_as: dict[str, str] | None = None,
) -> PolicyCallable:
    """
    Point the model at local notes that look relevant to the user's message.

    Searches markdown files under *paths* for the message's words, weighting
    rare words over common ones, and when notes score well enough ALLOWs with
    a ``context`` list of note paths and one-line summaries. The model is told
    the notes exist; it reads them itself. Never blocks: on no match or any
    error it abstains.

    :param paths: Note files or directories, e.g. ``["~/notes/questions"]``.
        ``~`` is expanded; missing paths are skipped.
    :param max_results: Maximum notes to list. Defaults to 5.
    :param min_score: Score a note must reach to be listed. Defaults to 6.0:
        two matched words each in fewer than 1 in 19 notes, or three each in
        fewer than 1 in 6.
    :param show_paths_as: Folder prefixes to rewrite in the listed paths, for
        when the server searches a copy of notes the agent keeps elsewhere, e.g.
        ``{"/srv/notes/vault": "~/vault"}``. The longest matching folder wins;
        paths outside every folder are shown as they are.
    :returns: An async policy callable that fires on ``request`` events.
    """
    # Agent YAML arguments skip schema checks; a bare string is one path.
    path_list = [paths] if isinstance(paths, str) else paths
    roots = [Path(path).expanduser() for path in path_list]
    shown_as = sorted(
        (
            (Path(prefix).expanduser(), shown.rstrip("/"))
            for prefix, shown in (show_paths_as or {}).items()
        ),
        key=lambda entry: len(entry[0].parts),
        reverse=True,
    )

    def display(note: Path) -> str:
        """Show *note* under the first (longest) mapped folder that holds it."""
        for prefix, shown in shown_as:
            if note.is_relative_to(prefix):
                return f"{shown}/{note.relative_to(prefix).as_posix()}"
        return str(note)

    async def evaluate(event: PolicyEvent) -> PolicyResponse | None:
        """
        Search the notes for the user's message and list the matches.

        :param event: Policy event dict.
        :returns: ALLOW with ``context`` when notes match; ``None`` otherwise.
        """
        if event.get("type") != "request":
            return None
        terms = _note_query_terms(request_user_text(event.get("data")))
        if not terms:
            return None
        try:
            matches = await asyncio.to_thread(
                _search_notes,
                [root for root in roots if root.exists()],
                terms,
                max_results=max_results,
                min_score=min_score,
                display=display,
            )
        except Exception:  # noqa: BLE001 — fail-open: notes are a hint, never a gate
            _log.exception("notes_context: note search failed")
            return None
        if not matches:
            return None
        header = "Possibly relevant notes; read them before answering from memory or searching:"
        return {"result": "ALLOW", "context": "\n".join([header, *matches])}

    return evaluate


# ── detect_thrashing ─────────────────────────────────────────────────────────

_THRASHING_HISTORY_KEY = "_thrashing_results"

_ERROR_PREFIXES: tuple[str, ...] = (
    "error:",
    "error -",
    "failed:",
    "traceback (most recent call last)",
    "exception:",
    "fatal:",
    "command failed",
    "permission denied",
    "no such file or directory",
    "enoent:",
    "eacces:",
    "eperm:",
)

_ERROR_JSON_RE = re.compile(r'^\s*\{[^}]*"error"\s*:', re.DOTALL)

# Claude Code's rendering of a non-zero shell exit: "Exit code 1\n<stderr>".
_EXIT_CODE_RE = re.compile(r"exit code [1-9]")


def _looks_like_error(result: str) -> bool:
    """Heuristically detect whether a tool result is an error.

    Checks for common error prefixes (case-insensitive) and
    JSON payloads with an ``"error"`` key.  Designed to be
    over-inclusive rather than under-inclusive — a false positive
    merely increments the error counter by one, which is harmless
    below the threshold; a false negative lets a real error slip
    past uncounted.

    :param result: The ``event["data"]["result"]`` string from a
        ``tool_result`` event.
    :returns: ``True`` when the result looks like an error.
    """
    if not result:
        return False
    lower = result[:500].lower().lstrip()
    if any(lower.startswith(p) for p in _ERROR_PREFIXES):
        return True
    if _EXIT_CODE_RE.match(lower):
        return True
    if _ERROR_JSON_RE.match(result[:500]):
        return True
    return False


def detect_thrashing(
    *,
    consecutive_threshold: int = 5,
    window: int = 10,
    window_error_rate: float = 0.8,
    action: str = "ASK",
) -> PolicyCallable:
    """Factory: detect when an agent is failing repeatedly.

    Fires on ``tool_result`` events.  Maintains a rolling window of
    recent tool-result outcomes (error / success) in ``session_state``
    and flags the agent when either:

    - the last *consecutive_threshold* results are all errors, **or**
    - the error rate within the last *window* results reaches or
      exceeds *window_error_rate*.

    On detection the policy returns *action* with a message telling
    the user the agent appears stuck.  The window is **not** reset on
    detection (unlike ``detect_task_switch``): the agent is likely to
    keep failing, so the policy should keep firing until the user
    intervenes or the agent recovers naturally (successful results
    push old errors out of the window).

    A result whose ``data["is_error"]`` is ``True`` counts as an error:
    the harness reported the failure itself. Any other result falls back
    to the text heuristic in :func:`_looks_like_error`, because many
    producers only have the output text.

    :param consecutive_threshold: Number of consecutive errors before
        the policy fires.  ``0`` disables the consecutive check.
        Defaults to ``5``.
    :param window: Rolling window size for the error-rate check.
        Must be ``>= 2`` when *window_error_rate* is set.  Defaults
        to ``10``.
    :param window_error_rate: Fraction of errors within the last
        *window* results that triggers the policy.  ``0.0`` disables
        the rate check.  Defaults to ``0.8`` (80%).
    :param action: Response when thrashing is detected — ``"ASK"``
        (default) or ``"DENY"``.
    :returns: A policy callable that fires on ``tool_result`` events.
    """
    normalised_action = _normalise_action(action, policy_name="detect_thrashing")

    def evaluate(event: PolicyEvent) -> PolicyResponse | None:
        """Track tool-result outcomes and flag sustained failure runs.

        Reads ``session_state[_THRASHING_HISTORY_KEY]`` for the
        rolling window and writes the updated window back via
        ``state_updates``.  Each entry is ``1`` (error) or ``0``
        (success).

        :param event: Policy event dict.
        :returns: *action* when thrashing is detected; ``None``
            (abstain) for non-``tool_result`` events; ALLOW with
            updated state otherwise.
        """
        if event.get("type") != "tool_result":
            return None

        data = event.get("data")
        if not isinstance(data, dict):
            return None
        result_str = data.get("result", "")
        if not isinstance(result_str, str):
            result_str = str(result_str)

        # A reported failure always counts, and everything else still goes
        # through the heuristic, so the flag can only add errors the text
        # check misses, never hide one it catches.
        reported_failure = data.get("is_error") is True
        is_error = 1 if reported_failure or _looks_like_error(result_str) else 0

        state = event.get("session_state") or {}
        raw_history = state.get(_THRASHING_HISTORY_KEY)
        if isinstance(raw_history, list) and all(isinstance(v, int) for v in raw_history):
            history: list[int] = raw_history
        else:
            history = []

        effective_window = max(window, 1)
        keep = max(effective_window, consecutive_threshold)
        updated = [*history, is_error][-keep:]

        state_update: PolicyResponse = {
            "result": "ALLOW",
            "state_updates": [
                {
                    "key": _THRASHING_HISTORY_KEY,
                    "action": "set",
                    "value": updated,
                }
            ],
        }

        # ── Consecutive check ──────────────────────────────────────
        if consecutive_threshold > 0 and len(updated) >= consecutive_threshold:
            tail = updated[-consecutive_threshold:]
            if all(v == 1 for v in tail):
                return {
                    "result": normalised_action,
                    "reason": (
                        f"The agent has hit {consecutive_threshold} consecutive "
                        f"tool errors. It may be stuck — review and redirect, "
                        f"or start a fresh session."
                    ),
                    "state_updates": state_update["state_updates"],
                }

        # ── Window rate check ──────────────────────────────────────
        if window_error_rate > 0.0 and len(updated) >= effective_window:
            rate_window = updated[-effective_window:]
            rate = sum(rate_window) / len(rate_window)
            if rate >= window_error_rate:
                pct = int(rate * 100)
                return {
                    "result": normalised_action,
                    "reason": (
                        f"The agent has a {pct}% error rate over the last "
                        f"{effective_window} tool calls. It may be stuck — "
                        f"review and redirect, or start a fresh session."
                    ),
                    "state_updates": state_update["state_updates"],
                }

        return state_update

    return evaluate


# ── Registry ──────────────────────────────────────────────────────────────────

POLICY_REGISTRY: list[dict[str, object]] = [
    {
        "handler": "omnigent.policies.builtins.context.notes_context",
        "kind": "factory",
        "name": "Notes Context",
        "description": (
            "Searches local markdown notes for the words in each user message and "
            "gives the model a short list of matching note paths and summaries "
            "alongside the message, so it checks what is already written down. "
            "Never blocks; abstains when nothing matches."
        ),
        "params_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Note files or directories searched recursively for *.md, "
                        "e.g. ['~/notes/questions']. Hidden folders and missing "
                        "paths are skipped."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum notes to list. Defaults to 5.",
                },
                "min_score": {
                    "type": "number",
                    "default": 6.0,
                    "description": (
                        "Score a note must reach to be listed. Each matched word scores "
                        "log(1 + notes / notes containing it), doubled in the file name. "
                        "Defaults to 6.0."
                    ),
                },
                "show_paths_as": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": (
                        "Folder prefixes to rewrite in the listed paths, when the server "
                        "searches a copy of notes the agent keeps elsewhere, e.g. "
                        "{'/srv/notes/vault': '~/vault'}. The longest matching folder wins."
                    ),
                },
            },
            "required": ["paths"],
        },
    },
    {
        "handler": "omnigent.policies.builtins.context.detect_task_switch",
        "kind": "factory",
        "name": "Detect Task Switch",
        "description": (
            "Uses the server-level LLM to classify each user message as a "
            "continuation of the current task or the start of a new, unrelated "
            "one. On a detected task switch, asks (or denies) with a recommendation "
            "to start a fresh session. Implements the 'Keep Context Lean' strategy: "
            "start fresh sessions when switching tasks rather than accumulating "
            "stale context. Requires an llm: config block on the server; "
            "abstains (fail-open) when no LLM client is available."
        ),
        "params_schema": {
            "type": "object",
            "properties": {
                "min_turns": {
                    "type": "integer",
                    "default": 2,
                    "description": (
                        "Number of prior user messages to accumulate before "
                        "the classifier starts firing. Defaults to 2."
                    ),
                },
                "history_window": {
                    "type": "integer",
                    "default": 4,
                    "description": (
                        "Maximum number of recent user messages kept as prior "
                        "context for the classifier. Older messages are dropped "
                        "as the window slides. Defaults to 10."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["ASK", "DENY"],
                    "default": "ASK",
                    "description": (
                        "Response when a task switch is detected. "
                        "ASK escalates to the user (default); "
                        "DENY blocks the request outright."
                    ),
                },
                "classification_prompt": {
                    "type": "string",
                    "description": (
                        "System prompt for the classifier. Must instruct the "
                        'model to return {"verdict": "CONTINUATION"|"TASK_SWITCH"}; '
                        "the output schema is enforced via structured output."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "handler": "omnigent.policies.builtins.context.detect_thrashing",
        "kind": "factory",
        "name": "Detect Agent Thrashing",
        "description": (
            "Detects when an agent is failing repeatedly by tracking tool-result "
            "outcomes in a rolling window. Fires when consecutive errors exceed a "
            "threshold or when the error rate within the window is too high. "
            "A failure the harness reports counts as an error. Otherwise "
            "detection is heuristic (common error prefixes, a non-zero exit "
            "code, and JSON error payloads). No server LLM required."
        ),
        "params_schema": {
            "type": "object",
            "properties": {
                "consecutive_threshold": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 0,
                    "description": (
                        "Number of consecutive tool errors before the policy "
                        "fires. Set to 0 to disable the consecutive check. "
                        "Defaults to 5."
                    ),
                },
                "window": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "description": (
                        "Rolling window size for the error-rate check. Defaults to 10."
                    ),
                },
                "window_error_rate": {
                    "type": "number",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Fraction of errors within the window that triggers "
                        "the policy (0.0–1.0). Set to 0 to disable the rate "
                        "check. Defaults to 0.8 (80%)."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["ASK", "DENY"],
                    "default": "ASK",
                    "description": (
                        "Response when thrashing is detected. "
                        "ASK escalates to the user (default); "
                        "DENY blocks the next tool result outright."
                    ),
                },
            },
            "required": [],
        },
    },
]
