"""Input sanitization sandbox.

Providers run open-source inference engines on personal machines. A prompt is
plain data to the model, but agent frameworks downstream of this grid often
template prompts into shell commands or code runners — so the orchestrator
refuses payloads that look like code-execution injection against providers.

Heuristic by design: a denylist can't be complete. The goal is to strip the
mass/obvious vector, with the Guild trust layer as the real boundary.
"""

from __future__ import annotations

import re

# Patterns aimed at the provider host, not at normal code discussion.
_DENY = [
    re.compile(r"rm\s+-rf\s+[/~]", re.I),
    re.compile(r"curl[^\n]{0,120}\|\s*(?:ba)?sh", re.I),
    re.compile(r"wget[^\n]{0,120}\|\s*(?:ba)?sh", re.I),
    re.compile(r"base64\s+(?:-d|--decode)[^\n]{0,80}\|\s*(?:ba)?sh", re.I),
    re.compile(r"\bmkfs\.\w+\s+/dev/", re.I),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", re.I),                  # fork bomb
    re.compile(r"\bdd\s+if=/dev/(?:zero|random)\s+of=/dev/\w+", re.I),
    re.compile(r"\b(?:ollama|vllm)\b[^\n]{0,60}\b(?:rm|delete|pull)\b[^\n]{0,80}--insecure", re.I),
    re.compile(r"file://(?:/etc/|/proc/|~/.ssh)", re.I),
    re.compile(r"\bnc\s+-e\s+/bin/(?:ba)?sh", re.I),                    # reverse shell
]

MAX_PAYLOAD_CHARS = 200_000


class SanitizationError(ValueError):
    pass


def scan_text(text: str) -> None:
    for pattern in _DENY:
        if pattern.search(text):
            raise SanitizationError(
                f"prompt rejected: matched host-execution pattern {pattern.pattern!r}"
            )


def sanitize_payload(payload: dict) -> dict:
    """Validate a consumer payload before it touches any provider machine."""
    total = 0
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if not isinstance(content, str):
            raise SanitizationError("message content must be a string")
        total += len(content)
        if total > MAX_PAYLOAD_CHARS:
            raise SanitizationError(f"payload exceeds {MAX_PAYLOAD_CHARS} chars")
        scan_text(content)
    return payload
