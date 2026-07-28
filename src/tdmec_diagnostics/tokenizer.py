"""Tokenizer length probe interface (Phase 2: deferred by default).

Tokenizer-only analysis is permitted only when lightweight, explicitly
documented, and does not trigger model-weight downloads. Otherwise execution
is deferred to Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class TokenizerLengthProbe(Protocol):
    @property
    def is_available(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str: ...

    def token_length(self, text: str) -> Optional[int]: ...


@dataclass
class NullTokenizerProbe:
    """Default probe: no tokenizer execution; Phase 3 deferral."""

    reason: str = (
        "Tokenizer execution deferred to Phase 3. Phase 2 does not download "
        "Qwen3 or any embedding/tokenizer model weights."
    )

    @property
    def is_available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self.reason

    def token_length(self, text: str) -> Optional[int]:
        return None


@dataclass
class WhitespaceTokenizerProbe:
    """Authorized lightweight probe: whitespace split only (no model download)."""

    @property
    def is_available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    def token_length(self, text: str) -> Optional[int]:
        if text is None:
            return None
        return len([t for t in str(text).split() if t])
