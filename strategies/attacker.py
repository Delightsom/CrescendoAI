"""Operator-supplied attack strategies for a multi-turn run.

Strategies choose the next user message. ``ScriptedAttacker`` replays a fixed
sequence authored by the operator; it does not invent follow-up prompts.
"""

from collections.abc import Sequence

from core.models import AttackStrategy, Message

__all__ = ["AttackStrategy", "ScriptedAttacker"]


class ScriptedAttacker:
    """Replay a fixed list of user messages, one per turn."""

    def __init__(self, prompts: Sequence[str]) -> None:
        cleaned = tuple(prompt.strip() for prompt in prompts)
        if not cleaned or any(not prompt for prompt in cleaned):
            raise ValueError("prompts must contain at least one non-empty message")
        self._prompts = cleaned

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    def next_prompt(self, history: Sequence[Message], turn_index: int) -> str | None:
        if turn_index < 0:
            raise ValueError("turn_index must be >= 0")
        if turn_index >= len(self._prompts):
            return None
        return self._prompts[turn_index]
