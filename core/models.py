"""Shared transcripts, run records, and the seams between modules."""

from enum import Enum
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class TargetResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    status_code: int
    latency_ms: float = Field(ge=0)


class TurnRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    prompt: str
    response: str | None = None
    status_code: int | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    ok: bool
    error: str | None = None


StoppedReason = Literal["max_turns", "attacker_exhausted", "empty_prompt", "target_error"]


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    turns: list[TurnRecord]
    stopped_reason: StoppedReason

    @property
    def ok(self) -> bool:
        return self.stopped_reason != "target_error" and all(turn.ok for turn in self.turns)


class TargetError(RuntimeError):
    """The target did not return a usable reply for this turn."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.latency_ms = latency_ms


class AttackStrategy(Protocol):
    def next_prompt(self, history: Sequence[Message], turn_index: int) -> str | None:
        """Return the next user message, or None when the strategy is finished."""


class Target(Protocol):
    def send(self, messages: Sequence[Message]) -> TargetResponse:
        """Send the transcript and return the assistant reply."""
