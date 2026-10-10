"""Run an attacker strategy against a chat target, one turn at a time."""

from collections.abc import Callable

from core.models import (
    AttackStrategy,
    Message,
    Role,
    RunResult,
    StoppedReason,
    Target,
    TargetError,
    TurnRecord,
)


class Orchestrator:
    """Replay up to ``max_turns`` user messages, keeping the full transcript."""

    def __init__(self, attacker: AttackStrategy, target: Target, *, max_turns: int) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.attacker = attacker
        self.target = target
        self.max_turns = max_turns

    def run(
        self,
        *,
        system_prompt: str | None = None,
        on_turn: Callable[[TurnRecord], None] | None = None,
    ) -> RunResult:
        history: list[Message] = []
        if system_prompt and system_prompt.strip():
            history.append(Message(role=Role.SYSTEM, content=system_prompt.strip()))

        records: list[TurnRecord] = []
        stopped_reason: StoppedReason = "max_turns"

        for index in range(self.max_turns):
            prompt = self.attacker.next_prompt(tuple(history), index)
            if prompt is None:
                stopped_reason = "attacker_exhausted"
                break

            text = prompt.strip()
            if not text:
                stopped_reason = "empty_prompt"
                break

            history.append(Message(role=Role.USER, content=text))
            record, failed = self._take_turn(index, text, history)
            records.append(record)
            if on_turn is not None:
                on_turn(record)
            if failed:
                stopped_reason = "target_error"
                break

        return RunResult(turns=records, stopped_reason=stopped_reason)

    def _take_turn(
        self,
        index: int,
        prompt: str,
        history: list[Message],
    ) -> tuple[TurnRecord, bool]:
        try:
            response = self.target.send(tuple(history))
        except TargetError as exc:
            record = TurnRecord(
                index=index,
                prompt=prompt,
                status_code=exc.status_code,
                latency_ms=exc.latency_ms,
                ok=False,
                error=str(exc),
            )
            return record, True

        history.append(Message(role=Role.ASSISTANT, content=response.content))
        record = TurnRecord(
            index=index,
            prompt=prompt,
            response=response.content,
            status_code=response.status_code,
            latency_ms=response.latency_ms,
            ok=True,
        )
        return record, False
