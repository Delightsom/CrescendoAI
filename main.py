"""CLI runner for a multi-turn conversation against a chat endpoint."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import SecretStr, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from core.models import RunResult, TurnRecord
from core.orchestrator import Orchestrator
from strategies.attacker import ScriptedAttacker
from targets.endpoint_client import EndpointClient, EndpointConfig

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crescendo",
        description=(
            "Run a multi-turn red-team conversation against an "
            "OpenAI-compatible chat endpoint."
        ),
    )
    parser.add_argument("--base-url", required=True, help="Endpoint origin, for example http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Model name sent in the chat-completions body")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token. Defaults to the CRESCENDO_API_KEY environment variable.",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds (default: 60)")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Stop after this many turns. Defaults to the number of prompts.",
    )
    parser.add_argument("--system", default=None, help="Optional system message sent with every request")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="User message for one turn. Repeat for each turn. Appended after --prompts-file.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="UTF-8 file with one prompt per line. Blank lines and # comments are ignored.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        prompts = load_prompts(args.prompt, args.prompts_file)
        max_turns = len(prompts) if args.max_turns is None else args.max_turns
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        api_key = _resolve_api_key(args.api_key)
        config = EndpointConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=SecretStr(api_key) if api_key else None,
            timeout_seconds=args.timeout,
        )
        attacker = ScriptedAttacker(prompts)
    except (OSError, ValueError, ValidationError) as exc:
        console.print(Panel(Text(_format_error(exc)), title="[red]Cannot start run[/red]", border_style="red"))
        return 2

    render_banner(
        console,
        config,
        prompt_count=len(attacker.prompts),
        max_turns=max_turns,
        system_prompt=args.system,
    )

    try:
        with EndpointClient(config) as target:
            orchestrator = Orchestrator(attacker, target, max_turns=max_turns)
            result = orchestrator.run(
                system_prompt=args.system,
                on_turn=lambda record: render_turn(console, record, max_turns),
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Run interrupted.[/yellow]")
        return 130

    render_summary(console, result)
    return 0 if result.ok else 1


def load_prompts(prompt_args: Sequence[str], prompts_file: str | None) -> list[str]:
    prompts: list[str] = []
    if prompts_file:
        text = Path(prompts_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            prompts.append(stripped)
    prompts.extend(prompt.strip() for prompt in prompt_args if prompt.strip())
    if not prompts:
        raise ValueError("Provide at least one --prompt or a --prompts-file with a prompt")
    return prompts


def _resolve_api_key(explicit: str | None) -> str | None:
    if explicit is not None:
        value = explicit.strip()
        if not value:
            raise ValueError("--api-key is empty")
        return value
    env = os.environ.get("CRESCENDO_API_KEY", "").strip()
    return env or None


def _format_error(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        lines = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"]) or "input"
            lines.append(f"{loc}: {error['msg']}")
        return "\n".join(lines)
    return str(exc)


def render_banner(
    console: Console,
    config: EndpointConfig,
    *,
    prompt_count: int,
    max_turns: int,
    system_prompt: str | None,
) -> None:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("Endpoint", Text(config.base_url))
    grid.add_row("Model", Text(config.model))
    grid.add_row("Auth", "bearer token" if config.api_key is not None else "none")
    grid.add_row("Prompts", str(prompt_count))
    grid.add_row("Max turns", str(max_turns))
    if system_prompt and system_prompt.strip():
        grid.add_row("System", Text(_preview(system_prompt.strip())))
    console.print(Panel(grid, title="[bold]CrescendoAI[/bold]", border_style="blue"))


def render_turn(console: Console, record: TurnRecord, total: int) -> None:
    console.print(Rule(Text(f"Turn {record.index + 1}/{total}", style="bold blue")))
    console.print(Panel(Text(record.prompt, style="cyan"), title="[cyan]Attacker[/cyan]", border_style="cyan"))
    if record.ok:
        meta = _latency(record.latency_ms)
        title = "[green]Target[/green]" if not meta else f"[green]Target[/green]  [yellow]{meta}[/yellow]"
        console.print(Panel(Text(record.response or "", style="green"), title=title, border_style="green"))
        return
    console.print(
        Panel(
            Text(record.error or "unknown error", style="red"),
            title="[red]Target error[/red]",
            border_style="red",
        )
    )


def render_summary(console: Console, result: RunResult) -> None:
    table = Table(title="Run summary", header_style="bold", border_style="blue")
    table.add_column("#", justify="right", style="blue")
    table.add_column("Status")
    table.add_column("HTTP", justify="right")
    table.add_column("Latency", justify="right", style="yellow")
    table.add_column("Prompt", style="cyan", overflow="fold")
    table.add_column("Reply", overflow="fold")

    for record in result.turns:
        status = Text("ok", style="green") if record.ok else Text("error", style="red")
        reply = record.response if record.ok else (record.error or "")
        table.add_row(
            str(record.index + 1),
            status,
            "" if record.status_code is None else str(record.status_code),
            _latency(record.latency_ms),
            Text(_preview(record.prompt)),
            Text(_preview(reply or "")),
        )

    console.print()
    console.print(table)
    border = "green" if result.ok else "red"
    console.print(
        Panel(
            Text(f"Stopped: {result.stopped_reason}\nSuccessful turns: {sum(1 for turn in result.turns if turn.ok)}/{len(result.turns)}"),
            title="[bold]Result[/bold]",
            border_style=border,
        )
    )


def _latency(latency_ms: float | None) -> str:
    if latency_ms is None:
        return ""
    return f"{latency_ms:.0f} ms"


def _preview(text: str, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


if __name__ == "__main__":
    sys.exit(main())
