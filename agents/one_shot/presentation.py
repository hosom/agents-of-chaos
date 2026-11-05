"""Shared terminal presentation with a lossless JSON output mode."""

import json
from collections.abc import Callable
from typing import Literal, TextIO, TypeVar

from pydantic import BaseModel
from rich import box
from rich.console import Console, RenderableType
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text

from .reporting import AgentResult

OutputFormat = Literal["auto", "pretty", "json"]
ResponseT = TypeVar("ResponseT", bound=BaseModel)
ToolRenderer = Callable[[str, str], RenderableType | None]


def render_content(content: str) -> RenderableType:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return Text(content)
    rendered = JSON.from_data(value)
    rendered.text.no_wrap = False
    rendered.text.overflow = "fold"
    return rendered


def print_result(
    result: AgentResult[ResponseT],
    stream: TextIO,
    *,
    output_format: OutputFormat = "auto",
    render_response: Callable[[ResponseT], RenderableType] | None = None,
) -> None:
    console = Console(file=stream, markup=False, highlight=False)
    if output_format == "json" or (
        output_format == "auto" and not stream.isatty()
    ):
        print(result.model_dump_json(indent=2), file=stream)
        return
    console.print()
    console.rule(Text("Result", style="bold green"))
    response = (
        render_response(result.response)
        if render_response is not None
        else render_content(result.response.model_dump_json())
    )
    console.print(
        Panel(
            response,
            title=Text("Response", style="bold cyan"),
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            render_content(result.diagnostics.model_dump_json()),
            title=Text("Diagnostics", style="bold blue"),
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
