"""Render public SDK events without changing the final response."""

import json
from typing import TextIO

from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantTurnStartData,
    SessionErrorData,
    SessionEvent,
    ToolExecutionCompleteData,
    ToolExecutionStartData,
)
from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text

from .presentation import ToolRenderer, render_content


def preview(text: str) -> str:
    compact = " ".join(text.split())
    return (
        compact if len(compact) <= 500 else compact[:500] + "... [truncated]"
    )


class ProgressPrinter:
    def __init__(
        self, stream: TextIO, *, tool_renderer: ToolRenderer | None = None
    ) -> None:
        self.console = Console(file=stream, markup=False, highlight=False)
        self.tool_renderer = tool_renderer
        self._message_id: str | None = None
        self._streamed_messages: set[str] = set()
        self._structured_messages: set[str] = set()
        self._tools: dict[str, str] = {}

    def finish(self) -> None:
        if self._message_id is not None:
            self.console.print()
            self._message_id = None

    def line(self, text: str, *, style: str = "dim") -> None:
        self.finish()
        self.console.print(Text(text, style=style))

    def card(
        self,
        title: str,
        content: RenderableType,
        *,
        style: str,
        subtitle: str | None = None,
    ) -> None:
        self.finish()
        self.console.print(
            Panel(
                content,
                title=Text(title, style=f"bold {style}"),
                subtitle=Text(subtitle, style=style) if subtitle else None,
                border_style=style,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def on_event(self, event: SessionEvent) -> None:
        data = event.data
        if isinstance(data, AssistantMessageDeltaData):
            if not data.delta_content:
                return
            if data.message_id in self._structured_messages:
                return
            if data.message_id not in self._streamed_messages:
                first_text = data.delta_content.lstrip()
                if not first_text:
                    return
                if first_text.startswith(("{", "[")):
                    self._structured_messages.add(data.message_id)
                    return
            if self._message_id != data.message_id:
                self.finish()
                self.console.print()
                self.console.print(Text("Assistant", style="bold cyan"))
                self._message_id = data.message_id
            self._streamed_messages.add(data.message_id)
            self.console.print(
                Text(data.delta_content), end="", soft_wrap=True
            )
        elif isinstance(data, AssistantMessageData):
            if self._message_id == data.message_id:
                self.finish()
            if (
                data.message_id not in self._streamed_messages
                and data.content.strip()
            ):
                try:
                    json.loads(data.content)
                except json.JSONDecodeError:
                    self.card("Assistant", Text(data.content), style="cyan")
                # Structured replies are shown by the final renderer.
            self._streamed_messages.discard(data.message_id)
            self._structured_messages.discard(data.message_id)
        elif isinstance(data, AssistantTurnStartData):
            self.line("Working...")
        elif isinstance(data, ToolExecutionStartData):
            self._tools[data.tool_call_id] = data.tool_name
            arguments = (
                render_content(preview(json.dumps(data.arguments)))
                if data.arguments is not None
                else Text("No arguments.", style="dim")
            )
            self.card(
                f"Tool: {data.tool_name}",
                arguments,
                style="blue",
                subtitle="Running",
            )
        elif isinstance(data, ToolExecutionCompleteData):
            name = self._tools.pop(data.tool_call_id, data.tool_call_id)
            if data.success:
                content = data.result.content if data.result else ""
                rendered = (
                    self.tool_renderer(name, content)
                    if self.tool_renderer is not None
                    else None
                )
                self.card(
                    name,
                    rendered
                    if rendered is not None
                    else render_content(preview(content)),
                    style="green",
                    subtitle="Completed",
                )
            else:
                self.card(
                    name,
                    Text(
                        data.error.message
                        if data.error
                        else "Tool execution failed."
                    ),
                    style="red",
                    subtitle="Failed",
                )
        elif isinstance(data, SessionErrorData):
            self.card("Error", Text(data.message), style="red")
