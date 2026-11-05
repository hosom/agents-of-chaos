import io
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from pydantic import BaseModel
from rich.console import Console
from rich.text import Text

from agents.one_shot.presentation import print_result
from agents.one_shot.reporting import (
    AgentResult,
    RunDiagnostics,
    ToolFailure,
)


class Reply(BaseModel):
    answer: str


class TerminalStream(io.StringIO):
    def isatty(self):
        return True


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.result = AgentResult(
            response=Reply(answer="[red]literal[/red]"),
            diagnostics=RunDiagnostics(
                session_id="runtime-session",
                tool_failures=[
                    ToolFailure(
                        timestamp=datetime(2026, 11, 5, tzinfo=timezone.utc),
                        tool_name="reveal_cell",
                        tool_args={"row": 0, "column": 0},
                        error="cell is already revealed",
                    )
                ],
            ),
        )

    def test_redirected_auto_output_remains_lossless_json(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}):
            print_result(self.result, stream)
        self.assertEqual(
            json.loads(stream.getvalue()),
            json.loads(self.result.model_dump_json()),
        )
        self.assertNotIn("\x1b[", stream.getvalue())

    def test_terminal_auto_output_has_colored_response_and_diagnostics(self):
        stream = TerminalStream()
        console = Console(
            file=stream,
            force_terminal=True,
            color_system="standard",
            no_color=False,
            width=100,
        )
        with patch(
            "agents.one_shot.presentation.Console", return_value=console
        ):
            print_result(self.result, stream)
        output = stream.getvalue()
        self.assertIn("\x1b[", output)
        for text in (
            "Response",
            "Diagnostics",
            "runtime-session",
            "reveal_cell",
            "cell is already revealed",
            "[red]literal[/red]",
        ):
            self.assertIn(text, output)

    def test_explicit_json_stays_plain_even_on_a_terminal(self):
        stream = TerminalStream()
        console = Console(file=stream, force_terminal=True)
        with patch(
            "agents.one_shot.presentation.Console", return_value=console
        ):
            print_result(self.result, stream, output_format="json")
        self.assertNotIn("\x1b[", stream.getvalue())
        self.assertEqual(
            json.loads(stream.getvalue()),
            json.loads(self.result.model_dump_json()),
        )

    def test_pretty_output_uses_agent_renderer_without_losing_diagnostics(
        self,
    ):
        stream = io.StringIO()
        seen = []

        def render(response):
            seen.append(response)
            return Text("A custom game board.")

        print_result(
            self.result,
            stream,
            output_format="pretty",
            render_response=render,
        )
        self.assertEqual(seen, [self.result.response])
        self.assertIn("A custom game board.", stream.getvalue())
        self.assertIn("runtime-session", stream.getvalue())
        self.assertIn("cell is already revealed", stream.getvalue())
        self.assertNotIn("\x1b[", stream.getvalue())

    def test_pretty_output_wraps_long_values_instead_of_clipping_them(self):
        long_text = "A long explanation. " * 20
        self.result.response.answer = long_text + "END_RESPONSE"
        failure = self.result.diagnostics.tool_failures[0]
        failure.tool_args = {"reason": long_text + "END_ARGUMENT"}
        failure.error = long_text + "END_ERROR"
        stream = io.StringIO()
        console = Console(file=stream, width=48)
        with patch(
            "agents.one_shot.presentation.Console", return_value=console
        ):
            print_result(self.result, stream, output_format="pretty")
        for marker in ("END_RESPONSE", "END_ARGUMENT", "END_ERROR"):
            self.assertIn(marker, stream.getvalue())


if __name__ == "__main__":
    unittest.main()
