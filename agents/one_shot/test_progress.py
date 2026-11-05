import io
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from copilot.session_events import SessionEvent
from rich.console import Console
from rich.text import Text

from agents.one_shot.progress import ProgressPrinter


class FlushedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushed_text = ""

    def flush(self):
        self.flushed_text = self.getvalue()
        super().flush()


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.stream = FlushedStream()
        self.printer = ProgressPrinter(self.stream)

    def emit(self, event_type, data):
        self.printer.on_event(
            SessionEvent.from_dict(
                {
                    "id": str(uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": event_type,
                    "data": data,
                }
            )
        )

    def test_text_flushes_incrementally_without_repeating_completed_message(
        self,
    ):
        for delta in ("", "Checking ", "the board."):
            self.emit(
                "assistant.message_delta",
                {"messageId": "message", "deltaContent": delta},
            )
            self.assertEqual(self.stream.flushed_text, self.stream.getvalue())
        self.assertEqual(
            self.stream.getvalue(), "\nAssistant\nChecking the board."
        )
        self.emit(
            "assistant.message",
            {"messageId": "message", "content": "Checking the board."},
        )
        self.assertEqual(
            self.stream.getvalue(), "\nAssistant\nChecking the board.\n"
        )
        self.emit(
            "assistant.message",
            {"messageId": "non-streamed", "content": "A complete message."},
        )
        self.assertIn("A complete message.", self.stream.getvalue())
        before = self.stream.getvalue()
        self.emit(
            "assistant.reasoning_delta",
            {"reasoningId": "reasoning", "deltaContent": "Not public text."},
        )
        self.assertEqual(self.stream.getvalue(), before)

    def test_interleaved_tools_close_text_and_show_outcomes_and_previews(self):
        self.emit(
            "assistant.message_delta",
            {"messageId": "message", "deltaContent": "Trying a move."},
        )
        self.emit(
            "tool.execution_start",
            {
                "toolCallId": "first",
                "toolName": "reveal_cell",
                "arguments": {"row": 0, "column": 0, "reason": "Test move."},
            },
        )
        self.emit(
            "tool.execution_start",
            {
                "toolCallId": "second",
                "toolName": "inspect_board",
                "arguments": {},
            },
        )
        self.emit(
            "tool.execution_complete",
            {
                "toolCallId": "second",
                "success": True,
                "result": {"content": "board\n" + "0" * 600},
            },
        )
        self.emit(
            "tool.execution_complete",
            {
                "toolCallId": "first",
                "success": False,
                "error": {"message": "cell is already revealed"},
            },
        )
        self.emit(
            "tool.execution_complete",
            {"toolCallId": "unseen", "success": True},
        )
        output = self.stream.getvalue()
        self.assertTrue(output.startswith("\nAssistant\nTrying a move.\n"))
        self.assertIn('"reason": "Test move."', output)
        self.assertIn("Tool: reveal_cell", output)
        self.assertIn("inspect_board", output)
        self.assertIn("Completed", output)
        self.assertIn("... [truncated]", output)
        self.assertNotIn("0" * 600, output)
        self.assertIn("cell is already revealed", output)
        self.assertIn("Failed", output)
        self.assertIn("unseen", output)

    def test_structured_reply_is_left_to_the_final_result_renderer(self):
        for delta in (" ", '{"answer":', '"done"}'):
            self.emit(
                "assistant.message_delta",
                {"messageId": "structured", "deltaContent": delta},
            )
        self.emit(
            "assistant.message",
            {"messageId": "structured", "content": '{"answer":"done"}'},
        )
        self.assertEqual(self.stream.getvalue(), "")
        self.emit(
            "assistant.message_delta",
            {"messageId": "prose", "deltaContent": "[Not JSON]"},
        )
        self.emit(
            "assistant.message",
            {"messageId": "prose", "content": "[Not JSON]"},
        )
        self.assertIn("[Not JSON]", self.stream.getvalue())

    def test_colorized_chat_keeps_model_text_literal(self):
        console = Console(
            file=self.stream,
            force_terminal=True,
            color_system="standard",
            no_color=False,
            width=100,
        )
        with patch("agents.one_shot.progress.Console", return_value=console):
            self.printer = ProgressPrinter(self.stream)
        self.emit(
            "assistant.message",
            {"messageId": "literal", "content": "[red]literal[/red]"},
        )
        output = self.stream.getvalue()
        self.assertIn("\x1b[", output)
        self.assertIn("[red]literal[/red]", output)

    def test_custom_tool_renderer_receives_full_result_before_truncation(self):
        content = '{"board":"' + "0" * 600 + '"}'
        calls = []

        def render(name, value):
            calls.append((name, value))
            return Text("A board view.")

        self.printer = ProgressPrinter(self.stream, tool_renderer=render)
        self.emit(
            "tool.execution_start",
            {"toolCallId": "board", "toolName": "inspect_board"},
        )
        self.emit(
            "tool.execution_complete",
            {
                "toolCallId": "board",
                "success": True,
                "result": {"content": content},
            },
        )
        self.assertEqual(calls, [("inspect_board", content)])
        self.assertIn("A board view.", self.stream.getvalue())
        self.assertNotIn("[truncated]", self.stream.getvalue())


if __name__ == "__main__":
    unittest.main()
