import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pyarrow.parquet as pq

from hooks.logging import log_tool_call as logger

SCRIPT = Path(__file__).with_name("log_tool_call.py")
ROOT = SCRIPT.parents[2]


class LoggingHookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = {
            "sessionId": "demo-session",
            "timestamp": 1730000000000,
            "cwd": "/demo",
        }

    def payload_for(self, event, **extra):
        payload = dict(self.payload)
        if event in ("postToolUse", "postToolUseFailure"):
            payload.update(
                toolName="bash", toolArgs={"command": "printf hello"}
            )
        if event == "postToolUse":
            payload["toolResult"] = {
                "resultType": "success",
                "textResultForLlm": "hello",
            }
        elif event == "postToolUseFailure":
            payload["error"] = "Command exited with status 1"
        elif event == "userPromptSubmitted":
            payload["prompt"] = "Analyze this session\nincluding tools."
        elif event == "errorOccurred":
            payload["error"] = {"name": "Error", "message": "example failure"}
        elif event == "sessionEnd":
            payload["reason"] = "user_exit"
        payload.update(extra)
        return payload

    def emit(self, event, **extra):
        raw = json.dumps(self.payload_for(event, **extra), ensure_ascii=False)
        logger.record_event(event, raw, self.root)
        return raw

    def parquet_records(self):
        return [
            record
            for path in sorted(self.root.glob("session-*/*.parquet"))
            for record in pq.read_table(path).to_pylist()
        ]

    def test_normalization_preserves_raw_payload_and_native_identifiers(self):
        payload = self.payload_for(
            "postToolUse",
            toolArgs={
                "command": "echo first\necho second",
                "nested": [1, True],
            },
            agentId="child-agent",
            messageId="native-message",
            toolCallId="native-call",
            futureField={"unknown": True},
        )
        raw = json.dumps(payload, indent=2) + "\n"
        record = logger.build_record("postToolUse", raw)
        self.assertEqual(record["rawPayload"], raw)
        self.assertEqual(json.loads(record["toolArgs"]), payload["toolArgs"])
        self.assertEqual(record["messageId"], "native-message")
        self.assertEqual(record["toolCallId"], "native-call")
        self.assertEqual(record["agentId"], "child-agent")
        self.assertIsNone(record["turnId"])

    def test_response_sizes_and_string_arguments(self):
        for text in ("", "hello", "\u00e9\U0001f600\n"):
            with self.subTest(text=text):
                payload = self.payload_for(
                    "postToolUse",
                    toolArgs='{"command":"echo hello"}',
                    toolResult={
                        "resultType": "success",
                        "textResultForLlm": text,
                    },
                )
                record = logger.build_record(
                    "postToolUse", json.dumps(payload)
                )
                self.assertEqual(
                    record["responseBytes"], len(text.encode("utf-8"))
                )
                self.assertEqual(
                    json.loads(record["toolArgs"]), payload["toolArgs"]
                )
                self.assertEqual(json.loads(record["rawPayload"]), payload)

    def test_invalid_inputs_are_reported(self):
        cases = [
            ("sessionStart", []),
            ("sessionStart", {}),
            ("sessionStart", dict(self.payload, timestamp=True)),
            ("sessionStart", dict(self.payload, messageId=123)),
            ("postToolUse", self.payload),
            ("postToolUse", self.payload_for("postToolUse", toolResult={})),
            ("postToolUseFailure", self.payload),
            ("userPromptSubmitted", self.payload),
            ("errorOccurred", self.payload),
            ("preToolUse", self.payload),
        ]
        for event, payload in cases:
            with self.subTest(event=event, payload=payload):
                with self.assertRaises(ValueError):
                    logger.build_record(event, json.dumps(payload))

    def test_all_events_round_trip_into_one_session_parquet(self):
        raws = [self.emit(event) for event in logger.EVENTS]
        self.assertEqual(list(self.root.rglob("*.jsonl")), [])
        self.assertEqual(
            [path.name for path in self.root.rglob("*.parquet")],
            ["session-demo-session.parquet"],
        )
        records = self.parquet_records()
        self.assertEqual(
            [row["event"] for row in records], list(logger.EVENTS)
        )
        self.assertEqual([row["rawPayload"] for row in records], raws)
        self.assertEqual(
            len({row["eventId"] for row in records}), len(records)
        )
        self.assertEqual(
            records[1]["prompt"],
            self.payload_for("userPromptSubmitted")["prompt"],
        )
        self.assertEqual(records[2]["responseBytes"], 5)
        self.assertEqual(records[3]["resultType"], "failure")
        self.assertEqual(
            records[3]["error"],
            self.payload_for("postToolUseFailure")["error"],
        )
        self.assertIsNone(records[3]["responseBytes"])
        self.assertEqual(
            json.loads(records[7]["error"]),
            self.payload_for("errorOccurred")["error"],
        )

    def test_session_journal_name_and_session_isolation(self):
        self.emit("sessionStart")
        self.emit("userPromptSubmitted", sessionId="another-session")
        journals = sorted(self.root.rglob("*.jsonl"))
        self.assertEqual(
            sorted(path.name for path in journals),
            ["session-another-session.jsonl", "session-demo-session.jsonl"],
        )
        self.emit("sessionEnd")
        self.assertEqual(
            [path.name for path in self.root.rglob("*.jsonl")],
            ["session-another-session.jsonl"],
        )
        self.emit("sessionEnd", sessionId="another-session")
        self.assertEqual(len(list(self.root.rglob("*.parquet"))), 2)

    def test_session_ids_cannot_escape_log_directory(self):
        self.emit("sessionStart", sessionId="../../other")
        journals = list(self.root.rglob("*.jsonl"))
        self.assertEqual(len(journals), 1)
        self.assertEqual(journals[0].name, "session-..%2F..%2Fother.jsonl")
        self.assertEqual(journals[0].parent.parent, self.root)

    def test_resumed_sessions_update_the_same_parquet(self):
        self.emit("sessionStart")
        self.emit("sessionEnd")
        original_ids = [row["eventId"] for row in self.parquet_records()]
        self.emit("sessionStart", source="resume")
        self.emit("userPromptSubmitted")
        self.emit("sessionEnd")
        records = self.parquet_records()
        self.assertEqual(len(list(self.root.rglob("*.parquet"))), 1)
        self.assertEqual(len(records), 5)
        self.assertEqual([row["eventId"] for row in records[:2]], original_ids)
        self.assertEqual(records[2]["source"], "resume")

    def test_conversion_failure_keeps_journal_and_previous_parquet(self):
        self.emit("sessionStart")
        self.emit("sessionEnd")
        path = next(self.root.rglob("*.parquet"))
        before = path.read_bytes()
        self.emit("userPromptSubmitted")
        with patch.object(
            logger.os, "replace", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.emit("sessionEnd")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(list(self.root.rglob("*.jsonl"))), 1)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
        logger.recover(self.root)
        self.assertEqual(len(self.parquet_records()), 4)
        self.assertEqual(list(self.root.rglob("*.jsonl")), [])

    def test_retry_after_publication_does_not_duplicate_rows(self):
        self.emit("sessionStart")
        original_unlink = Path.unlink

        def fail_journal_cleanup(path, *args, **kwargs):
            if path.suffix == ".jsonl":
                raise OSError("cleanup interrupted")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_journal_cleanup):
            with self.assertRaisesRegex(OSError, "cleanup interrupted"):
                self.emit("sessionEnd")
        self.assertEqual(len(self.parquet_records()), 2)
        self.emit("sessionStart", source="resume")
        logger.recover(self.root)
        logger.recover(self.root)
        records = self.parquet_records()
        self.assertEqual(len(records), 3)
        self.assertEqual(len({row["eventId"] for row in records}), 3)
        self.assertEqual(list(self.root.rglob("*.jsonl")), [])

    def test_missing_dependency_preserves_capture_until_recovery(self):
        self.emit("sessionStart")
        with patch.object(
            logger,
            "parquet_modules",
            side_effect=RuntimeError("PyArrow unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "PyArrow unavailable"):
                self.emit("sessionEnd")
            self.emit("userPromptSubmitted")
        self.assertEqual(len(list(self.root.rglob("*.jsonl"))), 2)
        logger.recover(self.root)
        self.assertEqual(
            [row["event"] for row in self.parquet_records()],
            ["sessionStart", "sessionEnd", "userPromptSubmitted"],
        )
        self.assertEqual(list(self.root.rglob("*.jsonl")), [])

    def test_recovery_handles_partial_journal_without_losing_source(self):
        self.emit("sessionStart")
        journal = next(self.root.rglob("*.jsonl"))
        with journal.open("a", encoding="utf-8") as output:
            output.write('{"interrupted":')
        with self.assertRaises(json.JSONDecodeError):
            logger.recover(self.root)
        self.assertEqual(len(list(self.root.rglob("*.jsonl"))), 1)
        self.assertEqual(list(self.root.rglob("*.parquet")), [])

    def test_multiple_parquet_batches_and_resume_boundaries(self):
        for index in range(1030):
            self.emit("userPromptSubmitted", prompt=str(index))
        self.emit("sessionEnd")
        self.emit("userPromptSubmitted", prompt="resumed")
        self.emit("sessionEnd")
        self.assertEqual(len(self.parquet_records()), 1033)
        self.assertEqual(self.parquet_records()[-2]["prompt"], "resumed")

    def test_concurrent_processes_and_recovery_preserve_all_events(self):
        code = dedent(
            """\
            import importlib.util
            import sys
            from pathlib import Path

            spec = importlib.util.spec_from_file_location(
                "logger", sys.argv[1]
            )
            logger = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(logger)
            logger.record_event(
                "userPromptSubmitted",
                sys.stdin.read(),
                Path(sys.argv[2]),
            )
            """
        )

        def invoke(index):
            raw = json.dumps(
                self.payload_for(
                    "userPromptSubmitted", prompt=f"{index}:" + "x" * 100000
                )
            )
            subprocess.run(
                [sys.executable, "-c", code, str(SCRIPT), str(self.root)],
                input=raw,
                text=True,
                check=True,
                capture_output=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(invoke, index) for index in range(16)]
            executor.submit(logger.recover, self.root).result()
            for future in futures:
                future.result()
        logger.recover(self.root)
        records = self.parquet_records()
        self.assertEqual(
            sorted(int(row["prompt"].split(":")[0]) for row in records),
            list(range(16)),
        )
        self.assertTrue(
            all(row["prompt"].endswith("x" * 100000) for row in records)
        )
        self.assertEqual(len(list(self.root.rglob("*.parquet"))), 1)

    def test_capture_continues_during_parquet_conversion(self):
        self.emit("sessionStart")
        entered = threading.Event()
        release = threading.Event()
        original_convert = logger.convert_journal

        def delayed_conversion(journal):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("conversion test was not released")
            return original_convert(journal)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            with patch.object(logger, "convert_journal", delayed_conversion):
                conversion = executor.submit(logger.recover, self.root)
                try:
                    self.assertTrue(entered.wait(5))
                    capture = executor.submit(self.emit, "userPromptSubmitted")
                    capture.result(timeout=5)
                finally:
                    release.set()
                conversion.result(timeout=5)
        self.assertEqual(len(self.parquet_records()), 1)
        self.assertEqual(len(list(self.root.rglob("*.jsonl"))), 1)
        logger.recover(self.root)
        self.assertEqual(len(self.parquet_records()), 2)

    def test_main_reports_parse_write_and_conversion_errors(self):
        for content, error in (
            ("not json", None),
            (json.dumps(self.payload), OSError("disk full")),
            (json.dumps(self.payload), RuntimeError("PyArrow unavailable")),
        ):
            with self.subTest(content=content, error=error):
                with patch.object(sys, "argv", [str(SCRIPT), "sessionStart"]):
                    with (
                        patch("sys.stdin") as stdin,
                        patch("sys.stderr") as stderr,
                    ):
                        stdin.read.return_value = content
                        with patch.object(
                            logger,
                            "record_event",
                            side_effect=error,
                            wraps=logger.record_event,
                        ):
                            self.assertEqual(logger.main(), 1)
                        self.assertIn(
                            "session analytics failed:",
                            stderr.write.call_args_list[0].args[0],
                        )

    def test_configured_commands_read_stdin_without_stdout(self):
        config = json.loads(
            (ROOT / ".github" / "hooks" / "logging.json").read_text()
        )
        self.assertEqual(config["version"], 1)
        self.assertEqual(set(config["hooks"]), set(logger.EVENTS))
        script = self.root / "hooks" / "logging" / SCRIPT.name
        script.parent.mkdir(parents=True)
        script.write_bytes(SCRIPT.read_bytes())
        for event, hooks in config["hooks"].items():
            for hook in hooks:
                windows = os.name == "nt"
                command = shlex.split(
                    hook["powershell" if windows else "bash"]
                )
                self.assertEqual(
                    command[0],
                    ".venv/Scripts/python.exe"
                    if windows
                    else ".venv/bin/python",
                )
                result = subprocess.run(
                    [sys.executable, *command[1:]],
                    cwd=self.root / hook["cwd"],
                    input=json.dumps(self.payload_for(event)),
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")
        paths = list((self.root / "logs" / "sessions").rglob("*.parquet"))
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            [row["event"] for row in pq.read_table(paths[0]).to_pylist()],
            list(logger.EVENTS),
        )
        result = subprocess.run(
            [sys.executable, str(script), "recover"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
