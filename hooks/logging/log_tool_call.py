"""Collect Copilot events and convert session journals to Parquet."""

import argparse
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

LOG_ROOT = Path(__file__).resolve().parents[2] / "logs" / "sessions"
PARQUET_BATCH_SIZE = 1024
EVENTS = (
    "sessionStart",
    "userPromptSubmitted",
    "postToolUse",
    "postToolUseFailure",
    "agentStop",
    "subagentStart",
    "subagentStop",
    "errorOccurred",
    "sessionEnd",
)
STRING_FIELDS = (
    "cwd",
    "agentId",
    "agentType",
    "parentAgentId",
    "messageId",
    "turnId",
    "toolCallId",
    "toolName",
    "prompt",
    "source",
    "reason",
    "stopReason",
    "agentName",
)


def build_record(event, raw_payload):
    if event not in EVENTS:
        raise ValueError(f"unsupported event: {event}")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("hook input must be a JSON object")
    if (
        not isinstance(payload.get("sessionId"), str)
        or not payload["sessionId"]
    ):
        raise ValueError("hook input must include a nonempty sessionId")

    record = {
        "schemaVersion": 1,
        "eventId": str(uuid.uuid4()),
        "loggedAt": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "sessionId": payload["sessionId"],
        "timestamp": payload.get("timestamp"),
        "rawPayload": raw_payload,
        "toolArgs": None,
        "responseBytes": None,
        "resultType": None,
        "error": None,
    }
    if (
        record["timestamp"] is not None
        and type(record["timestamp"]) is not int
    ):
        raise ValueError("timestamp must be an integer in Unix milliseconds")
    for key in STRING_FIELDS:
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        record[key] = value

    if event in ("postToolUse", "postToolUseFailure"):
        if not record["toolName"] or "toolArgs" not in payload:
            raise ValueError(
                "tool events require a nonempty toolName and toolArgs"
            )
        record["toolArgs"] = json.dumps(
            payload["toolArgs"], ensure_ascii=False
        )
    if event == "postToolUse":
        result = payload.get("toolResult")
        if not isinstance(result, dict) or not isinstance(
            result.get("textResultForLlm"), str
        ):
            raise ValueError(
                "postToolUse requires toolResult.textResultForLlm"
            )
        result_type = result.get("resultType")
        if result_type is not None and not isinstance(result_type, str):
            raise ValueError("toolResult.resultType must be a string")
        record["resultType"] = result_type
        record["responseBytes"] = len(
            result["textResultForLlm"].encode("utf-8")
        )
    elif event == "postToolUseFailure":
        if not isinstance(payload.get("error"), str):
            raise ValueError("postToolUseFailure requires an error string")
        record["resultType"] = "failure"
        record["error"] = payload["error"]
    elif event == "userPromptSubmitted" and not isinstance(
        payload.get("prompt"), str
    ):
        raise ValueError("userPromptSubmitted requires a prompt string")
    elif event == "errorOccurred":
        if not isinstance(payload.get("error"), dict):
            raise ValueError("errorOccurred requires an error object")
        record["error"] = json.dumps(payload["error"], ensure_ascii=False)

    return record


@contextmanager
def session_lock(directory, name=".lock"):
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / name, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b") as lock:
        if os.name == "nt":
            import msvcrt

            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_record(path, record):
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    data = line.encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if not written:
                raise OSError("incomplete journal write")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parquet_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        if error.name != "pyarrow":
            raise
        raise RuntimeError(
            "Parquet conversion requires PyArrow; "
            "install hooks/logging/requirements.txt"
        ) from error
    return pa, pq


def parquet_schema(pa):
    return pa.schema(
        [
            pa.field("schemaVersion", pa.int32(), nullable=False),
            pa.field("eventId", pa.string(), nullable=False),
            pa.field("event", pa.string(), nullable=False),
            pa.field("sessionId", pa.string(), nullable=False),
            pa.field("loggedAt", pa.string(), nullable=False),
            pa.field("timestamp", pa.int64()),
            *[pa.field(key, pa.string()) for key in STRING_FIELDS],
            pa.field("toolArgs", pa.string()),
            pa.field("resultType", pa.string()),
            pa.field("error", pa.string()),
            pa.field("responseBytes", pa.int64()),
            pa.field("rawPayload", pa.string(), nullable=False),
        ]
    )


def journal_batches(path, pa, schema):
    with path.open(encoding="utf-8") as journal:
        records = []
        for line in journal:
            records.append(json.loads(line))
            if len(records) == PARQUET_BATCH_SIZE:
                yield pa.Table.from_pylist(records, schema=schema)
                records = []
        if records:
            yield pa.Table.from_pylist(records, schema=schema)


def verify_parquet(path, journal, pa, pq, schema, start_row=0, row_count=None):
    """Compare journal rows with a window of Parquet data."""
    parquet = pq.ParquetFile(path)
    try:
        if parquet.schema_arrow.remove_metadata() != schema:
            raise ValueError(f"Parquet schema mismatch: {path}")
        expected_batches = iter(journal_batches(journal, pa, schema))
        expected = next(expected_batches, None)
        position = 0
        checked = 0
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            end = position + batch.num_rows
            if end <= start_row:
                position = end
                continue
            actual = pa.Table.from_batches([batch]).replace_schema_metadata(
                None
            )
            actual = actual.slice(max(0, start_row - position))
            position = end
            while actual.num_rows and (
                row_count is None or checked < row_count
            ):
                if expected is None:
                    raise ValueError(
                        f"Parquet contains unexpected rows: {path}"
                    )
                count = min(actual.num_rows, expected.num_rows)
                if row_count is not None:
                    count = min(count, row_count - checked)
                if not actual.slice(0, count).equals(expected.slice(0, count)):
                    raise ValueError(
                        f"Parquet contents differ from journal: {path}"
                    )
                checked += count
                actual = actual.slice(count)
                expected = expected.slice(count)
                if not expected.num_rows:
                    expected = next(expected_batches, None)
        if expected is not None or (
            row_count is not None and checked != row_count
        ):
            raise ValueError(f"Parquet contents differ from journal: {path}")
    finally:
        parquet.close()


def convert_journal(journal):
    """Publish each journal once to the session's Parquet file."""
    pa, pq = parquet_modules()
    schema = parquet_schema(pa)
    destination = journal.parent / f"{journal.parent.name}.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    previous = pq.ParquetFile(destination) if destination.exists() else None
    try:
        committed = {}
        previous_rows = 0
        if previous is not None:
            if previous.schema_arrow.remove_metadata() != schema:
                raise ValueError(f"Parquet schema mismatch: {destination}")
            metadata = previous.schema_arrow.metadata or {}
            if b"analytics.journals" not in metadata:
                raise ValueError(f"Missing conversion metadata: {destination}")
            committed = json.loads(metadata[b"analytics.journals"])
            previous_rows = previous.metadata.num_rows
        if journal.name in committed:
            offset, count = committed[journal.name]
            verify_parquet(destination, journal, pa, pq, schema, offset, count)
        else:
            # Journal IDs make cleanup retries safe after publication.
            count = sum(
                table.num_rows
                for table in journal_batches(journal, pa, schema)
            )
            committed[journal.name] = [previous_rows, count]
            output_schema = schema.with_metadata(
                {b"analytics.journals": json.dumps(committed).encode("utf-8")}
            )
            try:
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                )
                with os.fdopen(descriptor, "wb") as output:
                    with pq.ParquetWriter(
                        output, output_schema, compression="zstd"
                    ) as writer:
                        if previous is not None:
                            for batch in previous.iter_batches(
                                batch_size=PARQUET_BATCH_SIZE
                            ):
                                writer.write_batch(
                                    batch.replace_schema_metadata(
                                        output_schema.metadata
                                    )
                                )
                        for table in journal_batches(journal, pa, schema):
                            writer.write_table(
                                table.replace_schema_metadata(
                                    output_schema.metadata
                                )
                            )
                    output.flush()
                    os.fsync(output.fileno())
                verify_parquet(
                    temporary, journal, pa, pq, schema, previous_rows, count
                )
                if previous is not None:
                    previous.close()
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        if previous is not None:
            previous.close()
    journal.unlink()
    return destination


def convert_session(directory):
    with session_lock(directory, ".conversion.lock"):
        with session_lock(directory):
            pending = sorted(directory.glob("segment-*.jsonl"))
            journal = directory / f"{directory.name}.jsonl"
            if journal.exists():
                rotated = directory / f"segment-{uuid.uuid4()}.jsonl"
                journal.rename(rotated)
                pending.append(rotated)
        for journal in pending:
            convert_journal(journal)


def record_event(event, raw_payload, root=LOG_ROOT):
    record = build_record(event, raw_payload)
    session_name = f"session-{quote(record['sessionId'], safe='')}"
    directory = root / session_name
    with session_lock(directory):
        journal = directory / f"{session_name}.jsonl"
        append_record(journal, record)
    if event == "sessionEnd":
        convert_session(directory)


def recover(root=LOG_ROOT):
    if root.exists():
        for directory in sorted(root.glob("session-*")):
            if directory.is_dir():
                convert_session(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", choices=(*EVENTS, "recover"))
    args = parser.parse_args()
    try:
        if args.event == "recover":
            recover()
        else:
            record_event(args.event, sys.stdin.read())
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"session analytics failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
