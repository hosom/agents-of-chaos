"""Host-generated run diagnostics collected through SDK hooks."""

from datetime import datetime
from typing import Generic, TypeVar

from copilot import PostToolUseFailureHookInput
from pydantic import BaseModel, ConfigDict, Field, JsonValue

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ToolFailure(ReportModel):
    timestamp: datetime
    tool_name: str = Field(min_length=1)
    tool_args: JsonValue
    error: str


class RunDiagnostics(ReportModel):
    session_id: str = Field(min_length=1)
    tool_failures: list[ToolFailure]


class AgentResult(ReportModel, Generic[ResponseT]):
    response: ResponseT
    diagnostics: RunDiagnostics


class RunReporter:
    def __init__(self) -> None:
        self._failures: list[ToolFailure] = []
        self._collection_errors: list[str] = []

    def on_post_tool_use_failure(
        self,
        payload: PostToolUseFailureHookInput,
        _context: dict[str, str],
    ) -> None:
        try:
            failure = ToolFailure(
                timestamp=payload["timestamp"],
                tool_name=payload["toolName"],
                tool_args=payload["toolArgs"],
                error=payload["error"],
            )
        except (KeyError, ValueError, TypeError) as error:
            # SDK hook errors are swallowed; reject incomplete reports.
            self._collection_errors.append(str(error))
            return
        self._failures.append(failure)

    def diagnostics(self, session_id: str) -> RunDiagnostics:
        if self._collection_errors:
            raise RuntimeError(
                "Tool failure hook collection failed: "
                + "; ".join(self._collection_errors)
            )
        return RunDiagnostics(
            session_id=session_id,
            tool_failures=list(self._failures),
        )
