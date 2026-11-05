"""Run a YAML prompt with a typed response and hook diagnostics."""

import argparse
import asyncio
import importlib
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from string import Formatter
from typing import Annotated, Literal, TextIO, TypeVar

import yaml
from copilot import CopilotClient, Tool
from copilot.rpc import (
    PermissionDecisionApproveOnce,
    PermissionDecisionReject,
    SandboxConfig,
)
from copilot.session import (
    PermissionInvocation,
    PermissionRequest,
    PermissionRequestResult,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)
from pydantic.alias_generators import to_camel

from .presentation import ToolRenderer, print_result
from .progress import ProgressPrinter
from .reporting import AgentResult, RunReporter
from .tools import MATH_TOOLS

ResponseT = TypeVar("ResponseT", bound=BaseModel)
PARAMETER_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}
NonemptyString = Annotated[str, Field(min_length=1)]


class SandboxModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=False,
    )


class SandboxFilesystem(SandboxModel):
    readonly_paths: list[NonemptyString] | None = None
    readwrite_paths: list[NonemptyString] | None = None
    denied_paths: list[NonemptyString] | None = None
    clear_policy_on_exit: bool | None = None


class SandboxNetwork(SandboxModel):
    allowed_hosts: list[NonemptyString] | None = None
    blocked_hosts: list[NonemptyString] | None = None
    allow_outbound: bool | None = None
    allow_local_network: bool | None = None


class SandboxPolicy(SandboxModel):
    filesystem: SandboxFilesystem | None = None
    network: SandboxNetwork | None = None


class SandboxSettings(SandboxModel):
    enabled: bool
    allow_bypass: bool = False
    add_current_working_directory: bool | None = None
    allow_dev_tool_access: bool | None = None
    sandbox_mcp_servers: bool | None = None
    sandbox_lsp_servers: bool | None = None
    user_policy: SandboxPolicy | None = None

    def to_sdk(self) -> SandboxConfig:
        return SandboxConfig.from_dict(
            self.model_dump(by_alias=True, exclude_none=True)
        )


def permission_handler(
    allow_bypass: bool,
) -> Callable[
    [PermissionRequest, PermissionInvocation], PermissionRequestResult
]:
    def decide(
        request: PermissionRequest, _invocation: PermissionInvocation
    ) -> PermissionRequestResult:
        if getattr(request, "managed_approval_required", False):
            return PermissionDecisionReject(
                feedback=(
                    "This one-shot agent cannot provide managed "
                    "human approval."
                )
            )
        bypass = getattr(request, "request_sandbox_bypass", False) or getattr(
            request, "request_sandbox_permissive", False
        )
        if bypass and not allow_bypass:
            return PermissionDecisionReject(
                feedback="Sandbox bypass is disabled for this one-shot agent."
            )
        return PermissionDecisionApproveOnce()

    return decide


def template_fields(template: str) -> set[str]:
    fields = set()
    for _, name, format_spec, _ in Formatter().parse(template):
        if name is not None:
            if not name.isidentifier():
                raise ValueError(
                    "Template fields must be named parameters, "
                    "without attribute or index lookups"
                )
            fields.add(name)
            if format_spec:
                fields.update(template_fields(format_spec))
    return fields


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["string", "integer", "number", "boolean"]
    description: str = ""
    default: str | int | float | bool | None = None


class PromptDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    parameters: dict[str, Parameter] = Field(default_factory=dict)
    model: str | None = Field(default=None, min_length=1)
    tools: Literal["all"] | list[NonemptyString] = "all"
    sandbox: SandboxSettings | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> "PromptDefinition":
        if isinstance(self.tools, list):
            if len(self.tools) != len(set(self.tools)):
                raise ValueError("Tool allowlist must not contain duplicates")
            for name in self.tools:
                if name != name.strip() or "*" in name:
                    raise ValueError(
                        "Tool allowlists require exact names; use tools: all "
                        "to expose every tool"
                    )
        for name in self.parameters:
            if (
                not name.isidentifier()
                or name.startswith("_")
                or hasattr(BaseModel, name)
            ):
                raise ValueError(f"Invalid parameter name: {name}")
        referenced = template_fields(self.prompt)
        if self.system_prompt is not None:
            referenced.update(template_fields(self.system_prompt))
        unknown = referenced - self.parameters.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Undeclared template parameters: {names}")
        return self

    def render(
        self, parameters: Mapping[str, object]
    ) -> tuple[str, str | None]:
        fields = {}
        for name, parameter in self.parameters.items():
            default = (
                parameter.default
                if "default" in parameter.model_fields_set
                else ...
            )
            fields[name] = (
                PARAMETER_TYPES[parameter.type],
                Field(default=default, description=parameter.description),
            )
        parameter_model = create_model(
            "PromptParameters",
            __config__=ConfigDict(
                extra="forbid",
                strict=True,
                validate_default=True,
                allow_inf_nan=False,
            ),
            **fields,
        )
        values = parameter_model.model_validate(dict(parameters)).model_dump()
        prompt = self.prompt.format_map(values)
        system_prompt = (
            self.system_prompt.format_map(values)
            if self.system_prompt is not None
            else None
        )
        return prompt, system_prompt


def load_prompt(path: Path) -> PromptDefinition:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptDefinition.model_validate(content)


def load_response_model(reference: str) -> type[BaseModel]:
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError("Response model must use module:Class syntax")
    response_type = getattr(importlib.import_module(module), name, None)
    if not isinstance(response_type, type) or not issubclass(
        response_type, BaseModel
    ):
        raise ValueError(f"{reference} is not a Pydantic BaseModel subclass")
    return response_type


def parse_parameters(
    assignments: list[str], definition: PromptDefinition
) -> dict[str, object]:
    parameters = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or name not in definition.parameters:
            raise ValueError(
                f"Expected a declared parameter as NAME=VALUE: {assignment}"
            )
        if name in parameters:
            raise ValueError(f"Duplicate parameter: {name}")
        parameters[name] = (
            value
            if definition.parameters[name].type == "string"
            else yaml.safe_load(value)
        )
    return parameters


async def run_agent(
    definition: PromptDefinition,
    parameters: Mapping[str, object],
    response_type: type[ResponseT],
    *,
    timeout: float = 120.0,
    tools: list[Tool] | None = None,
    progress: TextIO | None = None,
    tool_renderer: ToolRenderer | None = None,
) -> AgentResult[ResponseT]:
    if not isinstance(response_type, type) or not issubclass(
        response_type, BaseModel
    ):
        raise TypeError("response_type must be a Pydantic BaseModel subclass")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive, finite number")
    prompt, system_prompt = definition.render(parameters)
    sandbox = definition.sandbox
    available_tools = None if definition.tools == "all" else definition.tools
    allow_bypass = bool(sandbox and sandbox.enabled and sandbox.allow_bypass)
    reporter = RunReporter()
    display = (
        ProgressPrinter(progress, tool_renderer=tool_renderer)
        if progress is not None
        else None
    )
    if display is not None:
        display.line("Connecting to Copilot...")
    async with CopilotClient() as client:
        async with await client.create_session(
            model=definition.model,
            streaming=display is not None,
            tools=MATH_TOOLS if tools is None else tools,
            available_tools=available_tools,
            sandbox_config=sandbox.to_sdk() if sandbox is not None else None,
            on_permission_request=permission_handler(
                allow_bypass=allow_bypass
            ),
            hooks={
                "on_post_tool_use_failure": reporter.on_post_tool_use_failure
            },
            enable_config_discovery=False,
            enable_file_hooks=True,
            enable_skills=False,
            skip_custom_instructions=True,
            system_message=(
                {"mode": "append", "content": system_prompt}
                if system_prompt is not None
                else None
            ),
        ) as session:
            if display is not None:
                display.line(f"Session {session.session_id}", style="dim cyan")
            unsubscribe = session.on(display.on_event) if display else None
            try:
                if isinstance(definition.tools, list) and definition.tools:
                    await session.rpc.tools.initialize_and_validate()
                    metadata = await session.rpc.tools.get_current_metadata()
                    if metadata.tools is None:
                        raise RuntimeError(
                            "Runtime did not provide initialized tool metadata"
                        )
                    offered = {
                        name
                        for tool in metadata.tools
                        for name in (tool.name, tool.namespaced_name)
                        if name is not None
                    }
                    missing = set(definition.tools) - offered
                    if missing:
                        names = ", ".join(sorted(missing))
                        raise ValueError(
                            f"Unknown or unavailable tools: {names}"
                        )
                response = await session.send_and_wait_typed(
                    prompt, response_type, timeout=timeout
                )
                return AgentResult(
                    response=response,
                    diagnostics=reporter.diagnostics(session.session_id),
                )
            finally:
                if unsubscribe is not None:
                    unsubscribe()
                if display is not None:
                    display.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", type=Path, help="YAML prompt definition")
    parser.add_argument(
        "--param", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument(
        "--response-model", required=True, metavar="MODULE:CLASS"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--quiet", action="store_true", help="Disable live progress on stderr."
    )
    parser.add_argument(
        "--output",
        choices=("auto", "pretty", "json"),
        default="auto",
        help="Final output: pretty on a terminal, JSON when redirected.",
    )
    args = parser.parse_args()
    try:
        definition = load_prompt(args.prompt)
        response_type = load_response_model(args.response_model)
        parameters = parse_parameters(args.param, definition)
        result = asyncio.run(
            run_agent(
                definition,
                parameters,
                response_type,
                timeout=args.timeout,
                progress=None if args.quiet else sys.stderr,
            )
        )
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        TimeoutError,
        ImportError,
        yaml.YAMLError,
    ) as error:
        print(f"one-shot agent failed: {error}", file=sys.stderr)
        return 1
    print_result(result, sys.stdout, output_format=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
