import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

from copilot.rpc import (
    CurrentToolMetadata,
    PermissionDecisionApproveOnce,
    PermissionDecisionReject,
    ToolsGetCurrentMetadataResult,
)
from copilot.session import CopilotSession
from copilot.session_events import (
    PermissionRequestRead,
    PermissionRequestShell,
    SessionEvent,
)
from pydantic import ValidationError

from agents.one_shot import agent
from agents.one_shot.models import MathSolution
from agents.one_shot.reporting import AgentResult, RunDiagnostics

EXAMPLE = Path(__file__).parent / "prompts" / "math.yaml"
VALID_SOLUTION = {
    "answer": "x = 2",
    "results": [{"name": "x", "value": 2.0}],
    "method": ["Divide both sides of 2x = 4 by 2."],
    "verification": ["Substitution gives 2 * 2 = 4."],
}


class PromptTests(unittest.TestCase):
    def test_example_defaults_and_parameter_content_are_preserved(self):
        definition = agent.load_prompt(EXAMPLE)
        problem = 'A problem with {braces}, "quotes", and\nmultiple lines.'
        prompt, system = definition.render({"problem": problem})
        self.assertIn(problem, prompt)
        self.assertIn("6 decimal places", prompt)
        self.assertIn("analytical math problems", system)
        default_prompt, _ = definition.render({})
        self.assertIn("Lagrange", default_prompt)

    def test_format_specs_conversions_nested_width_and_escaped_braces(self):
        definition = agent.PromptDefinition.model_validate(
            {
                "parameters": {
                    "count": {"type": "integer"},
                    "width": {"type": "integer", "default": 4},
                    "amount": {"type": "number"},
                    "label": {"type": "string"},
                    "enabled": {"type": "boolean"},
                },
                "prompt": (
                    '{{"count": {count:0{width}d}}} {amount:.2f} '
                    "{label!r} {enabled}"
                ),
                "system_prompt": "Handle {label}.",
            }
        )
        prompt, system = definition.render(
            {"count": 7, "amount": 1.234, "label": "demo", "enabled": True}
        )
        self.assertEqual(prompt, "{\"count\": 0007} 1.23 'demo' True")
        self.assertEqual(system, "Handle demo.")

    def test_missing_unknown_and_wrongly_typed_parameters_are_rejected(self):
        definition = agent.load_prompt(EXAMPLE)
        definition.parameters["problem"] = agent.Parameter(type="string")
        for parameters in (
            {},
            {"problem": "hello", "unknown": 1},
            {"problem": 1},
            {"problem": "hello", "digits": "3"},
            {"problem": "hello", "digits": True},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    definition.render(parameters)

    def test_invalid_definitions_fail_before_rendering(self):
        for definition in (
            {"prompt": "{missing}"},
            {"prompt": "hello", "system_prompt": "{missing}"},
            {
                "prompt": "{value.attr}",
                "parameters": {"value": {"type": "string"}},
            },
            {
                "prompt": "{value[0]}",
                "parameters": {"value": {"type": "string"}},
            },
            {"prompt": "{}"},
            {"prompt": "{broken"},
            {
                "prompt": "hello",
                "parameters": {"_hidden": {"type": "integer"}},
            },
            {
                "prompt": "hello",
                "parameters": {"model_dump": {"type": "string"}},
            },
            {"prompt": "hello", "parameters": {"value": {"type": "object"}}},
            {"prompt": "hello", "typo": True},
        ):
            with self.subTest(definition=definition):
                with self.assertRaises(ValidationError):
                    agent.PromptDefinition.model_validate(definition)

    def test_invalid_defaults_and_format_specs_are_rejected(self):
        definition = agent.PromptDefinition.model_validate(
            {
                "prompt": "{count:d}",
                "parameters": {"count": {"type": "integer", "default": "3"}},
            }
        )
        with self.assertRaises(ValidationError):
            definition.render({})
        definition = agent.PromptDefinition.model_validate(
            {
                "prompt": "{text:d}",
                "parameters": {"text": {"type": "string"}},
            }
        )
        with self.assertRaises(ValueError):
            definition.render({"text": "hello"})

    def test_cli_parameter_types_duplicates_and_equals_in_text(self):
        definition = agent.load_prompt(EXAMPLE)
        parameters = agent.parse_parameters(
            ["problem=yes: {literal}=value", "digits=2"], definition
        )
        self.assertEqual(
            parameters, {"problem": "yes: {literal}=value", "digits": 2}
        )
        for assignments in (
            ["unknown=1"],
            ["problem=one", "problem=two"],
            ["problem"],
        ):
            with self.subTest(assignments=assignments):
                with self.assertRaises(ValueError):
                    agent.parse_parameters(assignments, definition)

    def test_response_model_import_requires_a_pydantic_class(self):
        self.assertIs(
            agent.load_response_model("agents.one_shot.models:MathSolution"),
            MathSolution,
        )
        for reference in ("MathSolution", "json:loads", "json:Missing"):
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    agent.load_response_model(reference)

    def test_invalid_yaml_is_reported_by_cli_without_starting_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.yaml"
            path.write_text("- not a prompt definition", encoding="utf-8")
            arguments = [
                "agent",
                str(path),
                "--response-model",
                "agents.one_shot.models:MathSolution",
            ]
            stderr = io.StringIO()
            with patch.object(agent.sys, "argv", arguments):
                with patch.object(agent, "CopilotClient") as client:
                    with contextlib.redirect_stderr(stderr):
                        self.assertEqual(agent.main(), 1)
                    client.assert_not_called()
            self.assertIn("one-shot agent failed:", stderr.getvalue())

    def test_cli_prints_host_diagnostics_outside_the_model_response(self):
        result = AgentResult(
            response=MathSolution.model_validate(VALID_SOLUTION),
            diagnostics=RunDiagnostics(
                session_id="runtime-session", tool_failures=[]
            ),
        )
        stdout = io.StringIO()
        with (
            patch.object(
                agent.sys,
                "argv",
                [
                    "agent",
                    str(EXAMPLE),
                    "--response-model",
                    "agents.one_shot.models:MathSolution",
                ],
            ),
            patch.object(
                agent, "run_agent", new=AsyncMock(return_value=result)
            ),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(agent.main(), 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "response": VALID_SOLUTION,
                "diagnostics": {
                    "session_id": "runtime-session",
                    "tool_failures": [],
                },
            },
        )

    def test_cli_progress_uses_stderr_unless_quiet(self):
        result = AgentResult(
            response=MathSolution.model_validate(VALID_SOLUTION),
            diagnostics=RunDiagnostics(
                session_id="runtime-session", tool_failures=[]
            ),
        )
        for flags in (
            [],
            ["--quiet"],
            ["--output", "json"],
            ["--quiet", "--output", "pretty"],
        ):
            stderr = io.StringIO()
            stdout = io.StringIO()
            with (
                patch.object(
                    agent.sys,
                    "argv",
                    [
                        "agent",
                        str(EXAMPLE),
                        "--response-model",
                        "agents.one_shot.models:MathSolution",
                        *flags,
                    ],
                ),
                patch.object(
                    agent, "run_agent", new=AsyncMock(return_value=result)
                ) as run,
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(agent.main(), 0)
            self.assertIs(
                run.await_args.kwargs["progress"],
                None if "--quiet" in flags else stderr,
            )
            if "pretty" in flags:
                self.assertIn("Response", stdout.getvalue())
                self.assertIn("Diagnostics", stdout.getvalue())
            else:
                self.assertEqual(
                    json.loads(stdout.getvalue()),
                    json.loads(result.model_dump_json()),
                )

    def test_sandbox_settings_map_to_native_sdk_wire_fields(self):
        definition = agent.PromptDefinition.model_validate(
            {
                "prompt": "hello",
                "model": "example-model",
                "tools": ["view", "rg"],
                "sandbox": {
                    "enabled": True,
                    "allow_bypass": False,
                    "add_current_working_directory": False,
                    "allow_dev_tool_access": False,
                    "sandbox_mcp_servers": True,
                    "sandbox_lsp_servers": True,
                    "user_policy": {
                        "filesystem": {
                            "readonly_paths": ["/input"],
                            "readwrite_paths": ["/output"],
                            "denied_paths": ["/private"],
                            "clear_policy_on_exit": True,
                        },
                        "network": {
                            "allowed_hosts": ["api.github.com"],
                            "blocked_hosts": ["blocked.example"],
                            "allow_outbound": True,
                            "allow_local_network": False,
                        },
                    },
                },
            }
        )
        self.assertEqual(
            definition.sandbox.to_sdk().to_dict(),
            {
                "enabled": True,
                "allowBypass": False,
                "addCurrentWorkingDirectory": False,
                "allowDevToolAccess": False,
                "sandboxMcpServers": True,
                "sandboxLspServers": True,
                "userPolicy": {
                    "filesystem": {
                        "readonlyPaths": ["/input"],
                        "readwritePaths": ["/output"],
                        "deniedPaths": ["/private"],
                        "clearPolicyOnExit": True,
                    },
                    "network": {
                        "allowedHosts": ["api.github.com"],
                        "blockedHosts": ["blocked.example"],
                        "allowOutbound": True,
                        "allowLocalNetwork": False,
                    },
                },
            },
        )

    def test_invalid_session_settings_are_rejected(self):
        for settings in (
            {"tools": None},
            {"tools": "view"},
            {"tools": [""]},
            {"tools": [" view"]},
            {"tools": ["view", "view"]},
            {"tools": ["*"]},
            {"tools": [123]},
            {"sandbox": {}},
            {"sandbox": {"enabled": "true"}},
            {"sandbox": {"enabled": True, "allowBypass": True}},
            {"sandbox": {"enabled": True, "allow_bypass": "false"}},
            {"sandbox": {"enabled": True, "typo": False}},
            {
                "sandbox": {
                    "enabled": True,
                    "user_policy": {"network": {"allow_outbound": "false"}},
                }
            },
            {
                "sandbox": {
                    "enabled": True,
                    "user_policy": {
                        "filesystem": {"denied_paths": "/private"}
                    },
                }
            },
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValidationError):
                    agent.PromptDefinition.model_validate(
                        {"prompt": "hello", **settings}
                    )


class PermissionTests(unittest.TestCase):
    def test_regular_permissions_are_approved_noninteractively(self):
        request = PermissionRequestRead(intention="Read input", path="/input")
        result = agent.permission_handler(False)(
            request, {"session_id": "demo"}
        )
        self.assertIsInstance(result, PermissionDecisionApproveOnce)
        self.assertIsNot(result.approved_interactively, True)

    def test_managed_approval_is_rejected_with_bypass_enabled(self):
        request = PermissionRequestRead(
            intention="Read input",
            path="/input",
            managed_approval_required=True,
        )
        result = agent.permission_handler(True)(
            request, {"session_id": "demo"}
        )
        self.assertIsInstance(result, PermissionDecisionReject)
        self.assertIn("managed human approval", result.feedback)

    def test_bypass_requires_explicit_opt_in(self):
        request = PermissionRequestRead(
            intention="Read input", path="/input", request_sandbox_bypass=True
        )
        self.assertIsInstance(
            agent.permission_handler(False)(request, {"session_id": "demo"}),
            PermissionDecisionReject,
        )
        self.assertIsInstance(
            agent.permission_handler(True)(request, {"session_id": "demo"}),
            PermissionDecisionApproveOnce,
        )
        shell = PermissionRequestShell(
            can_offer_session_approval=False,
            commands=[],
            full_command_text="echo probe",
            has_write_file_redirection=False,
            intention="Run probe",
            possible_paths=[],
            possible_urls=[],
            request_sandbox_permissive=True,
        )
        self.assertIsInstance(
            agent.permission_handler(False)(shell, {"session_id": "demo"}),
            PermissionDecisionReject,
        )


class OneShotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.definition = agent.load_prompt(EXAMPLE)
        self.session = CopilotSession("demo-session", Mock())
        self.session.send_and_wait = AsyncMock(
            return_value=self.response(json.dumps(VALID_SOLUTION))
        )
        self.session.rpc.tools.initialize_and_validate = AsyncMock()
        self.session.rpc.tools.get_current_metadata = AsyncMock(
            return_value=ToolsGetCurrentMetadataResult(
                tools=[
                    CurrentToolMetadata(
                        description=tool.description, name=tool.name
                    )
                    for tool in agent.MATH_TOOLS
                ]
            )
        )
        self.session.disconnect = AsyncMock()
        self.client = MagicMock()
        self.client.__aenter__.return_value = self.client
        self.client.create_session = AsyncMock(return_value=self.session)
        patcher = patch.object(
            agent, "CopilotClient", return_value=self.client
        )
        self.factory = patcher.start()
        self.addCleanup(patcher.stop)

    def response(self, content):
        return SessionEvent.from_dict(
            {
                "id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "assistant.message",
                "data": {"messageId": str(uuid4()), "content": content},
            }
        )

    async def test_native_typed_sdk_returns_model_and_cleans_up(self):
        self.definition.model = "example-model"
        result = await agent.run_agent(
            self.definition,
            {"problem": "Solve 2x = 4."},
            MathSolution,
            timeout=30,
        )
        self.assertIsInstance(result, AgentResult)
        self.assertIsInstance(result.response, MathSolution)
        self.assertEqual(result.response.answer, "x = 2")
        self.assertEqual(result.diagnostics.session_id, "demo-session")
        self.assertEqual(result.diagnostics.tool_failures, [])
        self.factory.assert_called_once_with()
        self.client.create_session.assert_awaited_once()
        options = self.client.create_session.await_args.kwargs
        self.assertEqual(
            options["available_tools"],
            [tool.name for tool in agent.MATH_TOOLS],
        )
        self.assertIs(options["tools"], agent.MATH_TOOLS)
        self.assertIsNone(options["sandbox_config"])
        self.assertTrue(callable(options["on_permission_request"]))
        self.assertFalse(options["enable_config_discovery"])
        self.assertTrue(options["enable_file_hooks"])
        self.assertFalse(options["enable_skills"])
        self.assertTrue(options["skip_custom_instructions"])
        self.assertEqual(options["model"], "example-model")
        self.assertFalse(options["streaming"])
        self.assertIn(
            "analytical math problems", options["system_message"]["content"]
        )
        self.session.send_and_wait.assert_awaited_once()
        call = self.session.send_and_wait.await_args
        self.assertIn("Solve 2x = 4.", call.args[0])
        self.assertIs(call.kwargs["response_schema"], MathSolution)
        self.assertEqual(call.kwargs["timeout"], 30)
        self.session.disconnect.assert_awaited_once()
        self.client.__aexit__.assert_awaited_once()

    def event(self, event_type, data):
        return SessionEvent.from_dict(
            {
                "id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "data": data,
            }
        )

    async def test_progress_arrives_during_inference_and_keeps_diagnostics(
        self,
    ):
        progress = io.StringIO()
        stdout = io.StringIO()

        async def respond(*_args, **_kwargs):
            self.session._dispatch_event(
                self.event(
                    "assistant.message_delta",
                    {"messageId": "live", "deltaContent": "Checking a move."},
                )
            )
            self.assertIn("Checking a move.", progress.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            await self.dispatch_hook(
                "postToolUseFailure", self.failure_payload()
            )
            return self.response(json.dumps(VALID_SOLUTION))

        self.session.send_and_wait.side_effect = respond
        with contextlib.redirect_stdout(stdout):
            result = await agent.run_agent(
                self.definition, {}, MathSolution, progress=progress
            )
        self.assertTrue(
            self.client.create_session.await_args.kwargs["streaming"]
        )
        self.assertEqual(result.response.model_dump(), VALID_SOLUTION)
        self.assertEqual(len(result.diagnostics.tool_failures), 1)
        self.assertIn("Session demo-session", progress.getvalue())
        self.assertTrue(progress.getvalue().endswith("\n"))
        before = progress.getvalue()
        self.session._dispatch_event(
            self.event(
                "assistant.message",
                {"messageId": "late", "content": "Do not print after return."},
            )
        )
        self.assertEqual(progress.getvalue(), before)

    async def test_progress_unsubscribes_and_closes_partial_text_on_timeout(
        self,
    ):
        progress = io.StringIO()

        async def respond(*_args, **_kwargs):
            self.session._dispatch_event(
                self.event(
                    "assistant.message_delta",
                    {"messageId": "partial", "deltaContent": "Still working"},
                )
            )
            raise TimeoutError("response timed out")

        self.session.send_and_wait.side_effect = respond
        with self.assertRaises(TimeoutError):
            await agent.run_agent(
                self.definition, {}, MathSolution, progress=progress
            )
        self.assertTrue(progress.getvalue().endswith("Still working\n"))
        before = progress.getvalue()
        self.session._dispatch_event(
            self.event(
                "assistant.message",
                {
                    "messageId": "late",
                    "content": "Do not print after timeout.",
                },
            )
        )
        self.assertEqual(progress.getvalue(), before)
        self.session.disconnect.assert_awaited_once()
        self.client.__aexit__.assert_awaited_once()

    async def dispatch_hook(self, hook_type, payload):
        options = self.client.create_session.await_args.kwargs
        self.session._register_hooks(options["hooks"])
        return await self.session._handle_hooks_invoke(hook_type, payload)

    def failure_payload(self):
        return {
            "sessionId": "demo-session",
            "timestamp": 1735689600000,
            "cwd": "/demo",
            "toolName": "solve_linear_system",
            "toolArgs": {"matrix": [[0]], "vector": [1]},
            "error": "Singular matrix",
        }

    async def test_native_hooks_capture_each_failure_without_model_help(self):
        async def respond(*_args, **_kwargs):
            payload = self.failure_payload()
            self.assertIsNone(
                await self.dispatch_hook("postToolUseFailure", payload)
            )
            self.assertIsNone(
                await self.dispatch_hook("postToolUseFailure", payload)
            )
            await self.dispatch_hook(
                "postToolUse",
                {
                    **payload,
                    "toolResult": {
                        "resultType": "success",
                        "textResultForLlm": "Recovered successfully.",
                    },
                },
            )
            return self.response(json.dumps(VALID_SOLUTION))

        self.session.send_and_wait.side_effect = respond
        result = await agent.run_agent(self.definition, {}, MathSolution)
        failures = result.diagnostics.tool_failures
        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0], failures[1])
        self.assertEqual(failures[0].tool_name, "solve_linear_system")
        self.assertEqual(
            failures[0].tool_args, {"matrix": [[0]], "vector": [1]}
        )
        self.assertEqual(failures[0].error, "Singular matrix")
        self.assertEqual(
            failures[0].timestamp,
            datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(result.response.model_dump(), VALID_SOLUTION)
        self.assertIs(
            self.session.send_and_wait.await_args.kwargs["response_schema"],
            MathSolution,
        )

    async def test_failures_are_isolated_between_runs(self):
        async def respond(*_args, **_kwargs):
            await self.dispatch_hook(
                "postToolUseFailure", self.failure_payload()
            )
            return self.response(json.dumps(VALID_SOLUTION))

        self.session.send_and_wait.side_effect = respond
        first = await agent.run_agent(self.definition, {}, MathSolution)
        self.session.send_and_wait.side_effect = None
        second = await agent.run_agent(self.definition, {}, MathSolution)
        self.assertEqual(len(first.diagnostics.tool_failures), 1)
        self.assertEqual(second.diagnostics.tool_failures, [])

    async def test_malformed_hook_fails_instead_of_reporting_no_failures(self):
        async def respond(*_args, **_kwargs):
            await self.dispatch_hook(
                "postToolUseFailure", {**self.failure_payload(), "error": 42}
            )
            return self.response(json.dumps(VALID_SOLUTION))

        self.session.send_and_wait.side_effect = respond
        with self.assertRaisesRegex(
            RuntimeError, "Tool failure hook collection failed"
        ):
            await agent.run_agent(self.definition, {}, MathSolution)
        self.session.disconnect.assert_awaited_once()
        self.client.__aexit__.assert_awaited_once()

    async def test_all_tools_are_exposed_when_default_or_explicit(self):
        for values in ({}, {"tools": "all"}):
            with self.subTest(values=values):
                definition = agent.PromptDefinition.model_validate(
                    {"prompt": "hello", **values}
                )
                await agent.run_agent(definition, {}, MathSolution)
                self.assertIsNone(
                    self.client.create_session.await_args.kwargs[
                        "available_tools"
                    ]
                )

    async def test_per_run_tools_replace_math_tools_including_empty_list(self):
        for custom_tools in ([agent.MATH_TOOLS[0]], []):
            with self.subTest(tools=custom_tools):
                self.definition.tools = [tool.name for tool in custom_tools]
                await agent.run_agent(
                    self.definition, {}, MathSolution, tools=custom_tools
                )
                options = self.client.create_session.await_args.kwargs
                self.assertIs(options["tools"], custom_tools)
                self.assertEqual(
                    options["available_tools"], self.definition.tools
                )

    async def test_explicit_allowlist_and_sandbox_reach_sdk_before_send(self):
        definition = agent.PromptDefinition.model_validate(
            {
                "prompt": "hello",
                "tools": ["view"],
                "sandbox": {"enabled": True, "allow_bypass": True},
            }
        )
        tools = self.session.rpc.tools
        tools.initialize_and_validate = AsyncMock()
        tools.get_current_metadata = AsyncMock(
            return_value=ToolsGetCurrentMetadataResult(
                tools=[
                    CurrentToolMetadata(description="Read files", name="view")
                ]
            )
        )
        await agent.run_agent(definition, {}, MathSolution)
        options = self.client.create_session.await_args.kwargs
        self.assertEqual(options["available_tools"], ["view"])
        self.assertEqual(
            options["sandbox_config"].to_dict(),
            {"enabled": True, "allowBypass": True},
        )
        request = PermissionRequestRead(
            intention="Read input", path="/input", request_sandbox_bypass=True
        )
        self.assertIsInstance(
            options["on_permission_request"](request, {"session_id": "demo"}),
            PermissionDecisionApproveOnce,
        )
        tools.initialize_and_validate.assert_awaited_once()
        tools.get_current_metadata.assert_awaited_once()
        self.session.send_and_wait.assert_awaited_once()

    async def test_unavailable_tools_fail_before_model_inference(self):
        self.definition.tools = ["unavailable"]
        self.session.rpc.tools.initialize_and_validate = AsyncMock()
        self.session.rpc.tools.get_current_metadata = AsyncMock(
            return_value=ToolsGetCurrentMetadataResult(tools=[])
        )
        with self.assertRaisesRegex(
            ValueError, "Unknown or unavailable tools"
        ):
            await agent.run_agent(self.definition, {}, MathSolution)
        self.session.send_and_wait.assert_not_awaited()
        self.session.disconnect.assert_awaited_once()

    async def test_disabled_sandbox_does_not_authorize_bypass(self):
        self.definition.sandbox = agent.SandboxSettings(
            enabled=False, allow_bypass=True
        )
        await agent.run_agent(self.definition, {}, MathSolution)
        handler = self.client.create_session.await_args.kwargs[
            "on_permission_request"
        ]
        request = PermissionRequestRead(
            intention="Read input", path="/input", request_sandbox_bypass=True
        )
        self.assertIsInstance(
            handler(request, {"session_id": "demo"}), PermissionDecisionReject
        )

    async def test_invalid_output_is_rejected_by_native_pydantic_path(self):
        for content in (
            "not JSON",
            "{}",
            json.dumps({**VALID_SOLUTION, "answer": 42}),
            json.dumps({**VALID_SOLUTION, "extra": True}),
        ):
            with self.subTest(content=content):
                self.session.send_and_wait.return_value = self.response(
                    content
                )
                with self.assertRaises(ValidationError):
                    await agent.run_agent(self.definition, {}, MathSolution)
        self.assertEqual(self.session.disconnect.await_count, 4)
        self.assertEqual(self.client.__aexit__.await_count, 4)

    async def test_timeout_propagates_and_resources_are_released(self):
        self.session.send_and_wait.side_effect = TimeoutError(
            "response timed out"
        )
        with self.assertRaisesRegex(TimeoutError, "response timed out"):
            await agent.run_agent(self.definition, {}, MathSolution)
        self.session.disconnect.assert_awaited_once()
        self.client.__aexit__.assert_awaited_once()

    async def test_invalid_inputs_never_start_the_sdk(self):
        self.definition.parameters["problem"] = agent.Parameter(type="string")
        with self.assertRaises(ValidationError):
            await agent.run_agent(self.definition, {}, MathSolution)
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    await agent.run_agent(
                        self.definition,
                        {"problem": "Solve 2x = 4."},
                        MathSolution,
                        timeout=timeout,
                    )
        with self.assertRaises(TypeError):
            await agent.run_agent(self.definition, {}, dict)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
