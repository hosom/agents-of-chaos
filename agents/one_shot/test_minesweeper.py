import contextlib
import io
import json
import re
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from copilot import ToolInvocation
from pydantic import ValidationError
from rich.console import Console

from agents.one_shot import minesweeper
from agents.one_shot.agent import load_prompt
from agents.one_shot.minesweeper import (
    CellInput,
    FlagInput,
    GameConfig,
    MinesweeperGame,
    MinesweeperSolution,
)
from agents.one_shot.reporting import (
    AgentResult,
    RunDiagnostics,
    ToolFailure,
)


class GameTests(unittest.TestCase):
    def setUp(self):
        self.game = MinesweeperGame(GameConfig(seed=7))

    def test_omitted_seed_is_generated_per_configuration_and_can_be_replayed(
        self,
    ):
        with patch.object(
            minesweeper.secrets, "randbits", side_effect=[7, 8]
        ) as generate:
            first = GameConfig()
            second = GameConfig()
            explicit = GameConfig(seed=0)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(explicit.seed, 0)
        first_game = MinesweeperGame(first)
        second_game = MinesweeperGame(second)
        self.assertNotEqual(first_game._mines, second_game._mines)
        recorded = GameConfig.model_validate_json(first.model_dump_json())
        self.assertEqual(first_game._mines, MinesweeperGame(recorded)._mines)

    def test_seeded_opening_is_safe_and_does_not_expose_mines(self):
        state = self.game.state()
        self.assertEqual(state.board[0][0], "0")
        self.assertGreater(len(self.game._revealed), 1)
        self.assertFalse(self.game._revealed & self.game._mines)
        self.assertEqual(
            state.remaining_safe_cells,
            36 - 6 - len(self.game._revealed),
        )
        for row, column in self.game._mines:
            self.assertEqual(state.board[row][column], "#")
        for row, column in self.game._revealed:
            adjacent = sum(
                (r, c) in self.game._mines
                for r in range(max(0, row - 1), min(6, row + 2))
                for c in range(max(0, column - 1), min(6, column + 2))
            )
            self.assertEqual(state.board[row][column], str(adjacent))
        repeat = MinesweeperGame(GameConfig(seed=7))
        self.assertEqual(self.game._mines, repeat._mines)
        self.assertEqual(state, repeat.state())
        self.assertNotEqual(
            self.game._mines,
            MinesweeperGame(GameConfig(seed=8))._mines,
        )

    def test_flags_are_not_oracles_and_flood_fill_respects_them(self):
        flag = FlagInput(
            row=4, column=2, reason="Test a safe-cell flag.", flagged=True
        )
        self.game.flag(flag)
        self.assertEqual(self.game.state().board[4][2], "F")
        with self.assertRaisesRegex(ValueError, "remove the flag"):
            self.game.reveal(
                CellInput(row=4, column=2, reason="Invalid reveal.")
            )
        before = len(self.game._revealed)
        self.game.reveal(
            CellInput(row=5, column=2, reason="Test a zero-region reveal.")
        )
        self.assertGreater(len(self.game._revealed), before + 1)
        self.assertEqual(self.game.state().board[4][2], "F")
        self.game.flag(flag.model_copy(update={"flagged": False}))
        self.assertEqual(self.game.state().board[4][2], "#")
        self.game.reveal(
            CellInput(row=4, column=2, reason="Reveal after removing flag.")
        )
        self.assertEqual(self.game.moves[-2].action, "unflag")
        self.assertNotEqual(self.game.state().board[4][2], "#")

    def test_revealing_every_safe_cell_wins_without_flags(self):
        for row in range(6):
            for column in range(6):
                cell = (row, column)
                if (
                    cell not in self.game._mines
                    and cell not in self.game._revealed
                ):
                    self.game.reveal(
                        CellInput(row=row, column=column, reason="Test win.")
                    )
        self.assertEqual(self.game.status, "won")
        self.assertEqual(self.game.state().remaining_safe_cells, 0)
        self.assertFalse(self.game._flags)
        with self.assertRaisesRegex(ValueError, "already won"):
            row, column = next(iter(self.game._mines))
            self.game.reveal(
                CellInput(row=row, column=column, reason="Too late.")
            )

    def test_mine_hit_loses_and_only_exposes_detonated_mine(self):
        row, column = next(iter(self.game._mines))
        state = self.game.reveal(
            CellInput(row=row, column=column, reason="Test mine hit.")
        )
        self.assertEqual(state.status, "lost")
        self.assertEqual(state.board[row][column], "*")
        self.assertEqual(sum(line.count("*") for line in state.board), 1)
        self.assertEqual(state.remaining_safe_cells, 22)
        with self.assertRaisesRegex(ValueError, "already lost"):
            self.game.flag(
                FlagInput(row=5, column=0, reason="Too late.", flagged=True)
            )

    def test_invalid_moves_do_not_mutate_the_game(self):
        for row, column in ((6, 0), (0, 6), (0, 0)):
            before = self.game.state()
            moves = list(self.game.moves)
            with self.subTest(row=row, column=column):
                with self.assertRaises(ValueError):
                    self.game.reveal(
                        CellInput(row=row, column=column, reason="Invalid.")
                    )
                self.assertEqual(self.game.state(), before)
                self.assertEqual(self.game.moves, moves)
        flag = FlagInput(row=5, column=5, reason="Test.", flagged=True)
        self.game.flag(flag)
        with self.assertRaisesRegex(ValueError, "requested flag state"):
            self.game.flag(flag)
        self.assertEqual(len(self.game.moves), 2)

    def test_invalid_configuration_and_tool_inputs_are_rejected(self):
        for config in (
            {"rows": 2},
            {"columns": 17},
            {"mines": 0},
            {"rows": 3, "columns": 3, "mines": 6},
            {"seed": True},
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValidationError):
                    GameConfig.model_validate(config)
        for data in (
            {"row": -1, "column": 0, "reason": "Invalid."},
            {"row": True, "column": 0, "reason": "Invalid."},
            {"row": 1, "column": 0, "reason": ""},
        ):
            with self.assertRaises(ValidationError):
                CellInput.model_validate(data)

    def test_result_rejects_false_outcomes_and_uses_host_state(self):
        for status in ("won", "lost"):
            with self.assertRaisesRegex(ValueError, "agent reported"):
                self.game.result(
                    MinesweeperSolution(
                        status=status,
                        summary="Unsupported outcome.",
                        deductions=["An unsupported deduction."],
                    )
                )
        result = self.game.result(
            MinesweeperSolution(
                status="stuck",
                summary="Unresolved uncertainty.",
                deductions=["No further safe move was established."],
            )
        )
        self.assertEqual(result.state, self.game.state())
        self.assertEqual(result.moves, self.game.moves)
        self.assertIsNot(result.moves, self.game.moves)


class BoardPresentationTests(unittest.TestCase):
    def render(self, renderable):
        stream = io.StringIO()
        Console(file=stream, width=100).print(renderable)
        return stream.getvalue()

    def test_board_steps_show_actual_reveals_without_exposing_hidden_mines(
        self,
    ):
        game = MinesweeperGame(GameConfig(seed=7))
        before = self.render(
            minesweeper.render_tool_board(
                "inspect_board", game.state().model_dump_json()
            )
        )
        game.reveal(CellInput(row=2, column=2, reason="A proved safe cell."))
        after = self.render(
            minesweeper.render_tool_board(
                "reveal_cell", game.state().model_dump_json()
            )
        )
        self.assertIn("22 safe cells remaining", before)
        self.assertIn("21 safe cells remaining", after)
        self.assertEqual(before.count("#") - after.count("#"), 1)
        self.assertEqual(len(re.findall(r"\bM\b", before)), 1)
        self.assertEqual(len(re.findall(r"\bM\b", after)), 1)

    def test_final_won_board_marks_mines_and_includes_move_explanations(self):
        game = MinesweeperGame(GameConfig(rows=3, columns=3, mines=5, seed=7))
        result = game.result(
            MinesweeperSolution(
                status="won",
                summary="Won from the safe opening.",
                deductions=["All four safe cells were revealed."],
            )
        )
        output = self.render(minesweeper.render_game_result(result))
        self.assertIn("0 safe cells remaining", output)
        self.assertEqual(len(re.findall(r"\bM\b", output)), 6)
        self.assertIn("Key deductions", output)
        self.assertIn("All four safe cells were revealed.", output)
        self.assertIn("Automatic safe opening", output)
        self.assertIn("seed 7", output)

    def test_malformed_board_is_reported_instead_of_inventing_a_view(self):
        output = self.render(
            minesweeper.render_tool_board("reveal_cell", "not a board")
        )
        self.assertIn("Could not render board", output)
        self.assertIn("not a board", output)
        self.assertIsNone(
            minesweeper.render_tool_board("unrelated_tool", "{}")
        )


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_tool_handlers_return_state_and_explicit_errors(
        self,
    ):
        game = MinesweeperGame(GameConfig(seed=7))
        tools = {tool.name: tool for tool in game.tools()}

        async def invoke(name, arguments):
            return await tools[name].handler(
                ToolInvocation(
                    session_id="minesweeper-test",
                    tool_call_id="test-call",
                    tool_name=name,
                    arguments=arguments,
                )
            )

        result = await invoke("inspect_board", {})
        self.assertEqual(result.result_type, "success", result.error)
        self.assertEqual(
            json.loads(result.text_result_for_llm),
            game.state().model_dump(),
        )
        for name, arguments in (
            (
                "reveal_cell",
                {"row": 0, "column": 0, "reason": "Already open."},
            ),
            ("reveal_cell", {"row": 50, "column": 0, "reason": "Outside."}),
            ("flag_cell", {"row": 1, "column": 2, "flagged": True}),
            ("inspect_board", {"cheat": True}),
        ):
            with self.subTest(name=name, arguments=arguments):
                failure = await invoke(name, arguments)
                self.assertEqual(failure.result_type, "failure")
                self.assertTrue(failure.error)
        flag = await invoke(
            "flag_cell",
            {"row": 1, "column": 2, "flagged": True, "reason": "Test flag."},
        )
        self.assertEqual(flag.result_type, "success", flag.error)
        self.assertEqual(
            json.loads(flag.text_result_for_llm)["board"][1][2], "F"
        )
        reveal = await invoke(
            "reveal_cell",
            {"row": 2, "column": 2, "reason": "Test safe reveal."},
        )
        self.assertEqual(reveal.result_type, "success", reveal.error)
        self.assertEqual(
            json.loads(reveal.text_result_for_llm)["board"][2][2], "1"
        )

    async def test_play_uses_oneshot_prompt_and_isolates_games(self):
        observed_tools = []
        progress = io.StringIO()
        diagnostics = RunDiagnostics(
            session_id="game-session",
            tool_failures=[
                ToolFailure(
                    timestamp=datetime(2026, 11, 5, tzinfo=timezone.utc),
                    tool_name="reveal_cell",
                    tool_args={"row": 0, "column": 0},
                    error="cell is already revealed",
                )
            ],
        )

        async def solve(definition, parameters, response_type, **kwargs):
            self.assertEqual(definition, load_prompt(minesweeper.PROMPT))
            self.assertEqual(parameters, {"allow_guesses": True})
            self.assertIs(response_type, MinesweeperSolution)
            self.assertEqual(kwargs["timeout"], 12)
            self.assertIs(kwargs["progress"], progress)
            self.assertIs(
                kwargs["tool_renderer"], minesweeper.render_tool_board
            )
            tools = {tool.name: tool for tool in kwargs["tools"]}
            self.assertEqual(set(definition.tools), set(tools))
            observed_tools.append(tools["inspect_board"])
            return AgentResult(
                response=MinesweeperSolution(
                    status="stuck",
                    summary="Mocked model run.",
                    deductions=["No model inference in this test."],
                ),
                diagnostics=diagnostics,
            )

        with patch.object(minesweeper, "run_agent", side_effect=solve):
            first = await minesweeper.play(
                GameConfig(seed=7),
                allow_guesses=True,
                timeout=12,
                progress=progress,
            )
            second = await minesweeper.play(
                GameConfig(seed=7),
                allow_guesses=True,
                timeout=12,
                progress=progress,
            )
        self.assertEqual(first, second)
        self.assertIs(first.diagnostics, diagnostics)
        self.assertIsInstance(first.response, minesweeper.GameResult)
        self.assertIsNot(observed_tools[0], observed_tools[1])


class CliTests(unittest.TestCase):
    def test_cli_randomizes_omitted_seeds_and_records_explicit_seed_zero(self):
        async def play(config, **_kwargs):
            game = MinesweeperGame(config)
            return AgentResult(
                response=game.result(
                    MinesweeperSolution(
                        status=(
                            "stuck"
                            if game.status == "in_progress"
                            else game.status
                        ),
                        summary="Mocked model run.",
                        deductions=["No model inference in this test."],
                    )
                ),
                diagnostics=RunDiagnostics(
                    session_id="game-session", tool_failures=[]
                ),
            )

        with patch.object(
            minesweeper.secrets, "randbits", side_effect=[7, 8]
        ) as generate:
            for flags, expected_seed in (
                ([], 7),
                ([], 8),
                (["--seed", "0"], 0),
            ):
                stdout = io.StringIO()
                with (
                    patch.object(
                        minesweeper.sys,
                        "argv",
                        ["minesweeper", "--quiet", *flags],
                    ),
                    patch.object(
                        minesweeper, "play", new=AsyncMock(side_effect=play)
                    ),
                    contextlib.redirect_stdout(stdout),
                ):
                    self.assertEqual(minesweeper.main(), 0)
                self.assertEqual(
                    json.loads(stdout.getvalue())["response"]["config"][
                        "seed"
                    ],
                    expected_seed,
                )
        self.assertEqual(generate.call_count, 2)

    def test_cli_progress_uses_stderr_unless_quiet(self):
        game = MinesweeperGame(GameConfig(seed=7))
        result = AgentResult(
            response=game.result(
                MinesweeperSolution(
                    status="stuck",
                    summary="Mocked model run.",
                    deductions=["No model inference in this test."],
                )
            ),
            diagnostics=RunDiagnostics(
                session_id="game-session", tool_failures=[]
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
                patch.object(minesweeper.sys, "argv", ["minesweeper", *flags]),
                patch.object(
                    minesweeper, "play", new=AsyncMock(return_value=result)
                ) as play,
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(minesweeper.main(), 0)
            self.assertIs(
                play.await_args.kwargs["progress"],
                None if "--quiet" in flags else stderr,
            )
            if "pretty" in flags:
                self.assertIn("safe cells remaining", stdout.getvalue())
                self.assertIn("Moves", stdout.getvalue())
                self.assertIn("Diagnostics", stdout.getvalue())
            else:
                self.assertEqual(
                    json.loads(stdout.getvalue()),
                    json.loads(result.model_dump_json()),
                )

    def test_prompt_defaults_and_cli_errors(self):
        prompt, system = load_prompt(minesweeper.PROMPT).render({})
        self.assertIn("Guessing enabled: False", prompt)
        self.assertIn("report stuck", system)
        for failure in (
            ValueError("Invalid game."),
            TimeoutError("Timed out."),
        ):
            stderr = io.StringIO()
            with (
                patch.object(minesweeper.sys, "argv", ["minesweeper"]),
                patch.object(
                    minesweeper, "play", new=AsyncMock(side_effect=failure)
                ),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(minesweeper.main(), 1)
            self.assertIn("minesweeper agent failed:", stderr.getvalue())

    def test_invalid_cli_configuration_never_starts_the_agent(self):
        stderr = io.StringIO()
        with (
            patch.object(
                minesweeper.sys, "argv", ["minesweeper", "--mines", "100"]
            ),
            patch.object(minesweeper, "run_agent") as run,
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(minesweeper.main(), 1)
        run.assert_not_called()
        self.assertIn("opening", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
