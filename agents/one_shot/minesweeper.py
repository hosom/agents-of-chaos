"""Play Minesweeper through the existing one-shot agent runner."""

import argparse
import asyncio
import random
import secrets
import sys
from pathlib import Path
from typing import Literal, TextIO

from copilot import Tool, define_tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from rich import box
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from .agent import load_prompt, run_agent
from .presentation import print_result, render_content
from .reporting import AgentResult

PROMPT = Path(__file__).parent / "prompts" / "minesweeper.yaml"
Status = Literal["in_progress", "won", "lost"]


class GameModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GameConfig(GameModel):
    rows: int = Field(default=6, ge=3, le=16)
    columns: int = Field(default=6, ge=3, le=16)
    mines: int = Field(default=6, ge=1)
    seed: int = Field(default_factory=lambda: secrets.randbits(64))

    @model_validator(mode="after")
    def leave_safe_opening(self) -> "GameConfig":
        if self.mines > self.rows * self.columns - 4:
            raise ValueError("mines must leave the opening's 2x2 area safe")
        return self


class BoardState(GameModel):
    status: Status
    rows: int
    columns: int
    mines: int
    board: list[str]
    remaining_safe_cells: int


class Move(GameModel):
    action: Literal["reveal", "flag", "unflag"]
    row: int
    column: int
    reason: str = Field(min_length=1)


class InspectInput(GameModel):
    pass


class CellInput(GameModel):
    row: int = Field(ge=0, description="Zero-based row index.")
    column: int = Field(ge=0, description="Zero-based column index.")
    reason: str = Field(
        min_length=1,
        description="Explain the deduction, or explicitly label a guess.",
    )


class FlagInput(CellInput):
    flagged: bool = Field(
        description="True to mark a suspected mine, false to remove a flag."
    )


class MinesweeperSolution(GameModel):
    status: Literal["won", "lost", "stuck"]
    summary: str = Field(min_length=1)
    deductions: list[str] = Field(min_length=1)


class GameResult(GameModel):
    config: GameConfig
    analysis: MinesweeperSolution
    state: BoardState
    moves: list[Move]


class MinesweeperGame:
    def __init__(self, config: GameConfig):
        self.config = config
        candidates = [
            (row, column)
            for row in range(config.rows)
            for column in range(config.columns)
            if row > 1 or column > 1
        ]
        self._mines = set(
            random.Random(config.seed).sample(candidates, config.mines)
        )
        self._revealed: set[tuple[int, int]] = set()
        self._flags: set[tuple[int, int]] = set()
        self.status: Status = "in_progress"
        self.moves: list[Move] = []
        self.reveal(
            CellInput(
                row=0,
                column=0,
                reason="Automatic safe opening at (0, 0).",
            )
        )

    def neighbors(self, row: int, column: int):
        for r in range(max(0, row - 1), min(self.config.rows, row + 2)):
            for c in range(
                max(0, column - 1), min(self.config.columns, column + 2)
            ):
                if (r, c) != (row, column):
                    yield r, c

    def adjacent_mines(self, row: int, column: int) -> int:
        return sum(cell in self._mines for cell in self.neighbors(row, column))

    def state(self) -> BoardState:
        board = []
        for row in range(self.config.rows):
            line = ""
            for column in range(self.config.columns):
                cell = (row, column)
                if cell in self._flags:
                    line += "F"
                elif cell not in self._revealed:
                    line += "#"
                elif cell in self._mines:
                    line += "*"
                else:
                    line += str(self.adjacent_mines(row, column))
            board.append(line)
        return BoardState(
            status=self.status,
            rows=self.config.rows,
            columns=self.config.columns,
            mines=self.config.mines,
            board=board,
            remaining_safe_cells=(
                self.config.rows * self.config.columns
                - self.config.mines
                - len(self._revealed - self._mines)
            ),
        )

    def check_move(self, params: CellInput) -> tuple[int, int]:
        if self.status != "in_progress":
            raise ValueError(f"game is already {self.status}")
        if not (
            0 <= params.row < self.config.rows
            and 0 <= params.column < self.config.columns
        ):
            raise ValueError("cell is outside the board")
        cell = (params.row, params.column)
        if cell in self._revealed:
            raise ValueError("cell is already revealed")
        return cell

    def reveal(self, params: CellInput) -> BoardState:
        cell = self.check_move(params)
        if cell in self._flags:
            raise ValueError("remove the flag before revealing this cell")
        self.moves.append(Move(action="reveal", **params.model_dump()))
        if cell in self._mines:
            self._revealed.add(cell)
            self.status = "lost"
            return self.state()
        pending = [cell]
        while pending:
            current = pending.pop()
            if current in self._revealed or current in self._flags:
                continue
            self._revealed.add(current)
            if self.adjacent_mines(*current) == 0:
                pending.extend(self.neighbors(*current))
        if (
            len(self._revealed)
            == self.config.rows * self.config.columns - self.config.mines
        ):
            self.status = "won"
        return self.state()

    def flag(self, params: FlagInput) -> BoardState:
        cell = self.check_move(params)
        if (cell in self._flags) == params.flagged:
            raise ValueError("cell already has the requested flag state")
        if params.flagged:
            self._flags.add(cell)
        else:
            self._flags.remove(cell)
        self.moves.append(
            Move(
                action="flag" if params.flagged else "unflag",
                row=params.row,
                column=params.column,
                reason=params.reason,
            )
        )
        return self.state()

    def tools(self) -> list[Tool]:
        @define_tool(
            description=(
                "Inspect the visible Minesweeper board. Coordinates are "
                "zero-based. # is hidden, F is a player flag (not proof), "
                "0-8 count adjacent mines, * is a detonated mine. "
                "All eight surrounding cells count as neighbors."
            ),
            defer="never",
        )
        def inspect_board(params: InspectInput) -> BoardState:
            return self.state()

        @define_tool(
            description=(
                "Reveal one hidden, unflagged cell with a reason. "
                "Zeros automatically reveal their connected empty region "
                "and boundary clues. Hitting a mine loses the game. "
                "Returns only the visible board, never the hidden layout."
            ),
            defer="never",
        )
        def reveal_cell(params: CellInput) -> BoardState:
            return self.reveal(params)

        @define_tool(
            description=(
                "Place or remove a flag on a hidden cell with a reason. "
                "Flags are hypotheses, not confirmation of a mine. "
                "Win by revealing every safe cell, not by flagging mines."
            ),
            defer="never",
        )
        def flag_cell(params: FlagInput) -> BoardState:
            return self.flag(params)

        return [inspect_board, reveal_cell, flag_cell]

    def result(self, analysis: MinesweeperSolution) -> GameResult:
        expected = "stuck" if self.status == "in_progress" else self.status
        if analysis.status != expected:
            raise ValueError(
                f"agent reported {analysis.status}, but game is {self.status}"
            )
        return GameResult(
            config=self.config,
            analysis=analysis,
            state=self.state(),
            moves=list(self.moves),
        )


def render_board(state: BoardState) -> RenderableType:
    table = Table(
        box=box.SIMPLE,
        padding=(0, 1),
    )
    table.add_column("", style="dim", justify="right")
    for column in range(state.columns):
        table.add_column(str(column), justify="center", no_wrap=True)
    styles = {
        "0": "dim",
        "1": "bold blue",
        "2": "bold green",
        "3": "bold red",
        "4": "bold magenta",
        "5": "bold yellow",
        "6": "bold cyan",
        "7": "bold white",
        "8": "bright_black",
        "#": "dim",
        "F": "bold yellow",
        "M": "bold red",
        "*": "bold white on red",
    }
    for row, line in enumerate(state.board):
        cells = [
            "M" if state.status == "won" and cell == "#" else cell
            for cell in line
        ]
        table.add_row(
            str(row), *(Text(cell, style=styles[cell]) for cell in cells)
        )
    return Group(
        Text(
            f"{state.status.replace('_', ' ').title()} - "
            f"{state.remaining_safe_cells} safe cells remaining",
            style="bold green" if state.status == "won" else "bold cyan",
        ),
        table,
        Text(
            f"{state.mines} mines | # hidden | F flag | M mine | * detonated",
            style="dim",
        ),
    )


def render_tool_board(name: str, content: str) -> RenderableType | None:
    if name not in ("inspect_board", "reveal_cell", "flag_cell"):
        return None
    try:
        state = BoardState.model_validate_json(content)
    except ValidationError as error:
        return Group(
            Text(f"Could not render board: {error}", style="red"),
            render_content(content),
        )
    return render_board(state)


def render_game_result(result: GameResult) -> RenderableType:
    moves = Table(
        title="Moves",
        box=box.SIMPLE,
        show_lines=True,
        expand=True,
    )
    moves.add_column("#", justify="right", style="dim", no_wrap=True)
    moves.add_column("Action", style="cyan", no_wrap=True)
    moves.add_column("Cell", no_wrap=True)
    moves.add_column("Reason", ratio=1)
    for index, move in enumerate(result.moves, start=1):
        moves.add_row(
            str(index),
            move.action,
            f"({move.row}, {move.column})",
            Text(move.reason),
        )
    return Group(
        Text(result.analysis.summary),
        Text(
            f"{result.config.rows} x {result.config.columns}, "
            f"{result.config.mines} mines, seed {result.config.seed}",
            style="dim",
        ),
        render_board(result.state),
        Text("\nKey deductions", style="bold cyan"),
        *(Text(f"- {deduction}") for deduction in result.analysis.deductions),
        Text(""),
        moves,
    )


async def play(
    config: GameConfig,
    *,
    allow_guesses: bool = False,
    timeout: float = 300.0,
    progress: TextIO | None = None,
) -> AgentResult[GameResult]:
    game = MinesweeperGame(config)
    result = await run_agent(
        load_prompt(PROMPT),
        {"allow_guesses": allow_guesses},
        MinesweeperSolution,
        tools=game.tools(),
        timeout=timeout,
        progress=progress,
        tool_renderer=render_tool_board,
    )
    return AgentResult(
        response=game.result(result.response),
        diagnostics=result.diagnostics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--mines", type=int, default=6)
    parser.add_argument(
        "--seed",
        type=int,
        help="Reproduce a board layout; omitted to generate a fresh seed.",
    )
    parser.add_argument("--allow-guesses", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
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
        settings = {
            "rows": args.rows,
            "columns": args.columns,
            "mines": args.mines,
        }
        if args.seed is not None:
            settings["seed"] = args.seed
        config = GameConfig.model_validate(settings)
        result = asyncio.run(
            play(
                config,
                allow_guesses=args.allow_guesses,
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
    ) as error:
        print(f"minesweeper agent failed: {error}", file=sys.stderr)
        return 1
    print_result(
        result,
        sys.stdout,
        output_format=args.output,
        render_response=render_game_result,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
