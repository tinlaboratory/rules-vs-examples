from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Literal, Optional, Tuple

Player = Literal["A", "B"]
Outcome = Literal["A win", "B win", "continue"]
MoveType = Literal["place", "move"]

Coord = Tuple[int, int]


@dataclass(frozen=True)
class TapatanConfig:
    difficulty: str
    board_size: int
    pieces_per_player: int
    line_length: int
    max_movement_turns: int = 30


@dataclass
class TapatanState:
    config: TapatanConfig
    board: List[List[str]]  # board[y][x] in {"", "A", "B"}
    phase: Literal["placement", "movement"]
    to_move: Player
    placed_A: int
    placed_B: int
    history: List[str]


def get_tapatan_config(difficulty: str) -> TapatanConfig:
    d = difficulty.strip().lower()
    if d == "easy":
        return TapatanConfig("easy", 3, 3, 3, max_movement_turns=20)
    if d == "medium":
        return TapatanConfig("medium", 5, 4, 4, max_movement_turns=30)
    if d == "hard":
        return TapatanConfig("hard", 7, 5, 5, max_movement_turns=40)
    raise ValueError(f"Unknown difficulty: {difficulty}")


def in_bounds(coord: Coord, size: int) -> bool:
    x, y = coord
    return 0 <= x < size and 0 <= y < size


def get_neighbors(coord: Coord, config: TapatanConfig) -> List[Coord]:
    x, y = coord
    size = config.board_size
    neighbors: List[Coord] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if in_bounds((nx, ny), size):
            neighbors.append((nx, ny))
    if (x + y) % 2 == 0:
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if in_bounds((nx, ny), size):
                neighbors.append((nx, ny))
    return neighbors


def edge_exists(a: Coord, b: Coord, config: TapatanConfig) -> bool:
    return b in get_neighbors(a, config)


def format_place(player: Player, to_xy: Coord) -> str:
    x, y = to_xy
    return f"{player} place ({x},{y})"


def format_move(player: Player, from_xy: Coord, to_xy: Coord) -> str:
    x1, y1 = from_xy
    x2, y2 = to_xy
    return f"{player} move ({x1},{y1})->({x2},{y2})"


_PLACE_RE = re.compile(
    r"^([AB])\s+place\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$"
)
_MOVE_RE = re.compile(
    r"^([AB])\s+move\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*->\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$"
)


def parse_turn(turn_text: str) -> Tuple[Player, MoveType, Coord, Optional[Coord]]:
    text = (turn_text or "").strip()
    m = _PLACE_RE.match(text)
    if m:
        player = m.group(1)
        x = int(m.group(2))
        y = int(m.group(3))
        return player, "place", (x, y), None
    m = _MOVE_RE.match(text)
    if m:
        player = m.group(1)
        x1 = int(m.group(2))
        y1 = int(m.group(3))
        x2 = int(m.group(4))
        y2 = int(m.group(5))
        return player, "move", (x1, y1), (x2, y2)
    raise ValueError(f"Invalid move format: {turn_text!r}")


def split_moves(moves: str) -> List[str]:
    if not moves:
        return []
    if " ; " in moves:
        parts = moves.split(" ; ")
    else:
        parts = [p.strip() for p in moves.split(";")]
    return [p.strip() for p in parts if p.strip()]


def new_game(config: TapatanConfig) -> TapatanState:
    size = config.board_size
    board = [["" for _ in range(size)] for _ in range(size)]
    return TapatanState(
        config=config,
        board=board,
        phase="placement",
        to_move="A",
        placed_A=0,
        placed_B=0,
        history=[],
    )


def is_legal_turn(state: TapatanState, turn_text: str) -> bool:
    try:
        player, move_type, from_or_to, to_if_move = parse_turn(turn_text)
    except ValueError:
        return False

    if player != state.to_move:
        return False

    size = state.config.board_size
    pieces_per_player = state.config.pieces_per_player

    if move_type == "place":
        if state.phase != "placement":
            return False
        x, y = from_or_to
        if not in_bounds((x, y), size):
            return False
        if state.board[y][x] != "":
            return False
        if player == "A" and state.placed_A >= pieces_per_player:
            return False
        if player == "B" and state.placed_B >= pieces_per_player:
            return False
        return True

    if move_type == "move":
        if state.phase != "movement":
            return False
        if to_if_move is None:
            return False
        fx, fy = from_or_to
        tx, ty = to_if_move
        if not in_bounds((fx, fy), size) or not in_bounds((tx, ty), size):
            return False
        if state.board[fy][fx] != player:
            return False
        if state.board[ty][tx] != "":
            return False
        if not edge_exists((fx, fy), (tx, ty), state.config):
            return False
        return True

    return False


def apply_turn(state: TapatanState, turn_text: str) -> None:
    if not is_legal_turn(state, turn_text):
        raise ValueError(f"Illegal turn: {turn_text!r}")

    player, move_type, from_or_to, to_if_move = parse_turn(turn_text)

    if move_type == "place":
        x, y = from_or_to
        state.board[y][x] = player
        if player == "A":
            state.placed_A += 1
        else:
            state.placed_B += 1
        if (
            state.placed_A >= state.config.pieces_per_player
            and state.placed_B >= state.config.pieces_per_player
        ):
            state.phase = "movement"
    else:
        fx, fy = from_or_to
        tx, ty = to_if_move or from_or_to
        state.board[fy][fx] = ""
        state.board[ty][tx] = player

    state.history.append(turn_text.strip())
    state.to_move = "B" if player == "A" else "A"


def has_line(state: TapatanState, player: Player) -> bool:
    size = state.config.board_size
    target = state.config.line_length
    board = state.board
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for y in range(size):
        for x in range(size):
            if board[y][x] != player:
                continue
            for dx, dy in directions:
                count = 1
                cx, cy = x, y
                while count < target:
                    nx, ny = cx + dx, cy + dy
                    if not in_bounds((nx, ny), size):
                        break
                    if board[ny][nx] != player:
                        break
                    if not edge_exists((cx, cy), (nx, ny), state.config):
                        break
                    count += 1
                    cx, cy = nx, ny
                if count >= target:
                    return True
    return False


def find_line_direction(
    state: TapatanState, player: Player
) -> Optional[Tuple[int, int]]:
    """Return the ``(dx, dy)`` direction of a winning line, or ``None``.

    Diagonal wins have ``(1,1)`` or ``(1,-1)``; axis-aligned wins have
    ``(1,0)`` (horizontal) or ``(0,1)`` (vertical).
    """
    size = state.config.board_size
    target = state.config.line_length
    board = state.board
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for y in range(size):
        for x in range(size):
            if board[y][x] != player:
                continue
            for dx, dy in directions:
                count = 1
                cx, cy = x, y
                while count < target:
                    nx, ny = cx + dx, cy + dy
                    if not in_bounds((nx, ny), size):
                        break
                    if board[ny][nx] != player:
                        break
                    if not edge_exists((cx, cy), (nx, ny), state.config):
                        break
                    count += 1
                    cx, cy = nx, ny
                if count >= target:
                    return (dx, dy)
    return None


def max_line_length(state: TapatanState, player: Player) -> int:
    """Return the length of the longest contiguous line for *player*.

    Useful for detecting near-misses (``max_line == config.line_length - 1``).
    """
    size = state.config.board_size
    board = state.board
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    best = 0

    for y in range(size):
        for x in range(size):
            if board[y][x] != player:
                continue
            for dx, dy in directions:
                count = 1
                cx, cy = x, y
                while True:
                    nx, ny = cx + dx, cy + dy
                    if not in_bounds((nx, ny), size):
                        break
                    if board[ny][nx] != player:
                        break
                    if not edge_exists((cx, cy), (nx, ny), state.config):
                        break
                    count += 1
                    cx, cy = nx, ny
                if count > best:
                    best = count
    return best


def outcome_from_history(config: TapatanConfig, moves: str) -> Outcome:
    state = new_game(config)
    for turn in split_moves(moves):
        apply_turn(state, turn)
        mover = "A" if state.to_move == "B" else "B"
        if has_line(state, mover):
            return "A win" if mover == "A" else "B win"
    return "continue"
