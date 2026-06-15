#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = WORKSPACE_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tasks.tapatan_core import (  # noqa: E402
    TapatanConfig,
    TapatanState,
    apply_turn,
    edge_exists,
    format_move,
    format_place,
    get_neighbors,
    get_tapatan_config,
    has_line,
    in_bounds,
    new_game,
    outcome_from_history,
)

OutcomeLabel = str
Coord = Tuple[int, int]

VALID_OUTCOMES = ["A win", "B win", "continue"]


def _line_key(coords: Sequence[Coord]) -> Tuple[Coord, ...]:
    t = tuple(coords)
    rev = tuple(reversed(t))
    return t if t <= rev else rev


def compute_lines(config: TapatanConfig) -> List[List[Coord]]:
    size = config.board_size
    k = config.line_length
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    lines: List[List[Coord]] = []
    seen = set()

    for y in range(size):
        for x in range(size):
            for dx, dy in directions:
                coords: List[Coord] = []
                cx, cy = x, y
                valid = True
                for step in range(k):
                    if not in_bounds((cx, cy), size):
                        valid = False
                        break
                    coords.append((cx, cy))
                    if step < k - 1:
                        nx, ny = cx + dx, cy + dy
                        if not in_bounds((nx, ny), size):
                            valid = False
                            break
                        if not edge_exists((cx, cy), (nx, ny), config):
                            valid = False
                            break
                        cx, cy = nx, ny
                if valid and len(coords) == k:
                    key = _line_key(coords)
                    if key not in seen:
                        seen.add(key)
                        lines.append(coords)
    return lines


def _empty_cells(state) -> List[Coord]:
    size = state.config.board_size
    empties: List[Coord] = []
    for y in range(size):
        for x in range(size):
            if state.board[y][x] == "":
                empties.append((x, y))
    return empties


def _choose_safe_placement(state, player: str, rng: random.Random, forbidden: Optional[set] = None) -> Optional[Coord]:
    empties = _empty_cells(state)
    rng.shuffle(empties)
    for coord in empties:
        if forbidden and coord in forbidden:
            continue
        x, y = coord
        state.board[y][x] = player
        win = has_line(state, player)
        state.board[y][x] = ""
        if not win:
            return coord
    return None


def _legal_moves(state, player: str) -> List[Tuple[Coord, Coord]]:
    moves: List[Tuple[Coord, Coord]] = []
    size = state.config.board_size
    for y in range(size):
        for x in range(size):
            if state.board[y][x] != player:
                continue
            for nx, ny in get_neighbors((x, y), state.config):
                if state.board[ny][nx] == "":
                    moves.append(((x, y), (nx, ny)))
    return moves


def _safe_moves(state, player: str) -> List[Tuple[Coord, Coord]]:
    safe: List[Tuple[Coord, Coord]] = []
    for (fx, fy), (tx, ty) in _legal_moves(state, player):
        state.board[fy][fx] = ""
        state.board[ty][tx] = player
        win = has_line(state, player)
        state.board[fy][fx] = player
        state.board[ty][tx] = ""
        if not win:
            safe.append(((fx, fy), (tx, ty)))
    return safe


def _winning_moves(state, player: str) -> List[Tuple[Coord, Coord]]:
    wins: List[Tuple[Coord, Coord]] = []
    for (fx, fy), (tx, ty) in _legal_moves(state, player):
        state.board[fy][fx] = ""
        state.board[ty][tx] = player
        win = has_line(state, player)
        state.board[fy][fx] = player
        state.board[ty][tx] = ""
        if win:
            wins.append(((fx, fy), (tx, ty)))
    return wins


def _generate_nonwinning_placement_state(
    config: TapatanConfig,
    rng: random.Random,
    max_attempts: int = 2000,
):
    for _ in range(max_attempts):
        state = new_game(config)
        success = True
        while state.phase == "placement":
            player = state.to_move
            coord = _choose_safe_placement(state, player, rng)
            if coord is None:
                success = False
                break
            move = format_place(player, coord)
            try:
                apply_turn(state, move)
            except ValueError:
                success = False
                break
            if has_line(state, player):
                success = False
                break
        if success:
            return state
    return None


def generate_placement_win(
    config: TapatanConfig,
    winner: str,
    rng: random.Random,
    lines: Sequence[Sequence[Coord]],
    max_attempts: int = 2000,
) -> Optional[str]:
    for _ in range(max_attempts):
        state = new_game(config)
        line = list(rng.choice(lines))
        rng.shuffle(line)
        line_set = set(line)
        remaining = list(line)
        success = True

        while True:
            player = state.to_move
            if player == winner:
                if not remaining:
                    success = False
                    break
                coord = remaining.pop(0)
            else:
                coord = _choose_safe_placement(state, player, rng, forbidden=line_set)
                if coord is None:
                    success = False
                    break
            move = format_place(player, coord)
            try:
                apply_turn(state, move)
            except ValueError:
                success = False
                break

            mover = "A" if state.to_move == "B" else "B"
            if has_line(state, mover):
                if mover == winner:
                    return " ; ".join(state.history)
                success = False
                break

            if state.phase == "movement" and winner != mover:
                success = False
                break
    return None


def generate_movement_win(
    config: TapatanConfig,
    winner: str,
    rng: random.Random,
    lines: Sequence[Sequence[Coord]],
    max_attempts: int = 2000,
) -> Optional[str]:
    for _ in range(max_attempts):
        state = new_game(config)
        line = list(rng.choice(lines))
        missing_idx = rng.randrange(len(line))
        target = line[missing_idx]
        line_set = set(line)

        candidate_sources = [c for c in get_neighbors(target, config) if c not in line_set]
        if not candidate_sources:
            continue
        rng.shuffle(candidate_sources)
        source = candidate_sources[0]

        winner_positions = list(line_set - {target}) + [source]
        rng.shuffle(winner_positions)
        forbidden = set(winner_positions) | {target}

        success = True
        while state.phase == "placement":
            player = state.to_move
            if player == winner:
                if not winner_positions:
                    success = False
                    break
                coord = winner_positions.pop()
            else:
                coord = _choose_safe_placement(state, player, rng, forbidden=forbidden)
                if coord is None:
                    success = False
                    break
            move = format_place(player, coord)
            try:
                apply_turn(state, move)
            except ValueError:
                success = False
                break
            if has_line(state, player):
                success = False
                break

        if not success:
            continue

        if winner == "A":
            if state.to_move != "A":
                continue
            move = format_move("A", source, target)
            try:
                apply_turn(state, move)
            except ValueError:
                continue
            if has_line(state, "A"):
                return " ; ".join(state.history)
            continue

        # Winner is B; ensure A makes a safe move first.
        if state.to_move != "A":
            continue
        safe_moves = [m for m in _safe_moves(state, "A") if m[1] != target]
        if not safe_moves:
            continue
        from_xy, to_xy = rng.choice(safe_moves)
        try:
            apply_turn(state, format_move("A", from_xy, to_xy))
        except ValueError:
            continue
        try:
            apply_turn(state, format_move("B", source, target))
        except ValueError:
            continue
        if has_line(state, "B"):
            return " ; ".join(state.history)
    return None


def generate_movement_win_with_pre_moves(
    config: TapatanConfig,
    winner: str,
    rng: random.Random,
    pre_moves: int,
    cycle_options: Optional[Sequence[Tuple[Sequence[Coord], Coord, Coord, Coord]]] = None,
    max_attempts: int = 4000,
) -> Optional[str]:
    if winner == "A" and pre_moves % 2 != 0:
        raise ValueError("pre_moves must be even for A movement wins.")
    if winner == "B" and pre_moves % 2 != 1:
        raise ValueError("pre_moves must be odd for B movement wins.")

    def _attempt_cycle() -> Optional[str]:
        if not cycle_options:
            return None
        for _ in range(max_attempts):
            line, target, source, alt = rng.choice(cycle_options)
            line_set = set(line)
            state = new_game(config)

            winner_positions = list(line_set - {target}) + [source]
            rng.shuffle(winner_positions)
            forbidden = set(line_set) | {source, alt}

            success = True
            while state.phase == "placement":
                player = state.to_move
                if player == winner:
                    if not winner_positions:
                        success = False
                        break
                    coord = winner_positions.pop()
                else:
                    coord = _choose_safe_placement(state, player, rng, forbidden=forbidden)
                    if coord is None:
                        success = False
                        break
                move = format_place(player, coord)
                try:
                    apply_turn(state, move)
                except ValueError:
                    success = False
                    break
                if has_line(state, player):
                    success = False
                    break

            if not success:
                continue
            if state.to_move != "A":
                continue

            source_pos = source
            for _ in range(pre_moves):
                player = state.to_move
                if player == winner:
                    dest = alt if source_pos == source else source
                    if state.board[dest[1]][dest[0]] != "":
                        success = False
                        break
                    try:
                        apply_turn(state, format_move(player, source_pos, dest))
                    except ValueError:
                        success = False
                        break
                    source_pos = dest
                else:
                    safe_moves = [
                        m for m in _safe_moves(state, player)
                        if m[1] not in {target, source, alt}
                    ]
                    if not safe_moves:
                        success = False
                        break
                    from_xy, to_xy = rng.choice(safe_moves)
                    try:
                        apply_turn(state, format_move(player, from_xy, to_xy))
                    except ValueError:
                        success = False
                        break
                if has_line(state, player):
                    success = False
                    break

            if not success:
                continue
            if state.to_move != winner:
                continue
            if not edge_exists(source_pos, target, config):
                continue
            try:
                apply_turn(state, format_move(winner, source_pos, target))
            except ValueError:
                continue
            if has_line(state, winner):
                return " ; ".join(state.history)
        return None

    def _search(state, depth: int) -> Optional[List[str]]:
        if depth == 0:
            if state.to_move != winner:
                return None
            winning_moves = _winning_moves(state, winner)
            if not winning_moves:
                return None
            from_xy, to_xy = rng.choice(winning_moves)
            try:
                apply_turn(state, format_move(winner, from_xy, to_xy))
            except ValueError:
                return None
            if has_line(state, winner):
                return state.history
            return None

        player = state.to_move
        safe_moves = _safe_moves(state, player)
        rng.shuffle(safe_moves)
        # Limit branching to keep search tractable.
        for from_xy, to_xy in safe_moves[:20]:
            # Shallow copy of board/state for search depth.
            next_state = TapatanState(
                config=state.config,
                board=[row[:] for row in state.board],
                phase=state.phase,
                to_move=state.to_move,
                placed_A=state.placed_A,
                placed_B=state.placed_B,
                history=list(state.history),
            )
            try:
                apply_turn(next_state, format_move(player, from_xy, to_xy))
            except ValueError:
                continue
            result = _search(next_state, depth - 1)
            if result:
                return result
        return None

    cycle_result = _attempt_cycle()
    if cycle_result:
        return cycle_result

    for _ in range(max_attempts):
        state = _generate_nonwinning_placement_state(config, rng)
        if state is None:
            continue
        if state.to_move != "A":
            continue
        found = _search(state, pre_moves)
        if found:
            return " ; ".join(found)
    return None


def generate_continue_with_moves(
    config: TapatanConfig,
    rng: random.Random,
    movement_turns: int,
    max_attempts: int = 4000,
) -> Optional[str]:
    for _ in range(max_attempts):
        state = _generate_nonwinning_placement_state(config, rng)
        if state is None:
            continue
        success = True
        for _ in range(movement_turns):
            player = state.to_move
            safe_moves = _safe_moves(state, player)
            if not safe_moves:
                success = False
                break
            from_xy, to_xy = rng.choice(safe_moves)
            try:
                apply_turn(state, format_move(player, from_xy, to_xy))
            except ValueError:
                success = False
                break
            if has_line(state, player):
                success = False
                break
        if not success:
            continue
        moves_str = " ; ".join(state.history)
        if outcome_from_history(config, moves_str) == "continue":
            return moves_str
    return None


def generate_continue_placement(
    config: TapatanConfig,
    rng: random.Random,
    max_attempts: int = 2000,
) -> Optional[str]:
    for _ in range(max_attempts):
        state = _generate_nonwinning_placement_state(config, rng)
        if state is None:
            continue
        moves_str = " ; ".join(state.history)
        if outcome_from_history(config, moves_str) == "continue":
            return moves_str
    return None


def generate_continue_game(
    config: TapatanConfig,
    rng: random.Random,
    max_attempts: int = 2000,
    movement_prob: float = 0.5,
) -> Optional[str]:
    for _ in range(max_attempts):
        state = _generate_nonwinning_placement_state(config, rng)
        if state is None:
            continue

        if rng.random() > movement_prob:
            moves_str = " ; ".join(state.history)
            if outcome_from_history(config, moves_str) == "continue":
                return moves_str
            continue

        movement_turns = rng.randint(6, 10)
        for _ in range(movement_turns):
            player = state.to_move
            safe_moves = _safe_moves(state, player)
            if not safe_moves:
                break
            from_xy, to_xy = rng.choice(safe_moves)
            move = format_move(player, from_xy, to_xy)
            try:
                apply_turn(state, move)
            except ValueError:
                break
            if has_line(state, player):
                break

        moves_str = " ; ".join(state.history)
        if outcome_from_history(config, moves_str) == "continue":
            return moves_str
    return None


def _target_counts(total: int) -> Dict[str, int]:
    base = total // 3
    remainder = total - base * 3
    counts = {"A win": base, "B win": base, "continue": base}
    counts["continue"] += remainder
    return counts


def _distribute_counts(total: int, bins: Sequence[int]) -> Dict[int, int]:
    if not bins:
        return {}
    base = total // len(bins)
    remainder = total - base * len(bins)
    counts: Dict[int, int] = {}
    for idx, b in enumerate(bins):
        counts[b] = base + (1 if idx < remainder else 0)
    return counts


def _move_bins(config: TapatanConfig) -> Tuple[List[int], List[int], List[int]]:
    # Use smaller bins for easy, slightly larger for others.
    if config.difficulty == "easy":
        a_bins = [0, 2]  # even pre-moves for A
        b_bins = [1, 3]  # odd pre-moves for B
        c_bins = [1, 2, 3, 4]  # movement turns for continue
    elif config.difficulty == "hard":
        a_bins = [0, 2]
        b_bins = [1, 3]
        c_bins = [1, 2, 3, 4]
    else:
        a_bins = [0, 2, 4]
        b_bins = [1, 3, 5]
        c_bins = [1, 2, 3, 4, 5, 6]
    return a_bins, b_bins, c_bins


def _compute_cycle_options(
    config: TapatanConfig,
    lines: Sequence[Sequence[Coord]],
) -> List[Tuple[Sequence[Coord], Coord, Coord, Coord]]:
    options: List[Tuple[Sequence[Coord], Coord, Coord, Coord]] = []
    for line in lines:
        line_set = set(line)
        for target in line:
            for source in get_neighbors(target, config):
                if source in line_set:
                    continue
                for alt in get_neighbors(source, config):
                    if alt == target or alt == source:
                        continue
                    if alt in line_set:
                        continue
                    if not edge_exists(alt, target, config):
                        continue
                    options.append((line, target, source, alt))
    return options


def _build_example(
    idx: int,
    split: str,
    config: TapatanConfig,
    moves: str,
    result: OutcomeLabel,
) -> Dict[str, object]:
    return {
        "id": f"{split}_{idx:03d}",
        "difficulty": config.difficulty,
        "board_size": config.board_size,
        "pieces_per_player": config.pieces_per_player,
        "line_length": config.line_length,
        "moves": moves,
        "result": result,
    }


def generate_split(
    config: TapatanConfig,
    split: str,
    num_samples: int,
    rng: random.Random,
    signatures: set,
    max_attempts: int,
) -> List[Dict[str, object]]:
    lines = compute_lines(config)
    if not lines:
        raise RuntimeError(f"No valid lines found for difficulty {config.difficulty}.")
    cycle_options = _compute_cycle_options(config, lines)

    targets = _target_counts(num_samples)
    a_total = targets["A win"]
    b_total = targets["B win"]
    c_total = targets["continue"]

    a_place = a_total // 2
    a_move = a_total - a_place
    b_place = b_total // 2
    b_move = b_total - b_place
    c_place = c_total // 2
    c_move = c_total - c_place

    a_bins, b_bins, c_bins = _move_bins(config)
    a_move_targets = _distribute_counts(a_move, a_bins)
    b_move_targets = _distribute_counts(b_move, b_bins)
    c_move_targets = _distribute_counts(c_move, c_bins)

    examples: List[Dict[str, object]] = []

    def _accept_example(moves: str, label: str) -> bool:
        if outcome_from_history(config, moves) != label:
            return False
        signature = f"{config.difficulty}|{moves}|{label}"
        if signature in signatures:
            return False
        signatures.add(signature)
        examples.append(_build_example(0, split, config, moves, label))
        return True

    # Placement wins (fixed move counts)
    for winner, needed in [("A", a_place), ("B", b_place)]:
        count = 0
        attempts = 0
        label = f"{winner} win"
        while count < needed and attempts < max_attempts:
            attempts += 1
            moves = generate_placement_win(config, winner, rng, lines)
            if not moves:
                continue
            if _accept_example(moves, label):
                count += 1
        if count < needed:
            raise RuntimeError(
                f"Failed to generate {needed} placement wins for '{label}' after {attempts} attempts."
            )

    # Movement wins with balanced pre-move counts
    for winner, targets_map in [("A", a_move_targets), ("B", b_move_targets)]:
        label = f"{winner} win"
        for pre_moves, needed in targets_map.items():
            count = 0
            attempts = 0
            while count < needed and attempts < max_attempts:
                attempts += 1
                moves = generate_movement_win_with_pre_moves(
                    config,
                    winner,
                    rng,
                    pre_moves,
                    cycle_options=cycle_options,
                    max_attempts=2000,
                )
                if not moves:
                    continue
                if _accept_example(moves, label):
                    count += 1
            if count < needed:
                raise RuntimeError(
                    f"Failed to generate {needed} movement wins for '{label}' with pre_moves={pre_moves} "
                    f"after {attempts} attempts."
                )

    # Continue placement-only
    count = 0
    attempts = 0
    while count < c_place and attempts < max_attempts:
        attempts += 1
        moves = generate_continue_placement(config, rng)
        if not moves:
            continue
        if _accept_example(moves, "continue"):
            count += 1
    if count < c_place:
        raise RuntimeError(
            f"Failed to generate {c_place} placement-continue examples after {attempts} attempts."
        )

    # Continue with movement (balanced by movement turns)
    for movement_turns, needed in c_move_targets.items():
        count = 0
        attempts = 0
        while count < needed and attempts < max_attempts:
            attempts += 1
            moves = generate_continue_with_moves(
                config,
                rng,
                movement_turns,
                max_attempts=2000,
            )
            if not moves:
                continue
            if _accept_example(moves, "continue"):
                count += 1
        if count < needed:
            raise RuntimeError(
                f"Failed to generate {needed} movement-continue examples with movement_turns={movement_turns} "
                f"after {attempts} attempts."
            )

    rng.shuffle(examples)
    for idx, ex in enumerate(examples, start=1):
        ex["id"] = f"{split}_{idx:03d}"
    return examples[:num_samples]


def write_split(
    examples: Sequence[Dict[str, object]],
    output_dir: Path,
    split: str,
    num_samples: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"tapatan_{split}_n{num_samples}.json"
    with path.open("w") as f:
        json.dump(list(examples), f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tapatan datasets.")
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=200000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    for diff in difficulties:
        config = get_tapatan_config(diff)
        signatures: set = set()

        train_examples = generate_split(
            config=config,
            split="train",
            num_samples=args.train_samples,
            rng=rng,
            signatures=signatures,
            max_attempts=args.max_attempts,
        )

        test_examples = generate_split(
            config=config,
            split="test",
            num_samples=args.test_samples,
            rng=rng,
            signatures=signatures,
            max_attempts=args.max_attempts,
        )

        base_dir = WORKSPACE_ROOT / "data" / "tapatan" / diff
        train_path = write_split(train_examples, base_dir / "train", "train", args.train_samples)
        test_path = write_split(test_examples, base_dir / "test", "test", args.test_samples)

        print(f"[{diff}] Wrote {len(train_examples)} train examples -> {train_path}")
        print(f"[{diff}] Wrote {len(test_examples)} test examples -> {test_path}")


if __name__ == "__main__":
    main()
