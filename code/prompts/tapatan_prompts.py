from typing import Any, Dict, List

from prompts.markdown_loader import render_markdown_prompt
from tasks.tapatan_core import TapatanConfig


def _format_rules_list(rules: List[str]) -> str:
    return "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))


def get_rules_text(config: TapatanConfig) -> str:
    n = config.board_size
    p = config.pieces_per_player
    k = config.line_length
    rules = [
        f"The game is played on a {n}x{n} board of points with coordinates (x,y), where 0 <= x,y < {n}.",
        "Players A and B alternate turns. A moves first.",
        f"Each player has {p} pieces.",
        "Placement phase: on your turn, you must place one of your unused pieces on an empty coordinate.",
        "Movement phase: after both players have placed all their pieces, on your turn you must move one of your pieces to an adjacent empty coordinate.",
        "Adjacency matters both for legal moves in the movement phase and for checking whether a straight line is valid. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.",
        f"Win condition: after your move, if you have {k} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 6, you win immediately.",
        "Moves are written as 'A place (x,y)' or 'B move (x1,y1)->(x2,y2)'. Moves are separated by ' ; '.",
        "You must output a label that is exactly one of: A win, B win, or continue.",
    ]
    return _format_rules_list(rules)


def _example_line(example: Dict[str, Any]) -> str:
    moves = example.get("input") or example.get("moves") or ""
    result = example.get("correct_output") or example.get("result") or ""
    return f"{moves} → {result}"


def get_rule_based_prompt(config: TapatanConfig, test_input_moves: str) -> str:
    return render_markdown_prompt(
        "tapatan.md",
        ("Move Sequence Variant", "Rules", config.difficulty.title()),
        board_size=config.board_size,
        pieces_per_player=config.pieces_per_player,
        line_length=config.line_length,
        sequence=test_input_moves,
    )


def get_example_based_prompt(examples: List[Dict[str, Any]], test_input_moves: str) -> str:
    lines = [_example_line(example) for example in examples]
    prompt = "\n".join(lines) + ("\n\n" if lines else "")
    prompt += "Output exactly one label: A win, B win, or continue.\n"
    prompt += f"{test_input_moves} → "
    return prompt


def _board_example_block(example: Dict[str, Any]) -> str:
    board_state = example.get("board_state") or example.get("board") or ""
    result = example.get("correct_output") or example.get("result") or ""
    return f"BOARD:\n{board_state}\n\nLABEL: {result}"


def get_board_rule_based_prompt(config: TapatanConfig, test_board_state: str) -> str:
    return render_markdown_prompt(
        "tapatan.md",
        ("Final Board State Variant", "Rules", config.difficulty.title()),
        board_size=config.board_size,
        line_length=config.line_length,
        board_state=test_board_state,
    )


def get_board_example_based_prompt(
    examples: List[Dict[str, Any]], test_board_state: str
) -> str:
    blocks = [_board_example_block(example) for example in examples]
    prompt = "\n\n".join(blocks) + ("\n\n" if blocks else "")
    prompt += f"BOARD:\n{test_board_state}\n\nLABEL:"
    return prompt


def get_board_combined_prompt(
    config: TapatanConfig, examples: List[Dict[str, Any]], test_board_state: str
) -> str:
    rules_text = get_board_rule_based_prompt(config, "{board_state}").split(
        "\nBOARD:\n{board_state}\n\nLABEL:",
        1,
    )[0]
    example_text = "\n\n".join(_board_example_block(example) for example in examples)
    prompt = f"""{rules_text}

EXAMPLES:
{example_text}

BOARD:
{test_board_state}

LABEL:"""
    return prompt


def get_combined_prompt(
    config: TapatanConfig, examples: List[Dict[str, Any]], test_input_moves: str
) -> str:
    rules_text = get_rules_text(config)
    example_text = "\n".join(_example_line(example) for example in examples)
    task_intro = (
        "You will be learning a new board game. "
        "You will then be given one sequence of moves from that game. "
        "Determine the current game outcome."
    )
    prompt = f"""{task_intro}

RULES:
{rules_text}

EXAMPLES:
{example_text}

Output exactly one label: A win, B win, or continue.
{test_input_moves} → """
    return prompt
