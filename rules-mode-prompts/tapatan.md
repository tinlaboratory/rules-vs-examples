# Tapatan Prompt Templates

## Move Sequence Variant

### Rules

#### Easy
You will be learning a new board game. You will then be given one sequence of moves from that game. Determine the current game outcome.

RULES:
1. The game is played on a {board_size}x{board_size} board of points with coordinates (x,y), where 0 <= x,y < {board_size}.
2. Players A and B alternate turns. A moves first.
3. Each player has {pieces_per_player} pieces.
4. Placement phase: on your turn, you must place one of your unused pieces on an empty coordinate.
5. Movement phase: after both players have placed all their pieces, on your turn you must move one of your pieces to an adjacent empty coordinate.
6. Adjacency matters both for legal moves in the movement phase and for checking whether a straight line is valid. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.
7. Win condition: after your move, if you have {line_length} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 6, you win immediately.
8. Moves are written as 'A place (x,y)' or 'B move (x1,y1)->(x2,y2)'. Moves are separated by ' ; '.
9. You must output a label that is exactly one of: A win, B win, or continue.

SEQUENCE:
{sequence}

LABEL:

#### Medium
You will be learning a new board game. You will then be given one sequence of moves from that game. Determine the current game outcome.

RULES:
1. The game is played on a {board_size}x{board_size} board of points with coordinates (x,y), where 0 <= x,y < {board_size}.
2. Players A and B alternate turns. A moves first.
3. Each player has {pieces_per_player} pieces.
4. Placement phase: on your turn, you must place one of your unused pieces on an empty coordinate.
5. Movement phase: after both players have placed all their pieces, on your turn you must move one of your pieces to an adjacent empty coordinate.
6. Adjacency matters both for legal moves in the movement phase and for checking whether a straight line is valid. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.
7. Win condition: after your move, if you have {line_length} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 6, you win immediately.
8. Moves are written as 'A place (x,y)' or 'B move (x1,y1)->(x2,y2)'. Moves are separated by ' ; '.
9. You must output a label that is exactly one of: A win, B win, or continue.

SEQUENCE:
{sequence}

LABEL:

#### Hard
You will be learning a new board game. You will then be given one sequence of moves from that game. Determine the current game outcome.

RULES:
1. The game is played on a {board_size}x{board_size} board of points with coordinates (x,y), where 0 <= x,y < {board_size}.
2. Players A and B alternate turns. A moves first.
3. Each player has {pieces_per_player} pieces.
4. Placement phase: on your turn, you must place one of your unused pieces on an empty coordinate.
5. Movement phase: after both players have placed all their pieces, on your turn you must move one of your pieces to an adjacent empty coordinate.
6. Adjacency matters both for legal moves in the movement phase and for checking whether a straight line is valid. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.
7. Win condition: after your move, if you have {line_length} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 6, you win immediately.
8. Moves are written as 'A place (x,y)' or 'B move (x1,y1)->(x2,y2)'. Moves are separated by ' ; '.
9. You must output a label that is exactly one of: A win, B win, or continue.

SEQUENCE:
{sequence}

LABEL:

## Final Board State Variant

### Rules

#### Easy
You will be learning a new board game. You will then be given one final board state from that game. Determine the current game outcome.

RULES:
1. The board is a {board_size}x{board_size} grid of points with coordinates (x,y), where 0 <= x,y < {board_size}.
2. A board cell contains A for player A's piece, B for player B's piece, or . for an empty point.
3. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.
4. A player wins if they have {line_length} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 3.
5. If A has a valid winning line, output A win. If B has a valid winning line, output B win. If neither player has a valid winning line, output continue.
6. You must output a label that is exactly one of: A win, B win, or continue.

BOARD:
{board_state}

LABEL:

#### Medium
You will be learning a new board game. You will then be given one final board state from that game. Determine the current game outcome.

RULES:
1. The board is a {board_size}x{board_size} grid of points with coordinates (x,y), where 0 <= x,y < {board_size}.
2. A board cell contains A for player A's piece, B for player B's piece, or . for an empty point.
3. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.
4. A player wins if they have {line_length} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 3.
5. If A has a valid winning line, output A win. If B has a valid winning line, output B win. If neither player has a valid winning line, output continue.
6. You must output a label that is exactly one of: A win, B win, or continue.

BOARD:
{board_state}

LABEL:

#### Hard
You will be learning a new board game. You will then be given one final board state from that game. Determine the current game outcome.

RULES:
1. The board is a {board_size}x{board_size} grid of points with coordinates (x,y), where 0 <= x,y < {board_size}.
2. A board cell contains A for player A's piece, B for player B's piece, or . for an empty point.
3. Orthogonal neighbors are always adjacent. Diagonal neighbors are adjacent only from even-parity points where (x+y) is even: (x,y) connects to (x±1,y±1) when in bounds.
4. A player wins if they have {line_length} pieces in a straight line (horizontal, vertical, or diagonal) and each consecutive pair of points in that line is adjacent under rule 3.
5. If A has a valid winning line, output A win. If B has a valid winning line, output B win. If neither player has a valid winning line, output continue.
6. You must output a label that is exactly one of: A win, B win, or continue.

BOARD:
{board_state}

LABEL:
