import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    rows = len(board)
    cols = len(board[0])

    color = board[row][col]

    if color == 0:
        return ([], [])

    visited = set()
    liberties = set()
    group = []

    stack = [(row, col)]
    visited.add((row, col))

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while stack:
        r, c = stack.pop()
        group.append((r, c))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if board[nr][nc] == 0:
                    liberties.add((nr, nc))

                elif board[nr][nc] == color:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

    return (sorted(group), sorted(liberties))