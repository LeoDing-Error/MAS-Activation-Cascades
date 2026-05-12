from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

PRIMARY_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass(frozen=True)
class EvalTask:
    name: str
    difficulty: str
    prompt: str


HUMANEVAL_SUBSET: List[EvalTask] = [
    EvalTask("is_prime", "easy", "Write a Python function `is_prime(n: int) -> bool` that returns whether `n` is prime."),
    EvalTask("sum_list", "easy", "Write a Python function `sum_list(xs: list[int]) -> int` that returns the sum of all integers in `xs`."),
    EvalTask("reverse_string", "easy", "Write a Python function `reverse_string(s: str) -> str` that reverses a string."),
    EvalTask("find_pairs", "medium", "Write a Python function `find_pairs(nums: list[int], target: int) -> list[tuple[int, int]]` that returns index pairs summing to `target`."),
    EvalTask("is_palindrome", "medium", "Write a Python function `is_palindrome(s: str) -> bool` that ignores spaces and punctuation."),
    EvalTask("longest_common_prefix", "medium", "Write a Python function `longest_common_prefix(strings: list[str]) -> str` for a non-empty list of strings."),
    EvalTask("is_valid_sudoku", "hard", "Write a Python function `is_valid_sudoku(board: list[list[str]]) -> bool` that validates a 9x9 Sudoku board."),
    EvalTask("edit_distance", "hard", "Write a Python function `edit_distance(a: str, b: str) -> int` using dynamic programming."),
    EvalTask("generate_parentheses", "hard", "Write a Python function `generate_parentheses(n: int) -> list[str]` returning all balanced parentheses combinations."),
    EvalTask("LRUCache", "hard", "Implement an `LRUCache` class with `get` and `put` methods in Python."),
]


def parse_csv_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def parse_int_csv(raw: str | None) -> List[int] | None:
    items = parse_csv_list(raw)
    if items is None:
        return None
    return [int(item) for item in items]


def select_tasks(
    tasks: Sequence[EvalTask],
    *,
    task_names: Sequence[str] | None,
    task_indices: Sequence[int] | None,
    n_tasks: int | None,
) -> List[EvalTask]:
    if task_names is not None and task_indices is not None:
        raise ValueError("Specify either task_names or task_indices, not both")

    if task_names is not None:
        lookup = {task.name: task for task in tasks}
        selected: List[EvalTask] = []
        for name in task_names:
            if name not in lookup:
                raise ValueError(f"Unknown task name: {name}")
            selected.append(lookup[name])
        return selected

    if task_indices is not None:
        selected = []
        for index in task_indices:
            if index < 0 or index >= len(tasks):
                raise ValueError(f"Task index out of range: {index}")
            selected.append(tasks[index])
        return selected

    if n_tasks is None:
        return list(tasks)
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive when provided")
    return list(tasks[:n_tasks])


def default_tasks_for_experiment(experiment: str) -> List[EvalTask]:
    default_indices = {
        "1.2": [0],
        "1.3": [1],
        "1.4": [2],
    }
    return [HUMANEVAL_SUBSET[index] for index in default_indices[experiment]]
