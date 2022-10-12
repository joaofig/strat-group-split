import numpy as np

from numpy.random import default_rng
from numba import njit
from typing import Set, Tuple


def generate_problem(num_groups: int,
                     num_classes: int,
                     min_group_size: int,
                     max_group_size: int,
                     class_percent: np.array) -> np.ndarray:

    problem = np.zeros((num_groups, num_classes), dtype=int)

    rng = default_rng()
    group_sizes = rng.integers(low=min_group_size,
                               high=max_group_size,
                               size=num_groups)

    for i in range(num_groups):
        # Calculate the
        proportions = np.random.normal(class_percent, class_percent / 10)

        problem[i, :] = proportions * group_sizes[i]
    return problem


@njit
def calculate_cost(problem: np.ndarray,
                   solution: np.ndarray,
                   k: int) -> float:
    cost = 0.0
    total = np.sum(problem)
    class_sums = np.sum(problem, axis=0)
    num_classes = problem.shape[1]

    for i in range(k):
        idx = solution == i
        fold_sum = np.sum(problem[idx, :])

        # Start by calculating the fold imbalance cost
        cost += (fold_sum / total - 1.0 / k) ** 2

        # Now calculate the cost associated with the class imbalances
        for j in range(num_classes):
            cost += (np.sum(problem[idx, j]) / fold_sum - class_sums[j] / total) ** 2
    return cost


@njit
def generate_search_space(problem: np.ndarray,
                          solution: np.ndarray,
                          k: int) -> np.ndarray:
    num_groups = problem.shape[0]

    space = np.zeros((num_groups, k))
    sol = solution.copy()

    for i in range(num_groups):
        for j in range(k):
            if solution[i] == j:
                space[i,j] = np.infty
            else:
                sol[i] = j
                space[i, j] = calculate_cost(problem, sol, k)
        sol[i] = solution[i]
    return space


@njit
def solution_to_str(solution: np.ndarray) -> str:
    return "".join([str(n) for n in solution])


def generate_initial_solution(problem: np.ndarray,
                              k: int,
                              algo: str="k-bound") -> np.ndarray:
    num_groups = problem.shape[0]
    if algo == "k-bound":
        rng = default_rng()
        total = np.sum(problem)
        indices = rng.permutation(problem.shape[0])

        solution = np.zeros(num_groups, dtype=int)
        c = 0
        fold_total = 0
        for i in indices:
            group = np.sum(problem[i, :])
            if fold_total + group < total / k:
                fold_total += group
            else:
                c = (c + 1) % k
                fold_total = group
            solution[i] = c
    elif algo == "random":
        rng = default_rng()
        solution = rng.integers(low=0, high=k, size=num_groups)
    elif algo == "zeros":
        solution = np.zeros(num_groups, dtype=int)
    else:
        raise Exception("Invalid algorithm name")
    return solution


def solve(problem: np.ndarray,
          k=5,
          min_cost=1e-5,
          max_retry=100,
          verbose=False) -> np.ndarray:
    hist = set()
    retry = 0

    solution = generate_initial_solution(problem, k)
    incumbent = solution.copy()
    low_cost = calculate_cost(problem, solution, k)
    cost = 1.0
    while retry < max_retry and cost > min_cost:
        decision = generate_search_space(problem, solution, k=5)
        grp, cls = select_move(decision, solution, hist)

        if grp != -1:
            solution[grp] = cls
            cost = calculate_cost(problem, solution, k=5)
            if cost < low_cost:
                low_cost = cost
                incumbent = solution.copy()
                retry = 0
                if verbose:
                    print(cost)
            else:
                retry += 1
            hist.add(solution_to_str(solution))
    return incumbent


def select_move(decision: np.ndarray,
                solution: np.ndarray,
                history: Set) -> Tuple:
    candidates = np.argsort(decision, axis=None)

    for c in candidates:
        p = np.unravel_index(c, decision.shape)
        s = solution.copy()
        s[p[0]] = p[1]
        sol_str = solution_to_str(s)

        if sol_str not in history:
            return p
    return -1, -1  # No move found!


def main():
    problem = generate_problem(num_groups=500,
                               num_classes=4,
                               min_group_size=400,
                               max_group_size=2000,
                               class_percent=np.array([0.4, 0.3, 0.2, 0.1]))
    solution = solve(problem, k=5, verbose=True)

    print(np.sum(problem, axis=0) / np.sum(problem))
    print()

    folds = [problem[solution == i] for i in range(5)]
    fold_percents = np.array([np.sum(folds[i], axis=0) / np.sum(folds[i]) for i in range(5)])
    print(fold_percents)
    print()
    print([np.sum(folds[i]) / np.sum(problem) for i in range(5)])


if __name__ == "__main__":
    main()
