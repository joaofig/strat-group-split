import numpy as np

from numpy.random import default_rng
from collections import namedtuple
from tools import History, get_sort_index, hash_solution
from tools import calculate_cost, calculate_costs, calculate_cost_gradients
from tools import cosine_similarity, get_similarities

RANDOM_SEED = 5005  # 2310
Solution = namedtuple("Solution", "cost index")


def generate_problem(num_groups: int, num_classes: int,
                     min_group_size: int, max_group_size: int,
                     max_group_percent: float) -> np.ndarray:
    """
    Generates a problem matrix from the given parameters.
    :param num_groups: The number of data groups.
    :param num_classes: The number of classes.
    :param min_group_size: The minimum group size.
    :param max_group_size: The maximum group size.
    :param max_group_percent: The maximum class percent.
    :return: The problem matrix (N,C).
    """
    rng = default_rng(seed=RANDOM_SEED)
    problem = np.zeros((num_groups, num_classes))
    problem[:, 0] = rng.integers(low=min_group_size,
                                 high=max_group_size,
                                 size=num_groups)

    for i in range(1, num_groups):
        top_i = max_group_percent * problem[i, 0]
        for j in range(1, num_classes):
            h = top_i - problem[i, 1:j+1].sum()
            problem[i, j] = rng.integers(low=0, high=h)
    return problem


def print_problem(problem, solution=None):
    print("")
    if solution is None:
        for g in problem:
            print("\t".join([str(int(n)) for n in g]))
    else:
        for g in zip(problem, solution):
            print("{0}\t{1}".format("\t".join([str(int(n)) for n in g[0]]), g[1]))


def print_solution(problem, solution, k):
    """
    Prints the solution to the console.

    :param problem: The problem matrix.
    :param solution: Th solution vector.
    :param k: The number of folds.
    """
    num_classes = problem.shape[1]
    total_count = problem[:, 0].sum()

    print(calculate_cost(problem, solution, k))

    print("")
    print("K-Fold Partitioning")
    print(total_count, total_count / k)
    for i in range(k):
        index = solution == i
        print(i+1, problem[index, 0].sum())

    for j in range(1, num_classes):
        print("")
        print("Class {0}".format(j))

        for i in range(k):
            index = solution == i
            print(i+1, problem[index, j].sum() / problem[index, 0].sum())


# @jit(nopython=True)
def add_solution(tabu, solution_arr, cs, k, k0, k1, arr_ix):
    """

    :param tabu:
    :param solution_arr:
    :param cs:
    :param k: The number of folds in the problem (integer)
    :param k0:
    :param k1:
    :param arr_ix:
    :return:
    """
    sort_ix0 = get_sort_index(solution_arr, cs, k0, arr_ix)
    sort_ix1 = get_sort_index(solution_arr, cs, k1, arr_ix)

    n0 = sort_ix0.shape[0]
    n1 = sort_ix1.shape[0]

    i0, i1 = 0, 0
    solution_added = False
    while not solution_added and i0 < sort_ix0.shape[0] and i1 < sort_ix1.shape[0]:
        solution = np.copy(solution_arr)
        if n0 > 0 and n1 > 0:
            if cs[k0, sort_ix0[i0]] > cs[k1, sort_ix1[i1]]:
                solution[sort_ix0[i0]] = k1
                i0 += 1
            else:
                solution[sort_ix1[i1]] = k0
                i1 += 1
        elif n0 > 0:
            solution[sort_ix0[i0]] = k1
            i0 += 1
        else:
            solution[sort_ix1[i1]] = k0
            i1 += 1

        h = hash_solution(solution, k)
        if h not in tabu:
            tabu.add(h)
            solution_added = True
            solution_arr = solution

    return solution_arr, solution_added


def main():
    history = History()
    num_groups = 20             # Number of groups to simulate
    num_classes = 2             # Number of classes
    max_group_size = 10000      # Maximum group size
    max_group_percent = 0.4     # Maximum proportion for each class
    k = 5                       # Number of folds

    max_empty_iterations = 10000
    max_intensity_iterations = 10
    min_cost = 1000
    terminated = False

    tabu = set()

    problem = generate_problem(num_groups, num_classes,
                               min_group_size=10,
                               max_group_size=max_group_size,
                               max_group_percent=max_group_percent)

    rng = default_rng(seed=RANDOM_SEED)
    solution = rng.integers(low=0, high=k, size=num_groups)
    # solution = np.zeros(problem.shape[0], dtype=int)

    arr_ix = np.arange(num_groups, dtype=int)

    tabu.add(hash_solution(solution, k))

    incumbent_solution = solution
    incumbent_cost = calculate_cost(problem, incumbent_solution, k)
    print(incumbent_cost)
    history.add(solution)
    
    n = 0
    n_intensity = 0
    solution_added = False

    while not terminated:
        cost_grad = calculate_cost_gradients(problem, solution, k)

        cs = cosine_similarity(problem, cost_grad)

        sims = get_similarities(cost_grad)
        
        for sim in sims:
            k0, k1 = int(sim[1]), int(sim[2])   # Cast the indices from double to int

            solution, solution_added = add_solution(tabu, solution, cs, k, k0, k1, arr_ix)

            if solution_added:
                break   # Breaks the 'sims' loop

        if not solution_added:
            print("Solution not added!")

        cost = calculate_cost(problem, solution, k)
        if cost < incumbent_cost:
            print(cost)
            incumbent_cost = cost
            incumbent_solution = solution
            n = 0
            n_intensity = 0
            history.add(solution)

        n += 1
        n_intensity += 1
        if n > max_empty_iterations or incumbent_cost < min_cost:
            terminated = True

    print(calculate_costs(problem, incumbent_solution, k))
    print(len(tabu))
    print_solution(problem, incumbent_solution, k)

    # print_problem(problem, incumbent_solution)


if __name__ == "__main__":
    main()
