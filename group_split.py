import numpy as np
import time

from numpy.random import default_rng
from numba import jit
from collections import namedtuple

RANDOM_SEED = 23
Solution = namedtuple("Solution", "cost index")


@jit(nopython=True)
def index_to_str(idx):
    """
    Generates a string representation from an index array.
    @param idx: The NumPy boolean index array.
    @return: The string representation of the array.
    """
    num_chars = int(idx.shape[0] / 6 + 0.5)
    s = ""
    for i in range(num_chars):
        b = i * 6
        six = idx[b:b+6]
        c = 0
        for j in range(six.shape[0]):
            c = c * 2 + int(six[j])
        s = s + chr(c+32)
    return s


def create_initial_solution(sample_counts, p):
    """
    Creates the initial solution using a random shuffle
    @param sample_counts: The problem array.
    @param p: The validation set proportion.
    @return: A solution array.
    """
    rng = default_rng(seed=RANDOM_SEED)
    group_count = sample_counts.shape[0]
    idx = np.zeros(group_count, dtype=bool)
    rnd_idx = rng.permutation(group_count)
    start_count = 0
    sample_size = sample_counts[:, 0].sum()
    for i in range(group_count):
        start_count += sample_counts[rnd_idx[i], 0]
        idx[rnd_idx[i]] = True
        if start_count > sample_size * p:
            break
    return idx


@jit(nopython=True)
def calculate_cost(sample_counts, idx, p):
    """
    Calculates the cost of a given solution
    @param sample_counts: The problem array.
    @param idx: The solution array to evaluate.
    @param p: The train/validation split proportion.
    @return: The cost value.
    """
    train_count = sample_counts[~idx, 0].sum()
    valid_count = sample_counts[idx, 0].sum()
    total_count = train_count + valid_count

    cost = (valid_count - total_count * p) ** 2

    for i in range(1, sample_counts.shape[1]):
        r = sample_counts[:, i].sum() / total_count
        cost += (sample_counts[idx, i].sum() - r * sample_counts[idx, 0].sum()) ** 2
    return cost / 2.0


def calculate_cost_grad(sample_counts, idx, p):
    """
    Calculates the cost gradient of a given solution
    @param sample_counts: The problem array.
    @param idx: The solution array to evaluate.
    @param p: The train/validation split proportion.
    @return: The cost value.
    """
    grad = np.zeros(sample_counts.shape[1])

    total_count = sample_counts[:, 0].sum()
    valid_count = sample_counts[idx, 0].sum()

    grad[0] = total_count * p - valid_count

    for i in range(1, sample_counts.shape[1]):
        r = sample_counts[:, i].sum() / total_count
        grad[i] = r * sample_counts[idx, 0].sum() - sample_counts[idx, i].sum()
    return grad


def cosine_similarity(sample_counts, idx, cost_grad):
    """
    Calculates the cosine similarity vector between the problem array
    and the cost gradient vector
    @param sample_counts: The problem array.
    @param idx: The solution vector.
    @param cost_grad: The cost gradient vector.
    @return: The cosine similarity vector.
    """
    c = np.copy(sample_counts)
    c[idx] = -c[idx]            # Reverse direction of validation vectors
    a = np.inner(c, cost_grad)
    b = np.multiply(np.linalg.norm(c, axis=1), np.linalg.norm(cost_grad))
    return np.divide(a, b)


def euclidean_similarity(sample_counts, idx, cost_grad):
    c = np.copy(sample_counts)
    c[idx] = -c[idx]
    return np.linalg.norm(c - cost_grad, axis=1)


def generate_cosine_move(sample_counts, idx, p, expanded_set, intensify):
    """
    Generates a new move using the cosine similarity.
    @param sample_counts: The problem array.
    @param idx: The solution vector.
    @param p: The validation set proportion.
    @param expanded_set: The set of expanded solutions.
    @param intensify: Intensification / diversification flag.
    @return: The new solution vector.
    """
    cost_grad = calculate_cost_grad(sample_counts, idx, p)
    similarity = cosine_similarity(sample_counts, idx, cost_grad)
    sorted_ixs = np.argsort(similarity)
    if intensify:
        sorted_ixs = np.flip(sorted_ixs)
    for i in sorted_ixs:
        move = np.copy(idx)
        move[i] = not move[i]
        if index_to_str(move) not in expanded_set:
            return move
    return None


def generate_euclidean_move(sample_counts, idx, p, expanded_set, get_min):
    cost_grad = calculate_cost_grad(sample_counts, idx, p)
    similarity = euclidean_similarity(sample_counts, idx, cost_grad)
    sorted_ixs = np.argsort(similarity)
    if not get_min:
        sorted_ixs = np.flip(sorted_ixs)
    for i in sorted_ixs:
        move = np.copy(idx)
        move[i] = not move[i]
        if index_to_str(move) not in expanded_set:
            return move
    return None


def generate_moves(idx, expanded_set):
    """
    Generator for all acceptable moves from a previous solution.
    @param idx: The solution vector.
    @param expanded_set: The set of expanded solutions.
    """
    for i in range(idx.shape[0]):
        move = np.copy(idx)
        move[i] = not move[i]
        if index_to_str(move) not in expanded_set:
            yield move


def generate_counts(num_groups, num_classes,
                    min_group_size, max_group_size,
                    max_group_percent):
    """
    Generates a problem matrix from the given parameters.
    @param num_groups: The number of data groups.
    @param num_classes: The number of classes.
    @param min_group_size: The minimum group size.
    @param max_group_size: The maximum group size.
    @param max_group_percent: The maximum class percent.
    @return: The problem matrix.
    """
    rng = default_rng(seed=RANDOM_SEED)
    sample_cnt = np.zeros((num_groups, num_classes), dtype=int)
    sample_cnt[:, 0] = rng.integers(low=min_group_size, high=max_group_size, size=num_groups)

    for i in range(1, num_groups):
        for j in range(1, num_classes):
            sample_cnt[i, j] = rng.integers(low=0, high=max_group_percent * sample_cnt[i, 0] - sample_cnt[i, 1:j+1].sum())
    return sample_cnt


class BaseSolver(object):

    def __init__(self, problem, candidate, p):
        self.problem = problem
        self.p = p
        self.incumbent = Solution(calculate_cost(problem, candidate, p), candidate)


class SearchSolver(BaseSolver):

    def __init__(self, problem, candidate, p):
        super(SearchSolver, self).__init__(problem, candidate, p)

    def solve(self, min_cost, max_empty_iterations,
              max_intensity_iterations, verbose=True):
        """
        Uses the search solver to calculate the best split.
        @param min_cost: Minimum cost criterion.
        @param max_empty_iterations: Maximum number of non-improving iterations.
        @param max_intensity_iterations: Maximum number of intensity iterations.
        @param verbose: Verbose flag.
        @return: The incumbent solution.
        """
        terminated = False
        intensify = True
        expanded_set = set()
        solution = self.incumbent
        n = 0
        n_intensity = 0

        while not terminated:
            move_list = generate_moves(solution.index, expanded_set)
            cost_list = [Solution(calculate_cost(self.problem, move, self.p), move)
                         for move in move_list]
            cost_list.sort(key=lambda t: t[0], reverse=not intensify)
            intensify = True

            if len(cost_list) > 0:
                solution = cost_list[0]
                expanded_set.add(index_to_str(solution.index))
                if solution.cost < self.incumbent.cost:
                    self.incumbent = solution
                    n = 0
                    n_intensity = 0
                    intensify = True
                    if verbose:
                        print(self.incumbent.cost)
                else:
                    # Diversify?
                    if n_intensity > max_intensity_iterations:
                        intensify = False
                        n_intensity = 0
            else:
                terminated = True

            n += 1
            n_intensity += 1
            if n > max_empty_iterations or self.incumbent.cost < min_cost:
                terminated = True
        return self.incumbent


class GradientSolver(BaseSolver):

    def __init__(self, problem, candidate, p):
        super(GradientSolver, self).__init__(problem, candidate, p)

    def solve(self, min_cost, max_empty_iterations,
              max_intensity_iterations, verbose=True):
        """
        Uses the gradient solver to calculate the best split.
        @param min_cost: Minimum cost criterion.
        @param max_empty_iterations: Maximum number of non-improving iterations.
        @param max_intensity_iterations: Maximum number of intensity iterations.
        @param verbose: Verbose flag.
        @return: The incumbent solution.
        """
        terminated = False
        intensify = True
        expanded_set = set()
        solution = self.incumbent
        n = 0
        n_intensity = 0

        while not terminated:
            move = generate_cosine_move(self.problem, solution.index, self.p,
                                        expanded_set, intensify)
            intensify = True

            if move is not None:
                solution = Solution(calculate_cost(self.problem, move, self.p), move)
                expanded_set.add(index_to_str(solution.index))
                if solution.cost < self.incumbent.cost:
                    self.incumbent = solution
                    n = 0
                    n_intensity = 0

                    if verbose:
                        print(self.incumbent.cost)
                else:
                    if n_intensity > max_intensity_iterations:
                        intensify = False
                        n_intensity = 0
            else:
                terminated = True

            n += 1
            n_intensity += 1
            if n > max_empty_iterations or self.incumbent.cost < min_cost:
                terminated = True

        return self.incumbent


def print_solution(problem, solution, p):
    idx = solution.index
    valid_count = problem[idx, 0].sum()
    train_count = problem[~idx, 0].sum()
    total_count = train_count + valid_count

    print(solution.cost)
    print(p, valid_count / total_count)
    for i in range(1, problem.shape[1]):
        r = problem[:, i].sum() / total_count
        print(r, problem[idx, i].sum() / problem[idx, 0].sum())


def main():
    num_groups = 250            # Number of groups to simulate
    num_classes = 2             # Number of classes
    max_group_size = 10000      # Maximum group size
    max_group_perc = 0.4        # Maximum proportion for each class
    p = 0.3                     # Validation split proportion

    max_empty_iterations = 100
    max_intensity_iterations = 10
    min_cost = 10000

    sample_cnt = generate_counts(num_groups, num_classes,
                                 min_group_size=10,
                                 max_group_size=max_group_size,
                                 max_group_percent=max_group_perc)

    solution_arr = create_initial_solution(sample_cnt, p)

    s_solver = SearchSolver(sample_cnt, solution_arr, p)
    g_solver = GradientSolver(sample_cnt, solution_arr, p)

    start = time.time()
    solution = s_solver.solve(min_cost, max_empty_iterations, max_intensity_iterations, verbose=False)
    print(time.time() - start)

    print_solution(sample_cnt, solution, p)
    print()

    start = time.time()
    solution = g_solver.solve(min_cost, max_empty_iterations, max_intensity_iterations, verbose=False)
    print(time.time() - start)

    print_solution(sample_cnt, solution, p)


if __name__ == "__main__":
    main()
