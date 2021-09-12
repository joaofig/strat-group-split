import numpy as np

from numba import jit


@jit(nopython=True)
def index_to_str(idx):
    """
    Generates a string representation from an index array.

    :param idx: The NumPy boolean index array.
    :return: The string representation of the array.
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


@jit(nopython=True)
def hash_solution(solution: np.ndarray, k: int) -> str:
    """
    Calculates a string hash for the solution.
    :param solution: The solution vector.
    :param k: The number of folds.
    :return: The string hash.
    """
    s = ""
    for i in range(k-1):
        s = s + index_to_str(solution == i)
    return s


@jit(nopython=True)
def is_in(element, test_elements):
    """
    Predicate to test the inclusion of items in the first array on the second

    :param element: Array whose elements we want to test the inclusion for
    :param test_elements: Target array
    :return: Boolean array of the same size as `element` with the element-wise inclusion test results
    """
    unique = set(test_elements)
    result = np.zeros_like(element, dtype=np.bool_)
    for i in range(element.shape[0]):
        result[i] = element[i] in unique
    return result


@jit(nopython=True)
def get_sort_index(solution: np.ndarray, cs: np.ndarray, k: int, arr_ix: np.ndarray) -> np.ndarray:
    """

    :param solution: The solution vector.
    :param cs: The cosine similarity matrix.
    :param k: The selected fold index [0..K)
    :param arr_ix: A pre-calculated integer range [0..N).
    :return:
    """
    sort_ix = np.zeros((0,), dtype=np.int_)
    solution_indices_for_k = solution == k
    n = solution_indices_for_k.sum()

    # Check if there are any indexes for fold k
    if n > 0:
        # Get the descending sort indices for the similarities of fold k.
        # Lower similarities mean larger differences.
        sort_ix = np.flip(np.argsort(cs[k]))
        # Filter the solution indices that belong to fold k.
        sort_ix = sort_ix[is_in(sort_ix, arr_ix[solution_indices_for_k])]
    return sort_ix


@jit(nopython=True)
def cosine_similarity(problem: np.ndarray, cost_grad: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity vector between the problem array
    and the cost gradient vector.
    :param problem: The problem array.
    :param cost_grad: The cost gradient matrix.
    :return: The cosine similarity vector.
    """
    k = cost_grad.shape[0]
    s = np.zeros((k, problem.shape[0]))
    c = problem
    norm_c = np.zeros(problem.shape[0])
    for i in range(problem.shape[0]):
        norm_c[i] = np.linalg.norm(c[i])
    for i in range(k):
        g = cost_grad[i]
        a = np.dot(c, g)
        b = np.multiply(norm_c, np.linalg.norm(g))
        s[i, :] = np.divide(a, b)
    return s


@jit(nopython=True)
def vector_similarity(v0: np.ndarray, v1: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    :param v0: Vector
    :param v1: Vector
    :return: Similarity scalar.
    """
    a = np.dot(v0, v1)
    b = np.linalg.norm(v0) * np.linalg.norm(v1)
    return a / b * np.linalg.norm(v0 - v1)


@jit(nopython=True)
def get_lowest_similarity(cost_grads):
    """
    Calculates the fold index pair with the lowest cost gradient
    cosine similarity.
    :param cost_grads: K-dimensional cost gradient vector.
    :return: Fold index pair with the lowest cosine similarity.
    """
    sim = 1.0
    p = (-1, -1)
    n = cost_grads.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            s = vector_similarity(cost_grads[i], cost_grads[j])
            if s < sim:
                sim = s
                p = (i, j)
    return p


@jit(nopython=True)
def get_similarities(cost_grads):
    """
    Calculates the similarity array between all pairs of cost gradients
    :param cost_grads: The cost gradient matrix
    :return: The sorted similarity array (K,3) containing rows of
             [similarity, i, j] with i != j
    """
    n = cost_grads.shape[0]
    k_count = int(n * (n - 1) / 2)
    sims = np.zeros((k_count, 3))
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = vector_similarity(cost_grads[i], cost_grads[j])
            sims[k, 0] = s
            sims[k, 1] = i
            sims[k, 2] = j
            k += 1
    return sims[sims[:, 0].argsort()]


@jit(nopython=True)
def calculate_costs(problem: np.ndarray, solution: np.ndarray, k: int) -> np.ndarray:
    """
    Calculates the cost vector for the given solution.

    :param problem: The problem matrix.
    :param solution: The solution vector.
    :param k: The number of folds.
    :return: The K-dimensional cost vector.
    """
    c = problem.shape[1]
    costs = np.zeros(k)
    total_count = problem[:, 0].sum()

    for i in range(k):
        index = solution == i
        costs[i] = 0.5 * (problem[index, 0].sum() - total_count / k) ** 2
        stratum_sum = problem[index, 0].sum()
        for j in range(1, c):
            r = problem[:, j].sum() / total_count
            costs[i] += 0.5 * (problem[index, j].sum() - r * stratum_sum) ** 2
    return costs


@jit(nopython=True)
def calculate_cost(problem: np.ndarray, solution: np.ndarray, k: int) -> float:
    """
    Calculates the overall cost as the L2 norm of the cost vector.

    :param problem: The problem matrix.
    :param solution: The solution vector.
    :param k: The number of folds.
    :return: The scalar cost.
    """
    return np.linalg.norm(calculate_costs(problem, solution, k))


@jit(nopython=True)
def calculate_cost_gradients(problem: np.ndarray, solution: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the K cost gradients.

    :param problem: The problem matrix.
    :param solution: The solution vector.
    :param k: The number of folds.
    :return: The (K,C) gradient matrix.
    """
    c = problem.shape[1]
    gradients = np.zeros((k, c))
    total_count = problem[:, 0].sum()

    for i in range(k):
        index = solution == i
        gradients[i, 0] = problem[index, 0].sum() - total_count / k
        stratum_sum = problem[index, 0].sum()
        for j in range(1, c):
            r = problem[:, j].sum() / total_count
            gradients[i, j] = problem[index, j].sum() - r * stratum_sum
    return gradients
