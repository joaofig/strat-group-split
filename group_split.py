import numpy as np

from numpy.random import default_rng
from numba import jit


@jit(nopython=True)
def index_to_str(idx):
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


def create_initial_solution(sample_counts, group_count, p):
    rng = default_rng()
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
    train_count = sample_counts[~idx, 0].sum()
    valid_count = sample_counts[idx, 0].sum()
    total_count = train_count + valid_count

    cost = (valid_count - total_count * p) ** 2

    for i in range(1, sample_counts.shape[1]):
        r = sample_counts[:, i].sum() / total_count
        cost += (sample_counts[idx, i].sum() - r * sample_counts[idx, 0].sum()) ** 2

    return cost


def list_move_to_x(bool_idx, move_to_valid):
    if move_to_valid:
        idx = ~bool_idx
    else:
        idx = bool_idx
    indices = np.arange(bool_idx.shape[0], dtype=int)[idx]
    move_list = []
    for ix in indices:
        updated_idx = np.copy(bool_idx)
        updated_idx[ix] = move_to_valid
        move_list.append(updated_idx)
    return move_list


def main():
    num_groups = 500
    num_classes = 2
    max_group_size = 5000
    max_group_perc = 0.4
    p = 0.3
    max_empty_iterations = 500

    expanded_set = set()

    rng = default_rng()
    sample_cnt = np.zeros((num_groups, num_classes), dtype=int)
    sample_cnt[:, 0] = rng.integers(low=10, high=max_group_size, size=num_groups)
    for i in range(num_groups):
        sample_cnt[i, 1] = rng.integers(low=0, high=max_group_perc * sample_cnt[i, 0])

    idx = create_initial_solution(sample_cnt, num_groups, p)
    cost = calculate_cost(sample_cnt, idx, p)
    incumbent = (cost, idx)
    expanded_set.add(index_to_str(idx))

    print(cost)

    terminated = False

    iter = 0

    while not terminated:
        move_to_valid = (rng.integers(low=0, high=2) == 1)
        move_list = list_move_to_x(idx, move_to_valid=move_to_valid)
        cost_list = [(calculate_cost(sample_cnt, move, p), move)
                     for move in move_list
                     if index_to_str(move) not in expanded_set]
        cost_list.sort(key=lambda t: t[0])

        if len(cost_list) > 0:
            solution = cost_list[0]
            expanded_set.add(index_to_str(solution[1]))
            if solution[0] < incumbent[0]:
                incumbent = solution
                print(incumbent[0])
                iter = 0
            idx = solution[1]
        else:
            terminated = True

        iter += 1
        if iter > max_empty_iterations:
            terminated = True

    idx = incumbent[1]
    valid_count = sample_cnt[idx, 0].sum()
    train_count = sample_cnt[~idx, 0].sum()
    total_count = train_count + valid_count

    r = sample_cnt[:, 1].sum() / total_count

    print("------------")
    print(p, valid_count / total_count)
    print(r, sample_cnt[idx, 1].sum() / sample_cnt[idx, 0].sum())


if __name__ == "__main__":
    main()
