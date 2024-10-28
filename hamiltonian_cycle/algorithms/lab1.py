import numpy as np
import pandas as pd


def init_random_solution(
    ds: pd.DataFrame, dm: pd.DataFrame, start: int
) -> pd.DataFrame:
    return ds.sample(n=int(len(ds) * 0.5 + 0.5))


def init_nearest_neighbor_end(
    ds: pd.DataFrame, dm: pd.DataFrame, start: int
) -> pd.Series:
    size = int(len(ds) * 0.5 + 0.5)
    num_nodes = len(ds)

    dm = dm.copy()

    remaining_mask = np.ones(num_nodes, dtype=bool)
    solution = [start]
    remaining_mask[start] = False

    while len(solution) < size:
        last_node = solution[-1]

        distances_to_last_node = dm[last_node]
        distances_to_last_node[~remaining_mask] = np.inf

        nearest_node = np.argmin(distances_to_last_node)
        solution.append(nearest_node)
        remaining_mask[nearest_node] = False

    return ds.loc[solution]


def init_nearest_neighbor_best_position(
    ds: pd.DataFrame, dm: pd.DataFrame, start: int
) -> pd.Series:
    size = int(len(ds) * 0.5 + 0.5)
    num_nodes = len(ds)
    solution = [start]
    remaining_nodes = set(range(num_nodes)) - {start}

    while len(solution) < size:
        best_insertion_cost = float("inf")
        best_insertion = None

        for node in remaining_nodes:
            # Find the best position for the current node without closing the path
            for i in range(len(solution) + 1):
                if i == 0:
                    prev_node = solution[0]
                    insert_cost = dm[node, prev_node]
                elif i == len(solution):
                    prev_node = solution[-1]
                    insert_cost = dm[prev_node, node]
                else:
                    prev_node = solution[i - 1]
                    next_node = solution[i]
                    insert_cost = (
                        dm[prev_node, node]
                        + dm[node, next_node]
                        - dm[prev_node, next_node]
                    )

                if insert_cost < best_insertion_cost:
                    best_insertion_cost = insert_cost
                    best_insertion = (node, i)

        solution.insert(best_insertion[1], best_insertion[0])
        remaining_nodes.remove(best_insertion[0])

    return ds.loc[solution]


def init_greedy_cycle(ds: pd.DataFrame, dm: pd.DataFrame, start: int) -> pd.Series:
    size = int(len(ds) * 0.5 + 0.5)
    num_nodes = len(ds)

    dm = dm.copy()

    remaining_nodes = set(range(num_nodes))
    remaining_nodes.remove(start)
    solution = [start]

    nearest_node = np.argmin(dm[start, list(remaining_nodes)])
    nearest_node_idx = list(remaining_nodes)[nearest_node]
    solution.append(nearest_node_idx)
    remaining_nodes.remove(nearest_node_idx)

    while len(solution) < size:
        best_insertion_cost = float("inf")
        best_insertion = None

        for node_idx in remaining_nodes:
            for i in range(len(solution)):
                next_i = (i + 1) % len(solution)

                current_cost = (
                    dm[solution[i], node_idx]
                    + dm[node_idx, solution[next_i]]
                    - dm[solution[i], solution[next_i]]
                )

                if current_cost < best_insertion_cost:
                    best_insertion_cost = current_cost
                    best_insertion = (node_idx, i)

        solution.insert(best_insertion[1] + 1, best_insertion[0])
        remaining_nodes.remove(best_insertion[0])

    return ds.loc[solution]
