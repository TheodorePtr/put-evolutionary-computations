import numpy as np
import pandas as pd


def init_greedy_2regret_cycle(
    ds: pd.DataFrame, dm: pd.DataFrame, start: int
) -> pd.Series:
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
        best_regret = float("-inf")
        best_node = None
        best_insertion = None

        for node_idx in remaining_nodes:
            best_cost = float("inf")
            second_best_cost = float("inf")
            best_position = None

            for insertion_idx in range(len(solution)):
                next_idx = (insertion_idx + 1) % len(solution)

                current_cost = (
                    dm[solution[insertion_idx], node_idx]
                    + dm[node_idx, solution[next_idx]]
                    - dm[solution[insertion_idx], solution[next_idx]]
                )

                if current_cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = current_cost
                    best_position = insertion_idx
                elif current_cost < second_best_cost:
                    second_best_cost = current_cost

            regret = second_best_cost - best_cost

            if regret > best_regret:
                best_regret = regret
                best_node = node_idx
                best_insertion = best_position

        solution.insert(best_insertion + 1, best_node)
        remaining_nodes.remove(best_node)

    return ds.loc[solution]


def init_greedy_2regret_weighted_cycle(
    ds: pd.DataFrame,
    dm: pd.DataFrame,
    start: int,
    w_cost: float,
    w_regret: float,
    initial_solution: list = None,
) -> pd.Series:
    size = int(len(ds) * 0.5 + 0.5)
    num_nodes = len(ds)

    dm = dm.copy()

    # Start with the provided initial solution or construct one
    if initial_solution is None:
        remaining_nodes = set(range(num_nodes))
        remaining_nodes.remove(start)
        solution = [start]

        # Add the nearest node to start the solution
        nearest_node = np.argmin(dm[start, list(remaining_nodes)])
        nearest_node_idx = list(remaining_nodes)[nearest_node]
        solution.append(nearest_node_idx)
        remaining_nodes.remove(nearest_node_idx)
    else:
        solution = initial_solution.copy()
        remaining_nodes = set(range(num_nodes)) - set(solution)

    # Extend the solution until it reaches the desired size
    while len(solution) < size:
        best_combined_criterion = float("inf")
        best_node = None
        best_insertion = None

        for node_idx in remaining_nodes:
            best_cost = float("inf")
            second_best_cost = float("inf")
            best_position = None

            for insertion_idx in range(len(solution)):
                next_idx = (insertion_idx + 1) % len(solution)

                # Greedy cycle cost (objective improvement)
                current_cost = (
                    dm[solution[insertion_idx], node_idx]
                    + dm[node_idx, solution[next_idx]]
                    - dm[solution[insertion_idx], solution[next_idx]]
                )

                if current_cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = current_cost
                    best_position = insertion_idx
                elif current_cost < second_best_cost:
                    second_best_cost = current_cost

            # 2-regret calculation
            regret = second_best_cost - best_cost

            # Weighted sum criterion (greedy + 2-regret)
            combined_criterion = w_cost * best_cost - w_regret * regret

            if combined_criterion < best_combined_criterion:
                best_combined_criterion = combined_criterion
                best_node = node_idx
                best_insertion = best_position

        # Insert the best node at the best position
        solution.insert(best_insertion + 1, best_node)
        remaining_nodes.remove(best_node)

    return ds.loc[solution]
