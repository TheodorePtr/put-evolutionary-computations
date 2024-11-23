import random
from typing import Literal

import numpy as np
import pandas as pd

from hamiltonian_cycle.algorithms.lab1 import (
    init_greedy_cycle,
    init_random_solution,
)
from hamiltonian_cycle.costs import function_cost

SEARCH_STRATEGY_TYPE = Literal["greedy", "steepest"]
INTRA_SEARCH_TYPE = Literal["node", "edge"]


def objective_change_two_nodes(dm: np.ndarray, solution: list, i: int, j: int) -> float:
    if i == j:
        return np.inf

    n = len(solution)
    a, b = solution[i], solution[j]

    a_prev = solution[i - 1] if i > 0 else solution[-1]
    a_next = solution[(i + 1) % n]
    b_prev = solution[j - 1] if j > 0 else solution[-1]
    b_next = solution[(j + 1) % n]

    # Edges to be removed and added
    edges_removed = []
    edges_added = []

    # Remove edges connected to a and b
    if a_prev not in (a, b):
        edges_removed.append((a_prev, a))
        edges_added.append((a_prev, b))
    if a_next not in (a, b):
        edges_removed.append((a, a_next))
        edges_added.append((b, a_next))
    if b_prev not in (a, b):
        edges_removed.append((b_prev, b))
        edges_added.append((b_prev, a))
    if b_next not in (a, b):
        edges_removed.append((b, b_next))
        edges_added.append((a, b_next))

    # Calculate delta
    delta = -sum(dm[u, v] for u, v in edges_removed) + sum(
        dm[u, v] for u, v in edges_added
    )

    return delta


def objective_change_two_edges(dm: np.ndarray, solution: list, i: int, j: int) -> float:
    if i >= j or (i == 0 and j == len(solution) - 1):
        return np.inf

    n = len(solution)
    a_prev = solution[i - 1] if i > 0 else solution[-1]
    a = solution[i]
    b = solution[j]
    b_next = solution[(j + 1) % n]

    # Edges before and after reversal
    cost_before = dm[a_prev, a] + dm[b, b_next]
    cost_after = dm[a_prev, b] + dm[a, b_next]

    delta = cost_after - cost_before

    return delta


def objective_change_inter_route(
    dm: np.ndarray, solution: list, i: int, vacant_node: int, node_costs: list
) -> float:
    n = len(solution)
    node_in_solution = solution[i]
    if node_in_solution == vacant_node:
        return np.inf

    prev_node = solution[i - 1] if i > 0 else solution[-1]
    next_node = solution[(i + 1) % n]

    # Edge costs before and after the swap
    edge_cost_before = dm[prev_node, node_in_solution] + dm[node_in_solution, next_node]
    edge_cost_after = dm[prev_node, vacant_node] + dm[vacant_node, next_node]

    # Node costs before and after the swap
    node_cost_before = node_costs[node_in_solution]
    node_cost_after = node_costs[vacant_node]

    delta = (edge_cost_after - node_cost_after) - (edge_cost_before - node_cost_before)

    return delta


def two_nodes_exchange(solution: list, i: int, j: int) -> list:
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


def two_edges_exchange(solution: list, i: int, j: int) -> list:
    if i >= j:
        return solution.copy()  # No change if i >= j

    new_solution = solution.copy()
    new_solution[i : j + 1] = new_solution[i : j + 1][::-1]
    return new_solution


def inter_route_swap(
    solution: list,
    i: int,
    vacant_node: int,
    selected_nodes: set,
    non_selected_nodes: set,
) -> tuple:
    new_solution = solution.copy()
    node_in_solution = new_solution[i]
    new_solution[i] = vacant_node

    # Update the node sets
    selected_nodes = selected_nodes.copy()
    non_selected_nodes = non_selected_nodes.copy()

    selected_nodes.remove(node_in_solution)
    selected_nodes.add(vacant_node)
    non_selected_nodes.remove(vacant_node)
    non_selected_nodes.add(node_in_solution)

    return new_solution, selected_nodes, non_selected_nodes


def get_remaining_nodes(selected_nodes: set, num_nodes: int) -> set:
    return set(range(num_nodes)) - selected_nodes


def browse_intra_solutions(
    dm: np.ndarray,
    solution: list,
    intra_search: str,
    strategy: SEARCH_STRATEGY_TYPE,
    candidate_neighbors: np.ndarray | None,
) -> tuple:
    intra_neighbors = []
    n = len(solution)
    for i in range(n):
        if candidate_neighbors is not None:
            inter_candidates_id = [
                solution.index(j)
                for j in candidate_neighbors[i]
                if j in solution and solution.index(j) > i
            ]
            neighbors_to_check = inter_candidates_id
        else:
            neighbors_to_check = range(i + 1, n)

        for j in neighbors_to_check:
            if intra_search == "node":
                delta_nodes = objective_change_two_nodes(dm, solution, i, j)
                if delta_nodes < 0:
                    intra_neighbors.append((i, j, delta_nodes, "node"))
            elif intra_search == "edge":
                delta_edges = objective_change_two_edges(dm, solution, i, j)
                if delta_edges < 0:
                    intra_neighbors.append((i, j, delta_edges, "edge"))

            if strategy == "greedy" and intra_neighbors:
                return intra_neighbors
    return intra_neighbors


def browse_inter_solutions(
    dm: np.ndarray,
    solution: list,
    non_selected_nodes: set,
    costs: list,
    strategy: SEARCH_STRATEGY_TYPE,
    candidate_neighbors: np.ndarray | None,
) -> list:
    inter_neighbors = []
    for i in range(len(solution)):
        neighbors_to_check = non_selected_nodes
        if candidate_neighbors is not None:
            neighbors_to_check = neighbors_to_check.intersection(candidate_neighbors[i])

        for vacant_node in neighbors_to_check:
            inter_delta = objective_change_inter_route(
                dm, solution, i, vacant_node, costs
            )
            if inter_delta < 0:
                inter_neighbors.append((i, vacant_node, inter_delta, "inter"))
                if strategy == "greedy":
                    return inter_neighbors
    return inter_neighbors


def update_solution(
    solution: list,
    best_neighbor: tuple,
    selected_nodes: set,
    non_selected_nodes: set,
) -> tuple:
    neighbor_type = best_neighbor[-1]

    if neighbor_type == "node":
        i, j = best_neighbor[:2]
        solution = two_nodes_exchange(solution, int(i), int(j))
    elif neighbor_type == "edge":
        i, j = best_neighbor[:2]
        solution = two_edges_exchange(solution, int(i), int(j))
    elif neighbor_type == "inter":
        i, vacant_node = best_neighbor[:2]
        solution, selected_nodes, non_selected_nodes = inter_route_swap(
            solution, int(i), int(vacant_node), selected_nodes, non_selected_nodes
        )
    return solution, selected_nodes, non_selected_nodes


def local_search(
    ds: pd.DataFrame,
    dm: np.ndarray,
    initial_solution: list,
    strategy: SEARCH_STRATEGY_TYPE = "greedy",
    intra_search: INTRA_SEARCH_TYPE = "edge",
    use_candidates_heurtistic: bool = False,
    debug_mode: bool = True,
) -> pd.DataFrame:

    candidate_neighbors = None
    if use_candidates_heurtistic:
        candidate_neighbors = np.argsort(dm, axis=1)[:, 1:11]

    num_nodes = len(dm)
    initial_cost = function_cost(ds.loc[initial_solution])

    solution = initial_solution.copy()
    selected_nodes = set(initial_solution)
    non_selected_nodes = get_remaining_nodes(selected_nodes, num_nodes)

    while True:
        intra_neighbors = browse_intra_solutions(
            dm, solution, intra_search, strategy, candidate_neighbors
        )
        inter_neighbors = browse_inter_solutions(
            dm,
            solution,
            non_selected_nodes,
            ds["cost"].tolist(),
            strategy,
            candidate_neighbors,
        )

        all_neighbors = intra_neighbors + inter_neighbors

        if not all_neighbors:
            # No improvement found
            break

        if strategy == "greedy":
            random.shuffle(all_neighbors)
            best_neighbor = next((n for n in all_neighbors if n[2] < 0), None)
        elif strategy == "steepest":
            best_neighbor = min(all_neighbors, key=lambda x: x[2])

        if best_neighbor is None:
            # No improving neighbor found
            break

        # Save the old solution for debugging
        old_solution = solution.copy()

        # Update solution and cost
        solution, selected_nodes, non_selected_nodes = update_solution(
            solution, best_neighbor, selected_nodes, non_selected_nodes
        )
        initial_cost += best_neighbor[2]

        if debug_mode:
            # Calculate real improvement
            real_improvement = function_cost(ds.loc[old_solution]) - function_cost(
                ds.loc[solution]
            )

            if real_improvement != -best_neighbor[2]:
                print(f"Promised improvement: {best_neighbor[2]}")
                print(f"Real improvement: {real_improvement}")
                print(f"Operation: {best_neighbor[-1]}")
                print("===========")

            assert function_cost(ds.loc[solution]) < function_cost(ds.loc[old_solution])

    return ds.loc[solution]


def init_local_search(
    ds: pd.DataFrame,
    dm: np.ndarray,
    start: int,
    strategy: str,
    intra_search: str,
    debug_mode: bool = True,
    algo_to_enchance: str = "greedy_cycle",
) -> pd.DataFrame:
    if algo_to_enchance == "greedy_cycle":
        solution = list(init_greedy_cycle(ds, dm, start).index)
    elif algo_to_enchance == "random":
        solution = list(init_random_solution(ds, dm, start).index)
    solution = local_search(ds, dm, solution, strategy, intra_search, debug_mode)
    return solution
