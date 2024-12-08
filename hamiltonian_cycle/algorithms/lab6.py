import random

import numpy as np
import pandas as pd

from hamiltonian_cycle.algorithms.lab1 import init_random_solution
from hamiltonian_cycle.algorithms.lab3_4 import (
    NODE_ID,
    POSITION_IN_SOLUTION,
    LocalSearch,
    Node,
    get_remaining_nodes,
    # inter_route_swap,
)
from hamiltonian_cycle.costs import function_cost

# Global variables if necessary
# ds_global = ds_a
# dm_global = dm_a


class MultipleStartLocalSearch:
    def __init__(self, ds: pd.DataFrame, dm: np.ndarray, iterations: int = 200):
        self.ds = ds
        self.dm = dm
        self.iterations = iterations

    def __call__(self) -> pd.DataFrame:
        best_solution = None
        best_cost = float("inf")

        for _ in range(self.iterations):
            initial_solution = list(
                init_random_solution(ds=self.ds, dm=self.dm, start=0).index
            )

            local_search = LocalSearch(
                strategy="steepest",
                intra_search="edge",
                use_candidates_heurtistic=False,
                debug_mode=False,
            )
            solution = local_search(self.ds, self.dm, initial_solution)
            cost = function_cost(solution)

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        return best_solution


def two_nodes_exchange(solution: list[NODE_ID], node_a, node_b) -> list[NODE_ID]:
    new_solution = solution.copy()
    new_solution[node_a.solution_pos], new_solution[node_b.solution_pos] = (
        new_solution[node_b.solution_pos],
        new_solution[node_a.solution_pos],
    )
    return new_solution


def two_edges_exchange(solution: list[NODE_ID], node_a, node_b) -> list[NODE_ID]:
    new_solution = solution.copy()
    new_solution[node_a.solution_pos : node_b.solution_pos + 1] = reversed(
        new_solution[node_a.solution_pos : node_b.solution_pos + 1]
    )
    return new_solution


def inter_route_swap(
    solution: list[NODE_ID],
    node_a,
    node_b,
    selected_nodes: set[NODE_ID],
    non_selected_nodes: set[NODE_ID],
) -> tuple[list[NODE_ID], set[NODE_ID], set[NODE_ID]]:
    # Replace node_a with node_b in the solution
    new_solution = solution.copy()
    new_solution[node_a.solution_pos] = node_b.id

    # Update the sets
    selected_nodes = selected_nodes.copy()
    non_selected_nodes = non_selected_nodes.copy()

    selected_nodes.remove(node_a.id)
    selected_nodes.add(node_b.id)
    non_selected_nodes.remove(node_b.id)
    non_selected_nodes.add(node_a.id)

    return new_solution, selected_nodes, non_selected_nodes


def double_bridge_perturbation(solution: list[int]) -> list[int]:
    # Ensure positions are spread out
    pos = sorted(random.sample(range(1, len(solution)), 4))
    i, j, k, l = pos

    # Segments
    segment1 = solution[:i]
    segment2 = solution[i:j]
    segment3 = solution[j:k]
    segment4 = solution[k:l]
    segment5 = solution[l:]

    # Recombining segments
    new_solution = segment1 + segment3 + segment2 + segment4 + segment5
    return new_solution


def node_insertion_perturbation(
    solution: list[int], non_selected_nodes: set[NODE_ID], n_intra_insertions: int
) -> list[NODE_ID]:
    # We remove some nodes from the solution and replace them with nodes from non_selected_nodes
    to_remove_node_pos: list[POSITION_IN_SOLUTION] = random.sample(
        range(1, len(solution)), n_intra_insertions
    )
    to_insert_node_ids: list[NODE_ID] = random.sample(
        list(non_selected_nodes), n_intra_insertions
    )

    new_solution = solution.copy()
    for remove_pos, insert_id in zip(to_remove_node_pos, to_insert_node_ids):
        remove_node = Node(id=new_solution[remove_pos], solution_pos=remove_pos)

        # Before calling inter_route_swap, recalculate sets to ensure consistency
        selected_nodes = set(new_solution)
        non_selected_nodes = get_remaining_nodes(selected_nodes, len(new_solution))

        insert_node = Node(id=insert_id)
        new_solution, selected_nodes, non_selected_nodes = inter_route_swap(
            new_solution, remove_node, insert_node, selected_nodes, non_selected_nodes
        )

    return new_solution


def combined_perturbation(
    solution: list[NODE_ID],
    non_selected_nodes: set[NODE_ID],
    n_intra_insertions: int = 5,
) -> list[int]:
    """
    Applies a double-bridge move followed by randomized node insertions to perturb the solution.
    """
    # Apply the double-bridge move
    solution = double_bridge_perturbation(solution)

    # Recalculate sets for consistency after double bridge
    selected_nodes = set(solution)
    non_selected_nodes = get_remaining_nodes(selected_nodes, len(solution))

    # Now apply node insertion perturbation
    solution = node_insertion_perturbation(
        solution, non_selected_nodes, n_intra_insertions
    )

    return solution


class IteratedLocalSearch:
    def __init__(
        self,
        ds: pd.DataFrame,
        dm: np.ndarray,
        max_runtime: float,
        n_intra_insertions: int = 5,
    ):
        self.ds = ds
        self.dm = dm
        self.max_runtime = max_runtime
        self.n_intra_insertions = n_intra_insertions

    def _perturb(
        self,
        solution: list[NODE_ID],
        non_selected_nodes: set[NODE_ID],
    ) -> list[int]:
        # Use the combined perturbation
        return combined_perturbation(
            solution, non_selected_nodes, self.n_intra_insertions
        )

    def __call__(self):
        import time

        start_time = time.time()
        num_local_searches = 0
        num_successful_perturbations = 0

        def run_local_search(start_solution: list[NODE_ID]):
            local_search = LocalSearch(
                strategy="steepest",
                intra_search="edge",
                use_candidates_heurtistic=False,
                debug_mode=False,
            )
            # local_search returns a DataFrame with the selected solution's nodes
            return local_search(self.ds, self.dm, start_solution)

        initial_solution_df = init_random_solution(ds=self.ds, dm=self.dm, start=0)
        initial_solution_list = list(initial_solution_df.index)

        # Run local search on the initial solution
        best_solution_df = run_local_search(initial_solution_list)
        best_solution_list = list(best_solution_df.index)
        best_cost = function_cost(best_solution_df)

        # Initialize sets from the solution returned by local search
        selected_nodes = set(best_solution_list)
        non_selected_nodes = get_remaining_nodes(selected_nodes, len(self.dm))

        while time.time() - start_time < self.max_runtime:
            # Perturbation
            perturbed_solution_list = self._perturb(
                best_solution_list, non_selected_nodes
            )

            # After perturbation, recalculate sets
            selected_nodes = set(perturbed_solution_list)
            non_selected_nodes = get_remaining_nodes(selected_nodes, len(self.dm))

            # Local search on perturbed solution
            perturbed_solution_df = run_local_search(perturbed_solution_list)
            perturbed_solution_list = list(perturbed_solution_df.index)
            perturbed_cost = function_cost(perturbed_solution_df)
            num_local_searches += 1

            # Recalculate sets after local search, as solution may have changed
            selected_nodes = set(perturbed_solution_list)
            non_selected_nodes = get_remaining_nodes(selected_nodes, len(self.dm))

            # Acceptance criterion
            if perturbed_cost < best_cost:
                best_solution_df = perturbed_solution_df
                best_cost = perturbed_cost
                best_solution_list = perturbed_solution_list
                num_successful_perturbations += 1

        return best_solution_df, num_local_searches, num_successful_perturbations
