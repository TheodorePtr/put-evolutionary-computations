import random
import time

import numpy as np
import pandas as pd

from hamiltonian_cycle.algorithms.lab1 import init_random_solution
from hamiltonian_cycle.algorithms.lab2 import init_greedy_2regret_weighted_cycle
from hamiltonian_cycle.algorithms.lab3_4 import (
    LocalSearch,
    # inter_route_swap,
)
from hamiltonian_cycle.costs import function_cost


def find_k_longest_segments(distance_matrix, solution, segment_length, k):
    if segment_length < 2 or segment_length > len(solution):
        raise ValueError(
            "Segment length must be between 2 and the number of nodes in the solution."
        )

    longest_segments = []
    used_indices = set()

    for _ in range(k):
        max_distance = -np.inf
        best_segment = None

        for i in range(len(solution) - segment_length + 1):
            if any(j in used_indices for j in range(i, i + segment_length)):
                continue  # Skip overlapping segments

            segment_nodes = solution[i : i + segment_length]
            total_distance = sum(
                distance_matrix[segment_nodes[j], segment_nodes[j + 1]]
                for j in range(segment_length - 1)
            )

            if total_distance > max_distance:
                max_distance = total_distance
                best_segment = (i, i + segment_length - 1, total_distance)

        if best_segment is None:
            break  # No more segments can be found without overlapping

        # Mark the indices of the chosen segment as used
        start_idx, end_idx, _ = best_segment
        used_indices.update(range(start_idx, end_idx + 1))
        longest_segments.append(best_segment)

    return longest_segments


def eliminate_segments(solution, segments):
    # Flatten the indices of all segments to remove
    indices_to_remove = set()
    for start, end, _ in segments:
        indices_to_remove.update(range(start, end + 1))

    # Create a new solution excluding the indices to remove
    new_solution = [
        node for idx, node in enumerate(solution) if idx not in indices_to_remove
    ]

    return new_solution


def generate_random_random_segments_params(
    solution_length, min_percentage=0.2, max_percentage=0.3
):
    # Calculate target range of nodes to cover
    min_nodes = int(solution_length * min_percentage)
    max_nodes = int(solution_length * max_percentage)
    target_nodes = random.randint(min_nodes, max_nodes)

    # Generate random valid segment_length and k_segments
    possible_segment_lengths = [l for l in range(2, solution_length // 2 + 1)]
    while True:
        segment_length = random.choice(possible_segment_lengths)
        k_segments = target_nodes // segment_length

        # Ensure that the segments cover the target range and are feasible
        if k_segments > 0 and segment_length * k_segments <= target_nodes:
            break

    return segment_length, k_segments


class LargeNeighborhoodSearch:
    def __init__(
        self,
        ds: pd.DataFrame,
        dm: np.ndarray,
        max_runtime: float,
        w_cost: float,
        w_regret: float,
        apply_local_search: bool = True,
    ):
        self.ds = ds
        self.dm = dm
        self.max_runtime = max_runtime
        self.w_cost = w_cost
        self.w_regret = w_regret
        self.apply_local_search = apply_local_search

    def _destroy(self, solution: list[int]) -> list[int]:
        segment_length, k_segments = generate_random_random_segments_params(
            len(solution)
        )
        k_longest_segments = find_k_longest_segments(
            self.dm, solution, segment_length, k_segments
        )
        return eliminate_segments(solution, k_longest_segments)

    def _repair(self, destroyed_solution: list[int]) -> pd.DataFrame:
        repaired_solution_df = init_greedy_2regret_weighted_cycle(
            self.ds,
            self.dm,
            start=0,
            w_cost=self.w_cost,
            w_regret=self.w_regret,
            initial_solution=destroyed_solution,
        )
        return repaired_solution_df

    def _local_search(self, solution: list[int]) -> pd.DataFrame:
        local_search = LocalSearch(
            strategy="steepest",
            intra_search="edge",
            use_candidates_heurtistic=False,
            debug_mode=False,
        )
        local_search_solution = local_search(self.ds, self.dm, solution)
        return local_search_solution

    def __call__(self):
        start_time = time.time()
        num_iterations = 0

        # Initialize with a random solution and run local search
        initial_solution_df = init_random_solution(self.ds, self.dm, start=0)
        initial_solution_list = list(initial_solution_df.index)
        best_solution_df = self._local_search(initial_solution_list)
        best_solution_list = list(best_solution_df.index)
        best_cost = function_cost(best_solution_df)

        while time.time() - start_time < self.max_runtime:
            # Increment iteration counter
            num_iterations += 1

            # Destroy and repair solution
            destroyed_solution_list = self._destroy(best_solution_list)
            repaired_solution_df = self._repair(destroyed_solution_list)

            # Apply local search (optional based on configuration)
            if self.apply_local_search:
                repaired_solution_df = self._local_search(
                    list(repaired_solution_df.index)
                )

            # Evaluate repaired solution
            repaired_cost = function_cost(repaired_solution_df)

            # Accept the repaired solution if it's better
            if repaired_cost < best_cost:
                best_solution_list = list(repaired_solution_df.index)
                best_solution_df = repaired_solution_df
                best_cost = repaired_cost

        return best_solution_df, num_iterations
