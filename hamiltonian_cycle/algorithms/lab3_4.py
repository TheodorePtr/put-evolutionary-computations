import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from hamiltonian_cycle.costs import function_cost

SEARCH_STRATEGY_TYPE = Literal["greedy", "steepest"]

INTRA_MOVE_TYPE = Literal["node", "edge"]
MOVE_TYPE = Literal["node", "edge", "inter"]

NODE_ID = int
POSITION_IN_SOLUTION = int


@dataclass(frozen=True)
class Node:
    id: NODE_ID
    solution_pos: POSITION_IN_SOLUTION | None = None


@dataclass(frozen=True)
class MoveEstimation:
    move_type: MOVE_TYPE
    node_a: Node
    node_b: Node
    delta: float
    edges_removed: frozenset[tuple[NODE_ID, NODE_ID]]  # list of edges to be removed
    edges_added: frozenset[tuple[NODE_ID, NODE_ID]]  # list of edges to be added
    is_applicable: bool = True


class MoveEstimator:
    def __init__(self, dm: np.ndarray[float, float], ds: pd.DataFrame) -> None:
        self.dm = dm
        self.node_costs: list[float] = ds["cost"].to_list()

    def estimate_change_two_nodes_move(
        self, solution: list[NODE_ID], node_a: Node, node_b: Node
    ) -> MoveEstimation:
        assert node_a.solution_pos < node_b.solution_pos

        a_prev_id: NODE_ID = solution[node_a.solution_pos - 1]
        a_next_id: NODE_ID = solution[(node_a.solution_pos + 1) % len(solution)]
        b_prev_id: NODE_ID = solution[node_b.solution_pos - 1]
        b_next_id: NODE_ID = solution[(node_b.solution_pos + 1) % len(solution)]

        # Edges to be removed and added
        edges_removed: set[tuple[NODE_ID, NODE_ID]] = set()
        edges_added: set[tuple[NODE_ID, NODE_ID]] = set()

        # Remove edges connected to a and b
        if a_prev_id not in (node_a.id, node_b.id):
            edges_removed.add((a_prev_id, node_a.id))
            edges_added.add((a_prev_id, node_b.id))

        if a_next_id not in (node_a.id, node_b.id):
            edges_removed.add((node_a.id, a_next_id))
            edges_added.add((node_b.id, a_next_id))

        if b_prev_id not in (node_a.id, node_b.id):
            edges_removed.add((b_prev_id, node_b.id))
            edges_added.add((b_prev_id, node_a.id))

        if b_next_id not in (node_a.id, node_b.id):
            edges_removed.add((node_b.id, b_next_id))
            edges_added.add((node_a.id, b_next_id))

        # Calculate delta
        delta = -sum(self.dm[u, v] for u, v in edges_removed) + sum(
            self.dm[u, v] for u, v in edges_added
        )

        return MoveEstimation(
            move_type="node",
            node_a=node_a,
            node_b=node_b,
            delta=delta,
            edges_removed=frozenset(edges_removed),
            edges_added=frozenset(edges_added),
        )

    def estimate_change_two_edges_move(
        self, solution: list, node_a: Node, node_b: Node
    ) -> MoveEstimation:

        assert node_a.solution_pos < node_b.solution_pos

        if node_a.solution_pos == 0 and node_b.solution_pos == len(solution) - 1:
            return MoveEstimation(
                "edge", node_a, node_b, np.inf, frozenset([]), frozenset([])
            )

        a_prev_id = solution[node_a.solution_pos - 1]
        b_next_id = solution[(node_b.solution_pos + 1) % len(solution)]

        # Edges to be removed and added
        edges_removed = frozenset([(a_prev_id, node_a.id), (node_b.id, b_next_id)])
        edges_added = frozenset([(a_prev_id, node_b.id), (node_a.id, b_next_id)])

        # Edges before and after reversal
        cost_before = self.dm[a_prev_id, node_a.id] + self.dm[node_b.id, b_next_id]
        cost_after = self.dm[a_prev_id, node_b.id] + self.dm[node_a.id, b_next_id]

        delta = cost_after - cost_before

        return MoveEstimation(
            move_type="edge",
            node_a=node_a,
            node_b=node_b,
            delta=delta,
            edges_removed=edges_removed,
            edges_added=edges_added,
        )

    def estimate_inter_nodes_move(
        self,
        solution: list[NODE_ID],
        node_a: Node,
        node_b: Node,
    ) -> MoveEstimation:

        assert node_a.id != node_b.id

        a_prev_id = solution[node_a.solution_pos - 1]
        a_next_id = solution[(node_a.solution_pos + 1) % len(solution)]

        # Edge costs before and after the swap
        edge_cost_before = self.dm[a_prev_id, node_a.id] + self.dm[node_a.id, a_next_id]
        edge_cost_after = self.dm[a_prev_id, node_b.id] + self.dm[node_b.id, a_next_id]

        # Node costs before and after the swap
        node_cost_before = self.node_costs[node_a.id]
        node_cost_after = self.node_costs[node_b.id]

        # Edges to be removed and added
        edges_removed = frozenset([(a_prev_id, node_a.id), (node_a.id, a_next_id)])
        edges_added = frozenset([(a_prev_id, node_b.id), (node_b.id, a_next_id)])

        delta = (edge_cost_after - node_cost_after) - (
            edge_cost_before - node_cost_before
        )

        return MoveEstimation(
            move_type="inter",
            node_a=node_a,
            node_b=node_b,
            delta=delta,
            edges_removed=edges_removed,
            edges_added=edges_added,
        )


class LocalSearch:
    def __init__(
        self,
        strategy: SEARCH_STRATEGY_TYPE,
        intra_search: INTRA_MOVE_TYPE,
        use_candidates_heurtistic: bool = False,
        debug_mode: bool = True,
    ) -> None:
        self.strategy = strategy
        self.intra_search = intra_search
        self.use_candidates_heurtistic = use_candidates_heurtistic
        self.debug_mode = debug_mode

    def _shall_interupt(self, move_estimations: list[MoveEstimation]) -> bool:
        return self.strategy == "greedy" and len(move_estimations) > 0

    def _select_greedy_best_move(
        self, all_neighbors: set[MoveEstimation]
    ) -> MoveEstimation:
        all_neighbors_list = list(all_neighbors)
        random.shuffle(all_neighbors_list)
        return all_neighbors_list[0]

    def _browse_intra_moves(
        self,
        move_estimator: MoveEstimator,
        solution: list[NODE_ID],
    ) -> set[MoveEstimation]:
        intra_moves: set[MoveEstimation] = set()

        for node_a_pos in range(len(solution)):

            node_a = Node(
                id=solution[node_a_pos],
                solution_pos=node_a_pos,
            )
            node_b_positions = self._get_intra_solution_positions_to_traverse(
                solution, node_a
            )

            for node_b_pos in node_b_positions:
                node_b = Node(id=solution[node_b_pos], solution_pos=node_b_pos)
                if self.intra_search == "node":
                    move_estimation = move_estimator.estimate_change_two_nodes_move(
                        solution, node_a, node_b
                    )
                elif self.intra_search == "edge":
                    move_estimation = move_estimator.estimate_change_two_edges_move(
                        solution, node_a, node_b
                    )

                if move_estimation.delta < 0:
                    intra_moves.add(move_estimation)

                if self._shall_interupt(intra_moves):
                    return intra_moves

        return intra_moves

    def _get_intra_solution_positions_to_traverse(
        self, solution: list[NODE_ID], node_a: Node
    ) -> list[POSITION_IN_SOLUTION]:

        if self.use_candidates_heurtistic:
            positions_to_traverse = []
            for candidate_id in self.candidate_node_ids_per_node[node_a.id]:
                if (
                    candidate_id in solution
                    and solution.index(candidate_id) > node_a.solution_pos
                ):
                    positions_to_traverse.append(solution.index(candidate_id))
        else:
            positions_to_traverse = range(node_a.solution_pos + 1, len(solution))
        return positions_to_traverse

    def _browse_inter_moves(
        self,
        move_estimator: MoveEstimator,
        solution: list[NODE_ID],
        non_selected_nodes: set[NODE_ID],
    ) -> set[MoveEstimation]:
        inter_neighbors: set[MoveEstimation] = set()

        for node_a_pos in range(len(solution)):
            node_a = Node(
                id=solution[node_a_pos],
                solution_pos=node_a_pos,
            )

            vacant_node_ids = self._get_inter_solution_ids_to_traverse(
                non_selected_nodes, node_a_pos
            )

            for node_b_id in vacant_node_ids:
                node_b = Node(id=node_b_id)

                move_estimation = move_estimator.estimate_inter_nodes_move(
                    solution,
                    node_a,
                    node_b,
                )
                if move_estimation.delta < 0:
                    inter_neighbors.add(move_estimation)
                if self._shall_interupt(inter_neighbors):
                    return inter_neighbors
        return inter_neighbors

    def _get_inter_solution_ids_to_traverse(
        self, non_selected_nodes: set[NODE_ID], node_a_pos: POSITION_IN_SOLUTION
    ) -> set[NODE_ID]:
        neighbors_to_check = non_selected_nodes.copy()
        if self.use_candidates_heurtistic:
            neighbors_to_check = neighbors_to_check.intersection(
                self.candidate_node_ids_per_node[node_a_pos]
            )

        return neighbors_to_check

    def _update_solution(
        self,
        solution: list[NODE_ID],
        selected_move: MoveEstimation,
        selected_nodes: set[NODE_ID],
        non_selected_nodes: set[NODE_ID],
    ) -> tuple:

        if selected_move.move_type == "node":
            solution = two_nodes_exchange(
                solution, selected_move.node_a, selected_move.node_b
            )
        elif selected_move.move_type == "edge":
            solution = two_edges_exchange(
                solution, selected_move.node_a, selected_move.node_b
            )
        elif selected_move.move_type == "inter":
            solution, selected_nodes, non_selected_nodes = inter_route_swap(
                solution,
                selected_move.node_a,
                selected_move.node_b,
                selected_nodes,
                non_selected_nodes,
            )
        return solution, selected_nodes, non_selected_nodes

    def _store_candidate_node_ids(self, dm: np.ndarray[float, float]) -> None:
        self.candidate_node_ids_per_node: np.ndarray[NODE_ID, NODE_ID] = np.argsort(
            dm, axis=1
        )[:, 1:11]

    def __call__(
        self,
        ds: pd.DataFrame,
        dm: np.ndarray[float, float],
        initial_solution: list[NODE_ID],
    ) -> pd.DataFrame:

        if self.use_candidates_heurtistic:
            self._store_candidate_node_ids(dm)

        solution = initial_solution.copy()
        selected_nodes: set[NODE_ID] = set(initial_solution)
        non_selected_nodes: set[NODE_ID] = get_remaining_nodes(selected_nodes, len(dm))

        move_estimator = MoveEstimator(dm, ds)

        while True:
            intra_moves = self._browse_intra_moves(move_estimator, solution)
            inter_moves = self._browse_inter_moves(
                move_estimator, solution, non_selected_nodes
            )

            all_moves = intra_moves.union(inter_moves)

            # no improvment found
            if not all_moves:
                break

            if self.strategy == "greedy":
                selected_move = self._select_greedy_best_move(all_moves)
            elif self.strategy == "steepest":
                selected_move = min(all_moves, key=lambda move: move.delta)

            # Save the old solution for debugging
            if self.debug_mode:
                old_solution = solution.copy()

            # Update solution and cost
            solution, selected_nodes, non_selected_nodes = self._update_solution(
                solution, selected_move, selected_nodes, non_selected_nodes
            )
            # assert selected_nodes.isdisjoint(
            #     non_selected_nodes
            # ), "Selected and non-selected sets overlap."
            # assert selected_nodes.union(non_selected_nodes) == set(
            #     range(len(dm))
            # ), "Sets do not cover all nodes exactly."

            # # Also, check that all nodes in `solution` are indeed in `selected_nodes`
            # assert all(
            #     node_id in selected_nodes for node_id in solution
            # ), "Some solution nodes not in selected_nodes."

            if self.debug_mode:
                self._check_debug_assertions(ds, solution, selected_move, old_solution)

        return ds.loc[solution]

    def _invert_nodes_order(
        edges: list[tuple[NODE_ID, NODE_ID]]
    ) -> list[tuple[NODE_ID, NODE_ID]]:
        return [(v, u) for u, v in edges]

    def _check_debug_assertions(self, ds, solution, selected_move, old_solution):
        real_improvement = function_cost(ds.loc[old_solution]) - function_cost(
            ds.loc[solution]
        )

        if real_improvement != -selected_move.delta:
            print(f"Promised improvement: {selected_move.delta[2]}")
            print(f"Real improvement: {real_improvement}")
            print(f"Operation: {selected_move.move_type}")
            print("===========")

        assert function_cost(ds.loc[solution]) != function_cost(ds.loc[old_solution])


def get_remaining_nodes(selected_nodes: set[NODE_ID], num_nodes: int) -> set[NODE_ID]:
    return set(range(num_nodes)) - selected_nodes


def two_nodes_exchange(
    solution: list[NODE_ID], node_a: Node, node_b: Node
) -> list[NODE_ID]:
    new_solution = solution.copy()

    (
        new_solution[node_a.solution_pos],
        new_solution[node_b.solution_pos],
    ) = (
        new_solution[node_b.solution_pos],
        new_solution[node_a.solution_pos],
    )
    return new_solution


def two_edges_exchange(
    solution: list[NODE_ID], node_a: Node, node_b: Node
) -> list[NODE_ID]:
    new_solution = solution.copy()

    new_solution[node_a.solution_pos : node_b.solution_pos + 1] = new_solution[
        node_a.solution_pos : node_b.solution_pos + 1
    ][::-1]
    return new_solution


def inter_route_swap(
    solution: list[NODE_ID],
    node_a: Node,
    node_b: Node,
    selected_nodes: set[NODE_ID],
    non_selected_nodes: set[NODE_ID],
) -> tuple[list[NODE_ID], set[NODE_ID], set[NODE_ID]]:
    new_solution = solution.copy()

    new_solution[node_a.solution_pos] = node_b.id

    # Update the node sets
    selected_nodes = selected_nodes.copy()
    non_selected_nodes = non_selected_nodes.copy()

    selected_nodes.remove(node_a.id)
    selected_nodes.add(node_b.id)
    non_selected_nodes.remove(node_b.id)
    non_selected_nodes.add(node_a.id)

    return new_solution, selected_nodes, non_selected_nodes
