import pandas as pd

from hamiltonian_cycle.costs import function_cost
from hamiltonian_cycle.plots import plot_solution


def perform_experiment(
    ds: pd.DataFrame,
    dm: pd.DataFrame,
    title: str,
    func_to_optimize: callable,
    **kwargs: dict,
) -> None:
    ratings = []
    for i in range(200):
        greedy_cycle_solution = func_to_optimize(ds, dm, i, **kwargs)
        solution = list(greedy_cycle_solution.index)
        ratings.append((solution, function_cost(greedy_cycle_solution)))
    best = sorted(ratings, key=lambda x: x[1])[0]
    minimum = sorted(ratings, key=lambda x: x[1])[0][1]
    mean = sum([obj_function for _, obj_function in ratings]) / len(ratings)
    maximum = sorted(ratings, key=lambda x: x[1])[-1][1]

    print(f"Best solution: {best[0]}")
    print("Objective function statistics:")
    print(f"{minimum = }\n{mean = }\n{maximum = }")
    plot_solution(ds, best[0], title=title)
