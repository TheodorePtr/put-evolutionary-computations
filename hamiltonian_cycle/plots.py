import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_solution(ds_all: pd.DataFrame, solution: list, *, title: str = "") -> None:
    """
    Plots all nodes, highlighting the selected nodes and drawing paths between them.

    Args:
        ds_all (pd.DataFrame): DataFrame containing all nodes.
        solution (list): List of indices of selected nodes in the order of the solution.
        title (str): Title of the plot.
    """
    if ds_all["x"].max() > ds_all["y"].max():
        width, height = 13.0, 5.0
    else:
        width, height = 5.0, 13.0

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    sns.scatterplot(
        data=ds_all,
        x="x",
        y="y",
        hue="cost",
        palette="gray",
        size="cost",
        sizes=(50, 200),
        legend=False,
        ax=ax,
    )

    # Highlight selected nodes in red
    ds_selected = ds_all.loc[solution]
    sns.scatterplot(
        data=ds_selected,
        x="x",
        y="y",
        hue="cost",
        palette="Reds",
        size="cost",
        sizes=(50, 200),
        legend=False,
        ax=ax,
    )

    # Draw paths between selected nodes
    for i in range(-1, len(solution) - 1):
        idx1 = solution[i]
        idx2 = solution[i + 1]
        x_values = [ds_all.loc[idx1, "x"], ds_all.loc[idx2, "x"]]
        y_values = [ds_all.loc[idx1, "y"], ds_all.loc[idx2, "y"]]
        plt.plot(
            x_values,
            y_values,
            color="k",
            linestyle="-",
        )

    ax.set_title(title)
    plt.show()
