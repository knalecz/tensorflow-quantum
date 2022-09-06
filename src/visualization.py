# Na podstawie https://github.com/marcosdelcueto/Tutorial_Grid_Search

# https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas import read_csv
import numpy as np


GRID_SEARCH_RESULTS_FILENAME = "grid_search_results.csv"


def prepare_hyperparams_grid(input_filename):
    data = read_csv(input_filename)
    graph_x = data["A_1"].to_numpy()
    graph_y = data["A_2"].to_numpy()
    graph_z = data["B"].to_numpy()
    graph_performance = data["performance"].to_numpy()
    return graph_x, graph_y, graph_z, graph_performance


# https://stackoverflow.com/a/67774238
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


def plot_hyperparams_grid(graph_x, graph_y, graph_z, graph_performance):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        np.log10(graph_x),
        np.log10(graph_y),
        np.log10(graph_z),
        c=graph_performance,
        cmap="viridis",
        marker="o",
        s=[100] * len(graph_performance),
        edgecolors="black"
    )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.set_xlabel("A_1")
    ax.set_ylabel("A_2")
    ax.set_zlabel("B")

    plt.show()

#    file_name = 'hyperparams_grid_search.png'
#    plt.savefig(file_name,format='png',dpi=600)
#    plt.close()


if __name__ == "__main__":
    input_filename = GRID_SEARCH_RESULTS_FILENAME
    graph_x, graph_y, graph_z, graph_performance = prepare_hyperparams_grid(input_filename)
    plot_hyperparams_grid(graph_x, graph_y, graph_z, graph_performance)
