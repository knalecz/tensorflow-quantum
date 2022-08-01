# Na podstawie https://github.com/marcosdelcueto/Tutorial_Grid_Search

# https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html

import numpy as np
import matplotlib.pyplot as plt


def create_hyperparams_grid():
    graph_x = []
    graph_y = []
    graph_z = []
    graph_performance = []
    for a1_value in np.arange(-5.0, 2.0, 0.7):
        a1_value = pow(10, a1_value)
        graph_x_row = []
        graph_y_row = []
        graph_z_row = []
        graph_performance_row = []
        for a2_value in np.arange(-5.0, 2.0, 0.7):
            a2_value = pow(10, a2_value)
            for b_value in np.arange(-5.0, 2.0, 0.7):
                b_value = pow(10, b_value)
                hyperparams = (a1_value, a2_value, b_value)
                performance = 100
                graph_x_row.append(a1_value)
                graph_y_row.append(a2_value)
                graph_z_row.append(b_value)
                graph_performance_row.append(performance)

        graph_x.append(graph_x_row)
        graph_y.append(graph_y_row)
        graph_z.append(graph_z_row)
        graph_performance.append(graph_performance_row)
        print("")
    graph_x = np.array(graph_x)
    graph_y = np.array(graph_y)
    graph_z = np.array(graph_z)
    graph_performance = np.array(graph_performance)

    print(f"graph_x = {graph_x}")
    print(f"graph_y = {graph_y}")
    print(f"graph_z = {graph_z}")
    print(f"graph_performance = {graph_performance}")

    max_performance = np.max(graph_performance)
    pos_max_performance = np.argwhere(graph_z == np.min(graph_z))[0]
    print("Maximum performance: %.4f" % (max_performance))
    print("Optimum A_1: %f" % (graph_x[pos_max_performance[0], pos_max_performance[1]]))
    print("Optimum A_2: %f" % (graph_y[pos_max_performance[0], pos_max_performance[1]]))
    print("Optimum B: %f" % (graph_z[pos_max_performance[0], pos_max_performance[1]]))
    return graph_x, graph_y, graph_z, graph_performance


def plot_hyperparams_grid(graph_x, graph_y, graph_z, graph_performance):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    points = ax.scatter(
        graph_x,
        graph_y,
        graph_z,
        c=graph_performance,
        cmap="viridis",
        marker="o",
        s=[20 * n ** 2 / 100 for n in graph_performance],
    )

    ax.set_xlabel("A_1")
    ax.set_ylabel("A_2")
    ax.set_zlabel("B")

    plt.show()


#    file_name = 'Figure_hyperparams_grid.png'
#    plt.savefig(file_name,format='png',dpi=600)
#    plt.close()
