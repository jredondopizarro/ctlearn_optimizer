import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult


# taken from skopt.plots and slightly modified
def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
        args[i] [OptimizeResult, list of OptimizeResult, or tuple]:

            The result(s) for which to plot the convergence trace.

            - if OptimizeResult, then draw the corresponding single trace;
            - if list of OptimizeResult, then draw the corresponding
              convergence traces in transparency, along with the average
              convergence trace;
            - if tuple, then args[i][0] should be a string label and args[i][1]
              an OptimizeResult or a list of OptimizeResult.

        ax [Axes, optional]: The matplotlib axes on which to draw the plot,
            or None to create a new one.

        yscale [None or string, optional]: The scale for the y-axis.

    Returns

        ax: [Axes]: The matplotlib axes.
    """

    ax = kwargs.get("ax", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax = plt.gca()
    ax.set_title('Convergence plot')
    ax.set_xlabel('Number of iterations n')
    ax.set_ylabel('max(metric) after n iterations')
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            maxs = [np.max(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), maxs, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            maxs = [[np.max(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in maxs:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(maxs, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    ax.legend(loc='best')
    return ax
