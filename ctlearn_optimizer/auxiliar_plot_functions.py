

#taken from skopt.plots andslightly modified
def plot_convergence(*args):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import OptimizeResult

    ax = plt.gca()

    ax.set_title('Convergence plot')
    ax.set_xlabel('Number of iterations n')
    ax.set_ylabel('max(metric) after n iterations')
    ax.grid()

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.max(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.max(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    return ax
