import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skopt
from scipy.optimize import OptimizeResult


# taken from skopt.plots and slightly modified
def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters:
        args[i] (OptimizeResult, list of OptimizeResult, or tuple):

            The result(s) for which to plot the convergence trace.

            - if OptimizeResult, then draw the corresponding single trace;
            - if list of OptimizeResult, then draw the corresponding
              convergence traces in transparency, along with the average
              convergence trace;
            - if tuple, then args[i][0] should be a string label and args[i][1]
              an OptimizeResult or a list of OptimizeResult.

        ax (Axes, optional): The matplotlib axes on which to draw the plot,
            or None to create a new one.

        yscale (None or string, optional): The scale for the y-axis.

    Returns:
        Axes: The matplotlib axes.
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


# taken from neptunecontrib.hpo.utils
def df2result(df, metric_col, param_cols, param_types=None):
    """Convert df with metrics and hyperparams to the OptimizeResults format.

    It is a helper function that lets you use all the tools that expect
    OptimizeResult object like for example scikit-optimize plot_evaluations
    function.

    Parameters:
        df (pandas.DataFrame): Dataframe containing metric and hyperparameters.
        metric_col (str): Name of the metric column.
        param_cols (list): Names of the hyperparameter columns.
        param_types (list or None): Optional list of hyperparameter column
            types.
            By default it will treat all the columns as float but you can also
            pass str for categorical channels. Example: param_types=[float,
            str, float, float]

    Returns:
        scipy.optimize.OptimizeResult: Results object that contains
        the hyperparameter and metric information.

    """

    def _prep_df(df, param_cols, param_types):
        for col, col_type in zip(param_cols, param_types):
            df[col] = df[col].astype(col_type)
        return df

    def _convert_to_param_space(df, param_cols, param_types):
        dimensions = []
        for colname, col_type in zip(param_cols, param_types):
            if col_type == str:
                dimensions.append(skopt.space.Categorical(
                    categories=df[colname].unique(), name=colname))
            elif col_type == float:
                low, high = df[colname].min(), df[colname].max()
                dimensions.append(skopt.space.Real(low, high, name=colname))
            else:
                raise NotImplementedError
        skopt_space = skopt.Space(dimensions)
        return skopt_space

    if not param_types:
        param_types = [float for _ in param_cols]

    df = _prep_df(df, param_cols, param_types)
    param_space = _convert_to_param_space(df, param_cols, param_types)

    results = OptimizeResult()
    results.x_iters = df[param_cols].values
    results.func_vals = df[metric_col].to_list()
    results.x = results.x_iters[np.argmin(results.func_vals)]
    results.fun = np.min(results.func_vals)
    results.space = param_space
    return results
