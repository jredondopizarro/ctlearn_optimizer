# CTLearn Optimizer [GSOC 2019]

Framework for optimizing [CTLearn v0.3.0](https://github.com/ctlearn-project/ctlearn/tree/v031) 
models developed during the Google Summer of Code 2019.

This optimization utility uses [Tune](https://ray.readthedocs.io/en/latest/tune.html), 
a scalable framework for hyperparameter search and model training, and supports:

- Random search based optimization.
- Tree Parzen Estimators based optimization.
- Gaussian Processes based optimization.
- Genetic Algorithm based optimization.
- Parallel optimization (depending on available hardware resources).

## Authors

* **[Juan Alfonso Redondo Pizarro](https://github.com/jredondopizarro)**

##Dependencies

- CTLearn
- environment_kernels
- Hyperopt
- Matplotlib
- nbsphinx
- NumPy
- Pandas
- PyYAML 
- Ray
- Scikit-Optimize 
- SciPy
- Scikit-learn
- Seaborn
- setproctitle
- Sphinx
- sphinx_rtd_theme

### Installation, basic usage and configuration

Take a look at the the [documentation](https://ctlearn-optimizer.readthedocs.io/en/latest/index.html).





