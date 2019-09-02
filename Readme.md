# CTLearn Optimizer [GSOC 2019]

Framework for optimizing [CTLearn v0.3.0](https://github.com/ctlearn-project/ctlearn/tree/v031) models developed during the Google Summer of Code 2019.

This optimization utility uses [Tune](https://ray.readthedocs.io/en/latest/tune.html), a scalable framework for hyperparameter search and model training, and supports:

- Random search based optimization.
- Tree Parzen Estimators based optimization.
- Gaussian Processes based optimization.
- Genetic Algorithm based optimization.
- Parallel optimization (depending on available hardware resources).

Results and analysis of performed optimization runs are included in results_and_plots folder.

## Authors

* **[Juan Alfonso Redondo Pizarro](https://github.com/juan-redondo/ctlearn)**

### Prerequisites

```
CTLearn, , random, shutil, logging, time, pandas, skopt, hyperopt, sklearn, multiprocessing, os, re , yaml, numpy, logging, pickle, csv, timeit, argsparse, matplotlib, scipy
```

In addition, this package needs ray Python 3.6 snapshot version, that can be installed from [here](https://ray.readthedocs.io/en/latest/installation.html).

### Installation and basic usage

Take a look at the the documentation.

 




