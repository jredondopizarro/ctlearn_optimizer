.. ctlearn_optimizer documentation master file, created by
   sphinx-quickstart on Mon Aug 26 07:02:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CTLearn Optimizer's documentation!
*********************************************

CTLearn Optimizer is a framework for optimizing `CTLearn <https://github.com/ctlearn-project/ctlearn/>`_
models.

This optimization utility uses `Tune <https://ray.readthedocs.io/en/latest/tune.html>`_, 
a scalable framework for hyperparameter search and model training, and supports:

- Random search based optimization.
- Tree Parzen Estimators based optimization.
- Gaussian Processes based optimization.
- Genetic Algorithm based optimization.
- Parallel optimization (depending on available hardware resources).

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   installation
   usage
   settings
   analysis
   api
   contributing



