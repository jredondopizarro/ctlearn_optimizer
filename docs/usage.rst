***********
Basic usage
***********
==========================
Perfom an optimization run
==========================

Setting up the working directory and required files
===================================================

Set up the working directory for the optimization run:

.. code-block:: bash

    mkdir </directory/path>

There are two mandatory files that must be located in the working directory, 
examples are provided:

- ``CTLearn Optimizer YAML configuration file``, where all run settings are
  stored, organized into General, Optimizer, CTLearn and Hyperparameters sections.
  See :doc:`settings` for a description of every available setting and its 
  possible values in detail.

- ``CTLearn YAML configuration file``, required by CTLearn package to configure 
  its deep learning models. The name of this file needs to be specified in the 
  :ref:`pertinent setting <ctlearn-config-label>` of  CTLearn Optimizer configuration. 

  Generally, the user doesn't need to modify any setting of this file, as this 
  is done automatically by the CTLearn Optimizer framework based on the setting 
  values specified in the :ref:`CTLearn section <ctlearn-section-label>`  
  of the CTLearn Optimizer configuration file. Any other setting not present 
  in this section won't be automatically modified and will need to be edited 
  manually by the user.  

  See the example configuration file in `CTLearn's v0.3.0 repository 
  <https://github.com/ctlearn-project/ctlearn/tree/v031>`_ for more information 
  about the available configuration options.

Run the optimization
====================
This can be done either from the command line:

.. code-block:: bash

    python </installation/path>/src/ctlearn_optimizer/optimizer.py opt_config.yml

Or by importing ctlearn_optimizer as a module in a Python script:

.. code-block:: python

    from ctlearn_optimizer.optimizer import Optimizer
    import yaml

    with open('opt_config.yml', 'r') as opt_conf:
        config = yaml.load(opt_conf)

    opt = Optimizer(config)
    opt.optimize_ctlearn_model()

Where ``opt_config.yml`` is the CTLearn Optimizer YAML configuration file.

====================
Optimization results
====================

The outputs of an optimization run are:

- ``optimization.log`` file containg the following information about the 
  optimization run:

  - Optimization algorithm used.
  - Whether a trials file or a optimization results file of a previous run have
    been loaded.
  - Iteration.
  - Values of the hyperparameters and the metric to optimize in each iteration.
  - Best metric and hyperparameters set found at the end of the optimization run.

- ``optimization_results.csv`` file containing information about the loss, 
  iteration, metrics, hyperparameters and training_time throughout the 
  optimization run. This file is updated each iteration. See :doc:`analysis`
  for more information about the analysis of this file.

- ``trials.pkl`` file for resuming the optimization run later if required. This 
  file is updated each iteration.

- ``gp_model.pkl`` file, generated only if the optimization algorithm is
  gaussian processes. See :doc:`analysis` for more information about the 
  analysis of this file. This file is generated at the end of the optimization 
  run.

- ``ray_optimization_results`` folder to save Ray Tune training results to.

