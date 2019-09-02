==========
Basic usage
==========

Perfom optimization run
-----------------------

1. Create a working folder, cd to it

2. You will need both myconfig.yml (CTLearn config) and opt_config.yml (CTLearn Optimizer config) files in the folder, 
   examples are provided. Read configuration section for more details about the configuration of the optimizer.

3. Run python PATH/TO/PROJECT/ctlearn_optimizer/optimizer.py opt_config.yml.

4. Alternatively, import ctlearn_optimizer as a module in a Python script::

    from ctlearn_optimizer.optimizer import Optimizer
    with open(opt_config.yml, 'r') as opt_conf:
        config = yaml.load(opt_conf)

    opt = Optimizer(config)
    optimization_result = opt.optimize_ctlearn_model()

Analysis of the optimization results
---------------------------------

The outputs of ctlearn_optimizer are:

- optimization.log file containg information about the optimization run.
- optimization_results.csv file containing loss, iteration, hyperparameters, metrics and training_time. This file is updated each iteration.
- trials.pkl file for resuming from resume a previous runs. 

Take a look at the results_and_plots folder for more information about the analysis of the optimization_results.csv file.
