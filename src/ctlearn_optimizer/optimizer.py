import pickle
import argparse
import os
import csv
import multiprocessing
import random
from time import sleep
import pandas as pd
import yaml
import skopt
import ray
from ray.tune import run
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.automl import GeneticSearch
import hyperopt as hpo
import ctlearn_optimizer.bayesian_tpe as bayesian_tpe
import ctlearn_optimizer.bayesian_gp as bayesian_gp
import ctlearn_optimizer.genetic_algorithm as genetic_algorithm
import ctlearn_optimizer.common as common

# set dummy authentication key for multiprocessing
multiprocessing.current_process().authkey = b'1234'


class Optimizer:
    """ Basic class for an optimizer.

    Currently, only tree parzen estimators, random search, gaussian processes
    and genetic algorithm based optimization using Ray Tune is supported.

    """

    def __init__(self, opt_config):
        """ Initialize the optimizer:

            - Set optimizer attributes.
            - Set logger writing to both ``optimization.log`` and ``stdout``.
            - Load trials file from ``working_directory/trials.pkl``
              if required, thus allowing to resume a past optimization run.
            - Load optimization results file from
              ``working_directory/optimization_results.csv`` or create one at
              the same path as required. The results of the optimization run
              (loss, iteration, metrics, hyperparameters, time) are logged to
              this file for further analysis.

        Parameters:
            opt_config (dict): loaded optimization configuration file.

        Raises:
            NotImplementedError: if ``self.optimization_type`` is
                ``genetic_algorithm`` and ``self.reload_trials`` is ``True``.
        """

        # set working directory path
        gen_settings = opt_config['General_settings']
        self.working_directory = gen_settings.get('working_directory',
                                                  os.getcwd())

        # create new logger
        self.log_path = os.path.join(self.working_directory,
                                     'optimization.log')
        self.logger = common.set_logger(self.log_path)

        self.logger.info('Starting optimization run')

        # ray tune optimizator runs the objective function on a different
        # Python process, so we have to use multiprocessing manager to share
        # variables between processes.
        manager = multiprocessing.Manager()
        # set global run iteration (for taking into account reloaded trials)
        self.iteration = manager.Value('i', 0)
        # set current run iteration (always starts from 0)
        self.counter = manager.Value('i', 0)

        # set random state seed for reproducible results
        self.random_state = gen_settings.get('random_state', None)

        # set hardware resources
        self.num_cpus = gen_settings['num_cpus']
        self.num_cpus_per_trial = gen_settings['num_cpus_per_trial']
        self.num_gpus = gen_settings['num_gpus']
        self.num_gpus_per_trial = gen_settings['num_gpus_per_trial']

        # set file paths
        self.ctlearn_config_path = os.path.join(self.working_directory,
                                                gen_settings['ctlearn_config'])
        self.trials_file_path = os.path.join(self.working_directory,
                                             'trials.pkl')
        self.optim_results_path = os.path.join(self.working_directory,
                                               'optimization_results.csv')

        self.remove_training_folders = gen_settings.get(
            'remove_training_folders', True)
        # set optimizer configuration
        self.num_parallel_trials = gen_settings.get('num_parallel_trials', 1)
        self.mode = gen_settings.get('mode', 'max')
        self.n_initial_points = gen_settings.get('n_initial_points', 30)
        self.optimization_type = gen_settings['optimization_type']
        self.num_max_evals = gen_settings['num_max_evals']

        self.logger.info('Optimization algorithm:{}'.format(
            self.optimization_type))

        self.metric_to_optimize = gen_settings['metric_to_optimize']
        self.predict = gen_settings.get('predict', False)
        self.data_set_to_optimize = gen_settings.get('data_set_to_optimize',
                                                     'validation')
        if self.data_set_to_optimize == 'prediction':
            assert self.predict is True

        if self.optimization_type == 'genetic_algorithm':
            optimizer_info = opt_config['Optimizer_settings']
            self.ga_config = optimizer_info['genetic_algorithm_config']
        else:
            optimizer_info = opt_config.get('Optimizer_settings', None)
            self.ga_config = optimizer_info.get('genetic_algorithm_config',
                                                None)
        self.tpe_config = optimizer_info.get('tree_parzen_estimators_config',
                                             None)
        self.gp_config = optimizer_info.get('gaussian_processes_config', None)

        # set ctlearn basic configuration
        self.basic_config = opt_config['CTLearn_settings']

        # set hyperparameters logging and configuration
        hyperparameters_info = opt_config['Hyperparameters_settings']
        self.hyperparams_to_log = (hyperparameters_info
                                   ['hyperparameters_to_log'])
        self.hyperparameters_config = (hyperparameters_info['config'])
        self.hyperparams_to_optimize = (hyperparameters_info
                                        ['hyperparameters_to_optimize'])

        self.fixed_hyperparameters = hyperparameters_info.get(
            'fixed_hyperparameters', None)
        self.dependent_hyperparameters = hyperparameters_info.get(
            'dependent_hyperparameters', None)

        # set metrics logging and configuration
        self.list_metrics_val_to_log = gen_settings.get('metrics_val_to_log',
                                                        [])
        self.list_metrics_pred_to_log = gen_settings.get('metrics_pred_to_log',
                                                         [])
        if self.list_metrics_pred_to_log:
            assert self.predict is True
        else:
            assert self.data_set_to_optimize != 'prediction'

        self.user_defined_metric_val = gen_settings.get(
            'user_defined_metric_val', None)
        self.user_defined_metric_pred = gen_settings.get(
            'user_defined_metric_pred', None)

        # create hyperparameters space, used for trial reloading
        hyperparameter_space = self.create_space_hyperparams()

        # set trial reloading configuration and create optimization algorithm
        self.reload_trials = gen_settings.get('reload_trials', False)

        if self.reload_trials:
            if self.optimization_type == 'genetic_algorithm':
                raise NotImplementedError('Trial reloading is not currently \
                     supported by the genetic algorithm optimization')
            else:
                # reload trials file
                assert os.path.isfile(self.trials_file_path)
                with open(self.trials_file_path, 'rb') as input_file:
                    trials = pickle.load(input_file)

                if self.optimization_type == 'gaussian_processes':
                    # the gaussian processes based optimization needs an
                    # extra parameter self.gp_opt
                    self.gp_opt = common.restore(self)
                    # create optimization algorithm with reloaded trials
                    self.optimization_algorithm = \
                        self.create_optimization_algorithm(
                            hyperparameter_space)
                    # set global run iteration value to start from the
                    # number of trials of the reloaded run
                    self.iteration.value = len(trials.Xi)

                else:  # optimization type == tpe or random_search
                    # create optimization algorithm with reloaded trials
                    self.optimization_algorithm = \
                        self.create_optimization_algorithm(
                            hyperparameter_space)
                    common.restore(self)

                    if self.optimization_type == 'random_search':
                        self.optimization_algorithm.algo = hpo.rand.suggest

                    # set global run iteration value to start from the
                    # number of trials of the reloaded run
                    self.iteration.value = len(trials[0].trials)

                self.logger.info(
                    'A trials file with {} saved trials has been reloaded, \
                    new trials will be added'.format(self.iteration.value))
        else:
            self.logger.info('No trials file loaded, starting from scratch')
            self.gp_opt = None
            # create optimization algorithm
            self.optimization_algorithm = self.create_optimization_algorithm(
                hyperparameter_space)
            self.iteration.value = 0

        # set optimization results configuration
        reload_optimization_results = gen_settings.get(
            'reload_optimization_results', False)

        if reload_optimization_results:
            # reload optimization_results file
            assert os.path.isfile(self.optim_results_path)
            with open(self.optim_results_path, 'r') as file:
                existing_iters_csv = len(file.readlines()) - 1
            self.logger.info(
                'An optimization_results file with {} saved trials has been \
                reloaded, new trials will be added'.format(existing_iters_csv))

            if existing_iters_csv != self.iteration.value:
                self.logger.WARNING(
                    'Caution: the number of trials stored in the trials file \
                     and the number of trials stored in the \
                     optimization_results file does not match')

        else:  # create optimization_results.csv
            self.logger.info(
                'No optimization_results file loaded, starting from scratch')

        with open(self.optim_results_path, 'w') as file:
            writer = csv.writer(file)
            header = ['loss', 'iteration'] + self.hyperparams_to_log + \
                [elem + '_val' for elem in self.list_metrics_val_to_log] + \
                [elem + '_pred' for elem in self.list_metrics_pred_to_log] + \
                ['run_time']
            writer.writerow(header)

    def set_basic_config(self):
        """Set basic config and fixed hyperparameters in CTLearn config file.
        """

        common.set_basic_config(self)

    def create_space_hyperparams(self):
        """ Create space of hyperparameters following required syntax.

        Currently, only tree parzen estimators and random search spaces based
        on hyperopt, gaussian processes space based on skopt and
        genetic algorithm space based on ray.tune.automl are supported.

        Returns:
            space of hyperparameters following the syntax required by the
            optimization algorithm.

        Raises:
            NotImplementedError: if ``self.optimization_type`` is other than
                ``tree_parzen_estimators``, ``random_search``,
                ``gaussian_processes`` or ``genetic_algorithm``.
        """

        hyper_to_opt = self.hyperparams_to_optimize
        if self.optimization_type == 'tree_parzen_estimators':
            hyperparameter_space = bayesian_tpe.hyperopt_space(hyper_to_opt)
        elif self.optimization_type == 'random_search':
            hyperparameter_space = bayesian_tpe.hyperopt_space(hyper_to_opt)
        elif self.optimization_type == 'gaussian_processes':
            hyperparameter_space = bayesian_gp.skopt_space(hyper_to_opt)
        elif self.optimization_type == 'genetic_algorithm':
            hyperparameter_space = genetic_algorithm.gen_al_space(hyper_to_opt)
        else:
            raise NotImplementedError(
                'Other optimization types are not supported yet')
        return hyperparameter_space

    def create_optimization_algorithm(self, hyperparameter_space):
        """ Create optimization algorithm for Ray Tune.

        Currently, only tree parzen estimators, random search,
        gaussian processes and genetic algorithm based optimization using Ray
        Tune is supported.

        Parameters:
            space (dict, list or ray.tune.automl.search_space.SearchSpace):
                space of hyperparameters following the syntax required by
                the optimization algorithm.

        Returns:
            Optimization algorithm for Ray Tune.

        Raises:
            NotImplementedError: if self.optimization_type is other than
                ``tree_parzen_estimators``, ``random_search``,
                ``gaussian_processes`` or ``genetic_algorithm``.
        """

        if self.optimization_type == 'tree_parzen_estimators':
            algorithm = HyperOptSearch(
                hyperparameter_space,
                max_concurrent=self.num_parallel_trials,
                metric='loss',
                mode=self.mode,
                n_initial_points=self. n_initial_points,
                random_state_seed=self.random_state,
                gamma=self.tpe_config.get('gamma', 0.25))

        elif self.optimization_type == 'random_search':
            algorithm = HyperOptSearch(
                hyperparameter_space,
                max_concurrent=self.num_parallel_trials,
                metric='loss',
                mode=self.mode,
                random_state_seed=self.random_state)
            algorithm.algo = hpo.rand.suggest

        elif self.optimization_type == 'gaussian_processes':
            if not self.reload_trials:
                # the gaussian processes based optimization needs an
                # extra parameter self.gp_opt
                self.gp_opt = skopt.Optimizer(
                    hyperparameter_space,
                    n_initial_points=self.n_initial_points,
                    base_estimator=self.gp_config.get(
                        'base_estimator', 'GP'),
                    acq_func=self.gp_config.get(
                        'acq_function', 'gp_hedge'),
                    acq_optimizer=self.gp_config.get(
                        'acq_optimizer', 'auto'),
                    random_state=self.random_state,
                    acq_func_kwargs={'xi': self.gp_config.get('xi',
                                                              0.01),
                                     'kappa': self.gp_config.get('kappa',
                                                                 1.96)})

            hyperparams_names = [key for key in self.hyperparams_to_optimize]
            algorithm = SkOptSearch(self.gp_opt,
                                    hyperparams_names,
                                    max_concurrent=self.num_parallel_trials,
                                    metric='loss',
                                    mode=self.mode)

        elif self.optimization_type == 'genetic_algorithm':
            algorithm = GeneticSearch(
                hyperparameter_space,
                reward_attr='loss',
                max_generation=self.ga_config['max_generation'],
                population_size=self.ga_config['population_size'],
                population_decay=self.ga_config.get('population_decay', 0.95),
                keep_top_ratio=self.ga_config.get('keep_top_ratio', 0.2),
                selection_bound=self.ga_config.get('selection_bound', 0.4),
                crossover_bound=self.ga_config.get('crossover_bound', 0.4))

        else:
            raise NotImplementedError(
                'Other optimization types are not supported yet')

        return algorithm

    def get_ctlearn_metric_to_optimize(self, hyperparams):
        """ Evaluate a CTLearn model and return metric to optimize.

        Parameters:
            hyperparams (dict): set of hyperparameters to evaluate provided by
                the optimizer.

        Returns:
            float: metric to optimize.

        """

        # if more than one CTLearn model is going to be evaluated at the same
        # time, delay the execution of each model for a small random number of
        # seconds in order to avoid the simultaneus creation of CTLearn working
        # folders that have the same name, which would lead to errors
        if self.num_parallel_trials > 1:
            sleep(random.randint(1, 20))

        # evaluate a CTLearn model and return metric
        metric = common.ctlearn_objective(self, hyperparams)
        if self.optimization_type == 'genetic_algorithm':
            # the genetic algorithm internally maximizes the loss, so:
            if self.mode == 'min':
                metric *= -1

        return metric

    def optimize(self, objective_function):
        """ Start the optimization of ``objective_function`` using Ray Tune.

        Currently, only tree parzen estimators, random search,
        gaussian processes and genetic algorithm based optimization using Ray
        Tune is supported.

        Parameters:
            objective_function: function to optimize following the syntax:

                .. code:: python

                    def(hyperparams, reporter):
                        ...
                        # compute loss to optimize
                        ...
                        reporter(loss=loss)

        Returns:
            ExperimentAnalysis: object used for analyzing results from Tune
            ``run()``.

        """

        # start a Ray cluster with given resources and connect to it
        ray.init(num_cpus=self.num_cpus,
                 num_gpus=self.num_gpus)

        # set CTLearn basic config and fixed hyperparameters
        self.set_basic_config()

        # create custom trial names for Ray Tune
        def trial_str_creator(trial):
            return 'Iteration_{}: {}'.format(self.iteration.value + 1,
                                             trial.config)

        # execute optimization
        ray_result = run(
            objective_function,
            name='ray_exp',
            resources_per_trial={'cpu': self.num_cpus_per_trial,
                                 'gpu': self.num_gpus_per_trial},
            num_samples=self.num_max_evals,
            local_dir=os.path.join(self.working_directory,
                                   'ray_optimization_results'),
            search_alg=self.optimization_algorithm,
            verbose=1,
            queue_trials=True,
            trial_name_creator=ray.tune.function(trial_str_creator))

        # get best metric and hyperparameters found
        results_dataframe = pd.read_csv(self.optim_results_path)
        if self.data_set_to_optimize == 'validation':
            best_metric_label = self.metric_to_optimize + '_val'
        else:
            best_metric_label = self.metric_to_optimize + '_pred'
        if self.mode == 'max':
            best_metric = results_dataframe[best_metric_label].max()
        else:
            best_metric = results_dataframe[best_metric_label].max()

        best_hyperparameters = ray_result.get_best_config(metric='loss',
                                                          mode=self.mode)

        # log best metric and hyperparameters found
        self.logger.info(
            'Best metric found: {} = {:.4f}, with hyperparameters: {}'
            .format(best_metric_label, best_metric, best_hyperparameters))

        # create and save gaussian_processes model for further analysis
        if self.optimization_type == 'gaussian_processes':
            gp_model = skopt.utils.create_result(self.gp_opt.Xi,
                                                 self.gp_opt.yi,
                                                 self.gp_opt.space,
                                                 self.gp_opt.rng,
                                                 models=self.gp_opt.models)
            gp_mod_path = os.path.join(self.working_directory, 'gp_model.pkl')
            with open(gp_mod_path, 'wb') as output_file:
                pickle.dump(gp_model, output_file)
            self.logger.info('Gaussian_processes model saved')

        # save trials file
        if self.optimization_type != 'genetic_algorithm':
            common.save(self)
            self.logger.info('Trials file saved')

        self.logger.info('Optimization run finished')

        # terminate processes started by ray.init()
        ray.shutdown()

        return ray_result

    def optimize_ctlearn_model(self):
        """ Start the optimization of a CTLearn model using Ray Tune.

        Currently, only tree parzen estimators, random search,
        gaussian processes and genetic algorithm based optimization using Ray
        Tune is supported.

        Returns:
            ExperimentAnalysis: object used for analyzing results from Tune
            ``run()``.

        """

        def ctlearn_objective(hyperparams, reporter):
            loss = self.get_ctlearn_metric_to_optimize(hyperparams)
            reporter(loss=loss)

        result = self.optimize(ctlearn_objective)
        return result


#####################
# launch optimization
#####################
if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description=('Run CTLearn model optimization'))
    PARSER.add_argument(
        'opt_config',
        help='path to YAML file containing ctlearn_optimizer configuration')
    ARGS = PARSER.parse_args()

    with open(ARGS.opt_config, 'r') as opt_conf:
        OPT_CONF = yaml.load(opt_conf)

    OPT = Optimizer(OPT_CONF)
    OPT_RESULT = OPT.optimize_ctlearn_model()
