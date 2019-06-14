#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ctlearn_optimizer.bayesian_tpe as bayesian_tpe
import ctlearn_optimizer.common as common
import pickle
import logging
import argparse
import hyperopt
import numpy as np
import os
import csv
import yaml
from functools import partial
from shutil import copyfile


class optimizer:
    """ Basic class for an optimizer

    Methods:
        create_space_params: returns hyperparameters space with required style
        set_initial_config: sets basic config and fixed hyperparameters
        get_val_metrics: returns validation set metrics
        get_pred_metrics: returns prediction set metrics
        train: trains a CTlearn model
        predict: predicts using a trained CTLearn model
        objective: returns objective function to optimize with required style
        optimize: starts the optimization
    """

    def __init__(self, opt_config):
        """ Initialize the class

        Load trials file or create one as required. A trials file allows to resume
        an optimization run.
        Load checking_file or create one as required. The checking file is where
        the results of the optimization run (loss, iteration, metrics,
        hyperparameters, time) are stored.

        Args:
            opt_config: loaded optimization configuration file
        """

        self.counter = 0
        self.opt_config = opt_config
        self.random_state = opt_config.get('random_state', None)
        self.ctlearn_config = opt_config['ctlearn_config']
        self.n_startup_jobs = opt_config.get('n_startup_jobs',20)
        self.optimization_type = opt_config['optimization_type']
        self.num_max_evals = opt_config['num_max_evals']
        self.reload_trials = opt_config['reload_trials']
        self.reload_checking_file = opt_config['reload_checking_file']

        if self.opt_config['data_set_to_optimize'] == 'prediction':
            assert(self.opt_config['predict'] is True)

        #load trials file if reload_trials is True
        if self.reload_trials:
            assert(os.path.isfile('trials.pkl'))
            self.trials = pickle.load(open('trials.pkl', 'rb'))
            logging.info('Found trials.pkl file with {} saved trials'.format(len(self.trials.trials)))
            #set iteration and num_max_evals to match load trials
            self.num_max_evals += len(self.trials.trials)
            self.iteration = len(self.trials.trials)
        #else, create trials file
        else:
            self.trials = hyperopt.Trials()
            logging.info('No trials file loaded, starting from scratch')
            self.iteration = 0

        #load checking_file.csv if reload_checking_file is True
        if self.reload_checking_file:
            assert(os.path.isfile('./checking_file.csv'))
            with open('./checking_file.csv', 'r') as file:
                existing_iters_csv = len(file.readlines()) - 1

            logging.info('Found checking_file.csv with {} saved trials, new trials will be added'.format(existing_iters_csv))

            if existing_iters_csv != self.iteration:
                logging.info('Caution: the number of saved trials in trials.pkl and checking_file.csv files  does not match')
        #else, create checking_file.csv
        else:
            logging.info('No checking_file.csv file loaded, starting from scratch')

            hyperparams_to_log = self.opt_config['Hyperparameters']['Hyperparameters_to_log']
            list_metrics_val_to_log = self.opt_config.get('metrics_val_to_log',[])
            list_metrics_pred_to_log = self.opt_config.get('metrics_pred_to_log',[])
            with open('./checking_file.csv', 'w') as file:
                writer = csv.writer(file)
                header = ['loss','iteration']  + hyperparams_to_log + \
                            [element + '_val' for element in list_metrics_val_to_log] + \
                            [element + '_pred' for element in list_metrics_pred_to_log] + ['run_time']

                writer.writerow(header)

    def create_space_hyperparams(self):
        """ Returns hyperparameters space following required style

        Currently, only tree_parzen_estimators based BO and random_search
        using hyperopt are supported

        Returns:
            hyperparameters space following hyperopt syntax

        Raises:
            NotImplementedError if self.optimization_type is other than
            tree_parzen_estimators or random_search
        """

        if self.optimization_type == 'tree_parzen_estimators':
            hyperparameter_space = bayesian_tpe.hyperopt_space(self)
        elif self.optimization_type == 'random_search':
            hyperparameter_space = bayesian_tpe.hyperopt_space(self)
        else:
            raise NotImplementedError('Other optimization types are not supported yet')
        return hyperparameter_space

    def set_initial_config(self):
        """Sets basic config and fixed hyperparameters in ctlearn config file
        """
        common.set_initial_config(self)

    def get_pred_metrics(self):
        """Gets prediction set metrics

        Returns:
            dictionary containing prediction set metrics
        """
        return common.get_pred_metrics(self)

    def get_val_metrics(self):
        """Gets validation set metrics

        Returns:
            dictionary containing validation set metrics
        """
        return common.get_val_metrics(self)

    def train(self):
        """Trains a CTLearn model
        """
        common.train(self)

    def predict(self):
        """Predicts using a trained CTLearn model
        """
        common.predict(self)

    # set
    def objective(self, hyperparams):
        """ Returns objective function to optimize following required style

        Currently, only tree_parzen_estimators based BO and random_search
        using hyperopt are supported

        Returns:
            objective function for hyperopt input - output workflow

        Raises:
            NotImplementedError if self.optimization_type is other than
            tree_parzen_estimators or random_search
        """
        if self.optimization_type == 'tree_parzen_estimators':
            objective = bayesian_tpe.objective(self, hyperparams)
        elif self.optimization_type == 'random_search':
            objective = bayesian_tpe.objective(self, hyperparams)
        else:
            raise NotImplementedError('Other optimization types are not supported yet')
        return objective

    def optimize(self):
        """ Starts the optimization

        Currently, only tree_parzen_estimators based BO and random_search
        using hyperopt are supported

        Raises:
            NotImplementedError if self.optimization_type is other than
            tree_parzen_estimators or random_search
        """
        #set random state
        my_rstate = np.random.RandomState(self.random_state)

        #set initial config and get hyperparameter_space
        self.set_initial_config()
        hyperparameter_space = self.create_space_hyperparams()

        #select otimization algorithm
        if self.optimization_type == 'tree_parzen_estimators':
            algo = partial(hyperopt.tpe.suggest, n_startup_jobs=self.n_startup_jobs)
        if self.optimization_type == 'random_search':
            algo = hyperopt.rand.suggest

        #call hyperopt optimizator
        fmin = hyperopt.fmin(self.objective, hyperparameter_space,
                             algo, trials=self.trials,
                             max_evals=self.num_max_evals, rstate=my_rstate)

        #save trials file
        pickle.dump(self.trials, open('trials.pkl', 'wb'))
        logging.info('trials.pkl saved')
        logging.info('Optimization run finished')


###################
#launch optimization
###################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description=('Run Ctlearn model optimization'))
    parser.add_argument(
            'opt_config',
            help='path to YAML file containing ctlearn_optimizer configuration')
    args = parser.parse_args()

    log_file = f"optimization.log"
    logging.basicConfig(level=logging.INFO, filename=log_file)
    consoleHandler = logging.StreamHandler(os.sys.stdout)
    logging.getLogger().addHandler(consoleHandler)
    logging.info(f'Starting optimization run')

    with open(args.opt_config, 'r') as opt_config:
        opt_config = yaml.load(opt_config)

    model = optimizer(opt_config)
    model.optimize()
