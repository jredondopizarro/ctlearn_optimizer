import os
import re
import shutil
import csv
import pickle
import logging
import multiprocessing
from timeit import default_timer as timer
import numpy as np
import sklearn.metrics
import yaml
from ctlearn.run_model import run_model

# set dummy authentication key for multiprocessing
multiprocessing.current_process().authkey = b'1234'


def set_value(dictionary, value, *keys):
    """Modify the value the keys point to in a nested dictionary.

    The dictionary can be a nested dictionary containing lists, these lists can
    also contain nested dictionaries, and so on. The keys list can contain
    strings (which refer dictionary keys) and integers (which refer list
    indices). Dictionary cannot be empty.

    Parameters:
        dictionary [dict]: dictionary that contais the key-value pair the user
            wishes to modify.
        value [int, float, string]: value to set.
        keys [list]: list of keys containing strings and integers.

    Returns:
        modified dictionary [dict].

    Example:
        >>> dictionary = {'a':[0,{'b':1},0]}
        >>> value = 2
        >>> keys = ['a', 1, 'b']
        >>> set_value(dictionary, value, *keys) = {'a':[0,{'b':2},0]}

    Raises:
        TypeError: if type(dictionary) != dict.
    """

    if not isinstance(dictionary, dict):
        raise TypeError('set_value expects dict as first argument')

    _keys = keys[:-1]
    _element = dictionary
    for key in _keys:
        _element = _element[key]
    _element[keys[-1]] = value

    return dictionary


def create_nested_item(dictionary, *keys):
    """Create an empty item with specific keys and positions in a dictionary.

    The dictionary can be a nested dictionary containing lists, these lists can
    also contain nested dictionaries, and so on. The keys list can contain
    string (which refer dictionary keys) and integers (which refer list
    indices). The dictionary may or may be not empty.

    Parameters:
        dictionary [dict]: dictionary to modify.
        keys [list]: list of keys containing strings and integers.

    Returns:
        modified dictionary [dict].

    Example:
        >>> dictionary = {}
        >>> keys = ['a', 'b', 1, 'c', 2 , 'd']
        >>> create_nested_item(dictionary, *keys) =
        >>> {'a': {'b': [0, {'c': [0, 0, {'d': {}}]}]}}

    Raises:
        TypeError: if type(dictionary) != dict.
    """

    if not isinstance(dictionary, dict):
        raise TypeError('create_nested_item expects dict as first argument')

    _keys = keys
    _element = dictionary

    # iterate over the list of keys
    for counter, key in enumerate(_keys, 1):
        # set next_key value
        if counter < len(_keys):
            next_key = _keys[counter]
        else:
            next_key = None
        # key is str, therefore _element is dict
        if isinstance(key, str):
            if isinstance(_element, dict):
                # if key in the dictionary, access it
                if key in _element:
                    _element = _element[key]
                # else, create new item and access it
                else:
                    if isinstance(next_key, str):
                        _element.update({'{}'.format(key): {}})
                        _element = _element[key]
                    if isinstance(next_key, int):
                        _element.update({'{}'.format(key): []})
                        _element = _element[key]
                    if next_key is None:
                        _element.update({'{}'.format(key): {}})
        # key is int, therefore _element is list
        if isinstance(key, int):
            if isinstance(_element, list):
                # if list lenght is enought
                if len(_element) > key:
                    _dummy_element = _element[key]
                    # create new item
                    if (isinstance(next_key, str) and
                            not isinstance(_dummy_element, dict)):
                        _element[key] = {}
                    if (isinstance(next_key, int) and
                            not isinstance(_dummy_element, list)):
                        _element[key] = []
                # else, extend the list
                else:
                    while len(_element) < key + 1:
                        _element.append(0)
                    # create new item
                    if isinstance(next_key, str):
                        _element[key] = {}
                    if isinstance(next_key, int):
                        _element[key] = []
                # access the item
                _element = _element[key]

    return dictionary


def auxiliar_modify_params(self, hyperparams):
    """Modify the values of the hyperparameters in CTLearn configuration file.

    This function also modifies the logging model_directory of CTLearn and the
    prediction_file_path.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
        hyperparams [dict]: dictionary containing values of the
            hyperparameters.

    """

    # load ctlearn config file
    with open(self.ctlearn_config_path, 'r') as config:
        myconfig = yaml.load(config)

    # empty layers list in myconfig in order to get rid of previous
    # configurations
    myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'] = []

    # modify values of the hyperparameters in myconfig
    for param, value in hyperparams.items():
        if param in self.hyperparameters_config:
            # create hyperparameter empty item in myconfig
            create_nested_item(myconfig, *self.hyperparameters_config[param])
            # set hyperparameter value
            set_value(myconfig, value, *self.hyperparameters_config[param])

    # set model_directory and prediction_file_path
    myconfig['Logging']['model_directory'] = os.path.join(
        self.working_directory, 'run' + str(self.iteration.value))
    myconfig['Prediction']['prediction_file_path'] = os.path.join(
        self.working_directory, 'run' + str(self.iteration.value),
        'predictions_run{}.csv'.format(self.iteration.value))

    # dump ctlearn configuration
    with open(self.ctlearn_config_path, 'w') as config:
        yaml.dump(myconfig, config)


def get_pred_metrics(self):
    """Get CTLearn prediction metrics from the current CTLearn logging folder.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.

    Returns:
        metrics_pred_to_log [dict]:
            dictionary containing prediction set metrics to log to the
            optimization results file.

    """

    # load prediction file
    predictions_path = os.path.join(
        self.working_directory, 'run' + str(self.iteration.value),
        'predictions_run{}.csv'.format(self.iteration.value))

    # load prediction data
    predictions = np.genfromtxt(predictions_path, delimiter=',', names=True)
    labels = predictions['gamma_hadron_label'].astype(int)
    gamma_classifier_values = predictions['gamma']
    predicted_class = predictions['predicted_class'].astype(int)

    # compute metrics
    fpr, tpr, _thresholds = sklearn.metrics.roc_curve(
        labels, gamma_classifier_values, pos_label=0)
    auc = sklearn.metrics.auc(fpr, tpr)
    f1 = sklearn.metrics.f1_score(labels, predicted_class)
    acc = sklearn.metrics.accuracy_score(labels, predicted_class)
    bacc = sklearn.metrics.balanced_accuracy_score(labels, predicted_class)
    prec = sklearn.metrics.precision_score(labels, predicted_class)
    rec = sklearn.metrics.recall_score(labels, predicted_class)
    log_loss = sklearn.metrics.log_loss(labels, predicted_class)

    metrics_pred = {'auc': auc, 'acc': acc, 'bacc': bacc, 'f1': f1,
                    'prec': prec, 'rec': rec, 'log_loss': log_loss}

    # compute validation user defined metric if required
    if self.user_defined_metric_pred is not None:
        user_defined = eval(self.user_defined_metric_pred['expression'])
        metrics_pred.update(
            {self.user_defined_metric_pred['label']: user_defined})

    # return metrics_pred_to_log
    metrics_pred_to_log = {}
    for metric in self.list_metrics_pred_to_log:
        metrics_pred_to_log.update(
            {metric + '_pred': metrics_pred.get(metric)})

    return metrics_pred_to_log


def get_val_metrics(self):
    """Get CTLearn validation metrics from the current CTLearn logging folder.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.

    Returns:
        metrics_val_to_log [dict]:
            dictionary containing validation set metrics to log to the
            optimization results file.

    """

    # load training log file
    run_folder = os.path.join(self.working_directory,
                              'run' + str(self.iteration.value))
    for file in os.listdir(run_folder):
        if file.endswith('logfile.log'):
            with open(os.path.join(run_folder, file)) as log_file:
                contents = log_file.read()
                # ensure that prediction log file is not loaded
                if 'Training' in contents:
                    train_logfile = file

    # find required data
    with open(os.path.join(run_folder, train_logfile), 'r') as stream:
        r = re.compile('INFO:Saving dict for global step .*')
        matches = list(filter(r.match, stream))
        assert len(matches) > 0
        val_info = matches[-1]

    # extract validation metrics
    auc = float(re.findall(r'auc = [-+]?\d*\.*\d+', val_info)[0][6:])
    acc = float(re.findall(r'accuracy = [-+]?\d*\.*\d+', val_info)[0][11:])
    acc_gamma = float(re.findall(
        r'accuracy_gamma = [-+]?\d*\.*\d+', val_info)[0][17:])
    acc_proton = float(re.findall(
        r'accuracy_proton = [-+]?\d*\.*\d+', val_info)[0][18:])
    loss = float(re.findall(r'loss = [-+]?\d*\.*\d+', val_info)[0][7:])

    metrics_val = {'auc': auc, 'acc': acc, 'acc_gamma': acc_gamma,
                   'acc_proton': acc_proton, 'loss': loss}

    # compute prediction user defined metric
    if self.user_defined_metric_val is not None:
        user_defined = eval(
            self.user_defined_metric_val['expression'], metrics_val)
        metrics_val.update(
            {self.user_defined_metric_val['label']: user_defined})

    # return metrics_val_to_log
    metrics_val_to_log = {}
    for metric in self.list_metrics_val_to_log:
        metrics_val_to_log.update({metric + '_val': metrics_val.get(metric)})

    return metrics_val_to_log


def set_basic_config(self):
    """Set basic config and fixed hyperparameters in CTLearn config file.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
    """

    # load ctlearn config file
    with open(self.ctlearn_config_path, 'r') as config:
        myconfig = yaml.load(config)

    # set basic configuration
    myconfig['Training']['num_validations'] = (self.basic_config
                                               ['num_validations'])
    myconfig['Training']['num_training_steps_per_validation'] = (
        self.basic_config['num_training_steps_per_validation'])
    myconfig['Data']['Input']['batch_size'] = self.basic_config['batch_size']
    myconfig['Model']['model_directory'] = self.basic_config.get(
        'model_directory', 'null')
    myconfig['Data']['Loading']['validation_split'] = (self.basic_config
                                                       ['validation_split'])
    myconfig['Data']['Processing']['sorting'] = self.basic_config.get(
        'sorting', 'null')
    myconfig['Data']['Loading']['min_num_tels'] = self.basic_config.get(
        'min_num_tels', 1)
    myconfig['Data']['Loading']['example_type'] = (self.basic_config
                                                   ['example_type'])
    myconfig['Data']['Loading']['seed'] = self.basic_config.get('seed', None)
    if self.basic_config['model'] == 'cnn_rnn':
        myconfig['Model']['model']['module'] = 'cnn_rnn'
        myconfig['Model']['model']['function'] = 'cnn_rnn_model'
        assert self.basic_config['example_type'] == 'array'

    elif self.basic_config['example_type'] == 'single_tel':
        myconfig['Model']['model']['module'] = 'single_tel'
        myconfig['Model']['model']['function'] = 'single_tel_model'
        assert self.basic_config['example_type'] == 'single_tel'

    myconfig['Data']['Loading']['selected_tel_types'] = (
        self.basic_config['selected_tel_types'])

    aux_dict = {'SST:ASTRICam': {'camera_types': 'ASTRICam',
                                 'interpolation_image_shape': [56, 56, 1]},
                'SST:CHEC': {'camera_types': 'CHEC',
                             'interpolation_image_shape': [48, 48, 1]},
                'SST:DigiCam': {'camera_types': 'DigiCam',
                                'interpolation_image_shape': [96, 96, 1]},
                'MST:FlashCam': {'camera_types': 'FlashCam',
                                 'interpolation_image_shape': [112, 112, 1]},
                'LST:LSTCam': {'camera_types': 'LSTCam',
                               'interpolation_image_shape': [110, 110, 1]},
                'MST:NectarCam': {'camera_types': 'NectarCam',
                                  'interpolation_image_shape': [110, 110, 1]},
                'SCT:SCTCam': {'camera_types': 'SCTCam',
                               'interpolation_image_shape': [120, 120, 1]}}

    myconfig['Image Mapping']['camera_types'] = []
    myconfig['Image Mapping']['interpolation_image_shape'] = {}
    for tel_type in self.basic_config['selected_tel_types']:
        element = aux_dict[tel_type]
        myconfig['Image Mapping']['camera_types'].append(
            element['camera_types'])
        myconfig['Image Mapping']['interpolation_image_shape'].update(
            {element['camera_types']: element['interpolation_image_shape']})

    # set values of the fixed hyperparameters
    if self.fixed_hyperparameters is not None:
        for param, value in self.fixed_hyperparameters.items():
            create_nested_item(myconfig, *self.hyperparameters_config[param])
            set_value(myconfig, value, *self.hyperparameters_config[param])

    # dump ctlearn configuration
    with open(self.ctlearn_config_path, 'w') as config:
        yaml.dump(myconfig, config)


def train(self):
    """Run a CTlearn model training.

    Debug is set to False and log_to_file is set to True.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
    """

    # update file_list in ctlearn config
    with open(self.ctlearn_config_path, 'r') as config:
        myconfig = yaml.load(config)

    myconfig['Data']['file_list'] = os.path.join(
        self.working_directory, self.basic_config['training_file_list'])

    # dump ctlearn configuration
    with open(self.ctlearn_config_path, 'w') as config:
        yaml.dump(myconfig, config)

    # run training
    run_model(myconfig, mode='train', debug=False, log_to_file=True)


def predict(self):
    """Predict using a trained CTLearn model.

    Debug is set to False and log_to_file is set to True.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
    """

    # update file_list in ctlearn config
    with open(self.ctlearn_config_path, 'r') as config:
        myconfig = yaml.load(config)

    myconfig['Data']['file_list'] = os.path.join(
        self.working_directory, self.basic_config['prediction_file_list'])

    # modify ctlearn config to make sure that a prediction file will be created
    myconfig['Prediction']['export_as_file'] = True
    myconfig['Prediction']['true_labels_given'] = True

    # dump ctlearn configuration
    with open(self.ctlearn_config_path, 'w') as config:
        yaml.dump(myconfig, config)

    # run prediction
    run_model(myconfig, mode='predict', debug=False, log_to_file=True)


def modify_optimizable_params(self, hyperparams):
    """Update CTLearn config file with new hyperparameters at each iteration.

    This function takes the dictionary containing the values of the
    hyperparameters to optimize suggested by the optimizer, flattens and
    corrects the dictionary if required, then add the values of the dependent
    hyperparameters to the dictionary. Finally calls
    ``auxiliar_modify_params()`` to modify the hyperparameters.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
        hyperparams [dict]: flat or nested dictionary containing the values of
            the hyperparameters to optimize suggested by the optimizer.

    Returns:
        flat dictionary containing the values of the dependent hyperparameters
        and hyperparameters to optimize  [dict].
    """

    # flatten optimizable hyperparameters dict if required
    def aux_flat(hyperparams):
        flat_hyperparams = {}
        for key, item in hyperparams.items():
            if not isinstance(item, dict):
                flat_hyperparams.update({key: item})
            else:
                flat_hyperparams.update(aux_flat(item))
        return flat_hyperparams

    hyperparams = aux_flat(hyperparams)

    # correct hyperparameters_to_optimize keys (hyperopt space creator doesn't
    # support repeated labels, so a ! character is appended to each repeated
    # label)
    corrected_hyperparams = {}
    for key in hyperparams:
        if key.endswith('!'):
            dummy_key = key
            while dummy_key.endswith('!'):
                dummy_key = dummy_key[:-1]
            corrected_hyperparams.update({dummy_key: hyperparams[key]})
        else:
            corrected_hyperparams.update({key: hyperparams[key]})

    hyperparams = corrected_hyperparams

    # add dependent hyperparameters to the hyperparameters dict
    if self.dependent_hyperparameters is not None:
        for param, expression in self.dependent_hyperparameters.items():
            hyperparams.update({param: eval(expression, hyperparams)})

    # update myconfig with the values in hyperparams dict
    auxiliar_modify_params(self, hyperparams)

    return hyperparams


def save(self):
    """ Save trials of the current run at the working folder as ``trials.pkl``.

    Currently, trial saving for only tree_parzen_estimators, random_search or
    gaussian_processes based optimization using Ray Tune is supported.

    Raises:
        NotImplementedError: if self.optimization_type == 'genetic_algorithm'.
    """

    if self.optimization_type in ('tree_parzen_estimators',
                                  'random_search'):
        self.optimization_algorithm.save(self.trials_file_path)

    if self.optimization_type == 'gaussian_processes':
        with open(self.trials_file_path, 'wb') as output_file:
            pickle.dump(self.gp_opt, output_file)

    if self.optimization_type == 'genetic_algorithm':
        raise TypeError('trial saving is not currently \
            supported by the genetic algorithm optimization')


def restore(self):
    """ Load ``trials.pkl`` of a previous run from the ``working_directory``.

    Currently, trial loading for only tree_parzen_estimators, random_search or
    gaussian_processes based optimization using Ray Tune is supported.

    Returns:
        gp_opt_restored [skopt.optimizer.optimizer.Optimizer]:
            optimizer provided from Skopt (only if self.optimization.type ==
            'gaussian_processes').


    Raises:
        NotImplementedError: if self.optimization_type is genetic_algorithm.
    """

    if self.optimization_type in ('tree_parzen_estimators',
                                  'random_search'):
        self.optimization_algorithm.restore(self.trials_file_path)

    if self.optimization_type == 'gaussian_processes':
        with open(self.trials_file_path, 'rb') as input_file:
            gp_opt_restored = pickle.load(input_file)

    if self.optimization_type == 'genetic_algorithm':
        raise TypeError('trial loading is not currently \
                     supported by the genetic algorithm optimization')

    return gp_opt_restored if 'gp_opt_restored' in locals() else None


def set_logger(log_path):
    """ Set up new logger writing to both log_path and stdout.

    Ray Tune optimizator runs the objective function on a different Python
    process, so new loggers writing to the same file have to be created when
    necessary.

    Parameters:
        log_path [str]: path to log file.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    # log to file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # log to stdout
    console_handler = logging.StreamHandler(os.sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def optimization_results_logger(self, loss, hyperparams_dict, metrics_pred,
                                metrics_val, run_time):
    """ Write loss, hyperparameters, metrics and run_time to the results file.

    This function log the data to the optimization_results file stored as
    ``optimization_results.csv`` at ``working_directory``.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
        loss [float]: value to optimize.
        hyperparams_dict [dict]: values of the hyperparameters the user wishes
            to store.
        metrics_pred [dict]: values of the prediction set metrics the user
            wishes to store.
        metrics_val [dict]: values of the validation set metrics the user
            wishes to store.
        run_time [float]: execution time the user wishes to store.

    """

    with open(self.optim_results_path, 'a') as file:
        writer = csv.writer(file)
        row_hyperparams = []

        for element in self.hyperparams_to_log:
            if element in hyperparams_dict:
                row_hyperparams.append(hyperparams_dict[element])
            else:
                row_hyperparams.append(0)

        row = [loss, self.iteration.value] + row_hyperparams + \
            list(metrics_val.values()) + \
            list(metrics_pred.values()) + [run_time]
        writer.writerow(row)


def ctlearn_objective(self, hyperparams):
    """ Evaluate a CTLearn model and return metric to optimize.

    Train a CTLearn model and predict if necessary, get the metrics and log
    them to the optimization_results file. Also save trials file for resuming
    training if it has been interrupted.

    Parameters:
        self: ``ctlearn_optimizer.optimizer.Optimizer`` instance.
        hyperparams [dict]: values of the hyperparameters to evaluate
            suggested by the optimizer.

    Returns:
        loss [float]: metric to optimize.
    """
    # set up logger
    logger = set_logger(self.log_path)

    self.iteration.value += 1
    self.counter.value += 1

    logger.info('Current run iteration: {}' .format(self.counter.value))
    logger.info('Global iteration: {}' .format(self.iteration.value))

    # update values of the hyperparameters
    hyperparams_dict = modify_optimizable_params(self, hyperparams)

    start = timer()
    logger.info('Training')
    logger.info('Current hyperparameters: {}'. format(hyperparams_dict))

    # train ctlearn network
    train(self)
    logger = set_logger(self.log_path)
    logger.info('Training ended')
    run_time = timer() - start

    # get validation set metrics
    metrics_val = get_val_metrics(self)
    metrics_pred = {}

    # predict if required
    if self.data_set_to_optimize == 'prediction':
        logger.info('Predicting')
        predict(self)
        logger = set_logger(self.log_path)
        logger.info('Prediction ended')
        metrics_pred = get_pred_metrics(self)

    # set loss depending on metric and data set to optimize
    if self.data_set_to_optimize == 'validation':
        metric = self.metric_to_optimize + '_val'
        loss = metrics_val[metric]
        logger.info('{}: {:.4f}'.format(metric, metrics_val[metric]))

    elif self.data_set_to_optimize == 'prediction':
        metric = self.metric_to_optimize + '_pred'
        loss = metrics_pred[metric]
        logger.info('{}: {:.4f}'.format(metric, metrics_pred[metric]))

    # write loss, hyperparameters, metrics and run_time to the optimization
    # results file
    optimization_results_logger(self, loss, hyperparams_dict, metrics_pred,
                                metrics_val, run_time)

    # remove training folders in order to avoid space issues in long runs
    if self.remove_training_folders:
        run_folder = os.path.join(self.working_directory, 'run' +
                                  str(self.iteration.value))
        shutil.rmtree(run_folder, ignore_errors=True)

    # save trials file
    if self.optimization_type != 'genetic_algorithm':
        save(self)

    return loss
