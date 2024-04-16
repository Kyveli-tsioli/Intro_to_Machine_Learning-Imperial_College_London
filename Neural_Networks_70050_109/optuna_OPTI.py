import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from part2_house_value_regression import Regressor

import optuna ### hyperparameter optimisation
import warnings

class Optuna_Study:
    def train_evaluate(params_nn):
        """Runs NN and returns corresponding value, with given set of parameters

        Args:
            params_nn: parameters for NN
            x_train: training dataset

        Returns:
             error on val dataset
        """
        output_label = "median_house_value"
        data = pd.read_csv("housing.csv") 
        x = data.loc[:, data.columns != output_label]
        y = data.loc[:, [output_label]]
        x, x_test, y, y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)
        x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.25,train_size =0.75)
        
        regressor = Regressor(x_train,
                              params_nn['minibatch_size'],
                              params_nn['hidden_layers'],
                              params_nn['activations'],
                              params_nn['nb_epoch'],
                              params_nn['_optimizer'],
                              params_nn['lr'])
        
        regressor.fit(x_train, y_train)
        error = regressor.score(x_val, y_val)
        return error

    def objective(trial):
        layer_combinations = layer_combs 
  
        params_nn ={
            'minibatch_size': trial.suggest_categorical('minibatch_size', [50, 100, 200, 500, 1000]),
            'hidden_layers': trial.suggest_categorical('hidden_layers', layer_combinations),
            'activations': nn.ReLU,
            'nb_epoch': trial.suggest_categorical('nb_epoch', [50, 100, 200, 300, 400, 500]), 
            '_optimizer': trial.suggest_categorical('_optimizer', [optim.Adam, optim.SGD, optim.NAdam, optim.RMSprop]),
            'lr': trial.suggest_categorical('lr', [10**(-i) for i in range(1,5 + 1)]),
            }
        return Optuna_Study.train_evaluate(params_nn)
    
    def start_study(n_trials=1000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            study = optuna.create_study(study_name="Hyperparameter Optimisation", directions=['minimize'])
            study.optimize(Optuna_Study.objective, n_trials=n_trials)
    
            print("Best study:")
            best_trial = study.best_trial
            print(best_trial)
        
            return best_trial


def create_hidden_layers(n_combs=100):
    min_n_layers = 1
    max_n_layers = 4
    max_width_layers = 8
    min_width_layers = 4

    #List comprehension to generate all combinations of layer sizes
    layer_combinations = [
        list([2**np.random.randint(min_width_layers, max_width_layers + 1) for n in range(np.random.randint(min_n_layers, max_n_layers + 1))])
        for _ in range(n_combs)  
    ]
    return tuple(layer_combinations)

global layer_combs
layer_combs = create_hidden_layers()

Optuna_Study.start_study()
