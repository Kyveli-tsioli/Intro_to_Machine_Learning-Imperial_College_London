import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from part2_house_value_regression import Regressor
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import time
from tqdm import tqdm
from scipy.optimize import curve_fit

output_label = "median_house_value"
data = pd.read_csv("housing.csv")
x = data.loc[:, data.columns != output_label]
y = data.loc[:, [output_label]]
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, train_size=0.75)

"""
 Trial 234 finished with value: 50492.22482336348 and parameters: 
 {'minibatch_size': 1000, 
 'hidden_layers': [64, 64], 
 'nb_epoch': 500, 
 '_optimizer': <class 'torch.optim.adam.Adam'>, 
 'lr': 0.01}. 
 Best is trial 234 with value: 50492.22482336348.
"""

params_nn = {
    'minibatch_size': 200, #1000,
    'hidden_layers': [64,16], #[90, 50, 40],
    'minibatch_size': 1000,
    'hidden_layers': [64, 64],
    'activations': nn.ReLU,
    'nb_epoch': 500,
    '_optimizer': optim.Adam,
    'lr': 10 ** (-2)
}


def vary_parameters(params_nn, param_name="lr"):
    opt_list = [optim.Adam, optim.NAdam, optim.SGD, optim.RMSprop]
    lr_list = np.array([[10 ** (-i), 0.5 * 10 ** (-i)] for i in range(1, 5)]).flatten()
    epochs_list = [50, 100, 200, 300, 400, 500]

    errors = []  # Error for whole run
    error_epochs = []  # Errors for each epoch
    runtimes = []
    param_values = None

    if param_name == "lr":
        param_values = lr_list
    elif param_name == "opt":
        param_values = opt_list
    elif param_name == "epoch":
        param_values = epochs_list

    if param_values:  # just one variation
        for param_value in param_values:
            start_time = time.time()

            regressor = Regressor(x_train,
                                  params_nn['minibatch_size'],
                                  params_nn['hidden_layers'],
                                  params_nn['activations'],
                                  params_nn['nb_epoch'] if param_name != "epoch" else param_value,
                                  params_nn['_optimizer'] if param_name != "opt" else param_value,
                                  params_nn['lr'] if param_name != "lr" else param_value)

            regressor.fit(x_train, y_train)
            errors.append(regressor.score(x_val, y_val))
            loss = regressor.get_loss()
            error_epochs.append(loss)

            if param_name == "epoch":
                end_time = time.time()
                elapsed_time = end_time - start_time
                runtimes.append(elapsed_time)

    else:  # two hyper-parameter variations
        if param_name == "lr-opt":
            for opt in tqdm(opt_list):
                error_epochs_per_opt = []
                errors_per_opt = []
                for lr in lr_list:
                    regressor = Regressor(x_train,
                                          params_nn['minibatch_size'],
                                          params_nn['hidden_layers'],
                                          params_nn['activations'],
                                          params_nn['nb_epoch'],
                                          opt,
                                          lr)

                    regressor.fit(x_train, y_train)
                    errors_per_opt.append(regressor.score(x_val, y_val))
                    loss = regressor.get_loss()
                    error_epochs_per_opt.append(loss)
                errors.append(errors_per_opt)
                error_epochs.append(error_epochs_per_opt)

    return runtimes, errors, error_epochs

runtimes, errors, error_epochs = vary_parameters(params_nn, param_name="lr-opt")
#%%

def interpolate(x, y):
    interp_func = interp1d(x, y, kind='linear')
    x_interp = np.linspace(min(x), max(x), 10000)
    y_interp = interp_func(x_interp)
    return x_interp, y_interp

def plot_lr_opt_variation(errors):
    plt.figure(dpi=600, figsize=(7, 6))
    opt_list = ["Adam", "NAdam", "SGD", "RMSprop"]
    lr_list = np.array([[10 ** (-lr), 0.5 * 10 ** (-lr)] for lr in range(1, 5)]).flatten()
    for o, opt in enumerate(opt_list):
        x_interp, y_interp = interpolate(lr_list, errors[o])
        plt.xscale('log')
        plt.scatter(lr_list, errors[o], label=opt_list[o])
        # plt.ylim([40000, 140000])
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        plt.plot(x_interp, y_interp, linestyle='-', linewidth=0.8)
        #plt.minorticks_on()
        plt.grid(which='minor', linewidth='0.5', alpha=0.4, color='gray')
        plt.legend()
        plt.xlabel("Learning rate", fontsize=12)
        plt.ylabel("Validation loss (MSE)", fontsize=12)
    plt.show()
    
plot_lr_opt_variation(errors)

#%%

def vary_epoch(params_nn):
    epochs_list = [50, 100, 200, 300, 400, 500]

    errors = np.zeros(len(epochs_list))  # Error for whole run
    error_epochs = []  # Errors for each epoch
    runtimes = np.zeros(len(epochs_list))

    param_values = epochs_list
        
    for _ in range(10):

        for i, param_value in enumerate(param_values):
            start_time = time.time()

            regressor = Regressor(x_train,
                                  params_nn['minibatch_size'],
                                  params_nn['hidden_layers'],
                                  params_nn['activations'],
                                  param_value,
                                  params_nn['_optimizer'],
                                  params_nn['lr'])

            regressor.fit(x_train, y_train)
            errors[i] += regressor.score(x_val, y_val)
            loss = regressor.get_loss()
            error_epochs.append(loss)

            end_time = time.time()
            elapsed_time = end_time - start_time
            runtimes[i] += elapsed_time
            print(runtimes, errors)

    return runtimes/10, errors/10, error_epochs

runtimes, errors, error_epochs = vary_epoch(params_nn)

#%%

def epochs_runtime(runtimes, errors):
    plt.figure(dpi=600, figsize=(7, 6))
    epochs_list = [50, 100, 200, 300, 400, 500]
    x_interp, y_interp = interpolate(runtimes, errors)
    plt.scatter(runtimes, errors, marker='x', label="Epochs")
    
    for epoch, runtime, error in zip(epochs_list, runtimes, errors):
        offset = 50  # Adjust this value for the desired displacement
        plt.text(runtime, error + offset, str(epoch), fontsize=8, ha='right', va='bottom')

    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    plt.plot(x_interp, y_interp, linestyle='-', linewidth=0.8)
    plt.grid(which='minor', linewidth='0.5', alpha=0.4, color='gray')
    plt.xlabel("Run time (s)", fontsize=12)
    plt.ylabel("Validation loss (MSE)", fontsize=12, labelpad=10)
    plt.legend()
    #plt.ylim([52500, 59200])
    #plt.xlim([0,70])
    plt.show()
epochs_runtime(runtimes, errors)

#%%

def vary_minibatch(params_nn):
    error_epochs = []  # Errors for each epoch
    param_values = [50, 100, 200, 500, 1000]
    
    for i, param_value in tqdm(enumerate(param_values)):
        for n in tqdm(range(5)):
            regressor = Regressor(x_train,
                                  param_value,
                                  params_nn['hidden_layers'],
                                  params_nn['activations'],
                                  params_nn['nb_epoch'],
                                  params_nn['_optimizer'],
                                  params_nn['lr'])
    
            regressor.fit(x_train, y_train)
            regressor.score(x_val, y_val)
            loss = regressor.get_loss()
            if n == 0:
                param_error = np.array(loss)
            else:
                param_error += np.array(loss)
        error_epochs.append(param_error/5)
    return error_epochs
error_epochs = vary_minibatch(params_nn)

#%%
def plot_minibatch(errors):
    plt.figure(dpi=600, figsize=(7, 6))
    param_values = [50, 100, 200, 500, 1000]
    for i, e in enumerate(errors):
        epochs = np.linspace(1, 500, len(e))
        plt.plot(epochs, e, label=param_values[i], linewidth=0.8)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Normalised training loss (MSE)", fontsize=12, labelpad=10)
        plt.ylim([0, 0.04])
        #plt.xlim([0,70])
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        plt.grid(which='minor', linewidth='0.5', alpha=0.4, color='gray')
        plt.legend(loc='upper right')

plot_minibatch(error_epochs)
#%%

def vary_minibatch2(params_nn):
    param_values = [50, 100, 200, 500, 1000]
    errors = np.zeros(len(param_values))
    errrors_std = np.zeros(len(param_values))
    
    for i, param_value in tqdm(enumerate(param_values)):
        errors_param = []
        for n in range(5):
            regressor = Regressor(x_train,
                                  param_value,
                                  params_nn['hidden_layers'],
                                  params_nn['activations'],
                                  params_nn['nb_epoch'],
                                  params_nn['_optimizer'],
                                  params_nn['lr'])
    
            regressor.fit(x_train, y_train)
            errors_param.append(regressor.score(x_val, y_val))
        errors[i] = np.mean(errors_param)
        errrors_std[i] = np.std(errors_param)
    return errors, errrors_std

errors, errrors_std = vary_minibatch2(params_nn)
#%%
plt.figure(dpi=600, figsize=(7, 6))

x = [50, 100, 200, 500, 1000]

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

weights = 1 / errrors_std

initial_params = [np.max(errors), 0.1, np.min(errors)]  
params, covariance = curve_fit(exponential_decay, x, errors, p0=initial_params, sigma=weights)

x_fit = np.linspace(0, max(x)+100, 100)

y_fit = exponential_decay(x_fit, *params)
plt.plot(x_fit, y_fit, '--', label='Exponential Decay Fit', color='red')

plt.scatter(x, errors, zorder=200, marker="o", label=r"$\mu$ over 10 runs", color="black")
plt.errorbar(x, errors, fmt='none', yerr=errrors_std, capsize=5, color = "gray", linewidth=0.8, label=r"$\sigma$ over 10 runs")

plt.xlabel("Minibatch size", fontsize=12)
plt.ylabel("Validation loss (MSE)", fontsize=12, labelpad=10)
#plt.xlim([0,70])
plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
plt.grid(which='minor', linewidth='0.5', alpha=0.4, color='gray')
plt.legend(loc='upper right')
#plt.xlim([0,1100])

#%%

max_layers = 10
max_size = 10 #at least max_layers - 1

layers = [[2**i for i in range(max_size, max_size - n,-1)] for n in range(1, max_layers + 1)]

print(layers)
#%%

val = []
val_std = []
train = []
train_std = []

for i, layer in tqdm(enumerate(layers)):
    val_errors = []
    train_errors = []
    for n in tqdm(range(5)):
        regressor = Regressor(x_train,
                              params_nn['minibatch_size'],
                              layer,
                              params_nn['activations'],
                              params_nn['nb_epoch'],
                              params_nn['_optimizer'],
                              params_nn['lr'])

        regressor.fit(x_train, y_train)
        val_errors.append(regressor.score(x_val, y_val))
        train_errors.append(regressor.get_loss()[-1])
    val.append(np.mean(val_errors))
    val_std.append(np.std(val_errors))
    train.append(np.mean(train_errors))
    train_std.append(np.std(train_errors))

#%%

# Generate some example data
x = range(1,max_layers-1)
y1 = train[:-2]
y2 = val[:-2]
y1err = train_std[:-2]
y2err = val_std[:-2]

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(7,6), dpi=600)

# Plot the first set of data on the left y-axis
ax1.errorbar(x, y1, y1err, color='k')
ax1.set_xlabel("Layer size", fontsize=12)
ax1.set_ylabel("Normalised training loss (MSE)", fontsize=12, labelpad=10, color="k")
ax1.tick_params('y', colors='k')
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

# Create a second y-axis on the right side
ax2 = ax1.twinx()
ax2.errorbar(x, y2, y2err, color='#BD261A')
ax2.set_ylabel("Validation loss (MSE)", fontsize=12, labelpad=10, color="#BD261A")
ax2.tick_params('y', colors='#BD261A')

# Add a legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

#ax2.set_ylim([50000,60000])
#ax1.set_ylim([0,0.03])

plt.show()
#%%
y1 = np.array(train[:-2].copy())
y2 = np.array(val[:-2].copy())
y1err = np.array(train_std[:-2].copy()) 
y2err = np.array(val_std[:-2].copy()) 

#%%

fig, ax1 = plt.subplots(figsize=(7, 6), dpi=600)

ax1.plot(x, y1, color='k')
ax1.set_ylabel("Normalised training loss (MSE)", fontsize=12, labelpad=10, color="k")
ax1.tick_params('y', colors='k')
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(x, y2, color='#BD261A')
ax2.set_ylabel("Validation loss (MSE)", fontsize=12, labelpad=10, color="#BD261A")
ax2.tick_params('y', colors='#BD261A')

ax1.fill_between(x, y1 - y1err, y1 + y1err, color='lightgray', alpha=0.5)
ax2.fill_between(x, y2 - y2err, y2 + y2err, color='lightcoral', alpha=0.5)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#ax2.legend(lines + lines2, labels + labels2, loc='upper right')
ax2.set_ylim([50000,60000])
ax1.set_xlim([1,8])
ax1.set_xlabel("Nº of Hidden layers", fontsize=12)

plt.show()


#%% - BEST MODEL

epochs = [100*i for i in range(1,6)]

val = []
val_std = []
train = []
train_std = []
norm_train = []
norm_train_std = []
test = []
test_std = []

for epoch in tqdm(epochs):
    val_errors = []
    test_errors = []
    train_errors = []
    norm_train_errors = []
    for n in tqdm(range(5)):
        regressor = Regressor(x_train,
                              params_nn['minibatch_size'],
                              params_nn['hidden_layers'],
                              params_nn['activations'],
                              epoch,
                              params_nn['_optimizer'],
                              params_nn['lr'])

        regressor.fit(x_train, y_train)
        val_errors.append(regressor.score(x_val, y_val))
        test_errors.append(regressor.score(x_test, y_test))
        train_errors.append(regressor.score(x_train, y_train))
        norm_train_errors.append(regressor.get_loss()[-1])
    val.append(np.mean(val_errors))
    val_std.append(np.std(val_errors))
    test.append(np.mean(test_errors))
    test_std.append(np.std(test_errors))
    train.append(np.mean(train_errors))
    train_std.append(np.std(train_errors))
    norm_train.append(np.mean(norm_train_errors))
    norm_train_std.append(np.std(norm_train_errors))
    
#%%
x = [100*i for i in range(1,6)]

y1 = np.array(train.copy())
y2 = np.array(val.copy())
y1err = np.array(train_std.copy()) 
y2err = np.array(val_std.copy()) 
y3 = np.array(test.copy())
y3err = np.array(test_std.copy())
y4 = np.array(norm_train.copy())
y4err = np.array(norm_train_std.copy())

#%%

fig, ax1 = plt.subplots(figsize=(7, 6), dpi=600)

# Plot the first set of data on the left y-axis with error bars
#ax1.errorbar(x, y1, y1err, color='k', label='Train')
ax1.plot(x, y1, color='k')
ax1.set_ylabel("Normalised training loss (MSE)", fontsize=12, labelpad=10, color="k")
ax1.tick_params('y', colors='k')
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

# Create a second y-axis on the right side
ax2 = ax1.twinx()
# Plot the second set of data on the right y-axis with error bars
#ax2.errorbar(x, y2, y2err, color='#BD261A', label='Validation')
ax2.plot(x, y2, color='#BD261A')
ax2.plot(x, y3, color='b')
ax2.set_ylabel("Validation loss (MSE)", fontsize=12, labelpad=10, color="#BD261A")
ax2.tick_params('y', colors='#BD261A')

# Fill between the curves
ax1.fill_between(x, y1 - y1err, y1 + y1err, color='lightgray', alpha=0.5)
#ax2.fill_between(x, y2 - y2err, y2 + y2err, color='lightcoral', alpha=0.5)


ax2.fill_between(x, y3 - y3err, y3 + y3err, color='lightcoral', alpha=0.5)

# Add a legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
#ax2.set_ylim([50000,60000])
#ax1.set_xlim([1,8])
ax1.set_xlabel("Epochs", fontsize=12)

plt.show()

lr_list = np.array([[10 ** (-i), 0.5 * 10 ** (-i)] for i in range(1, 5)]).flatten()

plt.figure(dpi=600, figsize=(7,6))
plt.xscale('log')
plt.scatter(lr_list, errors, label = "data")
plt.ylim([40000,140000])
plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
#plt.plot(x_interp, y_interp, label='Interpolated Data', linestyle='--')
plt.minorticks_on()
plt.grid(which='minor', linewidth='0.5', alpha = 0.4, color='gray')
plt.legend()
plt.xlabel("learning rate")
plt.ylabel("Validation loss")
plt.show()
#%%


fig, ax1 = plt.subplots(figsize=(7, 6), dpi=600)

# Plot the first set of data on the left y-axis with error bars
#ax1.errorbar(x, y1, y1err, color='k', label='Train')

ax1.set_ylabel("Normalised training loss (MSE)", fontsize=12, labelpad=10, color="k")
ax1.tick_params('y', colors='k')
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

# Plot the second set of data on the right y-axis with error bars
#ax2.errorbar(x, y2, y2err, color='#BD261A', label='Validation')
ax1.plot(x, y1, color='k', label="Train")
ax1.plot(x, y2, color='#BD261A', label="Validation")
ax1.plot(x, y3, color='b', label="Test")
ax1.set_ylabel("Loss (MSE)", fontsize=12, labelpad=10, color="k")
ax1.tick_params('y', colors='k')

# Fill between the curves
ax1.fill_between(x, y1 - y1err, y1 + y1err, color='lightgray', alpha=0.5)
ax1.fill_between(x, y2 - y2err, y2 + y2err, color='lightcoral', alpha=0.5)
ax1.fill_between(x, y3 - y3err, y3 + y3err, color='blue', alpha=0.5)
ax1.legend(fontsize=12)
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_xlim([100,500])
#ax2.fill_between(x, y2 - y2err, y2 + y2err, color='lightcoral', alpha=0.5)









