##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from tqdm import tqdm


main_path = os.path.split(os.getcwd())[0] + '/Epidemiology_Behavior_Games'
config_path = main_path + '/config.csv'
config_data = pd.read_csv(config_path, sep=',', header=None, index_col=0)

results_path = config_data.loc['results_dir'][1]


##

t_max = 150
gamma = 1 / 7
t = np.linspace(0, t_max, t_max*2)
beta_coop = 6.5*7
sigma_coop = 0.95

beta_def = 0.4
sigma_def = 0.5

omega_search = np.linspace(0, 1, 100)
alpha_search = np.linspace(0, 2, 100)