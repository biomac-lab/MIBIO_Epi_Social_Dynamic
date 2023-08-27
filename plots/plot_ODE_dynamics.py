##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from tqdm import tqdm
from scipy.signal import find_peaks


main_path = os.path.join(os.path.split(os.getcwd())[0],'Epi_Social_Dynamic')
config_path = os.path.join(main_path,'config.csv')
config_data = pd.read_csv(config_path, sep=',', header=None, index_col=0)
results_path = config_data.loc['results_dir'][1]
plots_path = config_data.loc['plots_dir'][1]
parematric_df_dir = config_data.loc['parametric_df_dir'][1]


##
t_max = 200
gamma = 1 / 7
t = np.linspace(0, t_max, t_max*5)

def count_oscillations(sim):
    idx_peaks, dict_peaks = find_peaks(sim)
    return np.size(idx_peaks)


def graph_1D_experimentation(param_search, param_name:str):
    Imax_ = np.zeros(param_search.shape)
    Tmax_ = np.zeros(param_search.shape)
    Ifinal_ = np.zeros(param_search.shape)
    Dfinal_ = np.zeros(param_search.shape)
    Ioscillations_ = np.zeros(param_search.shape)
    Doscillations_ = np.zeros(param_search.shape)

    for idx1, p in enumerate(param_search):
        path_to_results = os.path.join(results_path, 'ode_results', '1D', 'ode_replicator_{}_{:0.2f}'.format(param_name, p)+'.csv')
        df_temp = pd.read_csv(path_to_results, index_col=0)
        
        Imax_[idx1] = np.max(df_temp[['I']])
        Tmax_[idx1] = np.array(df_temp[['time']])[np.argmax(df_temp[['I']])]
        Ifinal_[idx1] = np.array(df_temp[['I']])[-1]
        Dfinal_[idx1] = np.array(df_temp[['D']])[-1]
        Ioscillations_[idx1] = count_oscillations(np.array(df_temp[['I']])[:,0])
        Doscillations_[idx1] = count_oscillations(np.array(df_temp[['D']])[:,0])

    fig, ax = plt.subplots(2, 3, figsize=(14,10))

    ax[0,0].plot(param_search, Imax_)
    ax[0,0].set_title('Max. Infected')
    ax[0,0].grid()

    ax[1,0].plot(param_search, Tmax_)
    ax[1,0].set_xlabel(f'{param_name}')
    ax[1,0].set_title('Max. Infected Time')
    ax[1,0].grid()
    
    ax[0,1].plot(param_search, Ifinal_)
    ax[0,1].set_title('Final Infected')
    ax[0,1].grid()
    
    ax[1,1].plot(param_search, Dfinal_)
    ax[1,1].grid()
    ax[1,1].set_xlabel(f'{param_name}')
    ax[1,1].set_title('Final Defectors')

    ax[0,2].plot(param_search, Ioscillations_)
    ax[0,2].grid()
    ax[0,2].set_title('# Peaks of Infected')

    ax[1,2].plot(param_search, Doscillations_)
    ax[1,2].grid()
    ax[1,2].set_xlabel(f'{param_name}')
    ax[1,2].set_title('# Peaks of Defectors')

    if not os.path.isdir( os.path.join(results_path, plots_path, '1D') ):
                os.makedirs(os.path.join(results_path, plots_path, '1D'))
    
    plt.savefig(os.path.join(results_path, plots_path, '1D','plot_features_{}_exp.jpeg'.format(param_name)))
    plt.close()


df_parametric = pd.read_csv(os.path.join(main_path, parematric_df_dir), index_col=0)
beta_ = df_parametric[['beta']]
sigmaD_ = df_parametric[['sigmaD']]
sigmaC_ = df_parametric[['sigmaC']]

beta_search = np.linspace(beta_.loc['min'][0], beta_.loc['max'][0], int(beta_.loc['num'][0]))
sigmaD_search = np.linspace(sigmaD_.loc['min'][0], sigmaD_.loc['max'][0], int(sigmaD_.loc['num'][0]))
sigmaC_search = np.linspace(sigmaC_.loc['min'][0], sigmaC_.loc['max'][0], int(sigmaC_.loc['num'][0]))

graph_1D_experimentation(beta_search, 'beta')
graph_1D_experimentation(sigmaD_search, 'sigmaD')
graph_1D_experimentation(sigmaC_search, 'sigmaC')
