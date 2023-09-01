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

def graph_2D_experimentation(param_search1, param_search2, param_name1: str, param_name2: str):
    Imax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Tmax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ifinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Dfinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ioscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Doscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))

    param_ticks1 = np.linspace(param_search1[0], param_search1[-1], 6)
    param_ticks2 = np.linspace(param_search2[0], param_search2[-1], 6)

    for idx1, p1 in enumerate(param_search1):
        df_temp = df_parametric[[param_name1]]
        param_search1 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))

        for idx2, p2 in enumerate(param_search2):
            path_to_results = os.path.join(results_path, 'ode_results', '2D', 'ode_replicator_{}_{:0.2f}_{}_{:0.2f}'.format(param_name1, p1, param_name2, p2)+'.csv')
            df_temp = pd.read_csv(path_to_results, index_col=0)

            Imax_[idx1, idx2] = np.max(df_temp[['I']])
            Tmax_[idx1, idx2] = np.array(df_temp[['time']])[np.argmax(df_temp[['I']])]
            Ifinal_[idx1, idx2] = np.array(df_temp[['I']])[-1]
            Dfinal_[idx1, idx2] = np.array(df_temp[['D']])[-1]
            Ioscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['I']])[:,0])
            Doscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['D']])[:,0])
        
    fig, axes = plt.subplots(2, 3, figsize=(14,10))
    
    #heatmaps
    im1 = axes[0,0].matshow(Imax_, cmap='coolwarm')
    im2 = axes[1,0].matshow(Tmax_, cmap='coolwarm')
    im3 = axes[0,1].matshow(Ifinal_, cmap='coolwarm')
    im4 = axes[1,1].matshow(Dfinal_, cmap='coolwarm')
    im5 = axes[0,2].matshow(Ioscillations_, cmap='coolwarm')
    im6 = axes[1,2].matshow(Doscillations_, cmap='coolwarm')

    #axes[0,0].set_xticks(range(len(param_ticks2)))
    #axes[0,0].set_yticks(range(len(param_ticks1)))
    #axes[0,0].set_xticklabels(param_ticks2)
    #axes[0,0].set_yticklabels(param_ticks1)
    axes[0,0].set_title('Final Infected', y=-0.1)
    axes[0,0].set_ylabel(f'{param_name1}')
    plt.colorbar(im1, fraction=0.045, pad=0.05, ax=axes[0,0])

    #axes[1,0].set_xticks(range(len(param_ticks2)))
    #axes[1,0].set_yticks(range(len(param_ticks1)))
    #axes[1,0].set_xticklabels(param_ticks2)
    #axes[1,0].set_yticklabels(param_ticks1)
    #axes[1,0].set_title('Final Defectors', y=-0.1)
    axes[1,0].set_ylabel(f'{param_name1}')
    axes[1,0].set_xlabel(f'{param_name2}')
    plt.colorbar(im2, fraction=0.045, pad=0.05, ax=axes[1,0])
    
    #axes[0,1].set_xticks(range(len(param_ticks2)))
    #axes[0,1].set_yticks(range(len(param_ticks1)))
    #axes[0,1].set_xticklabels(param_ticks2)
    #axes[0,1].set_yticklabels(param_ticks1)
    axes[0,1].set_title('Max. Infected', y=-0.1)
    plt.colorbar(im3, fraction=0.045, pad=0.05, ax=axes[0,1])

    #axes[1,1].set_xticks(range(len(param_ticks2)))
    #axes[1,1].set_yticks(range(len(param_ticks1)))
    #axes[1,1].set_xticklabels(param_ticks2)
    #axes[1,1].set_yticklabels(param_ticks1)
    axes[1,1].set_title('Max. Infected Time', y=-0.1)
    axes[1,1].set_xlabel(f'{param_name2}')
    plt.colorbar(im4, fraction=0.045, pad=0.05, ax=axes[1,1])

    #axes[0,2].set_xticks(range(len(param_ticks2)))
    #axes[0,2].set_yticks(range(len(param_ticks1)))
    #axes[0,2].set_xticklabels(param_ticks2)
    #axes[0,2].set_yticklabels(param_ticks1)
    axes[0,2].set_title('# Peaks of Infected', y=-0.1)
    plt.colorbar(im5, fraction=0.045, pad=0.05, ax=axes[0,2])

    #axes[1,2].set_xticks(range(len(param_ticks2)))
    #axes[1,2].set_yticks(range(len(param_ticks1)))
    #axes[1,2].set_xticklabels(param_ticks2)
    #axes[1,2].set_yticklabels(param_ticks1)
    axes[1,2].set_title('# Peaks of Defectors', y=-0.1)
    axes[1,2].set_xlabel(f'{param_name2}')
    plt.colorbar(im6, fraction=0.045, pad=0.05, ax=axes[1,2])

    if not os.path.isdir( os.path.join(results_path, plots_path, '2D') ):
                os.makedirs(os.path.join(results_path, plots_path, '2D'))
    
    plt.savefig(os.path.join(results_path, plots_path, '2D','heatmap_features_{}_{}_exp.jpeg'.format(param_name1,param_name2)))
    plt.close()


             

df_parametric = pd.read_csv(os.path.join(main_path, parematric_df_dir), index_col=0)
beta_ = df_parametric[['beta']]
sigmaD_ = df_parametric[['sigmaD']]
sigmaC_ = df_parametric[['sigmaC']]
list_params = ['beta', 'sigmaD', 'sigmaC']

beta_search = np.linspace(beta_.loc['min'][0], beta_.loc['max'][0], int(beta_.loc['num'][0]))
sigmaD_search = np.linspace(sigmaD_.loc['min'][0], sigmaD_.loc['max'][0], int(sigmaD_.loc['num'][0]))
sigmaC_search = np.linspace(sigmaC_.loc['min'][0], sigmaC_.loc['max'][0], int(sigmaC_.loc['num'][0]))

graph_1D_experimentation(beta_search, 'beta')
graph_1D_experimentation(sigmaD_search, 'sigmaD')
graph_1D_experimentation(sigmaC_search, 'sigmaC')


for idx1, param_name1 in enumerate(list_params):
        df_temp = df_parametric[[param_name1]]
        param_search1 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))

        list_temp = list_params.copy()
        list_temp.remove(param_name1)
        for idx2, param_name2 in enumerate(list_temp):
            df_temp = df_parametric[[param_name2]]
            param_search2 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))
            graph_2D_experimentation(param_search1, param_search2, param_name1, param_name2)
        

