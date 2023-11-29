##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from tqdm import tqdm
from scipy.signal import find_peaks
import seaborn as sns


main_path = os.path.join(os.path.split(os.getcwd())[0],'Epi_Social_Dynamic')
config_path = os.path.join(main_path,'config.csv')

config_data = pd.read_csv(config_path, sep=',', header=None, index_col=0)
results_path = config_data.loc['results_dir'][1]
plots_path = config_data.loc['plots_dir'][1]
parematric_df_dir = config_data.loc['parametric_df_dir'][1]
df_parametric = pd.read_csv(os.path.join(main_path, parematric_df_dir), index_col=0)

##
t_max = 300
gamma = 1 / 7
alpha = 0.2
reward_matrix = np.array([[1.1, 1.1, 0.8, 0.7], [1.3, 1.3, 0.5, 0.3], [2, 1.8, 1, 1], [1.6, 1.4, 1, 1]])
t = np.linspace(0, t_max, t_max*5)
min_prominence = 0.001
beta_ = df_parametric[['beta']]
sigmaD_ = df_parametric[['sigmaD']]
sigmaC_ = df_parametric[['sigmaC']]
d_fract_ = df_parametric[['d_fract']]
list_params = ['beta', 'sigmaD', 'sigmaC']

dict_scenarios = {'static':(False, False, False, False), 
                  'dynamicS':(False, False, False, True), 
                  'dynamicS_SC':(True, False, False, True), 
                  'dynamicSI_SC':(True, False, True, True), 
                  'dynamicSI_SCPA':(True, True, True, True)}


def count_oscillations(sim, min_prominence):
    idx_peaks, dict_peaks = find_peaks(sim, prominence=min_prominence)
    return idx_peaks, len(idx_peaks)

def find_stability_time(sim, epsilon):
    last_val = sim[-1,0]
    stable_sim = sim[np.abs(sim-last_val) <= epsilon]
    return len(stable_sim)  

def graph_simulationFeatures(path_to_results, name_file, folder):
    df_temp = pd.read_csv(path_to_results+'.csv', index_col=0)
    I = np.array(df_temp[['I_c']]) + np.array(df_temp[['I_d']])
    S = np.array(df_temp[['S_c']]) + np.array(df_temp[['S_d']])
    C = np.array(df_temp[['S_c']]) + np.array(df_temp[['I_c']])
    D = np.array(df_temp[['S_d']]) + np.array(df_temp[['I_d']])
    time = np.array(df_temp[['time']])
    beta = np.round((df_temp[['beta']].iloc[0][0]),3)
    sigmaD = np.round((df_temp[['sigma_D']].iloc[0][0]),3)
    sigmaC = np.round((df_temp[['sigma_C']].iloc[0][0]),3)
    
    I_idxPeaks, I_numPeaks = count_oscillations(I[:,0], min_prominence)
    I_timeStable = find_stability_time(I, 0.0001)
    C_idxPeaks, C_numPeaks = count_oscillations(C[:,0], min_prominence)
    C_timeStable = find_stability_time(C, 0.0001)
    
    fig, axes = plt.subplots(2,1, figsize=(12,8))

    axes[0].plot(time, I, label='Infected', color='mediumorchid')
    axes[0].plot(time[-I_timeStable:], I[-I_timeStable:], label='final I = {:0.2f}'.format(I[-1,0]), color='mediumorchid', alpha=0.4, marker='o')
    axes[0].plot(time, S, label='Susceptible', color='dodgerblue')
    axes[0].scatter(time[np.argmax(I),0], np.max(I), label='Max. I = {:0.2f}'.format(np.max(I)), color='purple')
    axes[0].scatter(time[I_idxPeaks,0], I[I_idxPeaks,0], [500]*I_numPeaks, marker='|', label='# I Peaks = {:0.0f}'.format(I_numPeaks), color='indigo')
    axes[0].grid()
    axes[0].legend()
    axes[0].set_ylabel('Percentage [%]')
    axes[0].set_xlabel('Time [days]')

    axes[1].plot(time, C, label='Cooperators', color='limegreen')
    axes[1].plot(time[-C_timeStable:], C[-C_timeStable:], label='final C = {:0.2f}'.format(C[-1,0]), color='limegreen', alpha=0.4, marker='o')
    axes[1].plot(time, D, label='Defectors', color='orangered')
    axes[1].scatter(time[np.argmax(C)], np.max(C), label='Max. C = {:0.2f}'.format(np.max(C)), color='seagreen')
    axes[1].scatter(time[C_idxPeaks,0], C[C_idxPeaks,0], [500]*C_numPeaks, marker='|', label='# C Peaks = {:0.0f}'.format(C_numPeaks), color='darkgreen')
    axes[1].grid()
    axes[1].legend()
    axes[1].set_ylabel('Percentage [%]')
    axes[1].set_xlabel('Time [days]')

    fig.suptitle(f'${{\\beta}}$ = {beta} & ${{R_0}}$={np.round(alpha*beta/gamma,3)} \n ${{\sigma_D}}$ = {sigmaD} & ${{\sigma_C}}$={sigmaC}')
    
    if not os.path.isdir(os.path.join(main_path, plots_path, 'ODE_Simulations', folder) ):
                os.makedirs(os.path.join(main_path, plots_path, 'ODE_Simulations', folder))

    plt.savefig(os.path.join(main_path, 'plots', 'ODE_Simulations', folder, name_file+'.jpeg'), dpi=500)
    plt.close()

def graph_1D_experimentation(param_search, param_name:str, folder:str):
    Imax_ = np.zeros(param_search.shape)
    Tmax_ = np.zeros(param_search.shape)
    Ifinal_ = np.zeros(param_search.shape)
    Cfinal_ = np.zeros(param_search.shape)
    Ioscillations_ = np.zeros(param_search.shape)
    Coscillations_ = np.zeros(param_search.shape)

    for idx1, p in enumerate(param_search):
        if param_name == 'beta':
            str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_0.50'.format(p)  
        elif param_name == 'sigmaD':
            str_file = 'ode_coupled_beta_0.50_sigmaD_{:0.2f}_sigmaC_0.50'.format(p)
        else:
            str_file = 'ode_coupled_beta_0.50_sigmaD_0.50_sigmaC_{:0.2f}'.format(p)
        path_to_results = os.path.join(results_path, '1D', folder, str_file+'.csv')
        df_temp = pd.read_csv(path_to_results, index_col=0)
        df_temp['I'] = df_temp['I_c']+df_temp['I_d']
        df_temp['C'] = df_temp['S_c']+df_temp['I_c']

        Imax_[idx1] = np.max(df_temp[['I']])
        Tmax_[idx1] = np.array(df_temp[['time']])[np.argmax(df_temp[['I']])]
        Ifinal_[idx1] = np.array(df_temp[['I']])[-1]
        Cfinal_[idx1] = np.array(df_temp[['C']])[-1]
        Ioscillations_[idx1] = count_oscillations(np.array(df_temp[['I']])[:,0], min_prominence)[1]
        Coscillations_[idx1] = count_oscillations(np.array(df_temp[['C']])[:,0], min_prominence)[1]

    fig, ax = plt.subplots(2, 3, figsize=(14,10))
    if param_name == 'beta':
         str_xlabel = f'${{\\{param_name}}}$'
    elif param_name == 'sigmaD':
         str_xlabel = f'${{\sigma_D}}$'
    else:
         str_xlabel = f'${{\sigma_C}}$'

    ax[0,0].plot(param_search, Imax_)
    ax[0,0].set_title('Max. Infected')
    ax[0,0].grid()

    ax[1,0].plot(param_search, Tmax_)
    ax[1,0].set_xlabel(str_xlabel)
    ax[1,0].set_title('Max. Infected Time')
    ax[1,0].grid()
    
    ax[0,1].plot(param_search, Ifinal_)
    ax[0,1].set_title('Final Infected')
    ax[0,1].grid()
    
    ax[1,1].plot(param_search, Cfinal_)
    ax[1,1].grid()
    ax[1,1].set_xlabel(str_xlabel)
    ax[1,1].set_title('Final Cooperators')

    ax[0,2].plot(param_search, Ioscillations_)
    ax[0,2].grid()
    ax[0,2].set_title('# Peaks of Infected')

    ax[1,2].plot(param_search, Coscillations_)
    ax[1,2].grid()
    ax[1,2].set_xlabel(str_xlabel)
    ax[1,2].set_title('# Peaks of Cooperators')

    if not os.path.isdir( os.path.join(plots_path, '1D', folder) ):
                os.makedirs(os.path.join(plots_path, '1D', folder))
    
    plt.savefig(os.path.join(plots_path, '1D', folder,'plot_coupled_features_{}_exp.jpeg'.format(param_name)), dpi=450)
    plt.close()

def graph_IC_experimentation(initcond_search, param_search, param_name:str, folder:str):
    Imax_ = np.zeros((param_search.shape[0], initcond_search.shape[0]))
    Tmax_ = np.zeros((param_search.shape[0], initcond_search.shape[0]))
    Ifinal_ = np.zeros((param_search.shape[0], initcond_search.shape[0]))
    Cfinal_ = np.zeros((param_search.shape[0], initcond_search.shape[0]))
    Ioscillations_ = np.zeros((param_search.shape[0], initcond_search.shape[0]))
    Coscillations_ = np.zeros((param_search.shape[0], initcond_search.shape[0]))

    param_ticks1 = np.linspace(param_search[0], initcond_search[-1], 6)
    param_ticks2 = np.linspace(param_search[0], initcond_search[-1], 6)

    for idx1, p1 in enumerate(param_search):
        df_temp = df_parametric[[param_name]]

        for idx2, p2 in enumerate(initcond_search):
        
            if param_name == 'beta':
                str_file = 'ode_coupled_beta_{:0.2f}_IC_{:0.2f}'.format(p1, p2)
                
            elif param_name == 'sigmaD':
                str_file  = 'ode_coupled_sigmaD_{:0.2f}_IC_{:0.2f}'.format(p1, p2)

            elif param_name == 'sigmaC':    
                str_file  = 'ode_coupled_sigmaC_{:0.2f}_IC_{:0.2f}'.format(p1, p2)
                
            path_to_results = os.path.join(results_path, 'IC', folder,str_file+'.csv')
            df_temp = pd.read_csv(path_to_results, index_col=0)
            df_temp['I'] = df_temp['I_c']+df_temp['I_d']
            df_temp['C'] = df_temp['S_c']+df_temp['I_c']

            Imax_[idx1, idx2] = np.max(df_temp[['I']])
            Tmax_[idx1, idx2] = np.array(df_temp[['time']])[np.argmax(df_temp[['I']]),0]
            Ifinal_[idx1, idx2] = np.array(df_temp[['I']])[-1,0]
            Cfinal_[idx1, idx2] = np.array(df_temp[['C']])[-1,0]

            Ioscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['I']])[:,0], 0.001)[1]
            Coscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['C']])[:,0], 0.001)[1]
        
    fig, axes = plt.subplots(2, 3, figsize=(14,10))
    
    if param_name == 'beta':
         str_ylabel = f'${{\\{param_name}}}$'
    elif param_name == 'sigmaD':
         str_ylabel = f'${{\sigma_D}}$'
    else:
         str_ylabel = f'${{\sigma_C}}$'

    str_xlabel = r'$\%D0$'

    num_ticks = 5
    yticks = np.linspace(0, len(param_search) - 1, num_ticks, dtype=np.int8)
    xticks = np.linspace(0, len(initcond_search) -1, num_ticks, dtype=np.int8)
    # the content of labels of these yticks
    yticklabs = [param_search[int(idx)] for idx in yticks]
    xticklabs = [initcond_search[int(idx)] for idx in xticks]
    #heatmaps
    #im1 = axes[0,0](Imax_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Imax_, ax=axes[0,0], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[0,0].set_title('Max. Infected')
    axes[0,0].set_ylabel(str_ylabel)
    axes[0,0].set_yticks(yticks, labels=yticklabs)
    axes[0,0].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im1, fraction=0.045, pad=0.05, ax=axes[0,0])

    #im2 = axes[1,0].imshow(Tmax_, cmap='cool', vmin=0, vmax=t_max)
    sns.heatmap(Tmax_, ax=axes[1,0], cmap='spring', vmin=0, vmax=t_max, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[1,0].set_title('Time of Max. Infected')
    axes[1,0].set_ylabel(str_ylabel)
    axes[1,0].set_xlabel(str_xlabel)
    axes[1,0].set_yticks(yticks, labels=yticklabs)
    axes[1,0].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im2, fraction=0.045, pad=0.05, ax=axes[1,0])

    #im3 = axes[0,1].imshow(Ifinal_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Ifinal_, ax=axes[0,1], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[0,1].set_title('Final Infected')
    axes[0,1].set_yticks(yticks, labels=yticklabs)
    axes[0,1].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im3, fraction=0.045, pad=0.05, ax=axes[0,1])

    #im4 = axes[1,1].imshow(Cfinal_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Cfinal_, ax=axes[1,1], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[1,1].set_title('Final Cooperators')
    axes[1,1].set_xlabel(str_xlabel)
    axes[1,1].set_yticks(yticks, labels=yticklabs)
    axes[1,1].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im4, fraction=0.045, pad=0.05, ax=axes[1,1])

    #im5 = axes[0,2].imshow(Ioscillations_, cmap='summer', vmin=0, vmax=10)
    sns.heatmap(Ioscillations_, ax=axes[0,2], cmap='summer', vmin=0, vmax=10, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[0,2].set_title('# Peaks of Infected')
    axes[0,2].set_yticks(yticks, labels=yticklabs)
    axes[0,2].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im5, fraction=0.045, pad=0.05, ax=axes[0,2])

    #im6 = axes[1,2].imshow(Coscillations_, cmap='summer', vmin=0, vmax=10)
    sns.heatmap(Coscillations_, ax=axes[1,2], cmap='summer', vmin=0, vmax=10, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[1,2].set_title('# Peaks of Cooperators')
    axes[1,2].set_xlabel(str_xlabel)
    axes[1,2].set_yticks(yticks, labels=yticklabs)
    axes[1,2].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im6, fraction=0.045, pad=0.05, ax=axes[1,2])

    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, 'IC', folder) ):
                os.makedirs(os.path.join(plots_path, 'IC', folder))
    
    plt.savefig(os.path.join(plots_path, 'IC', folder,'heatmap_coupled_features_{}_exp.jpeg'.format(param_name)), dpi=400)
    plt.close() 

def graph_2D_experimentation(param_search1, param_search2, param_name1: str, param_name2: str, folder:str):
    Imax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Tmax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ifinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Cfinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ioscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Coscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))

    param_ticks1 = np.linspace(param_search1[0], param_search1[-1], 6)
    param_ticks2 = np.linspace(param_search2[0], param_search2[-1], 6)

    for idx1, p1 in enumerate(param_search1):
        df_temp = df_parametric[[param_name1]]
        param_search1 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))

        for idx2, p2 in enumerate(param_search2):
        
            if param_name1 == 'beta':
                if param_name2 == 'sigmaD':
                    str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_0.50'.format(p1, p2)
                else:
                    str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(p1, p2)
            elif param_name1 == 'sigmaD':
                if param_name2 == 'beta':
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_0.50'.format(p2, p1)
                else:
                    str_file  = 'ode_coupled_beta_0.50_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(p1, p2)
            elif param_name1 == 'sigmaC':
                if param_name2 == 'beta':
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(p2, p1)
                else:
                    str_file  = 'ode_coupled_beta_0.50_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(p2, p1)

            path_to_results = os.path.join(results_path, '2D', folder,str_file+'.csv')
            df_temp = pd.read_csv(path_to_results, index_col=0)
            df_temp['I'] = df_temp['I_c']+df_temp['I_d']
            df_temp['C'] = df_temp['S_c']+df_temp['I_c']

            Imax_[idx1, idx2] = np.max(df_temp[['I']])
            Tmax_[idx1, idx2] = np.array(df_temp[['time']])[np.argmax(df_temp[['I']]),0]
            Ifinal_[idx1, idx2] = np.array(df_temp[['I']])[-1,0]
            Cfinal_[idx1, idx2] = np.array(df_temp[['C']])[-1,0]

            Ioscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['I']])[:,0], 0.001)[1]
            Coscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['C']])[:,0], 0.001)[1]
        
    fig, axes = plt.subplots(2, 3, figsize=(14,10))
    
    if param_name1 == 'beta':
         str_ylabel = f'${{\\{param_name1}}}$'
    elif param_name1 == 'sigmaD':
         str_ylabel = f'${{\sigma_D}}$'
    else:
         str_ylabel = f'${{\sigma_C}}$'

    if param_name2 == 'beta':
         str_xlabel = f'${{\\{param_name2}}}$'
    elif param_name2 == 'sigmaD':
         str_xlabel = f'${{\sigma_D}}$'
    else:
         str_xlabel = f'${{\sigma_C}}$'

    num_ticks = 5
    yticks = np.linspace(0, len(param_search1) - 1, num_ticks, dtype=np.int8)
    xticks = np.linspace(0, len(param_search2) -1, num_ticks, dtype=np.int8)
    # the content of labels of these yticks
    yticklabs = [param_search1[int(idx)] for idx in yticks]
    xticklabs = [param_search2[int(idx)] for idx in xticks]
    #heatmaps
    #im1 = axes[0,0](Imax_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Imax_, ax=axes[0,0], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[0,0].set_title('Max. Infected')
    axes[0,0].set_ylabel(str_ylabel)
    axes[0,0].set_yticks(yticks, labels=yticklabs)
    axes[0,0].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im1, fraction=0.045, pad=0.05, ax=axes[0,0])

    #im2 = axes[1,0].imshow(Tmax_, cmap='cool', vmin=0, vmax=t_max)
    sns.heatmap(Tmax_, ax=axes[1,0], cmap='spring', vmin=0, vmax=t_max, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[1,0].set_title('Time of Max. Infected')
    axes[1,0].set_ylabel(str_ylabel)
    axes[1,0].set_xlabel(str_xlabel)
    axes[1,0].set_yticks(yticks, labels=yticklabs)
    axes[1,0].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im2, fraction=0.045, pad=0.05, ax=axes[1,0])

    #im3 = axes[0,1].imshow(Ifinal_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Ifinal_, ax=axes[0,1], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[0,1].set_title('Final Infected')
    axes[0,1].set_yticks(yticks, labels=yticklabs)
    axes[0,1].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im3, fraction=0.045, pad=0.05, ax=axes[0,1])

    #im4 = axes[1,1].imshow(Cfinal_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Cfinal_, ax=axes[1,1], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[1,1].set_title('Final Cooperators')
    axes[1,1].set_xlabel(str_xlabel)
    axes[1,1].set_yticks(yticks, labels=yticklabs)
    axes[1,1].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im4, fraction=0.045, pad=0.05, ax=axes[1,1])

    #im5 = axes[0,2].imshow(Ioscillations_, cmap='summer', vmin=0, vmax=10)
    sns.heatmap(Ioscillations_, ax=axes[0,2], cmap='summer', vmin=0, vmax=10, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[0,2].set_title('# Peaks of Infected')
    axes[0,2].set_yticks(yticks, labels=yticklabs)
    axes[0,2].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im5, fraction=0.045, pad=0.05, ax=axes[0,2])

    #im6 = axes[1,2].imshow(Coscillations_, cmap='summer', vmin=0, vmax=10)
    sns.heatmap(Coscillations_, ax=axes[1,2], cmap='summer', vmin=0, vmax=10, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    axes[1,2].set_title('# Peaks of Cooperators')
    axes[1,2].set_xlabel(str_xlabel)
    axes[1,2].set_yticks(yticks, labels=yticklabs)
    axes[1,2].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im6, fraction=0.045, pad=0.05, ax=axes[1,2])

    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, '2D', folder) ):
                os.makedirs(os.path.join(plots_path, '2D', folder))
    
    plt.savefig(os.path.join(plots_path, '2D', folder,'heatmap_coupled_features_{}_{}_exp.jpeg'.format(param_name1,param_name2)), dpi=400)
    plt.close()


sim1_path = os.path.join(main_path, results_path, 
                        '1D', 'dynamicSI_SCPA', 'ode_coupled_beta_0.25_sigmaD_0.50_sigmaC_0.50')           
sim2_path = os.path.join(main_path, results_path,
                        '1D', 'dynamicSI_SCPA', 'ode_coupled_beta_0.79_sigmaD_0.50_sigmaC_0.50')
sim3_path = os.path.join(main_path, results_path, 
                        '1D', 'dynamicSI_SCPA', 'ode_coupled_beta_0.50_sigmaD_0.50_sigmaC_0.25')
sim4_path = os.path.join(main_path, results_path, 
                        '1D', 'dynamicSI_SCPA', 'ode_coupled_beta_0.50_sigmaD_0.50_sigmaC_0.79')
sim5_path = os.path.join(main_path, results_path,
                        '1D', 'dynamicSI_SCPA', 'ode_coupled_beta_0.50_sigmaD_0.25_sigmaC_0.50')
sim6_path = os.path.join(main_path, results_path,
                        '1D', 'dynamicSI_SCPA', 'ode_coupled_beta_0.50_sigmaD_0.79_sigmaC_0.50')

graph_simulationFeatures(sim1_path, 'feats_ode_coupled_beta_0.25_sigmaD_0.50_sigmaC_0.50', 'dynamicSI_SCPA')
graph_simulationFeatures(sim2_path, 'feats_ode_coupled_beta_0.79_sigmaD_0.50_sigmaC_0.50', 'dynamicSI_SCPA')
graph_simulationFeatures(sim3_path, 'feats_ode_coupled_beta_0.50_sigmaD_0.50_sigmaC_0.25', 'dynamicSI_SCPA')
graph_simulationFeatures(sim4_path, 'feats_ode_coupled_beta_0.50_sigmaD_0.50_sigmaC_0.79', 'dynamicSI_SCPA')
graph_simulationFeatures(sim5_path, 'feats_ode_coupled_beta_0.50_sigmaD_0.25_sigmaC_0.50', 'dynamicSI_SCPA')
graph_simulationFeatures(sim6_path, 'feats_ode_coupled_beta_0.50_sigmaD_0.79_sigmaC_0.50', 'dynamicSI_SCPA')


beta_search = np.linspace(beta_.loc['min'][0], beta_.loc['max'][0], int(beta_.loc['num'][0]))
sigmaD_search = np.linspace(sigmaD_.loc['min'][0], sigmaD_.loc['max'][0], int(sigmaD_.loc['num'][0]))
sigmaC_search = np.linspace(sigmaC_.loc['min'][0], sigmaC_.loc['max'][0], int(sigmaC_.loc['num'][0]))
list_paramsSearch = [beta_search, sigmaD_search, sigmaC_search]
IC_search = np.linspace(d_fract_.loc['min'][0], d_fract_.loc['max'][0], int(d_fract_.loc['num'][0]))


for key_case, val_case in dict_scenarios.items():
    for idx, param in enumerate(list_params):
        graph_1D_experimentation(list_paramsSearch[idx], param, key_case)
        graph_IC_experimentation(IC_search, list_paramsSearch[idx], param, key_case)

for key_case, val_case in dict_scenarios.items():
    for idx1, param_name1 in enumerate(list_params):
            df_temp = df_parametric[[param_name1]]
            param_search1 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))

            list_temp = list_params.copy()
            list_temp.remove(param_name1)
            for idx2, param_name2 in enumerate(list_temp):
                df_temp = df_parametric[[param_name2]]
                param_search2 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))
                graph_2D_experimentation(param_search1, param_search2, param_name1, param_name2, key_case)
            

