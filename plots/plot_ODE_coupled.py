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
parematric_df_dir = config_data.loc['parametric_coupled_df_dir'][1]
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
#list_params = ['beta', 'sigmaD', 'sigmaC']
list_params = ['sigmaD', 'sigmaC']

#Selfcare - Public Awareness - Social Pressure - Dynamic I - Dynamic S
dict_scenarios = {'Null':(False, False, False, True, True),
                  'SP':(False, False, True, True, True),
                  'SC':(True, False, False, True, True),
                  'PA':(False, True, False, True, True),
                  'SP+SC':(True, False, True, True, True),
                  'SP+PA':(False, True, True, True, True),
                  'PA+SC':(True, True, False, True, True),
                  'SC+SP+PA':(True, True, True, True, True)}

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

def graph_1D_experimentation(prob_infect, param_search, param_name:str, folder:str):
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
            str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_0.50'.format(prob_infect,p)
        else:
            str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(prob_infect, p)
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

    fig.suptitle('Beta = {:0.2f}'.format(prob_infect))
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
    
    plt.savefig(os.path.join(plots_path, '1D', folder,'plot_coupled_features_beta_{:0.2f}_{}_exp.jpeg'.format(prob_infect,param_name)), dpi=450)
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

def graph_2D_experimentation(prob_infect, param_search1, param_search2, param_name1: str, param_name2: str, folder:str):
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
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(prob_infect,p1, p2)
            elif param_name1 == 'sigmaC':
                if param_name2 == 'beta':
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(prob_infect, p2, p1)
                else:
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(prob_infect,p2, p1)

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
    fig.suptitle('Beta = {:0.2f}'.format(prob_infect))
    
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
    gradient_Imax = np.gradient(Imax_)
    meanGradient_Imax_y = np.round(np.mean(np.abs(gradient_Imax[0])),3)
    meanGradient_Imax_x = np.round(np.mean(np.abs(gradient_Imax[1])),3)
    axes[0,0].set_title(f'Max. Infected \n {param_name1}={meanGradient_Imax_y} & {param_name2}={meanGradient_Imax_x}')
    axes[0,0].set_ylabel(str_ylabel)
    axes[0,0].set_yticks(yticks, labels=yticklabs)
    axes[0,0].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im1, fraction=0.045, pad=0.05, ax=axes[0,0])

    #im2 = axes[1,0].imshow(Tmax_, cmap='cool', vmin=0, vmax=t_max)
    sns.heatmap(Tmax_, ax=axes[1,0], cmap='spring', vmin=0, vmax=t_max, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    gradient_Tmax = np.gradient(Tmax_)
    meanGradient_Tmax_y = np.round(np.mean(np.abs(gradient_Tmax[0])),3)
    meanGradient_Tmax_x = np.round(np.mean(np.abs(gradient_Tmax[1])),3)
    axes[1,0].set_title(f'Time of Max. Infected \n {param_name1}={meanGradient_Tmax_y} & {param_name2}={meanGradient_Tmax_x}')
    axes[1,0].set_ylabel(str_ylabel)
    axes[1,0].set_xlabel(str_xlabel)
    axes[1,0].set_yticks(yticks, labels=yticklabs)
    axes[1,0].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im2, fraction=0.045, pad=0.05, ax=axes[1,0])

    #im3 = axes[0,1].imshow(Ifinal_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Ifinal_, ax=axes[0,1], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    gradient_Ifinal = np.gradient(Ifinal_)
    meanGradient_Ifinal_y = np.round(np.mean(np.abs(gradient_Ifinal[0])),3)
    meanGradient_Ifinal_x = np.round(np.mean(np.abs(gradient_Ifinal[1])),3)
    axes[0,1].set_title(f'Final Infected \n {param_name1}={meanGradient_Ifinal_y} & {param_name2}={ meanGradient_Ifinal_x}')
    axes[0,1].set_yticks(yticks, labels=yticklabs)
    axes[0,1].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im3, fraction=0.045, pad=0.05, ax=axes[0,1])

    #im4 = axes[1,1].imshow(Cfinal_, cmap='plasma', vmin=0, vmax=1)
    sns.heatmap(Cfinal_, ax=axes[1,1], cmap='plasma', vmin=0, vmax=1, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    gradient_Cfinal = np.gradient(Cfinal_)
    meanGradient_Cfinal_y = np.round(np.mean(np.abs(gradient_Cfinal[0])),3)
    meanGradient_Cfinal_x = np.round(np.mean(np.abs(gradient_Cfinal[1])),3)
    axes[1,1].set_title(f'Final Cooperators \n {param_name1}={meanGradient_Cfinal_y} & {param_name2}={meanGradient_Cfinal_x}')
    axes[1,1].set_xlabel(str_xlabel)
    axes[1,1].set_yticks(yticks, labels=yticklabs)
    axes[1,1].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im4, fraction=0.045, pad=0.05, ax=axes[1,1])

    #im5 = axes[0,2].imshow(Ioscillations_, cmap='summer', vmin=0, vmax=10)
    sns.heatmap(Ioscillations_, ax=axes[0,2], cmap='summer', vmin=0, vmax=10, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    gradient_Iosci = np.gradient(Ioscillations_)
    meanGradient_Iosci_y = np.round(np.mean(np.abs(gradient_Iosci[0])),3)
    meanGradient_Iosci_x = np.round(np.mean(np.abs(gradient_Iosci[1])),3)
    axes[0,2].set_title(f'# Peaks of Infected \n {param_name1}={meanGradient_Iosci_y} & {param_name2}={meanGradient_Iosci_x}')
    axes[0,2].set_yticks(yticks, labels=yticklabs)
    axes[0,2].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im5, fraction=0.045, pad=0.05, ax=axes[0,2])

    #im6 = axes[1,2].imshow(Coscillations_, cmap='summer', vmin=0, vmax=10)
    sns.heatmap(Coscillations_, ax=axes[1,2], cmap='summer', vmin=0, vmax=10, cbar=True,
                xticklabels=xticklabs, yticklabels=yticklabs)
    gradient_Cosci = np.gradient(Coscillations_)
    meanGradient_Cosci_y = np.round(np.mean(np.abs(gradient_Cosci[0])),3)
    meanGradient_Cosci_x = np.round(np.mean(np.abs(gradient_Cosci[1])),3)
    axes[1,2].set_title(f'# Peaks of Cooperators \n {param_name1}={meanGradient_Cosci_y} & {param_name2}={meanGradient_Cosci_x}')
    axes[1,2].set_xlabel(str_xlabel)
    axes[1,2].set_yticks(yticks, labels=yticklabs)
    axes[1,2].set_xticks(xticks, labels=xticklabs)
    #plt.colorbar(im6, fraction=0.045, pad=0.05, ax=axes[1,2])

    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, '2D', folder) ):
                os.makedirs(os.path.join(plots_path, '2D', folder))
    
    plt.savefig(os.path.join(plots_path, '2D', folder,'heatmap_coupled_features_beta_{:0.2f}_{}_{}_exp.jpeg'.format(prob_infect,param_name1,param_name2)), dpi=400)
    plt.close()

def compareModels_2D_experimentation(prob_infect, folder1:str, folder2:str, param_search1, param_search2, param_name1: str, param_name2: str):
    Imax_1 = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Tmax_1 = np.copy(Imax_1)
    Ifinal_1 = np.copy(Imax_1)
    Cfinal_1 = np.copy(Imax_1)
    Ioscillations_1 = np.copy(Imax_1)
    Coscillations_1 = np.copy(Imax_1)

    Imax_2 = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Tmax_2 = np.copy(Imax_2)
    Ifinal_2 = np.copy(Imax_2)
    Cfinal_2 = np.copy(Imax_2)
    Ioscillations_2 = np.copy(Imax_2)
    Coscillations_2 = np.copy(Imax_2)
    


    for idx1, p1 in enumerate(param_search1):
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
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(prob_infect,p1, p2)
            elif param_name1 == 'sigmaC':
                if param_name2 == 'beta':
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(p2, p1)
                else:
                    str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(prob_infect,p2, p1)

            path_to_results1 = os.path.join(results_path, '2D', folder1,str_file+'.csv')
            df_temp1 = pd.read_csv(path_to_results1, index_col=0)
            df_temp1['I'] = df_temp1['I_c']+df_temp1['I_d']
            df_temp1['C'] = df_temp1['S_c']+df_temp1['I_c']

            Imax_1[idx1, idx2] = np.max(df_temp1[['I']])
            Tmax_1[idx1, idx2] = np.array(df_temp1[['time']])[np.argmax(df_temp1[['I']]),0]
            Ifinal_1[idx1, idx2] = np.array(df_temp1[['I']])[-1,0]
            Cfinal_1[idx1, idx2] = np.array(df_temp1[['C']])[-1,0]

            Ioscillations_1[idx1, idx2] = count_oscillations(np.array(df_temp1[['I']])[:,0], 0.001)[1]
            Coscillations_1[idx1, idx2] = count_oscillations(np.array(df_temp1[['C']])[:,0], 0.001)[1]

            path_to_results2 = os.path.join(results_path, '2D', folder2,str_file+'.csv')
            df_temp2 = pd.read_csv(path_to_results2, index_col=0)
            df_temp2['I'] = df_temp2['I_c'] + df_temp2['I_d']
            df_temp2['C'] = df_temp2['S_c'] + df_temp2['I_c']

            Imax_2[idx1, idx2] = np.max(df_temp2[['I']])
            Tmax_2[idx1, idx2] = np.array(df_temp2[['time']])[np.argmax(df_temp2[['I']]),0]
            Ifinal_2[idx1, idx2] = np.array(df_temp2[['I']])[-1,0]
            Cfinal_2[idx1, idx2] = np.array(df_temp2[['C']])[-1,0]

            Ioscillations_2[idx1, idx2] = count_oscillations(np.array(df_temp2[['I']])[:,0], 0.001)[1]
            Coscillations_2[idx1, idx2] = count_oscillations(np.array(df_temp2[['C']])[:,0], 0.001)[1]
    
    Imax_1_norm = Imax_1/np.sum(Imax_1)
    Tmax_1_norm = Tmax_1/np.sum(Tmax_1)
    Ifinal_1_norm = Ifinal_1/np.sum(Ifinal_1)
    Cfinal_1_norm = Cfinal_1/np.sum(Cfinal_1)
    Iosci_1_norm = Ioscillations_1/np.sum(Ioscillations_1)
    Cosci_1_norm = Coscillations_1/np.sum(Coscillations_1)

    Imax_2_norm = Imax_2/np.sum(Imax_2)
    Tmax_2_norm = Tmax_2/np.sum(Tmax_2)
    Ifinal_2_norm = Ifinal_2/np.sum(Ifinal_2)
    Cfinal_2_norm = Cfinal_2/np.sum(Cfinal_2)
    Iosci_2_norm = Ioscillations_2/np.sum(Ioscillations_2)
    Cosci_2_norm = Coscillations_2/np.sum(Coscillations_2)

    MAD_Imax = np.mean(np.abs((Imax_2 - Imax_1)))
    MAD_Tmax = np.mean(np.abs((Tmax_2 - Tmax_1)))
    MAD_Ifinal = np.mean(np.abs((Ifinal_2  - Ifinal_1)))
    MAD_Cfinal = np.mean(np.abs((Cfinal_2 - Cfinal_1)))
    MAD_Iosc = np.mean(np.abs((Ioscillations_2 - Ioscillations_1)))
    MAD_Cosc = np.mean(np.abs((Coscillations_2 - Coscillations_1)))

    ratio_Imax = np.mean((Imax_2 / Imax_1))
    ratio_Tmax = np.mean((Tmax_2 / Tmax_1))
    ratio_Ifinal = np.mean((Ifinal_2  / Ifinal_1))
    ratio_Cfinal = np.mean((Cfinal_2 / Cfinal_1))
    ratio_Iosc = np.mean((Ioscillations_2 / Ioscillations_1))
    ratio_Cosc = np.mean((Coscillations_2 / Coscillations_1))

    DKL_Imax = np.sum(Imax_2_norm*np.log(Imax_2_norm/Imax_1_norm))
    DKL_Tmax = np.sum(Tmax_2_norm*np.log(Tmax_2_norm/Tmax_1_norm))
    DKL_Ifinal = np.sum(Ifinal_2_norm*np.log(Ifinal_2_norm/Ifinal_1_norm))
    DKL_Cfinal = np.sum(Cfinal_2_norm*np.log(Cfinal_2_norm/Cfinal_1_norm))
    DKL_Iosc = np.sum(Iosci_2_norm*np.log(Iosci_2_norm/Iosci_1_norm))
    DKL_Cosc = np.sum(Cosci_2_norm*np.log(Cosci_2_norm/Cosci_1_norm))

    MSE_Imax = np.mean(np.sqrt((Imax_2 - Imax_1)**2))
    MSE_Tmax = np.mean(np.sqrt((Tmax_2 - Tmax_1)**2))
    MSE_Ifinal = np.mean(np.sqrt((Ifinal_2  - Ifinal_1)**2))
    MSE_Cfinal = np.mean(np.sqrt((Cfinal_2 - Cfinal_1)**2))
    MSE_Iosc = np.mean(np.sqrt((Ioscillations_2 - Ioscillations_1)**2))
    MSE_Cosc = np.mean(np.sqrt((Coscillations_2 - Coscillations_1)**2))

    MSE_array = [folder2, MSE_Imax, MSE_Tmax, MSE_Ifinal, MSE_Cfinal, MSE_Iosc, MSE_Cosc]
    MAD_array = [folder2, MAD_Imax, MAD_Tmax, MAD_Ifinal, MAD_Cfinal, MAD_Iosc, MAD_Cosc]
    ratio_array = [folder2, ratio_Imax, ratio_Tmax, ratio_Ifinal, ratio_Cfinal, ratio_Iosc, ratio_Cosc]
    DKL_array = [folder2, DKL_Imax, DKL_Tmax, DKL_Ifinal, DKL_Cfinal, DKL_Iosc, DKL_Cosc]

    return MSE_array, MAD_array, ratio_array, DKL_array


beta_search = np.linspace(beta_.iloc[0,0], beta_.iloc[1,0], int(beta_.iloc[2,0]))
sigmaC_search = np.linspace(sigmaC_.iloc[0,0], sigmaC_.iloc[1,0], int(sigmaC_.iloc[2,0]))
sigmaD_search = np.linspace(sigmaD_.iloc[0,0], sigmaD_.iloc[1,0], int(sigmaD_.iloc[2,0]))
IC_search = np.linspace(d_fract_.loc['min'][0], d_fract_.loc['max'][0], int(d_fract_.loc['num'][0]))

#list_paramsSearch = [beta_search, sigmaD_search, sigmaC_search]
list_paramsSearch = [sigmaD_search, sigmaC_search]

#dict_paramSearch = {'beta': beta_search, 'sigmaD': sigmaD_search, 'sigmaC': sigmaC_search}
dict_paramSearch = {'sigmaD': sigmaD_search, 'sigmaC': sigmaC_search}

'''
for beta_temp in tqdm(beta_search):
    for key_case, val_case in dict_scenarios.items():
        for idx, param in enumerate(list_params):
            graph_1D_experimentation(beta_temp, list_paramsSearch[idx], param, key_case)
            graph_IC_experimentation(IC_search, list_paramsSearch[idx], param, key_case)
'''
for beta_temp in tqdm(beta_search):
    for key_case, val_case in dict_scenarios.items():
        graph_2D_experimentation(beta_temp, sigmaC_search, sigmaD_search, 'sigmaC', 'sigmaD', key_case)


phenomena = ['SP','PA','SC','SP+PA','SP+SC','PA+SC','SC+SP+PA']
for beta_temp in tqdm(beta_search):
    dfMSE_temp_null = pd.DataFrame(columns=['Model', 'Imax', 'Tmax', 'Ifinal', 'Cfinal', 'Iosc', 'Cosc'])
    dfMAD_temp_null = pd.DataFrame(columns=['Model', 'Imax', 'Tmax', 'Ifinal', 'Cfinal', 'Iosc', 'Cosc'])  
    dfRatio_temp_null = pd.DataFrame(columns=['Model', 'Imax', 'Tmax', 'Ifinal', 'Cfinal', 'Iosc', 'Cosc'])
    dfDKL_temp_null = pd.DataFrame(columns=['Model', 'Imax', 'Tmax', 'Ifinal', 'Cfinal', 'Iosc', 'Cosc']) 

    for pheno in phenomena:
        array_null = compareModels_2D_experimentation(beta_temp,'Null', pheno, sigmaC_search, sigmaD_search, 'sigmaC', 'sigmaD')
        dfMSE_temp_null.loc[len(dfMSE_temp_null)] = array_null[0]
        dfMAD_temp_null.loc[len(dfMAD_temp_null)] = array_null[1]
        dfRatio_temp_null.loc[len(dfRatio_temp_null)] = array_null[2]
        dfDKL_temp_null.loc[len(dfDKL_temp_null)] = array_null[3]

        if not os.path.isdir(os.path.join(results_path, 'Diff')):
            os.makedirs(os.path.join(results_path, 'Diff'))

        dfMSE_temp_null.to_csv(os.path.join(results_path, 'Diff', 'MSE_NullAddition_beta_{:0.2f}_{}_{}.csv'.format(beta_temp,'sigmaC', 'sigmaD')))
        dfMAD_temp_null.to_csv(os.path.join(results_path, 'Diff', 'MAD_NullAddition_beta_{:0.2f}_{}_{}.csv'.format(beta_temp,'sigmaC', 'sigmaD')))
        dfRatio_temp_null.to_csv(os.path.join(results_path, 'Diff', 'Ratio_NullAddition_beta_{:0.2f}_{}_{}.csv'.format(beta_temp,'sigmaC', 'sigmaD')))
        dfDKL_temp_null.to_csv(os.path.join(results_path, 'Diff', 'DKL_NullAddition_beta_{:0.2f}_{}_{}.csv'.format(beta_temp,'sigmaC', 'sigmaD')))

print('DONE COMPARISON OF MODELS')