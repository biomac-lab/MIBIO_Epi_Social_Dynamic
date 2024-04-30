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
#parametric_df_dir = config_data.loc['parametric_coupled_df_dir'][1]
parametric_df_dir = config_data.loc['parametric_coupled_paper_dir'][1]
df_parametric = pd.read_csv(os.path.join(main_path, parametric_df_dir), index_col=0)

##
t_max = 500
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

str_beta = f'${{\\{'beta'}}}$'
str_Imax = f'${{I_\max}}$'
str_Cfinal = f'${{C_\infty}}$'
str_Ifinal = f'${{I_\infty}}$'
str_Ioscillations = r'$I_{osc}$'
str_Delta = r'$\Delta$'

color_region1 = '#FFE66D'
color_region2  = '#ff6666'
color_region3 = '#2EC4B6'
color_region4 = '#5D2E8C'

colormap_regions = np.array([color_region1, color_region2, color_region3, color_region4])

#Selfcare - Public Awareness - Social Pressure - Dynamic I - Dynamic S
dict_scenarios = {'Null':(False, False, False, True, True),
                  'SP':(False, False, True, True, True),
                  'SC':(True, False, False, True, True),
                  'PA':(False, True, False, True, True),
                  'SP+SC':(True, False, True, True, True),
                  'SP+PA':(False, True, True, True, True),
                  'PA+SC':(True, True, False, True, True),
                  'SC+SP+PA':(True, True, True, True, True)}

dict_scenarios_simple = {'Null':(False, False, False, True, True),
                  'SC':(True, False, False, True, True),
                  'PA':(False, True, False, True, True)}

beta_search = np.linspace(beta_.iloc[0,0], beta_.iloc[1,0], int(beta_.iloc[2,0]))
sigmaC_search = np.linspace(sigmaC_.iloc[0,0], sigmaC_.iloc[1,0], int(sigmaC_.iloc[2,0]))
sigmaD_search = np.linspace(sigmaD_.iloc[0,0], sigmaD_.iloc[1,0], int(sigmaD_.iloc[2,0]))
IC_search = np.linspace(d_fract_.loc['min'][0], d_fract_.loc['max'][0], int(d_fract_.loc['num'][0]))

def count_oscillations(sim, min_prominence):
    idx_peaks, dict_peaks = find_peaks(sim, prominence=min_prominence)
    return idx_peaks, len(idx_peaks)

def find_stability_time(sim, epsilon):
    last_val = sim[-1,0]
    stable_sim = sim[np.abs(sim-last_val) <= epsilon]
    return len(stable_sim)  

def graph_paper_2D_experimentation(betas, param_search1, param_search2, param_name1: str, param_name2: str, folder:str):
    Imax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ifinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ioscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))

    fig, axes = plt.subplots(3, np.size(betas), figsize=(14,10), sharex=True)

    for b_idx, beta_temp in enumerate(betas):
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
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(beta_temp,p1, p2)
                elif param_name1 == 'sigmaC':
                    if param_name2 == 'beta':
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(beta_temp, p2, p1)
                    else:
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(beta_temp,p2, p1)

                path_to_results = os.path.join(results_path, '2D', folder,str_file+'.csv')
                df_temp = pd.read_csv(path_to_results, index_col=0)
                df_temp['I'] = df_temp['I_c']+df_temp['I_d']
                df_temp['C'] = df_temp['S_c']+df_temp['I_c']

                Imax_[idx1, idx2] = np.max(df_temp[['I']])
                Ifinal_[idx1, idx2] = np.array(df_temp[['I']])[-1,0]
                Ioscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['I']])[:,0], 0.001)[1]
                
        
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

        colorbar = False
        if b_idx == 2:
            colorbar = True

        #heatmaps
        #MAX INFECTED
        sns.heatmap(Imax_, ax=axes[0,b_idx], cmap='Reds', vmin=0, vmax=1, cbar=colorbar,
                    xticklabels=xticklabs, yticklabels=yticklabs)
        gradient_Imax = np.gradient(Imax_)
        meanGradient_Imax_y = np.round(np.mean(np.abs(gradient_Imax[0])),3)
        meanGradient_Imax_x = np.round(np.mean(np.abs(gradient_Imax[1])),3)
        axes[0,b_idx].set_title(f'${{\\{'beta'}}}$ = {beta_temp}')
        if b_idx == 0:
            axes[0,b_idx].set_ylabel(f'{str_Imax} \n {str_ylabel}')
        axes[0,b_idx].set_yticks(yticks, labels=yticklabs)
        axes[0,b_idx].set_xticks(xticks, labels=xticklabs)

        #FINAL INFECTED
        sns.heatmap(Ifinal_, ax=axes[1,b_idx], cmap='PuRd', vmin=0, vmax=1, cbar=colorbar,
                    xticklabels=xticklabs, yticklabels=yticklabs)
        gradient_Ifinal = np.gradient(Ifinal_)
        meanGradient_Ifinal_y = np.round(np.mean(np.abs(gradient_Ifinal[0])),3)
        meanGradient_Ifinal_x = np.round(np.mean(np.abs(gradient_Ifinal[1])),3)
        if b_idx == 0:
            axes[1,b_idx].set_ylabel(f'{str_Ifinal} \n {str_ylabel}')
        axes[1,b_idx].set_yticks(yticks, labels=yticklabs)
        axes[1,b_idx].set_xticks(xticks, labels=xticklabs)

        #PEAKS OF INFECTED
        
        sns.heatmap(Ioscillations_, ax=axes[2,b_idx], cmap='PuBuGn', vmin=0, vmax=10, cbar=colorbar,
                    xticklabels=xticklabs, yticklabels=yticklabs)
        gradient_Iosci = np.gradient(Ioscillations_)
        meanGradient_Iosci_y = np.round(np.mean(np.abs(gradient_Iosci[0])),3)
        meanGradient_Iosci_x = np.round(np.mean(np.abs(gradient_Iosci[1])),3)
        if b_idx == 0:
            axes[2,b_idx].set_ylabel(f'{str_Ioscillations} \n {str_ylabel}')
        axes[2,b_idx].set_xlabel(str_xlabel)
        axes[2,b_idx].set_yticks(yticks, labels=yticklabs)
        axes[2,b_idx].set_xticks(xticks, labels=xticklabs)


    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, 'paper', folder) ):
                os.makedirs(os.path.join(plots_path, 'paper', folder))
    
    plt.savefig(os.path.join(plots_path, 'paper', folder, f'fig1_htmp_inf (beta={betas[0]}->{betas[-1]}).jpeg'), dpi=400)
    plt.close()

def graph_correlation(prob_infect_search, param_search1, param_search2, param_name1: str, param_name2: str, folder:str): 
    
    fig, axes = plt.subplots(3, np.size(prob_infect_search), figsize=(14,10), sharex=True)

    for j, beta_temp in enumerate(prob_infect_search):
        categorical_region = list()
        Imax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
        Ifinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
        Ioscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
        Cfinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
        Imax_lst = list()
        Ifinal_lst = list()
        Ioscillations_lst = list()  
        Cfinal_lst = list()

        for idx1, p1 in enumerate(param_search1):
            df_temp = df_parametric[[param_name1]]
            param_search1 = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))

            for idx2, p2 in enumerate(param_search2):
            
                if param_name1 == 'beta':
                    if param_name2 == 'sigmaD':
                        sigmaD_temp = p2
                        sigmaC_temp = 0.5
                        str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_0.50'.format(p1, p2)
                    else:
                        sigmaD_temp = 0.5
                        sigmaC_temp = p2
                        str_file = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(p1, p2)
                elif param_name1 == 'sigmaD':
                    if param_name2 == 'beta':
                        sigmaD_temp = p1
                        sigmaC_temp = 0.5
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_0.50'.format(p2, p1)
                    else:
                        sigmaD_temp = p1
                        sigmaC_temp = p2
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(beta_temp,p1, p2)
                elif param_name1 == 'sigmaC':
                    if param_name2 == 'beta':
                        sigmaD_temp = 0.5
                        sigmaC_temp = p1
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_0.50_sigmaC_{:0.2f}'.format(beta_temp, p2, p1)
                    else:  
                        sigmaD_temp = p2   
                        sigmaC_temp = p1
                        str_file  = 'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}'.format(beta_temp,p2, p1)
                
                if sigmaC_temp < 0.5 and sigmaD_temp < 0.5: 
                    categorical_region.append(0)
                elif sigmaC_temp >= 0.5 and sigmaD_temp < 0.5:
                    categorical_region.append(1)  
                elif sigmaC_temp < 0.5 and sigmaD_temp >= 0.5:
                    categorical_region.append(2) 
                elif sigmaC_temp >= 0.5 and sigmaD_temp >= 0.5:
                    categorical_region.append(3)

                path_to_results = os.path.join(results_path, '2D', folder,str_file+'.csv')
                df_temp = pd.read_csv(path_to_results, index_col=0)
                df_temp['I'] = df_temp['I_c']+df_temp['I_d']
                df_temp['C'] = df_temp['S_c']+df_temp['I_c']

                Imax_[idx1, idx2] = np.max(df_temp[['I']])
                Ifinal_[idx1, idx2] = np.array(df_temp[['I']])[-1,0]
                Ioscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['I']])[:,0], 0.001)[1]
                Cfinal_[idx1, idx2] = np.array(df_temp[['C']])[-1,0]

                Imax_lst.append(np.max(df_temp[['I']]))
                Ifinal_lst.append(np.array(df_temp[['I']])[-1,0])
                Ioscillations_lst.append(count_oscillations(np.array(df_temp[['I']])[:,0], 0.001)[1])
                Cfinal_lst.append(np.array(df_temp[['C']])[-1,0])

        categorical_region = np.array(categorical_region)

        pearson_Imax = np.corrcoef(Cfinal_lst, Imax_lst)
        pearson_Ifinal = np.corrcoef(Cfinal_lst, Ifinal_lst)
        
        z_Imax = np.polyfit(Cfinal_lst, Imax_lst, 1)
        p_Imax = np.poly1d(z_Imax)

        z_Ifinal = np.polyfit(Cfinal_lst, Ifinal_lst, 1)
        p_Ifinal = np.poly1d(z_Ifinal)

        axes[0, j].scatter(Cfinal_lst, Imax_lst, color=colormap_regions[categorical_region], marker=',', alpha=0.25)
        axes[0, j].set_ylabel(str_Imax)
        axes[0, j].set_xlim([0,1])
        axes[0, j].grid(True)
        axes[0, j].set_title(f'{str_beta} = {beta_temp}')

        axes[1, j].scatter(Cfinal_lst, Ifinal_lst, color=colormap_regions[categorical_region], marker=',', alpha=0.25)
        axes[1, j].set_ylabel(str_Ifinal)
        axes[1, j].set_xlim([0,1])
        axes[1, j].grid(True)

        axes[2, j].scatter(Cfinal_lst, Ioscillations_lst, color=colormap_regions[categorical_region], marker=',', alpha=0.25)
        axes[2, j].set_xlabel(str_Cfinal)
        axes[2, j].set_xlim([0,1])
        axes[2, j].set_ylabel(str_Ioscillations)
        axes[2, j].grid(True)
                
    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, 'paper', folder) ):
                os.makedirs(os.path.join(plots_path, 'paper', folder))
    
    plt.savefig(os.path.join(plots_path, 'paper', folder, f'fig2_corr_infcoop (beta={prob_infect_search[0]}->{prob_infect_search[-1]}).jpeg'), dpi=400)
    
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

    MD_Imax = np.array(Imax_2 - Imax_1).flatten()
    MD_Tmax = np.array(Tmax_2 - Tmax_1).flatten()
    MD_Ifinal = np.array(Ifinal_2  - Ifinal_1).flatten()
    MD_Cfinal = np.array(Cfinal_2 - Cfinal_1).flatten()
    MD_Iosc = np.array(Ioscillations_2 - Ioscillations_1).flatten()
    MD_Cosc = np.array(Coscillations_2 - Coscillations_1).flatten()

    df_temp = pd.DataFrame(columns=['Model', 'diff_Imax', 'diff_Tmax', 'diff_Ifinal', 'diff_Cfinal', 'diff_Iosc', 'diff_Cosc'])
    
    df_temp['diff_Imax'] = MD_Imax
    df_temp['diff_Tmax'] = MD_Tmax
    df_temp['diff_Ifinal'] = MD_Ifinal
    df_temp['diff_Cfinal'] = MD_Cfinal
    df_temp['diff_Iosc'] = MD_Iosc
    df_temp['diff_Cosc'] = MD_Cosc


    df_temp['Model'] = [f'{folder2}']*len(df_temp)

    return df_temp

def graph_bars_diff(prob_infect_search):
    fig, axes = plt.subplots(3, np.size(prob_infect_search), figsize=(18,10))

    for b_idx, beta_temp in enumerate(prob_infect_search):
        path_to_results = os.path.join(results_path, 'Diff', 'Diff_NullAddition_beta_{:0.2f}.csv'.format(beta_temp))
        df_temp = pd.read_csv(path_to_results, index_col=0)

        bar_width = 0.25
        index = df_temp.index

        sns.histplot(df_temp, x='diff_Imax', hue='Model', fill=True, ax=axes[0,b_idx])
        axes[0,b_idx].set_title(f'{str_beta} = {beta_temp}')
        axes[0,b_idx].set_xlabel(f'{str_Delta} {str_Imax}')

        sns.histplot(df_temp, x='diff_Ifinal', hue='Model', fill=True, ax=axes[1,b_idx])
        axes[1,b_idx].set_xlabel(f'{str_Delta} {str_Ifinal}')
        
        sns.histplot(df_temp, x='diff_Iosc', hue='Model', fill=True, ax=axes[2,b_idx])
        axes[2,b_idx].set_xlabel(f'{str_Delta} {str_Ioscillations}')
        
        

    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, 'paper') ):
                os.makedirs(os.path.join(plots_path, 'paper'))
        
    plt.savefig(os.path.join(plots_path, 'paper', f'fig3_bchrt (beta={prob_infect_search[0]}->{prob_infect_search[-1]}).jpeg'), dpi=400)
    plt.close()

        

phenomena = ['PA','SC']
for beta_temp in tqdm(beta_search):
    diff_df = pd.DataFrame(columns=['Model', 'diff_Imax', 'diff_Tmax', 'diff_Ifinal', 'diff_Cfinal', 'diff_Iosc', 'diff_Cosc'])
    
    for pheno in phenomena:
        diff_temp = compareModels_2D_experimentation(beta_temp,'Null', pheno, sigmaC_search, sigmaD_search, 'sigmaC', 'sigmaD')
        diff_df = pd.concat([diff_df, diff_temp], axis=0)

        if not os.path.isdir(os.path.join(results_path, 'Diff')):
            os.makedirs(os.path.join(results_path, 'Diff'))

        diff_df.to_csv(os.path.join(results_path, 'Diff', 'Diff_NullAddition_beta_{:0.2f}.csv'.format(beta_temp)))

print('DONE COMPARISON OF MODELS')


for key_case, val_case in dict_scenarios.items():
    graph_paper_2D_experimentation(beta_search, sigmaD_search, sigmaC_search, 'sigmaD', 'sigmaC', key_case)
    graph_correlation(beta_search, sigmaD_search, sigmaC_search, 'sigmaD', 'sigmaC', key_case)

graph_bars_diff(beta_search)
    