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

beta_search = np.linspace(beta_.iloc[0,0], beta_.iloc[1,0], int(beta_.iloc[2,0]))
#beta_search = np.array([0.64, 1.23, 2.41])
sigmaC_search = np.linspace(sigmaC_.iloc[0,0], sigmaC_.iloc[1,0], int(sigmaC_.iloc[2,0]))
sigmaD_search = np.linspace(sigmaD_.iloc[0,0], sigmaD_.iloc[1,0], int(sigmaD_.iloc[2,0]))
IC_search = np.linspace(d_fract_.loc['min'][0], d_fract_.loc['max'][0], int(d_fract_.loc['num'][0]))

#list_paramsSearch = [beta_search, sigmaD_search, sigmaC_search]
list_paramsSearch = [sigmaD_search, sigmaC_search]

#dict_paramSearch = {'beta': beta_search, 'sigmaD': sigmaD_search, 'sigmaC': sigmaC_search}
dict_paramSearch = {'sigmaD': sigmaD_search, 'sigmaC': sigmaC_search}

def count_oscillations(sim, min_prominence):
    idx_peaks, dict_peaks = find_peaks(sim, prominence=min_prominence)
    return idx_peaks, len(idx_peaks)

def graph_correlation(prob_infect_search, param_search1, param_search2, param_name1: str, param_name2: str, folder:str):
    Imax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    #Tmax_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ifinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Cfinal_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    Ioscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    #Coscillations_ = np.zeros((param_search1.shape[0], param_search2.shape[0]))
    
    fig, axes = plt.subplots(3, np.size(prob_infect_search), figsize=(14,10))
    fig.suptitle('Dynamics = {}'.format(folder))

    for j, beta_temp in enumerate(prob_infect_search):
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
                #Tmax_[idx1, idx2] = np.array(df_temp[['time']])[np.argmax(df_temp[['I']]),0]
                Ifinal_[idx1, idx2] = np.array(df_temp[['I']])[-1,0]
                Cfinal_[idx1, idx2] = np.array(df_temp[['C']])[-1,0]

                Ioscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['I']])[:,0], 0.001)[1]
                #Coscillations_[idx1, idx2] = count_oscillations(np.array(df_temp[['C']])[:,0], 0.001)[1]

                Imax_array = Imax_.flatten()
                Ifinal_array = Ifinal_.flatten()
                Cfinal_array = Cfinal_.flatten()
                Ioscillations_array = Ioscillations_.flatten()

                pearson_Imax = np.corrcoef(Cfinal_array, Imax_array)
                pearson_Ifinal = np.corrcoef(Cfinal_array, Ifinal_array)
                pearson_Iosc = np.corrcoef(Cfinal_array, Ioscillations_array)
                
                z_Imax = np.polyfit(Cfinal_array, Imax_array, 1)
                p_Imax = np.poly1d(z_Imax)

                z_Ifinal = np.polyfit(Cfinal_array, Ifinal_array, 1)
                p_Ifinal = np.poly1d(z_Ifinal)

                z_Iosc = np.polyfit(Cfinal_array, Ioscillations_array, 1)
                p_Iosc = np.poly1d(z_Iosc)

                axes[0, j].scatter(Cfinal_array, Imax_array, color='teal')
                #axes[0, j].plot(Cfinal_array, p_Imax(Cfinal_array), color='black', linestyle='--')
                axes[0, j].set_xlabel('Cfinal')
                axes[0, j].set_ylabel('Imax')
                axes[0, j].grid(True)
                axes[0, j].set_title('Imax vs Cfinal \n (Pearson = {:0.2f})'.format(pearson_Imax[0,1]))

                axes[1, j].scatter(Cfinal_array, Ifinal_array, color='deeppink')
                #axes[1, j].plot(Cfinal_array, p_Ifinal(Cfinal_array), color='black', linestyle='--')
                axes[1, j].set_xlabel('Cfinal')
                axes[1, j].set_ylabel('Ifinal')
                axes[1, j].grid(True)
                axes[1, j].set_title('Ifinal vs Cfinal \n (Pearson = {:0.2f})'.format(pearson_Ifinal[0,1]))

                axes[2, j].scatter(Cfinal_array, Ioscillations_array, color='darkorange')
                #axes[2, j].plot(Cfinal_array, p_Iosc(Cfinal_array), color='black', linestyle='--')
                axes[2, j].set_xlabel('Cfinal')
                axes[2, j].set_ylabel('Ioscillations')
                axes[2, j].grid(True)
                axes[2, j].set_title('Ioscillations vs Cfinal \n (Pearson = {:0.2f})'.format(pearson_Iosc[0,1]))
    
    fig.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, '2D', folder) ):
                os.makedirs(os.path.join(plots_path, '2D', folder))
    
    plt.savefig(os.path.join(plots_path, '2D', folder,'corr_coupled_exp.jpeg'), dpi=400)
    plt.close()

def graph_bars_diff(prob_infect, comparison: str):
    path_to_results = os.path.join(results_path, 'Diff', '{}_NullAddition_beta_{:0.2f}_sigmaC_sigmaD.csv'.format(comparison,prob_infect))
    df_temp = pd.read_csv(path_to_results, index_col=0)

    bar_width = 0.25
    index = df_temp.index

    plt.figure(figsize=(10, 6))

    bar1 = plt.bar(index, df_temp['Imax'], bar_width, label='I max', color='skyblue')
    bar2 = plt.bar(index + bar_width, df_temp['Ifinal'], bar_width, label='I final', color='orange')
    bar3 = plt.bar(index + 2*bar_width, df_temp['Cfinal'], bar_width, label='C final', color='green')

    plt.xlabel('Addition of Phenomena')
    plt.ylabel('%')
    plt.title('Bar Chart of Categories')
    plt.xticks(index + bar_width / 3, df_temp['Model'])
    plt.legend()
    plt.tight_layout()

    if not os.path.isdir( os.path.join(plots_path, 'Diff') ):
                os.makedirs(os.path.join(plots_path, 'Diff'))
    
    plt.savefig(os.path.join(plots_path, 'Diff', 'barchart_{}_coupled_beta_{:0.2f}_exp.jpeg'.format(comparison,prob_infect)), dpi=400)
    plt.close()
     


'''
for key_case, val_case in dict_scenarios.items():
    graph_correlation(beta_search, sigmaC_search, sigmaD_search, 'sigmaC', 'sigmaD', key_case)

'''

for beta_temp in beta_search:
    #graph_bars_diff(beta_temp, 'MAD')
    graph_bars_diff(beta_temp, 'MSE')
