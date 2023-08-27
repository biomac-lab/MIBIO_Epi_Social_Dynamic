import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from tqdm import tqdm

main_path = os.path.join(os.path.split(os.getcwd())[0],'Epi_Social_Dynamic')
config_path = os.path.join(main_path, 'config.csv')
config_data = pd.read_csv(config_path, sep=',', header=None, index_col=0)
results_path  = config_data.loc['results_dir'][1]
plots_path = config_data.loc['plots_dir'][1]
parematric_df_dir = config_data.loc['parametric_df_dir'][1]


def SIS_replicator(x, t, beta_max, gamma, sigmaD, sigmaC=0, everything_dynamic:bool=False):
    global N

    S, I, xc, xd = x

    xr = [xc, xd]
    beta = beta_max*np.exp(-xr[0])

    dS = -beta*S*I + gamma*I
    dI = beta*S*I - gamma*I
    xdotSIS = [dS, dI]

    # Prisoner's dilemma
    # Payoff matrix
    g_infection = sigmaD*I
    g_susceptible = sigmaC*S

    a_CC = 1; a_DC = 0.75; a_CD = 0.5; a_DD = 0.25

    if everything_dynamic == False:
        A = np.array([[a_CC, a_CD],
                    [a_DC - g_infection, a_DD - g_infection]])
    else:
        A = np.array([[a_CC - g_susceptible, a_CD - g_susceptible],
                      [a_DC - g_infection, a_DD - g_infection]])

    A = A.squeeze()
    xdotREP = xr*(np.matmul(A,xr) - np.matmul(xr,np.matmul(A,xr)))

    dxdt = [xdotSIS[0], xdotSIS[1], xdotREP[0], xdotREP[1]]
    return dxdt


def run_sims_SIS_replicator(prob_infect, sigmaD, sigmaC:float=0, everything_dynamic:bool=False):
    defectFract = 0.5
    coopFract = 0.5
    N = 5000
    S = N-1
    I = 1
    C = coopFract
    D = defectFract

    y0 = [S/N, I/N, C, D]

    t_max = 300
    t = np.linspace(0, t_max, t_max*5)

    gamma = 1/7

    y = odeint(SIS_replicator, y0, t, args=(prob_infect, gamma, sigmaD, sigmaC, everything_dynamic))
    S_ = y[:,0]
    I_ = y[:,1]
    C_ = y[:,2]
    D_ = y[:,3]


    if everything_dynamic == False:
        pd_var = pd.DataFrame(columns=['time', 'beta', 'sigma_D', 'S', 'I', 'C', 'D'])
        pd_var['time'] = t
        pd_var['beta'] = prob_infect
        pd_var['sigma_D'] = sigmaD
        pd_var['S'] = S_
        pd_var['I'] = I_
        pd_var['C'] = C_
        pd_var['D'] = D_
    else:
        pd_var = pd.DataFrame(columns=['time', 'beta', 'sigma_D', 'sigma_C', 'S', 'I', 'C', 'D'])
        pd_var['time'] = t
        pd_var['beta'] = prob_infect
        pd_var['sigma_D'] = sigmaD
        pd_var['sigma_C'] = sigmaC
        pd_var['S'] = S_
        pd_var['I'] = I_
        pd_var['C'] = C_
        pd_var['D'] = D_

    return pd_var



df_parametric = pd.read_csv(os.path.join(main_path, parematric_df_dir), index_col=0)
beta_ = df_parametric[['beta']]
sigmaD_ = df_parametric[['sigmaD']]
sigmaC_ = df_parametric[['sigmaC']]

list_values = ['low', 'mean', 'high']

for idx1, val1 in enumerate(list_values):
    fig, ax = plt.subplots(3,3, figsize=(14,10))
    beta_temp = beta_.loc[val1][0]
    plt.suptitle(f'beta = {beta_temp}$')
    for idx2, val2 in enumerate(list_values):
        sigmaD_temp = sigmaD_.loc[val2][0]
        for idx3, val3 in enumerate(list_values):
            sigmaC_temp = sigmaC_.loc[val3][0]
            pd_temp = run_sims_SIS_replicator(beta_temp, sigmaD_temp, sigmaC_temp, True)

            ax[idx2, idx3].plot(pd_temp['time'], pd_temp['I'], label='Infected')
            ax[idx2,idx3].plot(pd_temp['time'], pd_temp['D'], label='Defector')
            ax[idx2,idx3].grid()
            ax[idx2,idx3].legend()

            if idx2 == 0:
                ax[idx2,idx3].set_title(f'sigma_D = {sigmaC_temp}') 
            if idx3 == 0:
                ax[idx2,idx3].set_ylabel(f'sigma_C = {sigmaD_temp} \n Fraction') 
            if idx2 == 2:
                ax[idx2,idx3].set_xlabel('Time') 

    if not os.path.isdir( os.path.join(results_path, plots_path, 'ODE_Simulations') ):
                os.makedirs(os.path.join(results_path, plots_path, 'ODE_Simulations'))
    
    plt.savefig(os.path.join(results_path, plots_path, 'ODE_Simulations','simu_ode_replicator_beta_{:0.2f}.jpeg'.format(beta_temp)))
    plt.close()

print('DONE SIMPLE SIMULATIONS')

### Save results
def exp_1D_SIS_replicator(param_search, param1:str):
    if not os.path.isdir( os.path.join(results_path, 'ode_results', '1D') ):
                os.makedirs(os.path.join(results_path, 'ode_results', '1D'))
    if param1 == 'beta':    
        for idx, p in tqdm(enumerate(param_search)):
            pd_var_res = run_sims_SIS_replicator(p, sigmaD_.loc['mean'][0], sigmaC_.loc['mean'][0], True)
            pd_var_res_ = pd_var_res.copy()
            
            pd_var_res_.to_csv(os.path.join(results_path, 'ode_results', '1D','ode_replicator_beta_{:0.2f}.csv'.format(p)))
        print('DONE beta EXPERIMENTATION')
    elif param1 == 'sigmaD':
        for idx, p in enumerate(param_search):
            pd_var_res = run_sims_SIS_replicator(beta_.loc['mean'][0], p, sigmaC_.loc['mean'][0], True)
            pd_var_res_ = pd_var_res.copy()
            
            pd_var_res_.to_csv(os.path.join(results_path, 'ode_results', '1D','ode_replicator_sigmaD_{:0.2f}.csv'.format(p)))    
        print('DONE sigmaD EXPERIMENTATION')
    elif param1 == 'sigmaC':
        for idx, p in enumerate(param_search):
            pd_var_res = run_sims_SIS_replicator(beta_.loc['mean'][0], sigmaD_.loc['mean'][0], p, True)
            pd_var_res_ = pd_var_res.copy()
            
            pd_var_res_.to_csv(os.path.join(results_path, 'ode_results', '1D','ode_replicator_sigmaC_{:0.2f}.csv'.format(p)))
        print('DONE sigmaC EXPERIMENTATION')

list_params = ['beta', 'sigmaD', 'sigmaC']

for idx, param_name in enumerate(list_params):
    df_temp = df_parametric[[param_name]]
    param_search = np.linspace(df_temp.loc['min'][0], df_temp.loc['max'][0], int(df_temp.loc['num'][0]))
    exp_1D_SIS_replicator(param_search, param_name)

print('DONE 1D Experimentations')


#def exp_2D_SIS_replicator():

#TODO Finish this function
#def exp_3D_SIS_replicator():






'''
t_max = 150
t = np.linspace(0, t_max, t_max*2)
sigma_search = np.linspace(0,1,100)
beta_search  = np.linspace(0,1,100)
for idx_p, prob in enumerate(tqdm(beta_search)):
    for idx_s, sigma in enumerate(sigma_search):

        pd_var_res = run_sims_SIS_replicator(sigma,prob)
        pd_var_res_ = pd_var_res.copy()

        if not os.path.isdir( os.path.join(results_path, 'ode_results') ):
            os.makedirs(os.path.join(results_path, 'ode_results'))

        pd_var_res_.to_csv(os.path.join(results_path, 'ode_results', 'ode_replicator_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(sigma,prob)))
##

beta_coop = 6.5/7
sigma_coop = 0.95

beta_def = 0.4
sigma_def = 0.5



omega_search = np.linspace(0, 1, 100)
alpha_search = np.linspace(0, 2, 100)

for idx_omega, omega_temp in enumerate(tqdm(omega_search)):
    for idx_alpha, alpha_temp in enumerate(alpha_search):

        pd_var_res = run_sims_SIS_replicator(sigma_coop, beta_coop, omega_temp, alpha_temp, True)
        pd_var_res_ = pd_var_res.copy()

        if not os.path.isdir( os.path.join(results_path, 'ode_results') ):
            os.makedirs(os.path.join(results_path, 'ode_results'))

        pd_var_res_.to_csv(os.path.join(results_path, 'ode_results',
                                        'ode_replicator_sigma_{:0.2f}_beta_{:0.2f}_omega_{:0.2f}_alpha_{:0.2f}.csv'.format(sigma_coop,beta_coop, omega_temp, alpha_temp)))

        pd_var_res = run_sims_SIS_replicator(sigma_def, beta_def, omega_temp, alpha_temp, True)
        pd_var_res_ = pd_var_res.copy()

        if not os.path.isdir(os.path.join(results_path, 'ode_results')):
            os.makedirs(os.path.join(results_path, 'ode_results'))

        pd_var_res_.to_csv(os.path.join(results_path, 'ode_results',
                                        'ode_replicator_sigma_{:0.2f}_beta_{:0.2f}_omega_{:0.2f}_alpha_{:0.2f}.csv'.format(sigma_def, beta_def, omega_temp, alpha_temp)))
'''