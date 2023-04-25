import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
from tqdm import tqdm



main_path = os.path.split(os.getcwd())[0] + '/Epidemiology_Behavior_Games'
config_path = main_path + '/config.csv'
config_data = pd.read_csv(config_path, sep=',', header=None, index_col=0)

results_path  = config_data.loc['results_dir'][1]


def SIS_replicator(x, t, beta_max, sigma, gamma, omega=0, alpha=1, everything_dynamic:bool=False):
    global N

    S, I, xc, xd = x

    xr = [xc, xd]
    beta = beta_max*np.exp(-xr[0])

    dS = -beta*S*I + gamma*I
    dI = beta*S*I - gamma*I
    xdotSIS = [dS, dI]

    # Prisoner's dilemma
    S_ = -0.5
    T_ = 1.5
    # Payoff matrix
    sigma_infection = sigma*I
    omega_susceptible = omega*(S)**alpha

    if everything_dynamic == False:
        A = np.array([[1, S_],
                    [T_-sigma_infection, 0-sigma_infection]])/3
    else:
        A = np.array([[1-omega_susceptible, S_-omega_susceptible],
                      [T_-sigma_infection, 0-sigma_infection]])/3


    xdotREP = xr*(np.matmul(A,xr) - np.matmul(xr,np.matmul(A,xr)))

    dxdt = [xdotSIS[0], xdotSIS[1], xdotREP[0], xdotREP[1]]

    return dxdt


def run_sims_SIS_replicator(sigma, prob_infect, omega:float=0, alpha:float=1, everything_dynamic:bool=False):
    defectFract = 0.5
    coopFract = 0.5
    N = 5000
    S = N-1
    I = 1
    C = coopFract
    D = defectFract

    y0 = [S/N, I/N, C, D]

    t_max = 150
    t = np.linspace(0, t_max, t_max*2)

    gamma = 1/7

    y = odeint(SIS_replicator, y0, t, args=(prob_infect, sigma, gamma, omega, alpha, everything_dynamic))
    S_ = y[:,0]
    I_ = y[:,1]
    C_ = y[:,2]
    D_ = y[:,3]


    if everything_dynamic == False:
        pd_var = pd.DataFrame(columns=['time', 'sigma', 'beta', 'S', 'I', 'C', 'D'])
        pd_var['time'] = t
        pd_var['sigma'] = sigma
        pd_var['beta'] = prob_infect
        pd_var['S'] = S_
        pd_var['I'] = I_
        pd_var['C'] = C_
        pd_var['D'] = D_
    else:
        pd_var = pd.DataFrame(columns=['time', 'sigma', 'beta', 'omega', 'alpha', 'S', 'I', 'C', 'D'])
        pd_var['time'] = t
        pd_var['sigma'] = sigma
        pd_var['beta'] = prob_infect
        pd_var['omega'] = omega
        pd_var['alpha'] = alpha
        pd_var['S'] = S_
        pd_var['I'] = I_
        pd_var['C'] = C_
        pd_var['D'] = D_

    return pd_var


### Save results

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