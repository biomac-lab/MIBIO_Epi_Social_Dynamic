import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os, pickle
from tqdm import tqdm
import sympy as sym

main_path = os.path.join(os.path.split(os.getcwd())[0],'Epi_Social_Dynamic')
config_path = os.path.join(main_path, 'config.csv')
config_data = pd.read_csv(config_path, sep=',', header=None, index_col=0)
results_path  = config_data.loc['results_dir'][1]
plots_path = config_data.loc['plots_dir'][1]
parematric_df_dir = config_data.loc['parametric_coupled_df_dir'][1]
gamma = 1/7
alpha = 0.2
reward_matrix = np.array([[1.1, 1.1, 0.8, 0.7], [1.3, 1.3, 0.5, 0.3], [2, 1.8, 1, 1], [1.6, 1.4, 1, 1]])
df_parametric = pd.read_csv(os.path.join(main_path, parematric_df_dir), index_col=0)
beta_ = df_parametric[['beta']]
sigmaD_ = df_parametric[['sigmaD']]
sigmaC_ = df_parametric[['sigmaC']]
d_fract_ = df_parametric[['d_fract']]

list_values = ['low', 'mean', 'high']

'''
dict_scenarios = {'static':(False, False, False, False, False), 
                  'dynamicS':(False, False, False, False, True), 
                  'dynamicS_SC':(True, False, False, False, True), 
                  'dynamicSI_SC':(True, False, False, True, True), 
                  'dynamicSI_SCPA':(True, True, False, True, True)}

dict_scenarios = {'Null':(False, False, False, True, True),
                    'Null+SP':(False, False, True, True, True),
                    'Null+SC':(True, False, False, True, True),
                    'Null+PA':(False, True, False, True, True),
                  'Full':(True, True, True, True, True),
                    'Full-SP':(True, True, False, True, True),
                    'Full-SC':(False, True, True, True, True),
                    'Full-PA':(True, False, True, True, True)}
'''
                    
#Selfcare - Public Awareness - Social Pressure - Dynamic I - Dynamic S
dict_scenarios = {'Null':(False, False, False, True, True),
                  'SP':(False, False, True, True, True),
                  'SC':(True, False, False, True, True),
                  'PA':(False, True, False, True, True),
                  'SP+SC':(True, False, True, True, True),
                  'SP+PA':(False, True, True, True, True),
                  'PA+SC':(True, True, False, True, True),
                  'SC+SP+PA':(True, True, True, True, True)}

def SIS_coupled(variables, t, beta_max, alpha, gamma, A, sigmaD, sigmaC, 
                selfcare:bool=True, public_awareness:bool=True, social_pressure:bool=True, dynamic_I:bool= True, dynamic_S:bool=True):
    global N

    S_c, S_d, I_c, I_d = variables

    beta = beta_max*np.exp(-(S_c + I_c))

    S_total = S_c + S_d 
    I_total = I_c + I_d 
    C_total = S_c + I_c
    D_total = S_d + I_d

    penalty_SC_Sc = 0
    penalty_SC_Sd = 0
    penalty_PA_Ic = 0
    penalty_PA_Id = 0
    penalty_SP_Sc = 0
    penalty_SP_Sd = 0

    if selfcare:
        penalty_SC_Sc = sigmaC*(S_total)
        penalty_SC_Sd = sigmaD*(I_total)
    if public_awareness:
        penalty_PA_Ic = sigmaC*(S_total)
        penalty_PA_Id = sigmaD*(I_total)
    if social_pressure:
        penalty_SP_Sc = sigmaC*(D_total)
        penalty_SP_Sd = sigmaD*(C_total)
        
    f_sc = (A[0,0]-penalty_SP_Sc)*S_c + (A[0,1]-penalty_SP_Sc)*S_d + (A[0,2]-penalty_SC_Sc)*I_c + (A[0,3]-penalty_SC_Sc)*I_d
    f_sd = (A[1,0]-penalty_SP_Sd)*S_c + (A[1,1]-penalty_SP_Sd)*S_d + (A[1,2]-penalty_SC_Sd)*I_c + (A[1,3]-penalty_SC_Sd)*I_d
    f_ic = (A[2,0]-penalty_PA_Ic)*S_c + (A[2,1]-penalty_PA_Ic)*S_d + A[2,2]*I_c + A[2,3]*I_d
    f_id = (A[3,0]-penalty_PA_Id)*S_c + (A[3,1]-penalty_PA_Id)*S_d + A[3,2]*I_c + A[3,3]*I_d
    
    fbar_s = f_sc*(S_c/S_total) + f_sd*(S_d/S_total)
    fbar_i = f_ic*(I_c/I_total) + f_id*(I_d/I_total)

    dS_cdt = -beta*S_c*(I_d + alpha*I_c) + gamma*I_c 
    dI_cdt = beta*S_c*(I_d + alpha*I_c) - gamma*I_c 
    dS_ddt = -beta*S_d*(I_d + alpha*I_c) + gamma*I_d 
    dI_ddt = beta*S_d*(I_d + alpha*I_c) - gamma*I_d 

    if dynamic_S:
        dS_cdt += S_c*(f_sc - fbar_s)
        dS_ddt += S_d*(f_sd - fbar_s)
    if dynamic_I:
        dI_cdt += I_c*(f_ic - fbar_i)
        dI_ddt += I_d*(f_id - fbar_i)

    return [dS_cdt, dS_ddt, dI_cdt, dI_ddt]

def run_sims_SIS_coupled(d_fract, prob_infect, alpha, A, sigmaD, sigmaC:float=0, 
                         selfcare=True, public_awareness=True, social_pressure=True, dynamic_I=True, dynamic_S=True):
    defectFract = d_fract
    coopFract = 1 - defectFract
    N = 5000
    I = 2
    S = N - I
    S_c = S*coopFract
    S_d = S*defectFract
    I_c = I*coopFract
    I_d = I*defectFract

    y0 = [S_c/N, S_d/N, I_c/N, I_d/N]

    t_max = 300
    t = np.linspace(0, t_max, t_max*8)
    gamma = 1/7

    y = odeint(SIS_coupled, y0, t, args=(prob_infect, alpha, gamma, A, sigmaD, sigmaC, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S))
    S_c_ = y[:,0]
    S_d_ = y[:,1]
    I_c_ = y[:,2]
    I_d_ = y[:,3]

    t_cc = A[2,0]
    t_cd = A[2,1]
    t_dc = A[3,0]
    t_dd = A[3,1]

    pd_var = pd.DataFrame(columns=['time', 'beta', 'alpha', 'sigma_D', 'sigma_C', 'S_c', 'S_d', 'I_c', 'I_d'])
    pd_var['time'] = t
    pd_var['beta'] = prob_infect
    pd_var['R0'] = calculateR0(y0[0], y0[1], prob_infect, alpha, gamma, sigmaC, t_cc, t_cd, t_dc, t_dd)
    pd_var['alpha'] = alpha
    pd_var['sigma_D'] = sigmaD
    pd_var['sigma_C'] = sigmaC
    pd_var['S_c'] = S_c_
    pd_var['S_d'] = S_d_
    pd_var['I_c'] = I_c_
    pd_var['I_d'] = I_d_

    return pd_var

def exp_1D_SIS_coupled(d_fract, prob_infect, param_search1, param1:str, folder:str, 
                       selfcare=True, public_awareness=True, social_pressure=True, dynamic_I=True, dynamic_S=True):
    if not os.path.isdir( os.path.join(results_path, '1D', folder) ):
        os.makedirs(os.path.join(results_path, '1D', folder))

    beta_mean = prob_infect
    sigmaD_mean = int(sigmaD_.iloc[5])
    sigmaC_mean = int(sigmaC_.iloc[5])

    if param1 == 'beta':    
        for idx, p in (enumerate(param_search1)):
            pd_var_res = run_sims_SIS_coupled(d_fract, p, alpha, reward_matrix, sigmaD_mean, sigmaC_mean, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
            pd_var_res_ = pd_var_res.copy()
            
            pd_var_res_.to_csv(os.path.join(results_path, 
    '1D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(p, sigmaD_mean, sigmaC_mean, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)))
        print('DONE beta EXPERIMENTATION')
    elif param1 == 'sigmaD':
        for idx, p in (enumerate(param_search1)):
            pd_var_res = run_sims_SIS_coupled(d_fract, beta_mean, alpha, reward_matrix, p, sigmaC_mean, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
            pd_var_res_ = pd_var_res.copy()
            
            pd_var_res_.to_csv(os.path.join(results_path, 
    '1D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(beta_mean, p, sigmaC_mean)))    
        print('DONE sigmaD EXPERIMENTATION')
    elif param1 == 'sigmaC':
        for idx, p in (enumerate(param_search1)):
            pd_var_res = run_sims_SIS_coupled(d_fract, beta_mean, alpha, reward_matrix, sigmaD_mean, p, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
            pd_var_res_ = pd_var_res.copy()
            
            pd_var_res_.to_csv(os.path.join(results_path,
    '1D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(beta_mean, sigmaD_mean, p)))
        print('DONE sigmaC EXPERIMENTATION')

def exp_2D_SIS_coupled(d_fract, prob_infect, param_search1, param_search2, param1: str, param2: str, folder:str, 
                       selfcare=True, public_awareness=True, social_pressure=True, dynamic_I=True, dynamic_S=True):
    if not os.path.isdir( os.path.join(results_path, '2D', folder) ):
                os.makedirs(os.path.join(results_path, '2D', folder))
    
    beta_mean = prob_infect
    sigmaD_mean = int(sigmaD_.iloc[5])
    sigmaC_mean = int(sigmaC_.iloc[5])
    
    if param1 == 'beta':
        if param2 == 'sigmaD':
            for idx1, p1 in (enumerate(param_search1)):
                 for idx2, p2 in (enumerate(param_search2)):
                        pd_var_res = run_sims_SIS_coupled(d_fract, p1, alpha, reward_matrix,p2,sigmaC_mean,selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
                        pd_var_res_ = pd_var_res.copy()

                        pd_var_res_.to_csv(os.path.join(results_path, 
            '2D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(p1,p2, sigmaC_mean)))
        elif param2 == 'sigmaC':
             for idx1, p1 in (enumerate(param_search1)):
                 for idx2, p2 in (enumerate(param_search2)):
                        pd_var_res = run_sims_SIS_coupled(d_fract, p1, alpha, reward_matrix,sigmaD_mean,p2,selfcare,public_awareness, social_pressure,dynamic_I,dynamic_S)
                        pd_var_res_ = pd_var_res.copy()

                        pd_var_res_.to_csv(os.path.join(results_path, 
            '2D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(p1,sigmaD_mean,p2)))

    elif param1 == 'sigmaD':
        if param2 == 'beta':
            for idx1, p1 in (enumerate(param_search1)):
                for idx2, p2 in (enumerate(param_search2)):
                    pd_var_res = run_sims_SIS_coupled(d_fract, p2,alpha,reward_matrix,p1,sigmaC_mean,selfcare,public_awareness, social_pressure,dynamic_I,dynamic_S)
                    pd_var_res_ = pd_var_res.copy()
            
                    pd_var_res_.to_csv(os.path.join(results_path, 
            '2D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(p2,p1,sigmaC_mean)))

        elif param2 == 'sigmaC':
            for idx1, p1 in (enumerate(param_search1)):
                for idx2, p2 in (enumerate(param_search2)):
                    pd_var_res = run_sims_SIS_coupled(d_fract, beta_mean,alpha,reward_matrix,p1,p2,selfcare,public_awareness, social_pressure,dynamic_I,dynamic_S)
                    pd_var_res_ = pd_var_res.copy()

                    pd_var_res_.to_csv(os.path.join(results_path, 
            '2D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(beta_mean,p1,p2)))

    elif param1 == 'sigmaC':
        if param2 == 'beta':
            for idx1, p1 in (enumerate(param_search1)):
                for idx2, p2 in (enumerate(param_search2)):
                    pd_var_res = run_sims_SIS_coupled(d_fract, p2,alpha,reward_matrix,sigmaD_mean,p1,selfcare,public_awareness, social_pressure,dynamic_I,dynamic_S)
                    pd_var_res_ = pd_var_res.copy()
            
                    pd_var_res_.to_csv(os.path.join(results_path, 
            '2D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(p2,sigmaD_mean,p1)))

        elif param2 == 'sigmaD':
            for idx1, p1 in (enumerate(param_search1)):
                for idx2, p2 in (enumerate(param_search2)):
                    pd_var_res = run_sims_SIS_coupled(d_fract, beta_mean,alpha,reward_matrix, p2, p1,selfcare,public_awareness, social_pressure,dynamic_I,dynamic_S)
                    pd_var_res_ = pd_var_res.copy()

                    pd_var_res_.to_csv(os.path.join(results_path, 
            '2D', folder,'ode_coupled_beta_{:0.2f}_sigmaD_{:0.2f}_sigmaC_{:0.2f}.csv'.format(beta_mean,p2,p1)))

def exp_IC_SIS_coupled(initcond_search, prob_infect, param_search1, param1:str, folder:str,
                        selfcare=True, public_awareness=True, social_pressure=True, dynamic_I=True, dynamic_S=True):
    if not os.path.isdir( os.path.join(results_path, 'IC', folder) ):
        os.makedirs(os.path.join(results_path, 'IC', folder))

    beta_mean = prob_infect
    sigmaD_mean = int(sigmaD_.iloc[5])
    sigmaC_mean = int(sigmaC_.iloc[5])

    if param1 == 'beta':    
        for idx, p in (enumerate(param_search1)):
            for idx2, d0 in enumerate(initcond_search):
                pd_var_res = run_sims_SIS_coupled(d0, p, alpha, reward_matrix, sigmaD_mean, sigmaC_mean, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
                pd_var_res_ = pd_var_res.copy()
                
                pd_var_res_.to_csv(os.path.join(results_path, 
        'IC', folder,'ode_coupled_beta_{:0.2f}_IC_{:0.2f}.csv'.format(p, d0)))
        print('DONE beta - Initial Conditions EXPERIMENTATION')
    elif param1 == 'sigmaD':
        for idx, p in (enumerate(param_search1)):
            for idx2, d0 in enumerate(initcond_search):
                pd_var_res = run_sims_SIS_coupled(d0, beta_mean, alpha, reward_matrix, p, sigmaC_mean, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
                pd_var_res_ = pd_var_res.copy()
                
                pd_var_res_.to_csv(os.path.join(results_path, 
        'IC', folder,'ode_coupled_sigmaD_{:0.2f}_IC_{:0.2f}.csv'.format(p, d0)))    
        print('DONE sigmaD - Initial Conditions EXPERIMENTATION')
    elif param1 == 'sigmaC':
        for idx, p in (enumerate(param_search1)):
            for idx2, d0 in enumerate(initcond_search):
                pd_var_res = run_sims_SIS_coupled(d0, beta_mean, alpha, reward_matrix, sigmaD_mean, p, selfcare, public_awareness, social_pressure, dynamic_I, dynamic_S)
                pd_var_res_ = pd_var_res.copy()
                
                pd_var_res_.to_csv(os.path.join(results_path,
        'IC', folder,'ode_coupled_sigmaC_{:0.2f}_IC_{:0.2f}.csv'.format(p, d0)))
        print('DONE sigmaC - Initial Conditions EXPERIMENTATION')

def calculateR0(S_c_v, S_d_v, beta_v, alpha_v, gamma_v, sigma_c_v, t_cc_v, t_cd_v, t_dc_v, t_dd_v):
    with open(os.path.join(main_path, 'run', 'R0.pickle'), 'rb') as file:
        expr_R0 = pickle.load(file)

    S_c, S_d, beta, alpha, gamma, sigma_c, t_cc, t_cd, t_dc, t_dd = sym.symbols('S_c S_d beta alpha gamma sigma_c t_cc t_cd t_dc t_dd')

    func_R0 = sym.lambdify([S_c, S_d, beta, alpha, gamma, sigma_c, t_cc, t_cd, t_dc, t_dd], expr_R0, 'numpy')
    R0 = func_R0(S_c_v, S_d_v, beta_v, alpha_v, gamma_v, sigma_c_v, t_cc_v, t_cd_v, t_dc_v, t_dd_v)

    return R0

for key_case, val_case in tqdm(dict_scenarios.items()):
    fig, ax = plt.subplots(3, 3, figsize=(14,10))
    for idx1, val1 in enumerate(list_values):
        beta_temp = beta_.loc[val1][0]
        for idx2, val2 in enumerate(list_values):
            d_fract_temp = d_fract_.loc[val2][0]
            sigmaD_temp = sigmaD_.loc['mean'][0]
            sigmaC_temp = sigmaC_.loc['mean'][0]
            pd_temp = run_sims_SIS_coupled(d_fract_temp,beta_temp, alpha, reward_matrix, sigmaD_temp, sigmaC_temp, val_case[0], val_case[1], val_case[2], val_case[3], val_case[4])

            ax[idx1, idx2].plot(pd_temp['time'], pd_temp['I_d'], label='Infected-Defector', color='darkred')
            ax[idx1,idx2].plot(pd_temp['time'], pd_temp['S_d'], label='Susceptible-Defector', color='mediumblue')
            ax[idx1, idx2].plot(pd_temp['time'], pd_temp['I_c'], label='Infected-Cooperator', color='orangered')
            ax[idx1,idx2].plot(pd_temp['time'], pd_temp['S_c'], label='Susceptible-Cooperator', color='cornflowerblue')
            ax[idx1,idx2].grid()
            ax[idx1,idx2].legend()

            ax[idx1, idx2].set_title(f'${{\sigma_C}}$ = {round(sigmaC_temp)} & ${{\sigma_D}}$ = {sigmaD_temp}')

            if idx1 == 0:
                ax[idx1,idx2].set_title(f'${{\%D0}}$ = {d_fract_temp} ') 
            if idx2 == 0:
                ax[idx1,idx2].set_ylabel(f'${{\\beta}}$ = {beta_temp}') 
            if idx1 == 2:
                ax[idx1,idx2].set_xlabel('Time [days]') 

    if not os.path.isdir( os.path.join(plots_path, 'ODE_Simulations', key_case) ):
                os.makedirs(os.path.join(plots_path, 'ODE_Simulations', key_case))
        
    plt.savefig(os.path.join(plots_path, 'ODE_Simulations', key_case,
                            'simu_ode_coupled_sigmaC_{:0.2f}_sigmaD_{:0.2f}.jpeg'.format(sigmaC_temp, sigmaD_temp)), dpi=400)
    plt.close()

'''
for key_case, val_case in tqdm(dict_scenarios.items()):
    for idx1, val1 in enumerate(list_values):
        fig, ax = plt.subplots(3,3, figsize=(14,10))
        beta_temp = beta_.loc[val1][0]
        plt.suptitle(f'${{\\beta}}$ = {beta_temp}')
        for idx2, val2 in enumerate(list_values):
            sigmaD_temp = sigmaD_.loc[val2][0]
            for idx3, val3 in enumerate(list_values):
                sigmaC_temp = sigmaC_.loc[val3][0]
                
                pd_temp = run_sims_SIS_coupled(0.9,beta_temp,alpha, reward_matrix, sigmaD_temp, sigmaC_temp, val_case[0], val_case[1], val_case[2], val_case[3], val_case[4])
                
                ax[idx2, idx3].plot(pd_temp['time'], pd_temp['I_d'], label='Infected-Defector', color='darkred')
                ax[idx2,idx3].plot(pd_temp['time'], pd_temp['S_d'], label='Susceptible-Defector', color='mediumblue')
                ax[idx2, idx3].plot(pd_temp['time'], pd_temp['I_c'], label='Infected-Cooperator', color='orangered')
                ax[idx2,idx3].plot(pd_temp['time'], pd_temp['S_c'], label='Susceptible-Cooperator', color='cornflowerblue')
                ax[idx2,idx3].grid()
                ax[idx2,idx3].legend()

                ax[idx2, idx3].set_title(f'${{R0}}$ = {round(pd_temp['R0'][0], 3)}')

                if idx2 == 0:
                    #ax[idx2,idx3].set_title(f'${{\sigma_D}}$ = {sigmaC_temp} & ${{R0}}$ = {round(pd_temp['R0'][0], 3)}') 
                    ax[idx2,idx3].set_title(f'${{\sigma_D}}$ = {sigmaC_temp} ') 
                if idx3 == 0:
                    ax[idx2,idx3].set_ylabel(f'${{\sigma_C}}$ = {sigmaD_temp} \n Fraction') 
                if idx2 == 2:
                    ax[idx2,idx3].set_xlabel('Time [days]') 

        if not os.path.isdir( os.path.join(plots_path, 'ODE_Simulations', key_case) ):
                    os.makedirs(os.path.join(plots_path, 'ODE_Simulations', key_case))
        
        plt.savefig(os.path.join(plots_path, 'ODE_Simulations', key_case,
                                'simu_ode_coupled_beta_{:0.2f}.jpeg'.format(beta_temp)), dpi=400)
        plt.close()
'''
print('DONE SIMPLE SIMULATIONS')

### Save results

#list_params = ['beta', 'sigmaD', 'sigmaC']
list_params = ['sigmaD', 'sigmaC']

IC_search = np.linspace(d_fract_.iloc[0,0], d_fract_.iloc[1,0], int(d_fract_.iloc[2,0]))
beta_search = np.linspace(beta_.iloc[0,0], beta_.iloc[1,0], int(beta_.iloc[2,0]))
sigmaD_search = np.linspace(sigmaD_.iloc[0,0], sigmaD_.iloc[1,0], int(sigmaD_.iloc[2,0]))
sigmaC_search = np.linspace(sigmaC_.iloc[0,0], sigmaC_.iloc[1,0], int(sigmaC_.iloc[2,0]))


for beta_temp in tqdm(beta_search):
    for key_case, val_case in dict_scenarios.items():
        for idx, param_name in enumerate(list_params):
            df_temp = df_parametric[[param_name]]
            param_search = np.linspace(df_temp.iloc[0,0], df_temp.iloc[1,0], int(df_temp.iloc[2,0]))
            exp_1D_SIS_coupled(0.9, beta_temp, param_search, param_name, key_case, val_case[0], val_case[1], val_case[2], val_case[3], val_case[4])
            exp_IC_SIS_coupled(IC_search, beta_temp, param_search, param_name, key_case, val_case[0], val_case[1], val_case[2], val_case[3], val_case[4])

print('DONE 1D Experimentations')
print('DONE IC Experimentations')

'''for beta_temp in tqdm(beta_search):
    for key_case, val_case in dict_scenarios.items():
        exp_2D_SIS_coupled(0.9, beta_temp, sigmaC_search, sigmaD_search, 'sigmaC', 'sigmaD', key_case, val_case[0], val_case[1], val_case[2], val_case[3], val_case[4])
        print(f'Finish {key_case} scenario')

print('DONE 2D Experimentations')
'''
#TODO Finish this function
#def exp_3D_SIS_replicator():

