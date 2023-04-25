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
def classify_game(R: float, S: float, T: float, P: float):
    if T > R and R > P and P > S: #Prisoner's Dilemma
        return 1
    elif T > R and R > S and S > P: #Hawk-Dove Game
        return 2
    elif R > T and T > P and P > S: #Stag Hunt Game
        return 3
    else: #Unnclassified
        return 4

def find_inflection_point(filepath: str):
    df_game = pd.read_csv(os.path.join(main_path, results_path, 'game_results', filepath))
    diff_type = np.diff(df_game['type'])
    i_max = np.argmax(diff_type)
    return i_max

def find_payoffs(filepath: str):
    df_simu = pd.read_csv(os.path.join(main_path, results_path, 'ode_results', filepath))
    reward = list(); sucker = list(); temptation = list(); punishment = list(); game=list()
    S_ = -0.5
    T_ = 1.5
    sigma_ = float(df_simu.iloc[0]['sigma'])
    beta_ = float(df_simu.iloc[0]['beta'])
    for index, row in df_simu.iterrows():
        I = row['I']
        N = row['I'] + row['S']
        reward.append(1)
        sucker.append(S_)
        temptation.append(T_ - sigma_*I/N)
        punishment.append(- sigma_*I/N)
        game.append(classify_game(reward[-1], sucker[-1], temptation[-1], punishment[-1]))

    data = {'reward': reward, 'sucker': sucker, 'temptation': temptation, 'punishment': punishment, 'type': game}
    df_game = pd.DataFrame(data)

    return df_game

##

t_max = 150
gamma = 1 / 7
t = np.linspace(0, t_max, t_max*2)
sigma_search = np.linspace(0,1,100)
beta_search  = np.linspace(0,1,100)

##
for idx_p, prob in enumerate(tqdm(beta_search)):
    for idx_s, sigma in enumerate(sigma_search):
        filepath_temp = 'ode_replicator_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(sigma, prob)
        df_game_temp = find_payoffs(filepath_temp)

        if not os.path.isdir(os.path.join(results_path, 'game_results') ):
            os.makedirs(os.path.join(results_path, 'game_results'))

        df_game_temp.to_csv(os.path.join(results_path, 'game_results', 'game_payoffs_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(sigma,prob)))


##

mat_inflection_time = np.zeros((len(sigma_search), len(beta_search)))
for idx_p, prob in enumerate(tqdm(beta_search)):
    for idx_s, sigma in enumerate(sigma_search):
        filepath_temp = 'game_payoffs_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(sigma, prob)
        i_max = find_inflection_point(filepath_temp)

        mat_inflection_time[idx_p, idx_s] = i_max

with open('inflection_point.npy', 'wb') as f:
    np.save(f, mat_inflection_time)

##

params1 = (0.90, 0.85)
params2 = (0.49, 0.40)
params3 = (0.80, 0.82)


df1 = pd.read_csv(os.path.join(results_path, 'game_results', 'game_payoffs_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(params1[0],params1[1])))
df2 = pd.read_csv(os.path.join(results_path, 'game_results', 'game_payoffs_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(params2[0],params2[1])))
df3 = pd.read_csv(os.path.join(results_path, 'game_results', 'game_payoffs_sigma_{:0.2f}_beta_{:0.2f}.csv'.format(params3[0],params3[1])))

plt.figure()
plt.plot(df1.index/150, df1['type'], label=r'\sigma=0.90 | \beta=0.85')
#plt.vlines(list(df1.index/150)[Imax], 0, 5, linestyles='dashed', color='k')
#plt.plot(df2.index/150, df2['type'], label=r'\sigma=0.49 | \beta=0.40')
#plt.plot(df3.index/150, df3['type'], label=r'\sigma=0.80 | \beta=0.82')
plt.yticks([1, 2, 3, 4], ['PD', 'HD', 'SH', 'UNKNOWN'])
plt.legend(loc='best')
plt.grid()
plt.ylabel('Game Type')
plt.xlabel('Time [Days]')
plt.show()


##

plt.figure()
plt.imshow(mat_inflection_time, cmap='RdYlGn')
plt.colorbar()
plt.show()

