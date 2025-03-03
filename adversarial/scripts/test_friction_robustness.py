import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import scipy, scipy.signal
import argparse
import os
import numpy as np
from test import test_const_adv
from IPython import embed
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import pandas as pd

def get_robustness(save_file, env_name, fric_fractions=[1.0], fric_bodies = [b'foot'], mass_fractions=[1.0], mass_bodies = [b'torso'], num_evals=5):
    print(sys.path)

    # hopper [b'foot'], halfcheetah [b'ffoot']
    savename = '/your-folder/'+save_file
    res_D = pickle.load(open(savename,'rb'))
    const_test_rew_summary = res_D['zero_test']
    rand_test_rew_summary = res_D['rand_test']
    step_test_rew_summary = res_D['step_test']
    rand_step_test_rew_summary = res_D['rand_step_test']
    adv_test_rew_summary = res_D['adv_test']

    policy = res_D['pro_policy']

    print('------------------------')
    P = policy
    #M=[];V=[];
    fric_fractions = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    mass_fractions = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    M = np.zeros((len(fric_fractions), len(mass_fractions)))
    V=np.zeros((len(fric_fractions), len(mass_fractions)))
    fis=np.zeros((len(fric_fractions), len(mass_fractions)))
    mis=np.zeros((len(fric_fractions), len(mass_fractions)))
    for fi,f in enumerate(fric_fractions):
        for mi,m in enumerate(mass_fractions):
            print('{}/{}'.format((fi*len(mass_fractions))+mi,len(mass_fractions)*len(fric_fractions)))
            env = normalize(GymEnv(env_name, 1.0));
            e = np.array(env.wrapped_env.env.model.geom_friction)
            fric_ind = env.wrapped_env.env.model.body_names.index(fric_bodies[0])
            e[fric_ind,0] = e[fric_ind,0]*f
            env.wrapped_env.env.model.geom_friction = e


            me = np.array(env.wrapped_env.env.model.body_mass)
            mass_ind = env.wrapped_env.env.model.body_names.index(mass_bodies[0])
            me[mass_ind,0] = me[mass_ind,0]*m
            env.wrapped_env.env.model.body_mass = me

            t = []
            for _ in range(num_evals):
                t.append(test_const_adv(env, P, 1000, 1))
            t=np.array(t)
            M[fi,mi] = t.mean()
            V[fi,mi] = t.std()
            fis[fi,mi] = e[fric_ind,0]
            mis[fi,mi] = me[mass_ind,0]
    return M,V,fis,mis

parser = argparse.ArgumentParser()
parser.add_argument('--save_file', type=str)
parser.add_argument('--env_name', type=str)

args = parser.parse_args()
M,V,fis,mis = get_robustness(args.save_file, args.env_name)
fric_fractions = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
mass_fractions = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

df = pd.DataFrame(M, columns = mass_fractions, index = fric_fractions)

df.to_csv('ppo_walker_worst_v2.csv')


print('M: ', M)
print('V: ', V)
print('fis: ', fis)
print('mis: ', mis)
