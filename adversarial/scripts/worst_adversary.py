import sys
sys.path.append('../../')
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
import matplotlib.pyplot as plt
import numpy as np
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv
import pickle
import argparse
import os
import gym
import random
import csv
import pandas as pd
import time
#from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', type=str)
parser.add_argument('--env', type=str, required=True, help='Name of adversarial environment')
parser.add_argument('--path_length', type=int, default=1000, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[100,100,100], help='layer definition')
parser.add_argument('--if_render', type=int, default=0, help='Should we render?')
parser.add_argument('--after_render', type=int, default=100, help='After how many to animate')
parser.add_argument('--n_exps', type=int, default=1, help='Number of training instances to run')
parser.add_argument('--n_itr', type=int, default=500, help='Number of iterations of the alternating optimization')
parser.add_argument('--n_pro_itr', type=int, default=1, help='Number of iterations for the portagonist')
parser.add_argument('--n_adv_itr', type=int, default=1, help='Number of interations for the adversary')
parser.add_argument('--batch_size', type=int, default=4000, help='Number of training samples for each iteration')
parser.add_argument('--save_every', type=int, default=100, help='Save checkpoint every save_every iterations')
parser.add_argument('--n_process', type=int, default=1, help='Number of parallel threads for sampling environment')
parser.add_argument('--adv_fraction', type=float, default=0.25, help='fraction of maximum adversarial force to be applied')
parser.add_argument('--step_size', type=float, default=0.01, help='kl step size for TRPO')
parser.add_argument('--gae_lambda', type=float, default=0.97, help='gae_lambda for learner')
parser.add_argument('--folder', type=str, default=os.environ['HOME'], help='folder to save result in')

args = parser.parse_args()

env_name = args.env
path_length = args.path_length
layer_size = tuple(args.layer_size)
ifRender = bool(args.if_render)
afterRender = args.after_render
n_exps = args.n_exps
n_itr = args.n_itr
n_pro_itr = args.n_pro_itr
n_adv_itr = args.n_adv_itr
batch_size = args.batch_size
save_every = args.save_every
n_process = args.n_process
adv_fraction = args.adv_fraction
step_size = args.step_size
gae_lambda = args.gae_lambda

savename = '/your-folder/'+ args.save_file
res_D = pickle.load(open(savename,'rb'))
# const_test_rew_summary = res_D['zero_test']
# rand_test_rew_summary = res_D['rand_test']
# step_test_rew_summary = res_D['step_test']
# rand_step_test_rew_summary = res_D['rand_step_test']
# adv_test_rew_summary = res_D['adv_test']




dicts = {}
start_time = time.time()
## Looping over experiments to carry out ##
for ne in range(n_exps):
    ## Environment definition ##
    ## The second argument in GymEnv defines the relative magnitude of adversary. For testing we set this to 1.0.
    env = normalize(GymEnv(env_name, adv_fraction))
    env_orig = normalize(GymEnv(env_name, 1.0))

    ## Protagonist policy definition ##
    pro_policy = res_D['pro_policy']


    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Zero Adversary for the protagonist training ##
    zero_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val = 0.0
    )

    # Adversary policy definition ##
    adv_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=False
    )

    # adv_policy = res_D['worst_adv_policy']
    adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Initializing the parallel sampler ##
    parallel_sampler.initialize(n_process)

    ## Optimizer for the Protagonist ##
    pro_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=zero_adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_pro_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=True
    )

    ## Optimizer for the Adversary ##
    adv_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_adv_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=False,
        scope='adversary_optim'
    )

    ## Setting up summaries for testing for a specific training instance ##
    pro_rews = []
    adv_rews = []
    all_rews = []
    const_testing_rews = []
    const_testing_rews.append(test_const_adv(env_orig, pro_policy, path_length=path_length))
    rand_testing_rews = []
    rand_testing_rews.append(test_rand_adv(env_orig, pro_policy, path_length=path_length))
    step_testing_rews = []
    step_testing_rews.append(test_step_adv(env_orig, pro_policy, path_length=path_length))
    rand_step_testing_rews = []
    rand_step_testing_rews.append(test_rand_step_adv(env_orig, pro_policy, path_length=path_length))
    adv_testing_rews = []
    adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))

    protag_rew = []
    ## Beginning alternating optimization ##
    for ni in range(n_itr):
        logger.log('\n\n\n####expNO{} global itr# {} n_pro_itr# {}####\n\n\n'.format(ne,ni,args.n_pro_itr))
        
        ## Train Adversary
        adv_algo.train()
        adv_rews += adv_algo.rews; all_rews += adv_algo.rews;
        logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))
        
      

        protag_rew.append(np.array(adv_algo.rews).mean())


        # if ni % 50 == 0:
        #     dicts['protag_rew'] = protag_rew
        #     # dict['advers_rew'] = advers_rew
                    

        #     df = pd.DataFrame(dicts) 
        #     df.to_csv('walker2d-v1_single.csv') 
        #     print('save reward csv')


    
    ## Shutting down the optimizer ##
    pro_algo.shutdown_worker()
    adv_algo.shutdown_worker()

    # ## Updating the test summaries over all training instances
    # const_test_rew_summary.append(const_testing_rews)
    # rand_test_rew_summary.append(rand_testing_rews)
    # step_test_rew_summary.append(step_testing_rews)
    # rand_step_test_rew_summary.append(rand_step_testing_rews)
    # adv_test_rew_summary.append(adv_testing_rews)

    # dicts['protag_rew'] = protag_rew
    # # dict['advers_rew'] = advers_rew
            

print('training time: ', time.time()-start_time)
