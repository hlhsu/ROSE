import sys
sys.path.append('../../')
from rllab.algos.multi_trpo import TRPO
from rllab.algos.multi_ppo import PPO
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

def concat_sample(sample_data, concat_sample_data):
    data_list = ['observations', 'actions', 'advantages', 'rewards', 'returns']
    if concat_sample_data == {}:
        for item in data_list:
            concat_sample_data[item] = sample_data[item]
        
        concat_sample_data['paths'] = sample_data['paths']
        concat_sample_data['agent_infos'] = sample_data['agent_infos']

        
    else:
        for item in data_list:
            concat_sample_data[item] = np.concatenate((concat_sample_data[item], sample_data[item]), axis=0)
        count = 0
        for p in (sample_data['paths']):
            count +=1
            concat_sample_data['paths'].append(p)

        total_count = 0
        for p in (concat_sample_data['paths']):
            total_count +=1
        
        concat_sample_data['agent_infos']['mean'] =np.concatenate((concat_sample_data['agent_infos']['mean'], sample_data['agent_infos']['mean']), axis=0)
        concat_sample_data['agent_infos']['log_std'] =np.concatenate((concat_sample_data['agent_infos']['log_std'], sample_data['agent_infos']['log_std']), axis=0)
    return concat_sample_data
   

    



## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True, help='Name of adversarial environment')
parser.add_argument('--path_length', type=int, default=1000, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[128,128,128], help='layer definition')
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
parser.add_argument('--algo_type', type=str, default='popu', help='popu, percent_all, percent_worst')
parser.add_argument('--rl_type', type=str, default='trpo', help= 'trpo, ppo')

## Parsing Arguments ##
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
save_dir = args.folder
algo_type = args.algo_type
rl_type = args.rl_type


if rl_type == 'trpo':
    RL = TRPO
    print('rl type: ', rl_type)
elif rl_type == 'ppo':
    RL = PPO
    print('rl type: ', rl_type)



## Initializing summaries for the tests ##
const_test_rew_summary = []
rand_test_rew_summary = []
step_test_rew_summary = []
rand_step_test_rew_summary = []
adv_test_rew_summary = []

## Preparing file to save results in ##
save_prefix = 'env-{}_Exp{}_Itr{}_BS{}_Adv{}_stp{}_lam{}_{}_{}_{}_v1'.format(env_name, n_exps, n_itr, batch_size, adv_fraction, step_size, gae_lambda, random.randint(0,1000000), algo_type, rl_type)
save_name = save_dir+'/'+save_prefix+'.p'

## Looping over experiments to carry out ##
dicts = {}
start_time = time.time()
for ne in range(n_exps):
    ## Environment definition ##
    ## The second argument in GymEnv defines the relative magnitude of adversary. For testing we set this to 1.0.
    env = normalize(GymEnv(env_name, adv_fraction))
    env_orig = normalize(GymEnv(env_name, 1.0))

    ## Protagonist policy definition ##
    pro_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=True
    )
    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Zero Adversary for the protagonist training ##
    zero_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val = 0.0
    )

    ## Adversary policy definition ##
    adv_policy_dict = dict()
    adv_baseline_dict = dict()
    for i in range(10):

        adv_policy_dict[i] = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=layer_size,
            is_protagonist=False
        )
        adv_baseline_dict[i] = LinearFeatureBaseline(env_spec=env.spec)


    ## Initializing the parallel sampler ##
    parallel_sampler.initialize(n_process)


    ## Optimizer for the Adversary ##
    adv_algo_dict = dict()
    adv_perform_dict = dict()

    for i in range(10): # number of adv num
        adv_algo_dict[i] = RL(
            env=env,
            pro_policy=pro_policy,
            adv_policy=adv_policy_dict[i],
            pro_baseline=pro_baseline,
            adv_baseline=adv_baseline_dict[i],
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
    # adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))

    protag_rew = []

    ## Beginning alternating optimization ##
    data_list = ['observations', 'actions', 'advantages', 'rewards', 'returns']
    pro_concat_sample_data = dict()
    for warm_i in range(10):
        sample_data = adv_algo_dict[warm_i].agent_sample_data(is_warm=True)
      

        adv_algo_dict[warm_i].train(sample_data)

        adv_policy_dict[warm_i] = adv_algo_dict[warm_i].policy
        adv_baseline_dict[warm_i] = adv_algo_dict[warm_i].baseline
        adv_algo_dict[warm_i].batch_size = int(batch_size/3)

        adv_perform_dict[warm_i] = test_learnt_adv(env, pro_policy, adv_policy_dict[warm_i], path_length=path_length)


        pro_algo = RL(
            env=env,
            pro_policy=pro_policy,
            adv_policy=adv_policy_dict[warm_i],
            pro_baseline=pro_baseline,
            adv_baseline=adv_baseline_dict[warm_i],
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_pro_itr,
            discount=0.995,
            gae_lambda=gae_lambda,
            step_size=step_size,
            is_protagonist=True
        )
        pro_sample_data = pro_algo.agent_sample_data(is_warm= True)
        pro_concat_sample_data = concat_sample(pro_sample_data, pro_concat_sample_data)
    
    pro_algo.train(pro_concat_sample_data)
    pro_policy = pro_algo.policy
    pro_baseline = pro_algo.baseline
    pro_env = pro_algo.env

     
       

    for ni in range(n_itr):
        logger.log('\n\n\n####expNO{} global itr# {} n_pro_itr# {}####\n\n\n'.format(ne,ni,args.n_pro_itr))

        print('algo type: ', algo_type)
        if algo_type == 'popu':
            adv_indexes = np.random.randint(len(adv_algo_dict), size = 3)
        elif algo_type == 'percent_worst' or algo_type == 'percent_all': 
            sorted_adv_perform_dict =  sorted(adv_perform_dict.items(), key=lambda item: item[1])
            sort_adv_perform_list = [item[0] for item in sorted_adv_perform_dict]
         
            adv_indexes = np.array(sort_adv_perform_list[:3])


        pro_concat_sample_data = dict()
        ## Train Adversary

        if algo_type == 'percent_all':
            for i in range(10):
                sample_data = adv_algo_dict[i].agent_sample_data()
   
                adv_algo_dict[i].train(sample_data)

                adv_policy_dict[i] = adv_algo_dict[i].policy
                adv_baseline_dict[i] = adv_algo_dict[i].baseline

              
                adv_perform_dict[i] = test_learnt_adv(env, pro_policy, adv_policy_dict[i], path_length=path_length)

        elif algo_type == 'percent_worst':  
            for i in list(adv_indexes):
                
                sample_data = adv_algo_dict[i].agent_sample_data()
    
                adv_algo_dict[i].train(sample_data)

                adv_policy_dict[i] = adv_algo_dict[i].policy
                adv_baseline_dict[i] = adv_algo_dict[i].baseline

            for i in range(10):
                adv_perform_dict[i] = test_learnt_adv(env, pro_policy, adv_policy_dict[i], path_length=path_length)
        elif algo_type == 'popu':
            for i in list(adv_indexes):
                sample_data = adv_algo_dict[i].agent_sample_data()
    
                adv_algo_dict[i].train(sample_data)

                adv_policy_dict[i] = adv_algo_dict[i].policy
                adv_baseline_dict[i] = adv_algo_dict[i].baseline
                adv_perform_dict[i] = test_learnt_adv(env, pro_policy, adv_policy_dict[i], path_length=path_length)

        ############################3 for pro agent ########################
        if algo_type != 'popu':
            

            sorted_adv_perform_dict =  sorted(adv_perform_dict.items(), key=lambda item: item[1])
            sort_adv_perform_list = [item[0] for item in sorted_adv_perform_dict]
     
            adv_indexes = np.array(sort_adv_perform_list[:3])
           

        for i in list(adv_indexes):

            pro_algo = RL(
            env=pro_env,
            pro_policy=pro_policy,
            adv_policy=adv_policy_dict[i],
            pro_baseline=pro_baseline,
            adv_baseline=adv_baseline_dict[i],
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_pro_itr,
            discount=0.995,
            gae_lambda=gae_lambda,
            step_size=step_size,
            is_protagonist=True
            )

            pro_sample_data = pro_algo.agent_sample_data()

            pro_concat_sample_data = concat_sample(pro_sample_data, pro_concat_sample_data)



        ## Train protagonist
        pro_algo.train(pro_concat_sample_data)
        pro_policy = pro_algo.policy
        pro_baseline = pro_algo.baseline
        pro_env = pro_algo.env

        


        pro_rews += pro_algo.rews; all_rews += pro_algo.rews
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
        protag_rew.append(pro_algo.rews[0])

        if algo_type != 'popu':
            for i in range(10):
                adv_perform_dict[i] = test_learnt_adv(env, pro_policy, adv_policy_dict[i], path_length=path_length)




        
        ## Test the learnt policies
        const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
        rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
        rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
        adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy_dict[i], path_length=path_length))

        if ni%afterRender==0 and ifRender==True:
            test_const_adv(env, pro_policy, path_length=path_length, n_traj=1, render=True);

        if ni!=0 and ni%save_every==0:
            ## SAVING CHECKPOINT INFO ##
            if algo_type=='popu':
                pickle.dump({'args': args,
                            'pro_policy': pro_policy,
                            'adv_policy': adv_policy_dict,
                            'zero_test': const_test_rew_summary,
                            'rand_test': rand_test_rew_summary,
                            'step_test': step_test_rew_summary,
                            'rand_step_test': rand_step_test_rew_summary,
                            'iter_save': ni,
                            'exp_save': ne,
                            'adv_test': adv_test_rew_summary}, open(save_name+'.temp','wb'))
            else:
                pickle.dump({'args': args,
                            'worst_adv_policy':adv_policy_dict[adv_indexes[0]],
                            'pro_policy': pro_policy,
                            'adv_policy': adv_policy_dict,
                            'zero_test': const_test_rew_summary,
                            'rand_test': rand_test_rew_summary,
                            'step_test': step_test_rew_summary,
                            'rand_step_test': rand_step_test_rew_summary,
                            'iter_save': ni,
                            'exp_save': ne,
                            'adv_test': adv_test_rew_summary}, open(save_name+'.temp','wb'))

    ## Shutting down the optimizer ##
    pro_algo.shutdown_worker()

    ## Updating the test summaries over all training instances
    const_test_rew_summary.append(const_testing_rews)
    rand_test_rew_summary.append(rand_testing_rews)
    step_test_rew_summary.append(step_testing_rews)
    rand_step_test_rew_summary.append(rand_step_testing_rews)
    adv_test_rew_summary.append(adv_testing_rews)

    

    dicts['protag_rew'] = protag_rew
            

    df = pd.DataFrame(dicts) 
    df.to_csv('new_{}_{}_{}_v1.csv'.format(rl_type, env_name, algo_type)) 
    print('save reward csv')

## SAVING INFO ##
if algo_type == 'popu':
    pickle.dump({'args': args,
                'pro_policy': pro_policy,
                'adv_policy': adv_policy_dict,
                'zero_test': const_test_rew_summary,
                'rand_test': rand_test_rew_summary,
                'step_test': step_test_rew_summary,
                'rand_step_test': rand_step_test_rew_summary,
                'adv_test': adv_test_rew_summary}, open(save_name,'wb'))
else:
    pickle.dump({'args': args,
                'worst_adv_policy':adv_policy_dict[adv_indexes[0]],
                'pro_policy': pro_policy,
                'adv_policy': adv_policy_dict,
                'zero_test': const_test_rew_summary,
                'rand_test': rand_test_rew_summary,
                'step_test': step_test_rew_summary,
                'rand_step_test': rand_step_test_rew_summary,
                'iter_save': ni,
                'exp_save': ne,
                'adv_test': adv_test_rew_summary}, open(save_name+'.temp','wb'))
    

logger.log('\n\n\n#### DONE ####\n\n\n')

print('training time: ', time.time()-start_time)
