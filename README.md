# Variational Adversarial Training Towards Policies with Improved Robustness

### <p align="center">[AISTATS 2025]</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=AEJgfq0AAAAJ&hl=en">Juncheng Dong</a><sup>*</sup> ·
  <a href="https://hlhsu.github.io/">Hao-Lun Hsu</a><sup>*</sup> ·
  <a href="[https://hlhsu.github.io/](https://people.duke.edu/~mp275/](https://people.duke.edu/~vt45/)">Vahid Tarokh</a> ·
  <a href="https://hlhsu.github.io/](https://people.duke.edu/~mp275/">Miroslav Pajic</a> ·
</p>
<p align="center">
Duke University (<sup>*</sup>indicates equal contribution)
</p>


This repo contains code for training RL agents with adversarial disturbance agents in our work on Advancing the Robustness of Optimal Policies through Strategic Design of Adversaries. We build heavily on the OpenAI rllab repo and utilize the repo: ([RARL](https://github.com/lerrel/rllab-adv/tree/master) for baseline comparison. We also implement from the repo ([SVPG](https://github.com/montrealrobotics/active-domainrand) 



## Installation instructions

Since we build upon the [rllab](https://github.com/openai/rllab) package for the optimizers, the installation process is similar to `rllab's` manual installation. Most of the packages are virtually installated in the anaconda `rllab3-adv` enivronment.

- Dependencies for scipy:

```
sudo apt-get build-dep python-scipy
```

- Install python modules:

```
conda env create -f environment.yml
```

- [Install MuJoCo](https://github.com/openai/mujoco-py)

- Add `rllab-adv` to your `PYTHONPATH`.

```
export PYTHONPATH=<PATH_TO_RLLAB_ADV>:$PYTHONPATH
```

## Example

```python
# Please change the save path to your own folder before running the example
# Enter the anaconda virtual environment
source activate rllab3-adv
# Train on HopperAdv-v1 using TRPO and single adversary
python adversarial/scripts/train_rarl_adversary.py --env HopperAdv-v1 --folder ~/rllab-adv/results --rl_type trpo --algo_type single

# Train on HopperAdv-v1 using PPO and ROSE-S
python adversarial/scripts/train_population_adversary.py --env HopperAdv-v1 --folder ~/rllab-adv/results --rl_type ppo --algo_type percent_worst
```


