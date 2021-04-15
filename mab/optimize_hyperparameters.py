import numpy as np
import os
import json
import random
import types
import inspect
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool
import time
from policies import *
from models import *
from utils import better_truncnorm
from run import *

path = os.environ['HOME']+"/data/"
#timestr = time.strftime("%Y%m%d-%H%M%S")
file_id = 1

def json_parse_objects(obj):
    if type(obj)==types.FunctionType:
        return inspect.getsource(obj).strip()
    else:
        return obj

def run_thread(token, mab_name, policy_name, file_id):
    np.random.seed(file_id+token)
    res = 1000
    reward_function = np.sin(np.linspace(0, 20, res)) * (np.linspace(0, 20, res) - 20) ** 2
    reward_function = (reward_function - reward_function.min()) / (
            reward_function.max() * 1.5 - reward_function.min()) + 0.11

    mab = {}
    mab['gaussian'] = GaussianMAB([(mu, 0.1) for mu in np.arange(1, 11) * 0.06 + 0.2])
    mab['nonstationary_trunc_gauss'] = NonStationaryTruncatedGaussianMAB(
        [(mu, 0.1) for mu in np.arange(1, 11) * 0.06 + 0.2])
    mab['gaussian_lipschitz'] = GaussianMAB(list(zip(reward_function,
                                                     [0.1] * len(reward_function))))
    mab['nonstationary_lipschitz'] = NonStationaryTruncatedGaussianMAB(list(zip(reward_function,
                                                                                [0.1] * len(reward_function))))
    mab['correlated'] = CorrelatedNonstationaryGaussianMAB([0.9] * 100, alpha=0.6, forgetting_rate=0.001)

    alpha = lambda n: 1 / n

    policies = {'zooming3': {'policy': adaptive_zooming3, 'label': 'Adaptive Zooming - 3rd attempt', 'data': [],
                             'params': {'alpha': np.concatenate((np.arange(1,10)*0.1,
                                                                 np.arange(1,10),
                                                                 np.arange(1,10)*10)),
                                        'beta': np.concatenate((np.arange(1,10)*0.1,
                                                                np.arange(1,10),
                                                                np.arange(1,10)*10)), 'active_set': []}},
                'eps': {'policy': epsilonGreedy, 'label': '$\epsilon$-greedy', 'data': [],
                        'params': {'epsilon': np.concatenate(([0],
                                                              np.arange(1,10)*0.01,
                                                              np.arange(1,10)*0.01+0.1)), 'alpha': alpha}},
                'exp3': {'policy': exp3, 'label': 'Exp3', 'data': [], 'params': {'eta': np.linspace(0,1,41),
                                                                                 'epsilon': np.linspace(0,1,41)}},
                'exp3_loss_est': {'policy': exp3_loss_est, 'label': 'Exp3_loss_est', 'data': [],
                                  'params': {'eta': np.linspace(0,1,41), 'epsilon': np.linspace(0,1,41)}},
                'exp3_s': {'policy': exp3_s, 'label': 'Exp3_S', 'data': [],
                           'params': {'eta': np.linspace(0,1,41), 'beta': np.linspace(0,0.5,11), 'epsilon': np.linspace(0,1,41)}},
                'zooming3_a': {'policy': adaptive_zooming3, 'label': 'Adaptive Zooming', 'data': [],
                                  'params': {'alpha': np.linspace(0,1000,41), 'beta': np.linspace(0,100,41), 'active_set': []}},
                'zooming3_b': {'policy': adaptive_zooming3, 'label': 'Adaptive Zooming', 'data': [],
                               'params': {'alpha': np.linspace(0,1000,41), 'beta': np.linspace(0,5,41), 'active_set': []}},
                }

    steps = 1000

    for _ in range(2000):
        # sample from hyperparameters randomly
        params = {k: (np.random.choice(v) if isinstance(v, Iterable) and len(v) > 0 else v)
                  for k, v in policies[policy_name]['params'].items()}
        result = run(mab[mab_name], policy=policies[policy_name]['policy'], steps=steps, **params)
        output = {'mab': mab_name, 'name': policy_name, 'result': result[-1, 2], 'result2': np.sum(np.cumsum(result[:, 1]))/steps}
        output.update(params)
        print(output)
        with open(path + "mab_hyperopt_{}_{}.json".format(file_id, token), "a") as f:
            json.dump(output, f, sort_keys=True, default=json_parse_objects)
            f.write("\n")

if __name__=="__main__":
    num_threads = 48

    file_id = random.randint(1000, 2000)
#    for mab_name in ['gaussian', 'nonstationary_trunc_gauss',
#                     'gaussian_lipschitz', 'nonstationary_lipschitz', 'correlated']:
    for mab_name in ['correlated']:
        for policy_name in ['eps', 'zooming3_a', 'zooming3_b', 'exp3', 'exp3_loss_est', 'exp3_s']:
            prt = partial(run_thread, mab_name=mab_name, policy_name=policy_name, file_id=file_id)
            Pool(num_threads).map(prt, range(num_threads))
