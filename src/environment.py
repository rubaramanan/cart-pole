from __future__ import division, absolute_import, print_function

import PIL.Image
import numpy as np
from tf_agents.environments import suite_gym, tf_py_environment

from conf.config import ReinforceConfig


def get_env(config: ReinforceConfig):
    # environment
    env = suite_gym.load(config.env_name)
    env.reset()
    PIL.Image.fromarray(env.render())
    print('Observation Spec:')
    print(env.time_step_spec().observation)
    print('Action Spec:')
    print(env.action_spec())
    time_step = env.reset()
    print('Time step:')
    print(time_step)
    action = np.array(1, dtype=np.int32)
    next_time_step = env.step(action)
    print('Next time step:')
    print(next_time_step)
    config.train_py_env = suite_gym.load(config.env_name)
    config.eval_py_env = suite_gym.load(config.env_name)
    config.train_env = tf_py_environment.TFPyEnvironment(config.train_py_env)
    config.eval_env = tf_py_environment.TFPyEnvironment(config.eval_py_env)
