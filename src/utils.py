from __future__ import division, absolute_import, print_function

import base64

import IPython
import imageio
from matplotlib import pyplot as plt
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy

from conf.config import ReinforceConfig


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_episode(config: ReinforceConfig):
    driver = py_driver.PyDriver(
        config.train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            config.tf_agent.collect_policy, use_tf_function=True),
        [config.rb_observer],
        max_episodes=config.collect_episodes_per_iteration)
    initial_time_step = config.train_py_env.reset()
    driver.run(initial_time_step)


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def write_evaluation_video(config: ReinforceConfig):
    with imageio.get_writer(config.video_filename, format='FFMPEG', fps=60) as video:
        for _ in range(config.num_episodes):
            time_step = config.eval_env.reset()
            video.append_data(config.eval_py_env.render())
            while not time_step.is_last():
                action_step = config.tf_agent.policy.action(time_step)
                time_step = config.eval_env.step(action_step.action)
                video.append_data(config.eval_py_env.render())


def plot_agent_step(config: ReinforceConfig, returns):
    steps = range(0, config.num_iterations + 1, config.eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=250)
