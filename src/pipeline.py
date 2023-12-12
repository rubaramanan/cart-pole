from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
from tf_agents.utils import common

from conf.config import ReinforceConfig
from src.agent import create_agent
from src.environment import get_env
from src.train import train_agent


def run_agent_pipeline(config: ReinforceConfig):
    # def get agent environment
    get_env(config)
    # create agent
    create_agent(config)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    config.tf_agent.train = common.function(config.tf_agent.train)
    # Reset the train step
    config.tf_agent.train_step_counter.assign(0)
    # train agent
    returns = train_agent(config)
    _ = [r for r in returns if r]
    config.reverb_server.stop()

    # write_evaluation_video(config)
    with imageio.get_writer(config.video_filename, format='FFMPEG', fps=60) as video:
        for _ in range(config.num_episodes):
            time_step = config.eval_env.reset()
            video.append_data(config.eval_py_env.render())
            while not time_step.is_last():
                action_step = config.tf_agent.policy.action(time_step)
                time_step = config.eval_env.step(action_step.action)
                video.append_data(config.eval_py_env.render())

    print(f"Please check the {config.video_filename} file , to evaluation recording.")

    # config.reverb_server.wait()
    # config.reverb_server = reverb.Server([table], port=42605)
    config.train_env.close()
    config.train_py_env.close()
    config.eval_env.close()
    config.eval_py_env.close()
    # config.reverb_server._port = 42605
