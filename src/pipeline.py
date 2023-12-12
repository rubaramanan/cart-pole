from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
from tf_agents.utils import common

from conf.config import ReinforceConfig
from src.agent import create_agent
from src.environment import get_env
from src.train import train_agent
from src.utils import write_evaluation_video


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

    write_evaluation_video(config)

    print(f"Please check the {config.video_filename} file , to evaluation recording.")

    config.train_env.close()
    config.train_py_env.close()
    config.eval_env.close()
    config.eval_py_env.close()
