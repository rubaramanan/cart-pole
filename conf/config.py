from dataclasses import dataclass
from typing import Any


@dataclass
class ReinforceConfig:
    env_name: str = "CartPole-v0"  # @param {type:"string"}
    num_iterations: int = 50  # @param {type:"integer"}
    collect_episodes_per_iteration: int = 2  # @param {type:"integer"}
    replay_buffer_capacity: int = 2000  # @param {type:"integer"}
    log_interval: int = 25  # @param {type:"integer"}
    num_eval_episodes: int = 10  # @param {type:"integer"}
    eval_interval: int = 50  # @param {type:"integer"}
    num_episodes: int = 3
    fc_layer_params: tuple = (100,)
    learning_rate: float = 0.001
    video_filename: str = 'imageio.mp4'
    train_env: Any = None
    train_py_env: Any = None
    eval_env: Any = None
    eval_py_env: Any = None
    tf_agent: Any = None
    rb_observer: Any = None
    replay_buffer: Any = None
    reverb_server: Any = None
