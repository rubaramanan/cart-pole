from __future__ import division, absolute_import, print_function

from conf.config import ReinforceConfig
from src.replay_buffer import get_replay_buffer
from src.utils import compute_avg_return, collect_episode


def train_agent(config: ReinforceConfig):
    # Evaluate the agent's policy once before training.
    config.reverb_server, config.replay_buffer, config.rb_observer = get_replay_buffer(config)
    avg_return = compute_avg_return(config.eval_env, config.tf_agent.policy, config.num_eval_episodes)
    returns = [avg_return]

    returns.extend(train_single_step(config) for _ in range(config.num_iterations))
    return returns

    # for _ in range(num_iterations):
    #
    #     train_single_step(returns)


def train_single_step(config: ReinforceConfig):
    returns = []
    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(config)
    # Use data from the buffer and update the agent's network.
    iterator = iter(config.replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = config.tf_agent.train(experience=trajectories)
    config.replay_buffer.clear()
    step = config.tf_agent.train_step_counter.numpy()
    if step % config.log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    if step % config.eval_interval == 0:
        avg_return = compute_avg_return(config.eval_env, config.tf_agent.policy, config.num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
