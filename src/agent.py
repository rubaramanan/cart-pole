from __future__ import division, absolute_import, print_function

import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network

from conf.config import ReinforceConfig


def create_agent(config: ReinforceConfig):
    # agent
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        config.train_env.observation_spec(),
        config.train_env.action_spec(),
        fc_layer_params=config.fc_layer_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    train_step_counter = tf.Variable(0)
    config.tf_agent = reinforce_agent.ReinforceAgent(
        config.train_env.time_step_spec(),
        config.train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    config.tf_agent.initialize()
