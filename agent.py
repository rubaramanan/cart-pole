from __future__ import division, absolute_import, print_function

import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network

from train import train_env, fc_layer_params, learning_rate


def create_agent():
    # agent
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()
