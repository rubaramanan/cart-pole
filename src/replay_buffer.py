from __future__ import division, absolute_import, print_function

import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec

from conf.config import ReinforceConfig


def get_replay_buffer(config: ReinforceConfig):
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        config.tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=config.replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)
    reverb_server = reverb.Server([table], port=42605)
    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        config.tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)
    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        config.replay_buffer_capacity
    )
    return reverb_server, replay_buffer, rb_observer
