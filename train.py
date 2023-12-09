from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
import matplotlib.pyplot as plt
import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from agent import create_agent
from environment import get_env
from utils import compute_avg_return, collect_episode, embed_mp4

# Set up a virtual display for rendering OpenAI gym environments.
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# Hyper parameters
env_name = "CartPole-v0"  # @param {type:"string"}
num_iterations = 250  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}

train_env, train_py_env, eval_env, eval_py_env = get_env(env_name)

tf_agent = create_agent()

# policies
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)

# try:
#     %%time
# except:
#     pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)

num_episodes = 3
video_filename = 'imageio.mp4'
with imageio.get_writer(video_filename, fps=60) as video:
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        video.append_data(eval_py_env.render())
        while not time_step.is_last():
            action_step = tf_agent.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(eval_py_env.render())

embed_mp4(video_filename)
