#! /usr/bin/env python

from __future__ import division
from exchange_data.models.resnet.model import Model as ResnetModel
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import alog
import argparse
import gym
import numpy as np
import tensorflow.keras.backend as K
import tgym.envs




class AtariProcessor(Processor):
    # def process_observation(self, observation):
    #     assert observation.ndim == 3  # (height, width, channel)
    #     img = Image.fromarray(observation)
    #     img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    #     processed_observation = np.array(img)
    #     assert processed_observation.shape == INPUT_SHAPE
    #     return processed_observation.astype('uint8')  # saves storage in experience memory
    #
    # def process_state_batch(self, batch):
    #     # We could perform this processing step in `process_observation`. In this case, however,
    #     # we would need to store a `float32` array instead, which is 4x more memory intensive than
    #     # an `uint8` array. This matters if we store 1M observations.
    #     processed_batch = batch.astype('float32') / 255.
    #     return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='orderbook-frame-env-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
kwargs = dict(
    database_name='binance_futures',
    depth=84,
    interval='6h',
    group_by='30s',
    sequence_length=32,
    symbol='ETCUSDT',
    window_size='2m',
    summary_interval=4,
    min_change=0.01
)

WINDOW_LENGTH = 4
INPUT_SHAPE = (WINDOW_LENGTH, kwargs['sequence_length'], kwargs['depth'] * 2, 1)

env = gym.make(args.env_name, **kwargs)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = INPUT_SHAPE
alog.info(input_shape)

model = ResnetModel(
    input_shape=input_shape,
    num_categories=2,
    include_last=False,
    batch_size=2,
    **kwargs
)

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=10000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(
    delta_clip=1.,
    gamma=.99,
    memory=memory,
    model=model,
    nb_actions=nb_actions,
    nb_steps_warmup=1000,
    policy=policy,
    processor=processor,
    target_model_update=1000,
    train_interval=12,
)

dqn.compile(Adam(lr=.0000025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
    weights_filename = f'dqn_{args.env_name}_weights.h5f'
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = f'dqn_{args.env_name}_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, verbose=2, callbacks=callbacks, nb_steps=1750000, log_interval=1000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)

elif args.mode == 'test':
    weights_filename = f'dqn_{args.env_name}_weights.h5f'
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
