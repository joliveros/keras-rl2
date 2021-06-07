from cached_property import cached_property

from examples.dqn_orderbook.processor import OrderBookFrameProcessor
from exchange_data.models.resnet.model import Model as ResnetModel
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.python.keras.callbacks import TensorBoard, History
from tensorflow.python.keras.optimizer_v2.adam import Adam
from pathlib import Path

import alog
import gym
import numpy as np
import tensorflow as tf


class SymbolAgent(object):
    def __init__(
        self,
        symbol,
        nb_steps,
        trial_id,
        env_name,
        env,
        test_env,
        cache_limit,
        window_length=36,
        value_max=1.0,
        value_min=0.1,
        lr=.000025,
        **kwargs
    ):
        kwargs['symbol'] = symbol
        self._kwargs = kwargs

        self.symbol = symbol
        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'
        self.trial_id = trial_id
        self.lr = lr
        self.nb_steps = nb_steps
        self.env_name = env_name
        self.env = env
        self.test_env = test_env

        INPUT_SHAPE = (window_length, kwargs['sequence_length'], kwargs['depth'] * 2, 1)
        np.random.seed(123)
        self.env.seed(123)
        nb_actions = self.env.action_space.n

        # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
        input_shape = INPUT_SHAPE
        alog.info(input_shape)

        model = ResnetModel(
            input_shape=input_shape,
            num_categories=2,
            include_last=False,
            **kwargs
        )

        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=cache_limit, window_length=window_length)
        processor = OrderBookFrameProcessor()

        # Select a policy. We use eps-greedy action selection, which means that a random action is selected
        # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
        # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
        # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
        # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                      attr='eps',
                                      nb_steps=self.nb_steps,
                                      value_max=value_max,
                                      value_min=value_min,
                                      value_test=0.0,
                                      )

        # The trade-off between exploration and exploitation is difficult and an on-going research topic.
        # If you want, you can experiment with the parameters or use a different policy. Another popular one
        # is Boltzmann-style exploration:
        # policy = BoltzmannQPolicy(tau=1.)
        # Feel free to give it a try!

        self.agent = DQNAgent(
            delta_clip=1.,
            gamma=.99,
            memory=memory,
            model=model,
            nb_actions=nb_actions,
            nb_steps_warmup=100,
            policy=policy,
            processor=processor,
            **kwargs,
        )

        self.agent.compile(Adam(lr=self.lr), metrics=['mae'])

    def run(self):
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
        weights_filename = f'dqn_{self.env_name}_weights.h5f'
        checkpoint_weights_filename = 'dqn_' + self.env_name + '_weights_{step}.h5f'

        tb_callback = TensorBoard(
            embeddings_freq=0,
            embeddings_metadata=None,
            histogram_freq=0,
            log_dir=str(Path(self.base_model_dir) / self.trial_id),
            profile_batch=2,
            update_freq='epoch',
            write_graph=True,
            write_images=False,
        )

        tb_callback._should_trace = True

        callbacks = [tb_callback]

        # callbacks += [FileLogger(log_filename, interval=100)]
        self.agent.fit(self.env, verbose=2, callbacks=callbacks, nb_steps=self.nb_steps, log_interval=1000)

        # After training is done, we save the final weights one more time.
        self.agent.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 1 episodes.
        history: History = self.agent.test(self.test_env, verbose=2, nb_episodes=2, visualize=False, nb_max_start_steps=10)

        ep_rewards = history.history['episode_reward']

        return sum(ep_rewards) / len(ep_rewards)
