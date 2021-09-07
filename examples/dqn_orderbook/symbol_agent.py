from bitmex_websocket.constants import NoValue
from examples.dqn_orderbook.processor import OrderBookFrameProcessor
from exchange_data.models.resnet.model import Model as ResnetModel
from pathlib import Path
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.python.keras.callbacks import TensorBoard, History
from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.adamax import Adamax

import alog
import gym
import numpy as np


class Optimizer(NoValue):
    Adadelta = 0
    Adam = 1
    Adamax = 2


class SymbolAgent(object):
    def __init__(
        self,
        env,
        nb_steps,
        symbol,
        policy_value_max,
        optimizer: int = 0,
        cache_limit=4000,
        lr=0.006226385,
        test_env=None,
        trial_id=0,
        window_length=4,
        **kwargs
    ):
        kwargs['symbol'] = symbol
        self._kwargs = kwargs

        self.symbol = symbol
        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'
        self.trial_id = trial_id
        self._optimizer = optimizer
        self.lr = lr
        self.nb_steps = nb_steps
        self.env = env
        self.test_env = test_env

        input_shape = (window_length, kwargs['sequence_length'], kwargs['depth'] * 2, 1)
        np.random.seed(123)
        self.env.seed(123)
        nb_actions = self.env.action_space.n

        model = ResnetModel(
            input_shape=input_shape,
            **kwargs
        )

        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=cache_limit, window_length=window_length)
        processor = OrderBookFrameProcessor()
        # policy = GreedyQPolicy()

        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr='eps',
            nb_steps=int(2000),
            value_max=policy_value_max,
            value_min=0.0,
            value_test=0.0
        )

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

        alog.info(self.optimizer)

        self.agent.compile(self.optimizer, metrics=['mae'])

    @property
    def optimizer(self):
        optimizer = Optimizer(self._optimizer)

        alog.info(optimizer)

        if optimizer == Optimizer.Adam:
            return Adam(lr=self.lr)
        elif optimizer == Optimizer.Adadelta:
            return Adadelta(learning_rate=self.lr)
        elif optimizer == Optimizer.Adamax:
            return Adamax(learning_rate=self.lr)

        raise Exception()

    @property
    def weights_filename(self):
        return self.base_model_dir + f'/{self.trial_id}/weights.h5f'

    def load_weights(self):
        self.agent.load_weights(self.weights_filename)

    def run(self):
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!

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
        self.agent.save_weights(self.weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 1 episodes.
        history: History = self.agent.test(self.test_env, verbose=2, nb_episodes=2, visualize=False, nb_max_start_steps=10)

        # ep_rewards = history.history['episode_reward']
        # nb_steps = history.history['nb_steps']
        # ep_rewards_avg = sum(ep_rewards) / len(ep_rewards)
        # nb_steps_avg = sum(nb_steps) / len(nb_steps)

        capital = [info['capital'].tolist() for info in history.history['info']]
        capital_avg = sum(capital) / len(capital)

        return capital_avg
