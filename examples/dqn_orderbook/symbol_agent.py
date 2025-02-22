import shutil

from bitmex_websocket.constants import NoValue
from optuna import Study, Trial

from examples.dqn_orderbook.processor import OrderBookFrameProcessor
# from exchange_data.models.video_cnn import Model
from exchange_data.models.resnet.model import Model
from pathlib import Path
from rl.agents import DQNAgent
from rl.callbacks import FileLogger
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.python.keras.callbacks import TensorBoard, History
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import SGD

import tensorflow as tf


import math
import alog
import gym
import numpy as np


class Optimizer(NoValue):
    Adadelta = 0
    Adam = 1
    Adamax = 2
    SGD = 3


class SymbolAgent(object):
    def __init__(
        self,
        env,
        beta_1,
        beta_2,
        nb_steps,
        nb_steps_2,
        symbol,
        policy_value_max,
        policy_value_min,
        train_recent_data,
        trial,
        clear_dir=False,
        trade_ratio=1/6,
        env2=None,
        optimizer: int = 2,
        cache_limit=6089,
        eps_greedy_policy_steps=34650,
        lr=1.852169e-07,
        test_env=None,
        window_length=1,
        action_repetition=3,
        **kwargs
    ):
        kwargs['symbol'] = symbol
        self.clear_dir = clear_dir
        self.trial: Trial = trial
        self._kwargs = kwargs
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.symbol = symbol
        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'
        self._optimizer = optimizer
        self.lr = lr
        self.nb_steps = nb_steps
        self.nb_steps_2 = nb_steps_2
        self.env = env
        self.env2 = env2
        self.eps_greedy_policy_steps = eps_greedy_policy_steps
        self.test_env = test_env
        self.train_recent_data = train_recent_data
        self.action_repetition = action_repetition
        self.trade_ratio = trade_ratio

        input_shape = (window_length, 72, 72, 1)
        
        self.env.seed(1)
        nb_actions = self.env.action_space.n

        model = Model(
            input_shape=input_shape,
            **kwargs
        )

        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=cache_limit, window_length=window_length)
        processor = OrderBookFrameProcessor()
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr='eps',
            nb_steps=int(self.eps_greedy_policy_steps),
            value_max=policy_value_max,
            value_min=policy_value_min,
            value_test=0.0
        )

        self.agent = DQNAgent(
            # delta_clip=1.,
            # dueling_type='max',
            # enable_double_dqn=True,
            # enable_dueling_network=True,
            # gamma=.99,
            memory=memory,
            model=model,
            nb_actions=nb_actions,
            nb_steps_warmup=800,
            policy=policy,
            processor=processor,
            **kwargs,
        )

        self.agent.compile(self.optimizer, metrics=['mae'])

    @property
    def optimizer(self):
        optimizer = Optimizer(self._optimizer)

        if optimizer == Optimizer.Adam:
            return Adam(lr=self.lr)
        elif optimizer == Optimizer.Adadelta:
            return Adadelta(learning_rate=self.lr)
        elif optimizer == Optimizer.Adamax:
            return Adamax(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        elif optimizer == Optimizer.SGD:
            return SGD(learning_rate=self.lr)
        raise Exception()

    @property
    def weights_filename(self):
        return self.base_model_dir + f'/{self.trial.number}/weights.h5f'

    def load_weights(self):
        self.agent.load_weights(self.weights_filename)

    @property
    def trial_dir(self):
        return str(Path(self.base_model_dir) / str(self.trial.number))

    def run(self):
        tb_callback = TensorBoard(
            embeddings_freq=0,
            embeddings_metadata=None,
            histogram_freq=0,
            log_dir=str(Path(self.base_model_dir) / str(self.trial.number)),
            profile_batch=2,
            update_freq='epoch',
            write_graph=True,
            write_images=False,
        )

        tb_callback._should_trace = True

        callbacks = [tb_callback]

        # callbacks += [FileLogger(log_filename, interval=100)]

        action_repetition = self.action_repetition

        self.agent.fit(self.env, verbose=2, callbacks=callbacks, nb_steps=self.nb_steps,
                       log_interval=1, action_repetition=action_repetition)

        if self.train_recent_data:
            self.agent.fit(self.env2, verbose=2, callbacks=callbacks, nb_steps=self.nb_steps_2, log_interval=1,
                           action_repetition=action_repetition)

        if self.clear_dir:
            shutil.rmtree(self.trial_dir)
        else:
            # After training is done, we save the final weights one more time.
            self.agent.save_weights(self.weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 1 episodes.
        history: History = self.agent.test(self.test_env,
                                           verbose=2,
                                           nb_episodes=2,
                                           visualize=False,
                                           nb_max_start_steps=10,
                                           action_repetition=action_repetition)

        # ep_rewards = history.history['episode_reward']
        # nb_steps = history.history['nb_steps']
        # ep_rewards_avg = sum(ep_rewards) / len(ep_rewards)
        # nb_steps_avg = sum(nb_steps) / len(nb_steps)

        capital = [info['capital'] / info['num_steps'] for info in history.history['info']]

        def flatten(l):
            return [item for sublist in l for item in sublist]

        capital_avg = sum(capital) / len(capital)

        trades = []
        trades += [info['trades'] for info in history.history['info']]
        trades = flatten(trades)
        trades = list({hash(t): t for t in trades}.values())
        pos_trades = len([t.pnl for t in trades if t.pnl > 0])
        trade_len = len(trades)
        alog.info(dict(pos_trades=pos_trades, trade_len=trade_len))

        if trade_len > 0:
            trade_ratio = pos_trades / trade_len
        else:
            trade_ratio = 0

        self.trial.set_user_attr('trades', len(trades))
        capital_ratio = (1 - self.trade_ratio)
        pos_trades = (pos_trades **(1/50)) - 1

        # return capital_avg

        return (capital_avg * capital_ratio) + (trade_ratio * self.trade_ratio)

