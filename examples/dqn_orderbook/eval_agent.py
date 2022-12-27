#! /usr/bin/env python
import sys

from cached_property import cached_property_with_ttl
from copy import deepcopy
from datetime import timedelta, datetime
from examples.dqn_orderbook.symbol_agent import SymbolAgent
from exchange_data.emitters import Messenger
from exchange_data.models.study_wrapper import StudyWrapper
from pytimeparse.timeparse import timeparse
from collections import deque

import alog
import click
import gym
import tensorflow as tf
import pandas as pd
import tgym.envs
import numpy as np

class NotEnoughTrialsException(Exception): pass


class SymbolEvalAgent(StudyWrapper, Messenger):

    def __init__(
            self,
            symbol,
            env_name,
            interval,
            eval_interval,
            valid_interval,
            memory,
            window_length,
            once,
            **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        Messenger.__init__(self)

        self.interval = interval
        self.once = once
        self._kwargs = kwargs.copy()
        self.window_length = window_length
        self.memory = memory
        self.valid_interval = valid_interval
        self.env_name = env_name
        self.symbol = symbol
        self.split_gpu()

        if once:
            self.emit()
            sys.exit(0)
        else:
            self.on(eval_interval, self.emit)
            self.sub([eval_interval])

    @property
    def best_trial_id(self):
        df = self.study.trials_dataframe()
        min_datetime_completed = datetime.utcnow() - timedelta(seconds=timeparse(self.valid_interval))
        df = df.loc[df['datetime_complete'] > min_datetime_completed]
        df = df[df['value'] > 0.0]

        pd.set_option('display.max_rows', len(df))

        alog.info(df[['user_attrs_tuned', 'user_attrs_tuned', 'value']])

        if not df.empty:
            df = df[df['user_attrs_tuned'] == False]

        # alog.info(df[['']])

        raise Exception()

        pd.set_option('display.max_rows', len(df) + 1)
       
        if df.shape[0] < 1:
            raise NotEnoughTrialsException()
        else:
            return df['number'].iloc[-1]

    @cached_property_with_ttl(ttl=60 * 15)
    def agent(self):
        params = self.best_trial_params
        params.pop('cache', None)
        params.pop('interval', None)
        params.pop('nb_steps', None)
        params.pop('offset_interval', None)
        params.pop('random_frame_start', None)

        agent = SymbolAgent(
            env=self.env,
            nb_steps=2,
            policy_value_max=0.25,
            random_frame_start=False,
            trial_id=self.best_trial_id,
            study=self.study,
            **params)

        agent.load_weights()

        return agent.agent

    @property
    def best_trial_params(self):
        best_trial_id = self.best_trial_id

        df = self.study.trials_dataframe()

        trial_row = df.loc[df['number'] == best_trial_id].tail(1)

        return trial_row['user_attrs_params'].to_dict()[best_trial_id]

    @property
    def env(self):
        params = self.best_trial_params
        params = {**params, **self._kwargs}

        params.pop('cache', None)
        params.pop('interval', None)
        params.pop('offset_interval', None)
        params.pop('random_frame_start', None)

        params['is_test'] = True

        interval = self.interval_for_env(params)

        return gym.make(self.env_name,
                        random_frame_start=False,
                        interval=interval,
                        **params)

    def interval_for_env(self, params):
        group_by = params['group_by']
        sequence_length = params['sequence_length']
        seconds = timeparse(group_by) * \
                  (sequence_length + self.window_length)

        interval = timeparse(self.interval)

        interval = f'{seconds + interval}s'

        return interval

    def emit(self, *args):
        env = self.env
        env.reset()

        done = False
        _obs = deque(maxlen=self.window_length)

        prediction = 0
        action = 0

        while not done:
            obs, reward, _done, meta = env.step(prediction)
            action = meta['action']
            prediction = int(self.agent.forward(deepcopy(obs)))
            self.agent.backward(0.0, terminal=False)
            done = _done

        pd.set_option('display.max_rows', 12)

        alog.info(env.frame)

        alog.info(action)

        self.publish(f'{self.symbol}_prediction', str(action))

    def split_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0 and self.memory > 0:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [
                    tf.config.
                        LogicalDeviceConfiguration(memory_limit=self.memory),
                ])


@click.command()
@click.argument('symbol', type=str)
@click.option('--database-name', default='binance_futures', type=str)
@click.option('--env-name', default='orderbook-frame-env-v0', type=str)
@click.option('--eval-interval', '-e', default='2m', type=str)
@click.option('--interval', '-i', default='1m', type=str)
@click.option('--memory', '-m', default=400, type=int)
@click.option('--window-length', '-w', default=3, type=int)
@click.option('--offset-interval', default='0h', type=str)
@click.option('--summary-interval', default=1, type=int)
@click.option('--valid-interval', default='30m', type=str)
@click.option('--once', is_flag=True)
def main(**kwargs):
    SymbolEvalAgent(**kwargs)


if __name__ == '__main__':
    main()
