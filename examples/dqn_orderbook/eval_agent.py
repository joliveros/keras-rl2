#! /usr/bin/env python

from cached_property import cached_property_with_ttl
from copy import deepcopy
from datetime import timedelta, datetime
from examples.dqn_orderbook.symbol_agent import SymbolAgent
from exchange_data.emitters import Messenger
from exchange_data.models.resnet.study_wrapper import StudyWrapper
from pytimeparse.timeparse import timeparse

import alog
import click
import gym
import tensorflow as tf
import pandas as pd
import tgym.envs


class NotEnoughTrialsException(Exception): pass


class SymbolEvalAgent(StudyWrapper, Messenger):

    def __init__(
            self,
            symbol,
            env_name,
            eval_interval,
            valid_interval,
            memory,
            **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        Messenger.__init__(self)

        self._kwargs = kwargs.copy()
        self.memory = memory
        self.valid_interval = valid_interval
        self.env_name = env_name
        self.symbol = symbol
        self.split_gpu()
        self.on(eval_interval, self.emit)
        self.sub([eval_interval])

    @property
    def best_trial_id(self):
        df = self.study.trials_dataframe()
        min_datetime_completed = datetime.utcnow() - timedelta(seconds=timeparse(self.valid_interval))
        df = df.loc[df['datetime_complete'] > min_datetime_completed]

        pd.set_option('display.max_rows', len(df) + 1)

        alog.info(df)

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

        alog.info(alog.pformat(params))

        agent = SymbolAgent(
            env=self.env,
            nb_steps=2,
            policy_value_max=0.25,
            random_frame_start=False,
            trial_id=self.best_trial_id,
            **params)

        agent.load_weights()

        return agent.agent

    @property
    def best_trial_params(self):
        best_trial_id = self.best_trial_id
        df = self.study.trials_dataframe()

        trial_row = df.iloc[best_trial_id]

        return trial_row['user_attrs_params']

    @property
    def env(self):
        params = self.best_trial_params
        params = {**params, **self._kwargs}

        params.pop('cache', None)
        params.pop('interval', None)
        params.pop('offset_interval', None)
        params.pop('random_frame_start', None)

        interval = self.interval_for_env(params)

        return gym.make(self.env_name,
                        random_frame_start=False,
                        interval=interval,
                        **params)

    def interval_for_env(self, params):
        group_by = params['group_by']
        sequence_length = params['sequence_length']
        interval = f'{timeparse(group_by) * sequence_length * self._kwargs["window_length"]}s'
        return interval

    def emit(self, *args):
        env = self.env
        env.reset()

        done = False

        while not done:
            obs, reward, _done, meta = env.step(0)
            prediction = str(self.agent.forward(deepcopy(obs)))
            self.agent.backward(0.0, terminal=False)
            done = _done

        alog.info(prediction)

        self.publish(f'{self.symbol}_prediction', prediction)

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
@click.option('--eval-interval', '-e', default='30s', type=str)
@click.option('--interval', '-i', default='1m', type=str)
@click.option('--memory', '-m', default=400, type=int)
@click.option('--window-length', default=3, type=int)
@click.option('--offset-interval', default='0h', type=str)
@click.option('--summary-interval', default=1, type=int)
@click.option('--valid-interval', default='30m', type=str)
def main(**kwargs):
    SymbolEvalAgent(**kwargs)


if __name__ == '__main__':
    main()
