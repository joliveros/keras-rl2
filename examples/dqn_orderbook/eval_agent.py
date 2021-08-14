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
import tgym.envs

class NotEnoughTrialsException(Exception): pass


class SymbolEvalAgent(StudyWrapper, Messenger):

    def __init__(self, symbol, env_name, eval_interval, valid_interval, memory, **kwargs):
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

        alog.info(df)

        if df.shape[0] < 3:
            raise NotEnoughTrialsException()
        else:
            return df['value'].idxmax()

    @cached_property_with_ttl(ttl=60*4)
    def agent(self):
        agent = SymbolAgent(symbol=self.symbol,
                            trial_id=self.best_trial_id,
                            env=self.env,
                            nb_steps=2,
                            **self._kwargs)

        agent.load_weights()

        return agent.agent

    @property
    def env(self):
        kwargs = self._kwargs.copy()
        return gym.make(self.env_name, symbol=self.symbol, **kwargs)

    def emit(self, *args):
        observation = deepcopy(self.env.reset())
        self.publish(f'{self.symbol}_prediction', str(self.agent.forward(observation)))

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
@click.option('--cache', is_flag=True)
@click.option('--database-name', default='binance_futures', type=str)
@click.option('--depth', '-d', default=28, type=int)
@click.option('--env-name', default='orderbook-frame-env-v0', type=str)
@click.option('--eval-interval', '-e', default='30s', type=str)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='1m', type=str)
@click.option('--leverage', default=1.0, type=float)
@click.option('--max-loss', default=-0.01, type=float)
@click.option('--max-negative-pnl', default=-10/100, type=float)
@click.option('--max-summary', default=30, type=int)
@click.option('--memory', '-m', default=400, type=int)
@click.option('--min-capital', default=1.0, type=float)
@click.option('--min-change', default=0.001, type=float)
@click.option('--offset-interval', default='0h', type=str)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--sequence-length', '-l', default=22, type=int)
@click.option('--summary-interval', default=4, type=int)
@click.option('--valid-interval', default='30m', type=str)
@click.option('--window-size', '-w', default='4m', type=str)
def main(**kwargs):
    SymbolEvalAgent(**kwargs)


if __name__ == '__main__':
    main()
