import json
import zlib

from examples.dqn_orderbook.symbol_agent import SymbolAgent
from exchange_data import settings
from exchange_data.models.study_wrapper import StudyWrapper
from optuna import Trial
from pathlib import Path
from pytimeparse.timeparse import timeparse
from redlock import RedLockError, RedLock
from tensorflow_datasets.core.utils import nullcontext

import alog
import gym
import re
import shutil
import tensorflow as tf
import time as t
import pandas as pd


class SymbolTuner(StudyWrapper):
    current_lock_ix = 0
    hparams = None
    run_count = 0

    def __init__(self,
                 export_best,
                 session_limit,
                 env_name,
                 retry,
                 min_capital,
                 memory,
                 train_recent_data=False,
                 num_locks=2,
                 **kwargs):

        self._kwargs = kwargs.copy()
        self._kwargs['train_recent_data'] = train_recent_data

        super().__init__(**kwargs)

        StudyWrapper.__init__(self, **kwargs)

        self.train_recent_data = train_recent_data
        self.retry = retry
        self.export_best = export_best
        self.min_capital = min_capital
        self.memory = memory
        self.num_locks = num_locks
        self.env_name = env_name
        self.export_dir.mkdir(exist_ok=True)

        Path(self.best_model_dir).mkdir(parents=True, exist_ok=True)

        self.split_gpu()

        self.study.optimize(self.run, n_trials=session_limit)

    @property
    def best_model_dir(self):
        return f'{Path.home()}/.exchange-data/best_exported_models/' \
               f'{self.symbol}_export/1'

    def clear(self):
        alog.info('### clear runs ###')
        try:
            self._clear()
        except RedLockError:
            pass

    def _clear(self):
        with RedLock('clear_lock', [dict(
                host=settings.REDIS_HOST,
                db=0
        )],
                     retry_delay=timeparse('15s'),
                     retry_times=12, ttl=timeparse('1h') * 1000):
            self.study_db_path.unlink()
            self.clear_dirs()

    def clear_dirs(self):
        shutil.rmtree(str(self.export_dir), ignore_errors=True)
        Path(self.export_dir).mkdir()

        shutil.rmtree(
            f'{Path.home()}/.exchange-data/models/{self.symbol}_params',
            ignore_errors=True)

        base_dir = Path(self.base_model_dir)
        shutil.rmtree(str(base_dir), ignore_errors=True)

        if not base_dir.exists():
            base_dir.mkdir()

    @property
    def export_dir(self):
        return Path(f'{Path.home()}/.exchange-data/models/' \
                    f'{self.symbol}_export')

    @property
    def run_dir(self):
        return f'{Path.home()}/.exchange-data/models/{self.symbol}_params/' \
               f'{self.trial.number}'

    @property
    def train_lock(self):
        if self.num_locks == 0:
            return nullcontext()

        lock_name = f'train_lock_{self.current_lock_ix}'

        alog.info(f'### lock name {lock_name} ####')

        self.current_lock_ix += 1

        if self.current_lock_ix > self.num_locks - 1:
            self.current_lock_ix = 0

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=timeparse('15s'),
                       retry_times=12, ttl=timeparse('1h') * 1000)

    def run(self, *args):
        retry_relay = 10

        try:
            if self.run_count > 1:
                t.sleep(retry_relay)
            with self.train_lock:
                return self._run(*args)
            self.run_count += 1

        except RedLockError as e:
            alog.info(e)
            t.sleep(retry_relay)
            return self.run(*args)

    @property
    def env(self):
        kwargs = self._kwargs.copy()
        return gym.make(self.env_name, **kwargs)

    @property
    def env2(self):
        kwargs = self._kwargs.copy()
        kwargs['interval'] = kwargs['interval2']
        return gym.make(self.env_name, **kwargs)

    @property
    def test_env(self):
        kwargs = self._kwargs.copy()
        test_interval = kwargs['test_interval']
        kwargs['interval'] = test_interval
        kwargs['offset_interval'] = '0h'
        kwargs['is_test'] = True
        kwargs['random_frame_start'] = False
        kwargs['max_short_position_length'] = -1
        kwargs['trading_fee'] = 0.0004

        return gym.make(self.env_name, **kwargs)

    @property
    def agent(self):
        tune = self._kwargs.get('tune', False)

        hparams = dict()

        if tune:
            self.trial.set_user_attr('tuned', True)

            hparams = dict(
                memory_interval=self.trial.suggest_int('memory_interval', 1, 399),
                delta_clip=self.trial.suggest_uniform('delta_clip', 0, 99),
                gamma=self.trial.suggest_uniform('gamma', 0, 0.9999),
                enable_double_dqn=self.trial.suggest_categorical('enable_double_dqn', [True, False]),
                macd_diff_enabled=self.trial.suggest_categorical('macd_diff_enabled', [True, False]),
                dueling_type=self.trial.suggest_categorical('dueling_type', ['avg', 'max', 'naive']),
                base_filter_size=self.trial.suggest_int('base_filter_size', 4, 32),
                dense_width=self.trial.suggest_int('dense_width', 4, 396),
                block_kernel=self.trial.suggest_int('block_kernel', 1, 7),
                num_dense=self.trial.suggest_int('num_dense', 0, 5),
                # _offset_interval=self.trial.suggest_int('offset_interval', 1, 12),
                # interval_minutes=self.trial.suggest_int('interval_minutes', 1, 24 * 7),
                # interval_minutes2=self.trial.suggest_int('interval_minutes2', 4, 4 * 6),
                kernel_size=self.trial.suggest_int('kernel_size', 1, 7),
                max_pooling_kernel=self.trial.suggest_int('max_pooling_kernel', 1, 21),
                max_pooling_strides=self.trial.suggest_int('max_pooling_strides', 1, 16),
                padding=self.trial.suggest_int('padding', 1, 8),
                strides=self.trial.suggest_int('strides', 1, 36),
                eps_greedy_policy_steps=self.trial.suggest_int('eps_greedy_policy_steps', 1000, 1000000, log=True),
                num_lstm=self.trial.suggest_int('num_lstm', 0, 4),
                lstm_size=self.trial.suggest_int('lstm_size', 16, 224),
                # trade_ratio=self.trial.suggest_float('trade_ratio', 0, 1.0),
                beta_1=self.trial.suggest_uniform('beta_1', 0.0, 0.99999),
                beta_2=self.trial.suggest_uniform('beta_2', 0.0, 0.99999),
                # fee_ratio = self.trial.suggest_float('fee_ratio', 0.9, 2.0),
                # trading_fee = self.trial.suggest_float('trading_fee', 0.0004, 0.005),
                policy_value_max = self.trial.suggest_float('policy_value_max', 0.001, 0.9),
                batch_size = self.trial.suggest_int('batch_size', 8, 32),
                # lr=self.trial.suggest_uniform('lr', 1e-12, 1e-02),
                # depth = self.trial.suggest_int('depth', 2, 81),
                # self._kwargs['offset_interval'] = f'{hparams["_offset_interval"] * 60}m'
                # self._kwargs['interval2'] = f'{hparams["interval_minutes2"] * 15}m'
                max_flat_position_length=self.trial.suggest_int('max_flat_position_length', 1, 200),
                # max_negative_pnl = self.trial.suggest_float('max_negative_pnl', -20/100, -0.5/100),
                # max_position_length = self.trial.suggest_int('max_position_length', 0, 72),
                max_short_position_length=self.trial.suggest_int('max_short_position_length', 1, 200),
                nb_steps = self.trial.suggest_int('nb_steps', 5000, 60000, log=True),
                # nb_steps_2 = self.trial.suggest_int('nb_steps_2', 1000, int(5e4)),
                num_conv=self.trial.suggest_int('num_conv', 15, 31),
                # round_decimals = self.trial.suggest_int('round_decimals', 2, 3),
                # sequence_length = self.trial.suggest_int('sequence_length', 2, 96),
                # train_recent_data = self.trial.suggest_categorical('train_recent_data', [True, False]),
                # window_length = self.trial.suggest_int('window_length', 1, 2),
                min_change = self.trial.suggest_float('min_change', 0.0, 0.02),
                cache_limit=self.trial.suggest_int('cache_limit', 100, 10000),
                train_interval=self.trial.suggest_int('train_interval', 2, 8000, log=True),
                target_model_update=self.trial.suggest_int('target_model_update', 2, 8000, log=True),
                window_factor=self.trial.suggest_float('window_factor', 0.001, 10, log=True),
                gap_enabled = self.trial.suggest_categorical('gap_enabled', [True, False]),
                # max_change = self.trial.suggest_float('max_change', 0.001, 0.02),
                # min_flat_change = self.trial.suggest_float('min_flat_change', -0.01, 0.0),
                # action_repetition = self.trial.suggest_int('action_repetition', 1, 12),
                reward_ratio=self.trial.suggest_float('reward_ratio', 1, 1000, log=True),
                # window_slow = self.trial.suggest_int('window_slow', 12, 64),
                # window_fast = self.trial.suggest_int('window_fast', 12, 64),
                # window_sign = self.trial.suggest_int('window_sign', 12, 64)
            )

            # hparams['interval'] = f'{hparams["interval_minutes"] * 60}m'

        else:
            self.trial.set_user_attr('tuned', False)
            self.trial.suggest_int('test_num', 1, 2)
            

        if not tune:
            try:
                params = self.best_tuned_trial_params
                for param in params:
                    self._kwargs[param] = params[param]

                self._kwargs['cache'] = False

            except ValueError:
                pass

        kwargs = self._kwargs.copy()

        alog.info(alog.pformat(kwargs))

        for param in hparams:
            kwargs[param] = hparams[param]

        kwargs['action_repetition'] = 1
        # kwargs['batch_size'] = 32
        kwargs['max_change'] = 0.01
        # kwargs['min_change'] = 0.0
        kwargs['min_flat_change'] = -0.001
        kwargs['random_frame_start'] = False
        kwargs['trading_fee'] = 0.0004
        kwargs['trade_ratio'] = 1/8

        # kwargs['base_filter_size'] = 30
        # kwargs['beta_1'] = 0.8153017891164606
        # kwargs['beta_2'] = 0.6890738409577996
        # kwargs['block_kernel'] = 2
        # kwargs['cache_limit'] = 1076
        # kwargs['delta_clip'] = 37.39432606215123
        # kwargs['dense_width'] = 291
        # kwargs['dueling_type'] = 'avg'
        # kwargs['enable_double_dqn'] = False
        # kwargs['eps_greedy_policy_steps'] = 36571
        # kwargs['gamma'] = 0.2392985908403873
        # kwargs['kernel_size'] = 4
        # kwargs['lr'] = 0.0037977337352434453
        # kwargs['lstm_size'] = 37
        # kwargs['max_flat_position_length'] = 38
        # kwargs['max_pooling_kernel'] = 5
        # kwargs['max_pooling_strides'] = 16
        # kwargs['max_short_position_length'] = 27
        # kwargs['memory_interval'] = 34
        # kwargs['num_conv'] = 31
        # kwargs['num_dense'] = 3
        # kwargs['num_lstm'] = 2
        # kwargs['padding'] = 3
        # kwargs['policy_value_max'] = 0.4266285795619068
        # kwargs['reward_ratio'] = 166.29033402213398
        # kwargs['strides'] = 8
        # kwargs['target_model_update'] = 135
        # kwargs['train_interval'] = 107
        # kwargs['window_factor'] = 2.494463725032405


        # kwargs['interval'] = '2h'
        # kwargs['nb_steps'] = 120 * 2

        self._kwargs = kwargs

        env = self.env
        env.reset()

        # post env init params
        kwargs['quantile'] = env.quantile
        kwargs['trade_volume_max'] = env.trade_volume_max
        kwargs['change_max'] = env.change_max

        # env2 = self.env2
        # env2.reset()

        test_env = self.test_env
        test_env.reset()

        if not self.trial.user_attrs['tuned']:
            if 'tuned' in kwargs:
                del kwargs['tuned']
            if 'tune' in kwargs:
                del kwargs['tune']

        for param in kwargs:
            if param not in hparams:
                self.trial.set_user_attr(param, kwargs[param])

        alog.info(alog.pformat({**self.trial.user_attrs, **self.trial.params}))

        params = dict(
            env=env,
            # env2=env2,
            env_name=self.env_name,
            # policy_value_max=0.5,
            short_reward_enabled=False,
            test_env=test_env,
            trial=self.trial,
            **kwargs
        )

        alog.info(alog.pformat(params))

        return SymbolAgent(**params)

    def _run(self, trial: Trial):
        self.trial = trial

        try:
            if 'exported_model_path' in self.study.best_trial.user_attrs:
                exported_model_path = self.study.best_trial.user_attrs['exported_model_path']
                if self.export_best and self.study.best_trial.value > self.min_capital:
                    self.save_best_params()
                    shutil.rmtree(self.best_model_dir, ignore_errors=True)
                    shutil.copytree(exported_model_path, self.best_model_dir)
        except ValueError as err:
            pass

        try:
            result = self.agent.run()
        except Exception as err:
            if self.retry:
                result = -1.0
            else:
                raise err

        return result

    @property
    def model_version(self):
        return re.match(r'.+\/(\d+)$', self.exported_model_path).group(1)

    def split_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0 and self.memory > 0:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [
                    tf.config.
                    LogicalDeviceConfiguration(memory_limit=self.memory),
                ])
