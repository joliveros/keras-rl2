from cached_property import cached_property
from numpy import NaN

from examples.dqn_orderbook.symbol_agent import SymbolAgent, Optimizer
from exchange_data import settings
from exchange_data.emitters import Messenger
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


class SymbolTuner(StudyWrapper):
    current_lock_ix = 0
    hparams = None
    run_count = 0

    def __init__(self,
                 export_best,
                 session_limit,
                 clear_runs,
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
        self.clear_runs = clear_runs
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
        tune = self._kwargs['tune']
        hparams = dict()

        if tune:
            self.trial.set_user_attr('tuned', True)

            hparams = dict(
                memory_interval=self.trial.suggest_int('memory_interval', 1, 399),
                delta_clip=self.trial.suggest_uniform('delta_clip', 0, 99),
                gamma=self.trial.suggest_uniform('gamma', 0, 0.9999),
                # enable_double_dqn=self.trial.suggest_categorical('enable_double_dqn', [True, False]),
                dueling_type=self.trial.suggest_categorical('dueling_type', ['avg', 'max', 'naive']),
                base_filter_size=self.trial.suggest_int('base_filter_size', 4, 72),
                dense_width=self.trial.suggest_int('dense_width', 4, 396),
                block_kernel=self.trial.suggest_int('block_kernel', 1, 4),
                num_dense=self.trial.suggest_int('num_dense', 0, 5),
                # _offset_interval=self.trial.suggest_int('offset_interval', 1, 12),
                # interval_minutes=self.trial.suggest_int('interval_minutes', 1, 24 * 7),
                # interval_minutes2=self.trial.suggest_int('interval_minutes2', 4, 4 * 6),
                kernel_size=self.trial.suggest_int('kernel_size', 1, 4),
                # max_pooling_kernel=self.trial.suggest_int('max_pooling_kernel', 1, 12),
                # max_pooling_strides=self.trial.suggest_int('max_pooling_strides', 1, 16),
                # padding=self.trial.suggest_int('padding', 1, 8),
                strides=self.trial.suggest_int('strides', 1, 16),
                eps_greedy_policy_steps=self.trial.suggest_int('eps_greedy_policy_steps', 1000, 1000000, log=True),
                num_lstm=self.trial.suggest_int('num_lstm', 0, 4),
                lstm_size=self.trial.suggest_int('lstm_size', 16, 224),
            )

            # self._kwargs['trade_ratio'] = self.trial.suggest_float('trade_ratio', 0, 1.0)
            # self._kwargs['beta_1'] = self.trial.suggest_uniform('beta_1', 0.0, 0.99999)
            # self._kwargs['beta_2'] = self.trial.suggest_uniform('beta_2', 0.0, 0.99999)
            # self._kwargs['fee_ratio'] = self.trial.suggest_float('fee_ratio', 0.9, 2.0)
            # self._kwargs['trading_fee'] = self.trial.suggest_float('trading_fee', 0.0004, 0.005)
            # self._kwargs['policy_value_max'] = self.trial.suggest_float('policy_value_max', 0.001, 0.9)
            # self._kwargs['batch_size'] = self.trial.suggest_int('batch_size', 8, 64)
            self._kwargs['lr'] = self.trial.suggest_uniform('lr', 1e-12, 1e-02)
            # self._kwargs['depth'] = self.trial.suggest_int('depth', 12, 256)
            # self._kwargs['offset_interval'] = f'{hparams["_offset_interval"] * 60}m'
            # self._kwargs['interval'] = f'{hparams["interval_minutes"] * 60}m'
            # self._kwargs['interval2'] = f'{hparams["interval_minutes2"] * 15}m'
            self._kwargs['max_flat_position_length'] \
                 = self.trial.suggest_int('max_flat_position_length', 1, 200)
            # self._kwargs['max_negative_pnl'] = self.trial.suggest_float('max_negative_pnl', -20/100, -0.5/100)
            # self._kwargs['max_position_length'] = self.trial.suggest_int('max_position_length', 0, 72)
            self._kwargs['max_short_position_length'] \
                = self.trial.suggest_int('max_short_position_length', 1, 200)
            self._kwargs['nb_steps'] = self.trial.suggest_int('nb_steps', 5000, 100000, log=True)
            # self._kwargs['nb_steps_2'] = self.trial.suggest_int('nb_steps_2', 1000, int(5e4))
            self._kwargs['num_conv'] = self.trial.suggest_int('num_conv', 1, 15)
            self._kwargs['round_decimals'] = self.trial.suggest_int('round_decimals', 2, 3)
            # self._kwargs['sequence_length'] = self.trial.suggest_int('sequence_length', 2, 96)
            # self._kwargs['train_recent_data'] = self.trial.suggest_categorical('train_recent_data', [True, False])
            # self._kwargs['window_length'] = self.trial.suggest_int('window_length', 1, 2)
            # self._kwargs['min_change'] = self.trial.suggest_float('min_change', 0.0, 0.02)
            # self._kwargs['cache_limit'] = self.trial.suggest_int('cache_limit', 100, 10000)
            self._kwargs['train_interval'] = self.trial.suggest_int('train_interval', 2, 8000, log=True)
            self._kwargs['target_model_update'] = self.trial.suggest_int('target_model_update', 2, 8000, log=True)
            # self._kwargs['gap_enabled'] = self.trial.suggest_categorical('gap_enabled', [True, False])
            # self._kwargs['max_change'] = self.trial.suggest_float('max_change', 0.001, 0.02)
            # self._kwargs['min_flat_change'] = self.trial.suggest_float('min_flat_change', -0.01, 0.0)
            # self._kwargs['action_repetition'] = self.trial.suggest_int('action_repetition', 1, 12)
            self._kwargs['reward_ratio'] \
                = self.trial.suggest_float('reward_ratio', 1, 1000, log=True)

        else:
            self.trial.set_user_attr('tuned', False)
            self.trial.suggest_int('test_num', 1, 2)

        params = dict()
        params['action_repetition'] = 1
        params['base_filter_size'] = 18
        params['batch_size'] = 18
        params['beta_1'] = 0.5692086887313117
        params['beta_2'] = 0.4380206740466757
        params['block_kernel'] = 2
        params['cache_limit'] = 8978
        params['delta_clip'] = 42.41585116321283
        params['dense_width'] = 44
        params['dueling_type'] = 'max'
        params['enable_double_dqn'] = False
        params['eps_greedy_policy_steps'] = 657084
        params['gamma'] = 0.7963338411045654
        params['kernel_size'] = 1
        params['lr'] = 0.0003085875867916756
        params['lstm_size'] = 130
        params['max_change'] = 0.01
        params['max_flat_position_length'] = 128
        params['max_position_length'] = 31
        params['max_short_position_length'] = 128
        params['memory_interval'] = 277
        params['min_change'] = 0.0
        params['min_flat_change'] = -0.001
        params['num_conv'] = 15
        params['num_dense'] = 0
        params['num_lstm'] = 2
        params['random_frame_start'] = False
        params['strides'] = 16
        params['target_model_update'] = 146
        params['trading_fee'] = 0.0004 * 4
        params['train_interval'] = 15
        params['reward_ratio'] = 30.919582800647163

        kwargs = self._kwargs.copy()
        kwargs = {**kwargs, **hparams}

        alog.info(alog.pformat(kwargs))

        for param in params:
            if param not in kwargs:
                kwargs[param] = params[param]

        env = self.env
        env.reset()
        kwargs['quantile'] = env.quantile
        kwargs['trade_volume_max'] = env.trade_volume_max
        kwargs['change_max'] = env.change_max

        # env2 = self.env2
        # env2.reset()

        test_env = self.test_env
        test_env.reset()

        self.trial.set_user_attr('params', kwargs)

        params = dict(
            env=env,
            # env2=env2,
            env_name=self.env_name,
            policy_value_max=0.5,
            short_reward_enabled=False,
            test_env=test_env,
            trial_id=str(self.trial.number),
            study=self.study,
            **kwargs
        )

        alog.info(alog.pformat(params))

        return SymbolAgent(**params)

    def _run(self, trial: Trial):
        self.trial = trial

        if trial.number > self.clear_runs > 0:
            exported_model_path = self.study.best_trial.user_attrs['exported_model_path']
            if self.export_best and self.study.best_trial.value > self.min_capital:
                self.save_best_params()
                shutil.rmtree(self.best_model_dir, ignore_errors=True)
                shutil.copytree(exported_model_path, self.best_model_dir)

            self.clear()

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


