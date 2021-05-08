import re
import shutil
import time as t
from pathlib import Path

import alog
import tensorflow as tf
from exchange_data import settings
from exchange_data.emitters import Messenger
from exchange_data.models.resnet.study_wrapper import StudyWrapper
from optuna import Trial
from pip._vendor.contextlib2 import nullcontext
from pytimeparse.timeparse import timeparse
from redlock import RedLockError, RedLock

from examples.dqn_orderbook.symbol_agent import SymbolAgent


class SymbolTuner(StudyWrapper, Messenger):
    current_lock_ix = 0
    hparams = None
    run_count = 0

    def __init__(self,
                 export_best,
                 session_limit,
                 clear_runs,
                 backtest_interval,
                 min_capital,
                 memory,
                 num_locks=2,
                 **kwargs):

        self._kwargs = kwargs

        super().__init__(**kwargs)

        StudyWrapper.__init__(self, **kwargs)
        Messenger.__init__(self, **kwargs)

        self.export_best = export_best
        self.clear_runs = clear_runs
        self.min_capital = min_capital
        self.memory = memory
        self.num_locks = num_locks
        self.export_dir.mkdir(exist_ok=True)

        self.split_gpu()

        Path(self.best_model_dir).mkdir(parents=True, exist_ok=True)

        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'

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
                self._run(*args)
            self.run_count += 1

        except RedLockError as e:
            alog.info(e)
            t.sleep(retry_relay)
            return self.run(*args)

    def _run(self, trial: Trial):
        self.trial = trial

        if self.clear_runs < trial.number and self.clear_runs > 0:
            exported_model_path = self.study.best_trial.user_attrs['exported_model_path']
            if self.export_best and self.study.best_trial.value > self.min_capital:
                self.save_best_params()
                shutil.rmtree(self.best_model_dir, ignore_errors=True)
                shutil.copytree(exported_model_path, self.best_model_dir)

            self.clear()

        tf.keras.backend.clear_session()

        with tf.summary.create_file_writer(self.run_dir).as_default():

            params = dict(
                trial_id=str(trial.number),
                nb_steps=50,
                **self._kwargs
            )

            alog.info(alog.pformat(params))

            agent = SymbolAgent(**params)
            result = agent.run()

            # self.model_dir = agent.model_dir
            #
            # accuracy = result.get('accuracy')
            # global_step = result.get('global_step')
            # self.exported_model_path = result.get('exported_model_path')
            # trial.set_user_attr('exported_model_path', self.exported_model_path)
            # trial.set_user_attr('model_version', self.model_version)
            # trial.set_user_attr('quantile', self.quantile)
            #
            # tf.summary.scalar('accuracy', accuracy, step=global_step)

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