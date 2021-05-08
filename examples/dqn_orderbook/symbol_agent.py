import alog
import gym
import numpy as np
from exchange_data.models.resnet.model import Model as ResnetModel
from tensorflow.python.keras.optimizer_v2.adam import Adam

from examples.dqn_orderbook.processor import OrderBookFrameProcessor
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


class SymbolAgent(object):
    def __init__(self, env_name, nb_steps, lr=.0000025, **kwargs):
        self.env_name = env_name
        self.lr = lr
        self.env = env = gym.make(env_name, **kwargs)
        self.nb_steps = nb_steps

        WINDOW_LENGTH = 4
        INPUT_SHAPE = (WINDOW_LENGTH, kwargs['sequence_length'], kwargs['depth'] * 2, 1)
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
        processor = OrderBookFrameProcessor()

        # Select a policy. We use eps-greedy action selection, which means that a random action is selected
        # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
        # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
        # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
        # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=self.nb_steps)

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
            nb_steps_warmup=1000,
            policy=policy,
            processor=processor,
            target_model_update=1000,
            train_interval=12,
        )

        self.agent.compile(Adam(lr=self.lr), metrics=['mae'])

    def run(self):
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
        weights_filename = f'dqn_{self.env_name}_weights.h5f'
        checkpoint_weights_filename = 'dqn_' + self.env_name + '_weights_{step}.h5f'
        log_filename = f'dqn_{self.env_name}_log.json'
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        self.agent.fit(self.env, verbose=2, callbacks=callbacks, nb_steps=self.nb_steps, log_interval=1000)

        # After training is done, we save the final weights one more time.
        self.agent.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 1 episodes.
        self.agent.test(self.env, verbose=2, nb_episodes=1, visualize=False, nb_max_start_steps=10)
