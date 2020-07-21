import os
import gin.tf

from dopamine.discrete_domains import run_experiment

#where to store training logs
BASE_PATH = 'C:/Users/Holm/Documents/dopamine/experiments'  # @param

#which arcade environment?
GAME = 'Breakout'    # 'Pong'  # @param

#define where to store log data
DQN_PATH = os.path.join(BASE_PATH, 'dqn')

dqn_config = """
# Hyperparameters used for reporting DQN results in Bellemare et al. (2017).
import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables

DQNAgent.gamma = 0.99
DQNAgent.update_horizon = 1
DQNAgent.min_replay_history = 50000  # agent steps
DQNAgent.update_period = 4
DQNAgent.target_update_period = 10000  # agent steps
DQNAgent.epsilon_train = 0.01
DQNAgent.epsilon_eval = 0.001
DQNAgent.epsilon_decay_period = 1000000  # agent steps
DQNAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
DQNAgent.optimizer = @tf.train.RMSPropOptimizer()

tf.train.RMSPropOptimizer.learning_rate = 0.00025
tf.train.RMSPropOptimizer.decay = 0.95
tf.train.RMSPropOptimizer.momentum = 0.0
tf.train.RMSPropOptimizer.epsilon = 0.00001
tf.train.RMSPropOptimizer.centered = True

atari_lib.create_atari_environment.game_name = "{}"
# Deterministic ALE version used in the DQN Nature paper (Mnih et al., 2015).
atari_lib.create_atari_environment.sticky_actions = False
create_agent.agent_name = 'dqn'
Runner.num_iterations = 200 # 200
Runner.training_steps = 250000 #   250000  # agent steps
Runner.evaluation_steps = 125000 # 125000  # agent steps
Runner.max_steps_per_episode = 27000 # 27000  # agent steps

AtariPreprocessing.terminal_on_life_loss = True

WrappedReplayBuffer.replay_capacity = 1000000
WrappedReplayBuffer.batch_size = 32
""".format(GAME)


if __name__ == '__main__':
  gin.parse_config(dqn_config, skip_unknown=False)

  # train our runner
  dqn_runner = run_experiment.create_runner(DQN_PATH, schedule='continuous_train')
  print('Will train DQN agent, please be patient, may be a while...')
  dqn_runner.run_experiment()
  print('Done training!')

