import gymnasium as gym
import sinergym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# Set up wandb initialization
wandb.init(project='sinergym', entity='Huron-Yin', name='experiment_DQN')

# Create the environment
env = gym.make('Eplus-5zone-hot-discrete-stochastic-v1')
env = sinergym.utils.wrappers.NormalizeObservation(env)
env = sinergym.utils.wrappers.LoggerWrapper(env, logger_class=sinergym.utils.logger.CSVLogger, flag=True)

# Wrap environment in a DummyVecEnv for compatibility
env = DummyVecEnv([lambda: env])

# Define the evaluation environment (could be the same as training env for simplicity)
eval_env = gym.make('Eplus-5zone-hot-discrete-stochastic-v1')
eval_env = sinergym.utils.wrappers.NormalizeObservation(eval_env)
eval_env = sinergym.utils.wrappers.LoggerWrapper(eval_env, logger_class=sinergym.utils.logger.CSVLogger, flag=True)
eval_env = DummyVecEnv([lambda: eval_env])

# DQN parameters
model = DQN('MlpPolicy', env, learning_rate=1e-4, buffer_size=1000000, learning_starts=50000,
            batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=10000,
            exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05,
            max_grad_norm=10, verbose=1, tensorboard_log="./dqn_tensorboard/", seed=3, device='auto')

# Evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

# Connect the WandbCallback to log training data
model.learn(total_timesteps=int(5e4), callback=[WandbCallback(), eval_callback])

# Save the final model
model.save("final_dqn_model")

# Close the environment
env.close()
eval_env.close()

# Finish wandb run
wandb.finish()