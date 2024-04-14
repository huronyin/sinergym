import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
 
import sinergym
from sinergym.utils.wrappers import (LoggerWrapper, NormalizeAction, NormalizeObservation)
 
# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-smalldatacenter-mixed-continuous-v1')
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
 
def plot_rewards(rewards, filename='rewards_plot.png'):
    """Plot the cumulative rewards per episode and save to a file."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.title('Cumulative Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.savefig(filename)  # Save the figure as a file
    print(f"Plot saved to {filename}")
 
# Store cumulative rewards to plot later
episode_rewards = []
 
# Execute interactions during 10 episodes
for i in range(200):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        # Random action control
        a = env.action_space.sample()
        # Read observation and reward
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        # If this timestep is a new month start
        if info['month'] != current_month:
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    episode_rewards.append(sum(rewards))
    print('Episode ', i, 'Mean reward: ', np.mean(rewards), 'Cumulative reward: ', sum(rewards))
 
# Plot and save the rewards after all episodes are done
plot_rewards(episode_rewards, 'cumulative_rewards.png')
 
# Close the environment
env.close()
