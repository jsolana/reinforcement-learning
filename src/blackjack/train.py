import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt

from agent import BlackjackAgent
import utils

from config import (
    LEARNING_RATE,
    N_EPISODES,
    START_EPSILON,
    FINAL_EPSILON,
    DISCOUNT_FACTOR,
    ROLLING_LENGTH
)

epsilon_decay = (
    START_EPSILON / (N_EPISODES / 2)
)

# Environment
env = gym.make("Blackjack-v1", sab=False)

env = gym.wrappers.RecordEpisodeStatistics(
    env,
    buffer_length=N_EPISODES
)

# Agent
agent = BlackjackAgent(
    env=env,
    learning_rate=LEARNING_RATE,
    initial_epsilon=START_EPSILON,
    epsilon_decay=epsilon_decay,
    final_epsilon=FINAL_EPSILON,
    discount_factor=DISCOUNT_FACTOR,
)

# Training loop
for episode in tqdm(range(N_EPISODES)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(
            obs,
            action,
            reward,
            terminated,
            next_obs
        )
        obs = next_obs
    agent.decay_epsilon()

# Save trained policy
utils.save_q_table(agent.q_values,
    "blackjack_q_table.pkl"
)

print("Q-table saved.")

# =========================
# PLOTS
# =========================

fig, axs = plt.subplots(
    ncols=3,
    figsize=(15, 5)
)

# Rewards
reward_avg = utils.get_moving_average(
    env.return_queue,
    ROLLING_LENGTH
)

axs[0].plot(reward_avg)
axs[0].set_title("Episode Rewards")

# Episode lengths
length_avg = utils.get_moving_average(
    env.length_queue,
    ROLLING_LENGTH
)

axs[1].plot(length_avg)
axs[1].set_title("Episode Lengths")

# TD Error
error_avg = utils.get_moving_average(
    agent.training_error,
    ROLLING_LENGTH,
    mode="same"
)

axs[2].plot(error_avg)
axs[2].set_title("Training Error")

plt.tight_layout()
plt.show()