import gymnasium as gym
from agent import BlackjackAgent

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
)

env = gym.make(
    "Blackjack-v1"
)

agent = BlackjackAgent(
    env=env,
    learning_rate=LEARNING_RATE,
    initial_epsilon=0,
    epsilon_decay=0,
    final_epsilon=0,
    discount_factor=DISCOUNT_FACTOR,
)

# Load trained Q-table
agent.load_q_table(
    "blackjack_q_table.pkl"
)

print("Q-table loaded.")

# Disable exploration
agent.epsilon = 0

# Play games
for episode in range(100):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        result = {1: "WIN  🟢", 0: "DRAW 🟡", -1: "LOSE 🔴"}[reward]
        print(f"Episode {episode + 1}: {result} (reward={reward})")
env.close()