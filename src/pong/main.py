import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")

print(f"🎮 Action space:  {env.action_space}")
#print("📊 Observation space:", env.observation_space)

obs, info = env.reset()

try:
    while True:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action: {action} | Reward: {reward} | Info: {info}")

        if terminated or truncated:
            print("🔁 Resetting environment...")
            obs, info = env.reset()

except EOFError:
    print("\n👋 Ctrl+D detectado. Cerrando entorno...")

finally:
    env.close()
    print("✅ Environment cerrado correctamente")