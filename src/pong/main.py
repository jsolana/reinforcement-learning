import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import utils

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")


# ---------------------------------------------------
# SNAPSHOT UTILITIES
# ---------------------------------------------------

def show_snapshots(obs, processed, frame_diff):
    """
    Display raw, processed, and difference frames using utils visualization helpers.
    """

    # Show raw frame
    utils.show_image(obs, title="Raw Pong Frame")

    # Show processed frame
    utils.show_image(
        processed,
        title="Processed Frame",
        shape=(80, 80),
        cmap="gray"
    )

    # Show frame difference
    utils.show_image(
        frame_diff,
        title="Frame Difference",
        shape=(80, 80),
        cmap="gray"
    )


def test_snapshots(obs, processed, frame_diff):
    """
    Save raw, processed, and difference frames to disk for debugging and inspection.
    """

    # Save raw frame
    utils.save_image(obs, "raw_frame")

    # Save processed frame
    utils.save_image(
        processed,
        "processed_frame",
        shape=(80, 80),
        cmap="gray"
    )

    # Save difference frame
    utils.save_image(
        frame_diff,
        "difference_frame",
        shape=(80, 80),
        cmap="gray"
    )


# ---------------------------------------------------
# RESET ENVIRONMENT
# ---------------------------------------------------

obs, info = env.reset()

# Show raw frame
# utils.show_image(obs, title="Raw Pong Frame")

# ---------------------------------------------------
# PREPROCESS FIRST FRAME
# ---------------------------------------------------

processed = utils.preprocess_frame(obs)

# utils.show_image(
#     processed,
#     title="Processed Frame",
#     shape=(80, 80),
#     cmap="gray"
# )

# ---------------------------------------------------
# EPISODE LOOP
# ---------------------------------------------------

try:
    done = False
    total_reward = 0
    steps = 0
    previous_frame = None

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # Preprocess + frame difference
        frame_diff, previous_frame = utils.preprocess_with_difference(
            obs,
            previous_frame
        )

        total_reward += reward
        steps += 1

        done = terminated or truncated

    # ---------------------------------------------------
    # EPISODE RESULTS
    # ---------------------------------------------------

    print("Episode finished")
    print("Steps:", steps)
    print("Total reward:", total_reward)

    # ---------------------------------------------------
    # SNAPSHOTS
    # ---------------------------------------------------

    # show_snapshots(obs, processed, frame_diff)
    # test_snapshots(obs, processed, frame_diff)

# ---------------------------------------------------
# SAFETY EXIT
# ---------------------------------------------------

except KeyboardInterrupt:
    print("\n👋 Exit...")

finally:
    env.close()