import time
from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

ENV_ID = "ALE/Breakout-v5"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "member3_runs" / "exp10_combined" / "final_model"
N_EPISODES = 3

gym.register_envs(ale_py)


def main() -> None:
    model = DQN.load(str(MODEL_PATH))

    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=42,
        env_kwargs={"render_mode": "human"},
    )
    env = VecFrameStack(env, n_stack=4)

    try:
        for episode in range(1, N_EPISODES + 1):
            obs = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                time.sleep(0.01)

            print(f"Episode {episode}: Total Reward = {total_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
