import os
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

# ── CHANGE THIS BLOCK PER EXPERIMENT ────────────────────────
EXP_NAME          = "exp1_baseline"
LEARNING_RATE     = 1e-4
GAMMA             = 0.99
BATCH_SIZE        = 32
BUFFER_SIZE       = 100_000
EXPLORATION_FRAC  = 0.1
EXPLORATION_INIT  = 1.0
EXPLORATION_FINAL = 0.01
TOTAL_STEPS       = 150_000
# ─────────────────────────────────────────────────────────────

ENV_ID     = "ALE/Breakout-v5"
POLICY     = "CnnPolicy"
DRIVE_PATH = f"/content/drive/MyDrive/atari-dqn/models/{EXP_NAME}"
os.makedirs(DRIVE_PATH, exist_ok=True)

vec_env  = make_atari_env(ENV_ID, n_envs=1, seed=42)
vec_env  = VecFrameStack(vec_env, n_stack=4)
eval_env = make_atari_env(ENV_ID, n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

model = DQN(
    policy                   = POLICY,
    env                      = vec_env,
    learning_rate            = LEARNING_RATE,
    buffer_size              = BUFFER_SIZE,
    learning_starts          = 10_000,
    batch_size               = BATCH_SIZE,
    gamma                    = GAMMA,
    train_freq               = 4,
    target_update_interval   = 1000,
    exploration_fraction     = EXPLORATION_FRAC,
    exploration_initial_eps  = EXPLORATION_INIT,
    exploration_final_eps    = EXPLORATION_FINAL,
    verbose                  = 1,
    tensorboard_log          = f"{DRIVE_PATH}/tb_logs/"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = f"{DRIVE_PATH}/best/",
    log_path             = f"{DRIVE_PATH}/eval_logs/",
    eval_freq            = 10_000,
    n_eval_episodes      = 5,
    deterministic        = True,
    render               = False,
    verbose              = 1
)

checkpoint_callback = CheckpointCallback(
    save_freq   = 50_000,
    save_path   = f"{DRIVE_PATH}/checkpoints/",
    name_prefix = EXP_NAME,
    verbose     = 1
)

print(f"Training {EXP_NAME} | lr={LEARNING_RATE} | gamma={GAMMA} | steps={TOTAL_STEPS}")
model.learn(
    total_timesteps = TOTAL_STEPS,
    callback        = CallbackList([eval_callback, checkpoint_callback]),
    progress_bar    = True
)

model.save(f"{DRIVE_PATH}/final_model")
print(f"✅ Done! Saved to {DRIVE_PATH}/final_model.zip")
vec_env.close()
eval_env.close()
