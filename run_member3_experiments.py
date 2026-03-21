import os
import json
from pathlib import Path
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

# Define your 10 experiments
EXPERIMENTS = [
    {
        "name": "exp1_baseline",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp2_lower_lr",
        "lr": 5e-5,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp3_higher_lr",
        "lr": 2e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp4_lower_gamma",
        "lr": 1e-4,
        "gamma": 0.95,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp5_much_lower_gamma",
        "lr": 1e-4,
        "gamma": 0.90,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp6_larger_batch",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 64,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp7_smaller_batch",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 16,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
    {
        "name": "exp8_higher_epsilon_end",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 0.1,
    },
    {
        "name": "exp9_slower_decay",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.2,
    },
    {
        "name": "exp10_combined",
        "lr": 5e-4,
        "gamma": 0.95,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
    },
]

ENV_ID = "ALE/Breakout-v5"
TOTAL_STEPS = 150_000

def run_experiment(config):
    exp_name = config["name"]
    output_dir = Path("member3_runs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"LR: {config['lr']}, Gamma: {config['gamma']}, Batch: {config['batch']}")
    print(f"{'='*60}")
    
    # Create environment
    vec_env = make_atari_env(ENV_ID, n_envs=1, seed=42)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=exp_name,
        verbose=0,
    )
    
    # Create and train model
    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=config["lr"],
        gamma=config["gamma"],
        batch_size=config["batch"],
        buffer_size=100_000,
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=config["eps_decay"],
        exploration_initial_eps=config["eps_start"],
        exploration_final_eps=config["eps_end"],
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )
    
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        progress_bar=True,
    )
    
    # Save final model
    model.save(str(output_dir / "final_model"))
    vec_env.close()
    eval_env.close()
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ {exp_name} completed. Saved to {output_dir}")

if __name__ == "__main__":
    for i, config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/10] Starting experiment...")
        try:
            run_experiment(config)
        except Exception as e:
            print(f"✗ Error in {config['name']}: {e}")
            continue
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)