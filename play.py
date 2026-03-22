{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Sng43/atari-dqn/blob/play.py/play.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Play.py Script (Breakout Game)"
      ],
      "metadata": {
        "id": "pH7Gq7fU7ooe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Defined the required libraries, set the saved DQN model path, specified the Atari environment, and prepared the evaluation settings needed to load the trained agent consistently."
      ],
      "metadata": {
        "id": "c86WhoMGz1iS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -U gymnasium ale-py \"stable-baselines3[extra]\""
      ],
      "metadata": {
        "collapsed": true,
        "id": "eltMwf-03xqs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import gymnasium as gym\n",
        "import os\n",
        "import ale_py\n",
        "\n",
        "gym.register_envs(ale_py)\n",
        "\n",
        "print(\"ALE environments registered.\")"
      ],
      "metadata": {
        "collapsed": true,
        "id": "ikFq5rrm4EoZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from stable_baselines3 import DQN\n",
        "from stable_baselines3.common.env_util import make_atari_env\n",
        "from stable_baselines3.common.vec_env import VecFrameStack\n"
      ],
      "metadata": {
        "collapsed": true,
        "id": "yfP6fz810KHF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "MODEL_PATH = \"/content/drive/MyDrive/final_model\"\n",
        "ENV_ID = \"ALE/Breakout-v5\"\n",
        "N_EPISODES = 5\n",
        "SEED = 42"
      ],
      "metadata": {
        "id": "w0EcTe9HSlWi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "if not os.path.exists(MODEL_PATH + \".zip\"):\n",
        "    raise FileNotFoundError(\n",
        "        f\"Model not found: {MODEL_PATH}.zip\\n\"\n",
        "        \"Check the exact Google Drive folder and filename.\"\n",
        "    )\n",
        "\n",
        "model = DQN.load(MODEL_PATH)\n",
        "print(f\"Loaded model from {MODEL_PATH}.zip\")"
      ],
      "metadata": {
        "collapsed": true,
        "id": "mL_0x5Hj5i6_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Created the Atari evaluation environment with human rendering and stacked four frames so the observation format matched the CNN-based DQN model used during training."
      ],
      "metadata": {
        "id": "UWjDxdrW0sW1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "env = make_atari_env(\n",
        "    ENV_ID,\n",
        "    n_envs=1,\n",
        "    seed=SEED,\n",
        "    env_kwargs={\"render_mode\": \"human\"}\n",
        ")\n",
        "\n",
        "env = VecFrameStack(env, n_stack=4)"
      ],
      "metadata": {
        "id": "6jpULsqa1TfG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Loaded the trained DQN model into the prepared environment so the saved policy could be evaluated using the same observation structure from training."
      ],
      "metadata": {
        "id": "5FR_L6dl4mnt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = DQN.load(MODEL_PATH, env=env)\n",
        "print(\"Model loaded successfully.\")"
      ],
      "metadata": {
        "id": "GvoDhdGe4Khh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Ran several evaluation episodes using deterministic action selection, which serves as greedy policy execution in Stable-Baselines3, while displaying the game and printing total rewards."
      ],
      "metadata": {
        "id": "AhWVSkoO7UqN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for episode in range(N_EPISODES):\n",
        "    obs = env.reset()\n",
        "    done = [False]\n",
        "    total_reward = 0\n",
        "\n",
        "    while not done[0]:\n",
        "        action, _ = model.predict(obs, deterministic=True)\n",
        "        obs, rewards, done, info = env.step(action)\n",
        "        total_reward += rewards[0]\n",
        "\n",
        "    print(f\"Episode {episode + 1}: Total Reward = {total_reward}\")\n",
        "\n",
        "env.close()"
      ],
      "metadata": {
        "id": "1vLYATRi4raD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Hyperparameter Experiment\n",
        "\n",
        "**Batch Size and Buffer Size**"
      ],
      "metadata": {
        "id": "TKOfrz1WNyEL"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Setup Inports**"
      ],
      "metadata": {
        "id": "Kc-7rGQeOA_f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import ale_py\n",
        "import gymnasium as gym\n",
        "gym.register_envs(ale_py)\n",
        "\n",
        "import pandas as pd\n",
        "\n",
        "from stable_baselines3 import DQN\n",
        "from stable_baselines3.common.env_util import make_atari_env\n",
        "from stable_baselines3.common.vec_env import VecFrameStack\n",
        "from stable_baselines3.common.evaluation import evaluate_policy"
      ],
      "metadata": {
        "id": "io_l6jCRN9fP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Defined the shared training constants so every hyperparameter experiment could be executed consistently under the same conditions."
      ],
      "metadata": {
        "id": "AN47G_iqOSLq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ENV_ID = \"ALE/Breakout-v5\"\n",
        "POLICY = \"CnnPolicy\"\n",
        "LEARNING_RATE = 0.0005\n",
        "GAMMA = 0.99\n",
        "EXPLORATION_FRAC = 0.1\n",
        "EXPLORATION_INIT = 1.0\n",
        "EXPLORATION_FINAL = 0.01\n",
        "TOTAL_STEPS = 30_000\n",
        "SEED = 42\n",
        "\n",
        "SAVE_ROOT = \"/content/drive/MyDrive/atari-dqn/member2_experiments\"\n",
        "os.makedirs(SAVE_ROOT, exist_ok=True)"
      ],
      "metadata": {
        "id": "mWrtNegDOPGs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Created a reusable experiment function that trains one DQN configuration, evaluates its average reward, saves the model, and returns structured results for later comparison and documentation."
      ],
      "metadata": {
        "id": "ADdKEn7_OehS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def run_experiment(exp_name, batch_size, buffer_size):\n",
        "    exp_path = os.path.join(SAVE_ROOT, exp_name)\n",
        "    os.makedirs(exp_path, exist_ok=True)\n",
        "\n",
        "    train_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED)\n",
        "    train_env = VecFrameStack(train_env, n_stack=4)\n",
        "\n",
        "    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED + 1)\n",
        "    eval_env = VecFrameStack(eval_env, n_stack=4)\n",
        "\n",
        "    model = DQN(\n",
        "        policy=POLICY,\n",
        "        env=train_env,\n",
        "        learning_rate=LEARNING_RATE,\n",
        "        buffer_size=buffer_size,\n",
        "        learning_starts=10_000,\n",
        "        batch_size=batch_size,\n",
        "        gamma=GAMMA,\n",
        "        train_freq=4,\n",
        "        target_update_interval=1000,\n",
        "        exploration_fraction=EXPLORATION_FRAC,\n",
        "        exploration_initial_eps=EXPLORATION_INIT,\n",
        "        exploration_final_eps=EXPLORATION_FINAL,\n",
        "        verbose=0,\n",
        "        tensorboard_log=f\"{exp_path}/tb_logs/\"\n",
        "    )\n",
        "\n",
        "    print(f\"Running {exp_name} | batch={batch_size} | buffer={buffer_size}\")\n",
        "    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)\n",
        "    model.save(f\"{exp_path}/{exp_name}_model\")\n",
        "\n",
        "    mean_reward, std_reward = evaluate_policy(\n",
        "        model,\n",
        "        eval_env,\n",
        "        n_eval_episodes=3,\n",
        "        deterministic=True\n",
        "    )\n",
        "\n",
        "    train_env.close()\n",
        "    eval_env.close()\n",
        "\n",
        "    return {\n",
        "        \"experiment\": exp_name,\n",
        "        \"batch_size\": batch_size,\n",
        "        \"buffer_size\": buffer_size,\n",
        "        \"learning_rate\": LEARNING_RATE,\n",
        "        \"gamma\": GAMMA,\n",
        "        \"epsilon_start\": EXPLORATION_INIT,\n",
        "        \"epsilon_end\": EXPLORATION_FINAL,\n",
        "        \"epsilon_decay_fraction\": EXPLORATION_FRAC,\n",
        "        \"mean_reward\": mean_reward,\n",
        "        \"std_reward\": std_reward\n",
        "    }"
      ],
      "metadata": {
        "id": "ELqd66QVOa3G"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Initialized an empty results container to store each experiment outcome, making it easier to combine all ten runs into one comparison table afterward."
      ],
      "metadata": {
        "id": "b54SBI7EO0Qq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "results = []"
      ],
      "metadata": {
        "id": "vJnjRvagOxER"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 1 - Batch = 16**"
      ],
      "metadata": {
        "id": "19YNKNkGPBeb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp1 = run_experiment(\n",
        "    exp_name=\"member2_exp1_batch16\",\n",
        "    batch_size=16,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp1)\n",
        "pd.DataFrame([exp1])"
      ],
      "metadata": {
        "collapsed": true,
        "id": "R_SBK_UtO7au"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 2 -cBatch = 32**"
      ],
      "metadata": {
        "id": "s8xdugplTD3p"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp2 = run_experiment(\n",
        "    exp_name=\"member2_exp2_batch32\",\n",
        "    batch_size=32,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp2)\n",
        "pd.DataFrame([exp2])"
      ],
      "metadata": {
        "id": "HvtBjybgPKqD",
        "collapsed": true
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 3 - batch = 64**"
      ],
      "metadata": {
        "id": "P1FNjFSHTNME"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp3 = run_experiment(\n",
        "    exp_name=\"member2_exp3_batch64\",\n",
        "    batch_size=64,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp3)\n",
        "pd.DataFrame([exp3])"
      ],
      "metadata": {
        "id": "2TB5iPgNTL_b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 4 - Batch = 128**"
      ],
      "metadata": {
        "id": "-n6b4qKgTYls"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp4 = run_experiment(\n",
        "    exp_name=\"member2_exp4_batch128\",\n",
        "    batch_size=128,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp4)\n",
        "pd.DataFrame([exp4])"
      ],
      "metadata": {
        "id": "B7NviG_nTWel"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 5 - Batch = 32, Buffer = 10,000**"
      ],
      "metadata": {
        "id": "CVWLOBMaTkgj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp5 = run_experiment(\n",
        "    exp_name=\"member2_exp5_batch32_buffer10k\",\n",
        "    batch_size=32,\n",
        "    buffer_size=10_000\n",
        ")\n",
        "results.append(exp5)\n",
        "pd.DataFrame([exp5])"
      ],
      "metadata": {
        "id": "SKrBaDRqTi4k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiments 6 - Batch = 32, Buffer = 25,000**"
      ],
      "metadata": {
        "id": "cFFGxdJWUngh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp6 = run_experiment(\n",
        "    exp_name=\"member2_exp6_batch32_buffer25k\",\n",
        "    batch_size=32,\n",
        "    buffer_size=25_000\n",
        ")\n",
        "results.append(exp6)\n",
        "pd.DataFrame([exp6])"
      ],
      "metadata": {
        "id": "NVinLPgLUdQM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 7 - Batch = 32, Buffer = 75, 000**"
      ],
      "metadata": {
        "id": "0k7bvVsQU35J"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp7 = run_experiment(\n",
        "    exp_name=\"member2_exp7_batch32_buffer75k\",\n",
        "    batch_size=32,\n",
        "    buffer_size=75_000\n",
        ")\n",
        "results.append(exp7)\n",
        "pd.DataFrame([exp7])"
      ],
      "metadata": {
        "id": "68ERns16Uztw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 8 - Batch = 32, Buffer = 100,000**"
      ],
      "metadata": {
        "id": "EWC5Vr6eVJh8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp8 = run_experiment(\n",
        "    exp_name=\"member2_exp8_batch32_buffer100k\",\n",
        "    batch_size=32,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp8)\n",
        "pd.DataFrame([exp8])"
      ],
      "metadata": {
        "id": "b1OlPYknVHSQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 9 - Batch = 64, Buffer = 100,000**"
      ],
      "metadata": {
        "id": "YUQWMEs2VaPV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp9 = run_experiment(\n",
        "    exp_name=\"member2_exp9_batch64_buffer100k\",\n",
        "    batch_size=64,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp9)\n",
        "pd.DataFrame([exp9])"
      ],
      "metadata": {
        "id": "9NxuP8bpVY0D"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Experiment 10 - Batch = 128, Buffer = 100,000**"
      ],
      "metadata": {
        "id": "xJ3OVQ7qVmPC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "exp10 = run_experiment(\n",
        "    exp_name=\"member2_exp10_batch128_buffer100k\",\n",
        "    batch_size=128,\n",
        "    buffer_size=100_000\n",
        ")\n",
        "results.append(exp10)\n",
        "pd.DataFrame([exp10])"
      ],
      "metadata": {
        "id": "VRnUBzfLVlGl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Final Combined Results Table**\n",
        "\n",
        "Combined all ten Member Two experiments into one table and added behavior notes, making the final submission easier to compare, interpret, and document clearly."
      ],
      "metadata": {
        "id": "UTq8E9DIV1Ly"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "results_df = pd.DataFrame(results)\n",
        "\n",
        "def noted_behavior(row):\n",
        "    if row[\"mean_reward\"] >= 20:\n",
        "        return \"Strong reward performance with stable learning behavior.\"\n",
        "    elif row[\"mean_reward\"] >= 10:\n",
        "        return \"Moderate performance with acceptable training stability.\"\n",
        "    else:\n",
        "        return \"Lower performance; may require longer training or better tuning.\"\n",
        "\n",
        "results_df[\"noted_behavior\"] = results_df.apply(noted_behavior, axis=1)\n",
        "results_df"
      ],
      "metadata": {
        "id": "FIU75_VDVxAG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "results_df.to_csv(f\"{SAVE_ROOT}/member2_hyperparameter_results.csv\", index=False)\n",
        "print(\"Saved results to CSV successfully.\")"
      ],
      "metadata": {
        "id": "wRSpqM6mXP_V"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}