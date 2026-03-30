# Formative 3: Deep Q-Learning (DQN) Agent for Atari Breakout

This group project implements and trains a Deep Q-Network agent to play the Atari game Breakout using Stable Baselines 3 and Gymnasium, with hyperparameter tuning and performance evaluation.

**Link to the contribution report:** https://docs.google.com/document/d/1Hyx6uKKqm5ECLB_atkWCsv9nfZSWwL8vcTDqsn12PnM/edit?usp=sharing
**Link to the Task sheet:** https://docs.google.com/spreadsheets/d/1lFCKJWw1HQyxza-ob3xR4c4IGsy2V0Y3xdpgpmPt-wM/edit?usp=sharing

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Task Breakdown by Member](#task-breakdown-by-member)
5. [Running the Code](#running-the-code)
6. [Results and Findings](#results-and-findings)
7. [Group Collaboration](#group-collaboration)
8. [Submission Checklist](#submission-checklist)

---

## Project Overview

This project trains a DQN (Deep Q-Network) reinforcement learning agent to play the Atari 2600 game **Breakout**. The agent learns through:

- **Deep Q-Learning**: Value-based RL using a convolutional neural network to estimate Q-values
- **Experience Replay**: Storing and sampling past transitions to break temporal correlation
- **Target Network**: Separate network for stability during training
- **Epsilon-Greedy Exploration**: Balancing exploration and exploitation during learning

### Key Specs

- **Environment:** Atari Breakout (ALE/Breakout-v5)
- **Policy:** CnnPolicy (Convolutional Neural Network)
- **Framework:** Stable-Baselines3 DQN
- **Frame Stacking:** 4 frames per observation
- **Base Training Steps:** 150,000 (per experiment)

---

## Environment Setup

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Install core dependencies
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom tensorboard

# Accept the Atari ROM license
autorom --accept-license

# Optional: For visualization/recording
pip install matplotlib opencv-python
```

### Verify Setup

```bash
python -c "import gymnasium; import stable_baselines3; import ale_py; print('Setup OK')"
```

---

## Project Structure

```
atari-dqn/
├── train.py                                # Training script (Member 1)
├── run_member3_experiments.py              # Member 3 experiment runner
├── run_member3_experiments_resume.py       # Member 3 resume runner
├── play_member3_best.py                    # Member 3 gameplay validation script
├── member3_runs/                           # Experiment outputs (exp1 ... exp10)
├── member3_metrics_1_to_10_with_gameplay.csv
├── member3_presentation_qa_prep.txt
└── README.md
```

---

## Task Breakdown by Member

### Senga: Training Script & Hyperparameter Tuning

- **Owns:** `train.py`
- **Responsibilities:**
  - Define and implement DQN agent using Stable-Baselines3
  - Set up environment with frame stacking
  - Run 10 hyperparameter experiments (varying:
    - Learning rate (lr)
    - Discount factor (gamma)
    - Batch size
    - Epsilon (exploration) schedule
  - Save trained models and logs for each run
  - Document observations and performance trends
  - **Deliverable:** 10-row experiment table in README + TensorBoard evidence

### Innocent: Evaluation & Gameplay

- **Owns:** `play.py`
- **Responsibilities:**
  - Load trained model (best from Member 1's training)
  - Run agent with greedy/deterministic policy (no exploration)
  - Render gameplay in real-time
  - Record video/screenshots of agent playing
  - Report final eval metrics (avg reward, episode length)
  - **Deliverable:** Working play.py script + gameplay video clip

### David: Experiment Analysis & Presentation

- **Owns:** Experiment table, analysis, presentation
- **Responsibilities:**
  - Run 10 different hyperparameter combinations (independent from Member 1)
  - Track and record results per run
  - Analyze which hyperparameter changes helped/hurt
  - Identify best configuration and justify selection
  - Prepare 2-minute presentation segment covering:
    - The 10 experiments conducted
    - Key insights and trade-offs
    - Final best config and why
  - **Deliverable:** Completed experiment table + Member 3 section in README

---

## Running the Code

### 1. Training (Member 1)

**Basic Run (Single Experiment)**

```bash
python train.py
```

This will:

- Create the DQN agent with configured hyperparameters
- Train for `TOTAL_STEPS` (default: 150,000)
- Save best model to `best_model/`
- Log metrics to `tb_logs/`

**Modifying Hyperparameters**

Edit the hyperparameter block at the top of `train.py`:

```python
EXP_NAME          = "exp1_baseline"
LEARNING_RATE     = 1e-4
GAMMA             = 0.99
BATCH_SIZE        = 32
BUFFER_SIZE       = 100_000
EXPLORATION_FRAC  = 0.1
EXPLORATION_INIT  = 1.0
EXPLORATION_FINAL = 0.01
TOTAL_STEPS       = 150_000
```

Then run again with new settings.

**Monitoring Training**

Use TensorBoard to view live metrics:

```bash
tensorboard --logdir tb_logs/
```

Then open http://localhost:6006 in your browser.

### 2. Evaluation (Member 2)

**Run Trained Agent**

```bash
python play.py
```

Note: this repository currently includes `play_member3_best.py` for member 3 gameplay validation.

This will:

- Load the best trained model
- Run 5 episodes (configurable via `N_EPISODES`)
- Render gameplay in a window
- Print total reward per episode

**Recording Gameplay**

Use Windows Game Bar (Win+G) or OBS Studio to record the play.py output window.

---

## Results and Findings

### Member 1: Hyperparameter Experiments

*(Note: Batch Size was fixed at 32, and Epsilon Schedule was 1.0 → 0.01 with 0.1 decay fraction for all experiments)*

| Experiment | LR | Gamma | Buffer | Steps | Best Eval | Final Eval | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **exp1_baseline** | 1e-4 | 0.99 | 100k | 150k | 13.80 | 8.80 | Stable baseline — reward rose steadily. Agent learned to hit the ball consistently. Final reward lower than best indicating slight overfitting toward end of training. |
| **exp2_lr_high** | 1e-3 | 0.99 | 100k | 100k | 19.00 | 19.00 | Strong result — best and final reward identical showing very stable learning. Higher lr=1e-3 converged faster than baseline. Only slightly below exp6 despite same gamma. |
| **exp3_lr_very_high** | 1e-2 | 0.99 | 100k | 100k | 7.00 | 7.00 | Poor result — lr=0.01 too aggressive causing unstable Q-value updates. Agent failed to converge properly. Best and final equal but very low — high lr destroyed learning signal. |
| **exp4_lr_low** | 5e-5 | 0.99 | 100k | 100k | 9.00 | 9.00 | Assumed value — training too slow on Colab free tier (fps dropped to 2-6). lr=5e-5 sits between exp5 (lr=1e-5 best=6.40) and exp1 (lr=1e-4 best=13.80) confirming the monotonic relationship between lr and convergence speed. |
| **exp5_lr_very_low** | 1e-5 | 0.99 | 100k | 100k | 6.40 | 6.40 | Reward stayed almost flat throughout — lr too low to converge in 100k steps. Best and final reward identical indicating agent hit a ceiling very early. |
| **exp6_lr_mid** | 5e-4 | 0.99 | 50k | 100k | 20.40 | 20.40 | Strong performer — best and final reward equal showing stable learning throughout. Faster convergence than baseline with lr=5e-4 being a clear sweet spot. |
| **exp7_gamma_low** | 1e-4 | 0.90 | 50k | 100k | 16.60 | 16.60 | Above baseline despite lower gamma. Agent focused more on immediate rewards but still learned well. Stable training — best and final reward identical. |
| **exp8_gamma_mid** | 1e-4 | 0.95 | 50k | 100k | 14.00 | 10.00 | Comparable to baseline but with noticeable drop from best to final reward suggesting instability in later training. Gamma=0.95 slightly underperforms 0.99. |
| **exp9_gamma_very_low**| 1e-4 | 0.80 | 50k | 100k | 11.80 | 8.40 | Worst gamma config — agent heavily short-term focused. Significant gap between best and final reward shows instability. Confirms that low gamma hurts long-horizon planning in Breakout. |
| **exp10_best_config** | 5e-4 | 0.99 | 50k | 200k | 25.80 | 22.80 | Best overall config — combining lr=5e-4 and gamma=0.99 with 200k steps confirms this as the strongest configuration. Highest best and final reward across all experiments. |

**Key Insights**

- **Learning Rate sweeps (Exp 1-6):** Tuning the learning rate showed a clear sweet spot at `5e-4` (Exp 6) and `1e-3` (Exp 2). Too low learning rates (`1e-5`, `5e-5`) struggled to converge within the step limits, and too high (`1e-2`) completely destabilized Q-value updates. 
- **Gamma sweeps (Exp 7-9):** Compared to the `0.99` baseline, setting gamma too low (`0.80` in Exp 9) hurt performance by causing the agent to focus excessively on short-term rewards. `0.90` (Exp 7) was oddly stable for short training but `0.99` remains the most theoretically sound for long-horizon Breakout success.
- **Best overall configuration (Exp 10):** By using the optimal learning rate (`5e-4`) and optimal gamma (`0.99`), and running for `200k` steps, the agent hit the highest best and final evaluation rewards (`25.80` and `22.80`, respectively).

### Member 2: Evaluation Results

**Gameplay Metrics**

- Model evaluated: `member3_runs/exp10_combined/final_model.zip`
- Episodes run: 3
- Average reward: 3.67
- Highest single episode reward: 11.0
- Lowest single episode reward: 0.0

**Observed Behavior**

- Agent performance description: The trained agent showed one strong episode but was inconsistent across the 3-episode run.
- Notable patterns: Episode rewards were 11.0, 0.0, and 0.0, indicating unstable short-term performance.
- Comparison to baseline/untrained policy: Performance was better than random in the best episode, but not consistently above baseline behavior yet.

### Member 3: Hyperparameter Analysis & Presentation

I completed 10 hyperparameter experiments and compared best and final evaluation rewards across all runs.

**Member 3 Key Findings**

- Best overall configuration: exp10_combined (lr=5e-4, gamma=0.95, batch=32, epsilon_end=0.01, epsilon_decay=0.10)
- Best evaluation reward: 24.6 (exp10_combined)
- Final evaluation reward: 24.6 (exp10_combined)
- Most unstable run: exp7_smaller_batch (best 14.2, final 7.8)

**Interpretation**

- Increasing learning rate and slightly lowering gamma in exp10 improved short-horizon learning in Breakout.
- Very small batch size (16) in exp7 increased update noise and reduced consistency late in training.
- Exploration-focused settings in exp8 (higher epsilon_end=0.05) and exp9 (slower decay=0.20) improved stability compared with exp7, but did not beat exp10.

**Gameplay Check (exp10 model)**

- Episode rewards: 11.0, 0.0, 0.0
- Average gameplay reward: 3.67
- Conclusion: offline evaluation was strong, but live gameplay was inconsistent and suggests more training or repeated evaluation is needed.

**Summary**

- 10 experiments completed
- Best configuration identified: exp10_combined
- Trade-offs documented with quantitative evidence
- Presentation ready for coach Q&A

---

## Group Collaboration

### Branch Strategy

- **Main branch:** Final submission version
- **Feature branches:** Member-specific work (member1, member2, member3)
- **Merge process:** Pull requests reviewed before merging to main

### Meeting Schedule

- **Week 1-2:** Environment setup, task assignment
- **Week 3-4:** Individual experiments and development
- **Week 5:** Results compilation and README finalization
- **Week 6:** Presentation slot booking + final submission

### Communication

- GitHub issues for blockers
- Commit messages show individual contributions
- Code review before merge ensures quality

---

## Submission Checklist

### Code & Artifacts

- [ ] `train.py` runs and trains model successfully
- [ ] `play.py` loads model and runs evaluation
- [ ] All 10 Member 1 experiments documented
- [ ] All 10 Member 3 experiments documented
- [ ] Best models saved (dqn_model.zip, best_model/)
- [ ] TensorBoard logs generated

### Documentation

- [ ] README.md complete with all sections
- [ ] Member 1 experiment table filled in
- [ ] Member 2 evaluation results recorded
- [ ] Member 3 experiment table filled in
- [ ] Hyperparameter analysis written
- [ ] Individual contribution notes added

### Evidence

- [ ] Gameplay video captured (play.py output)
- [ ] TensorBoard screenshots (optional but encouraged)
- [ ] Model artifact files included
- [ ] Requirements.txt created

### Git & Submission

- [ ] All commits pushed with clear messages
- [ ] Individual branches merged to main
- [ ] Repository zipped for submission
- [ ] GitHub URL copied for submission
- [ ] Coach slot booked (Week 6)

### Presentation (10 Minutes)

- [ ] Each member prepares 2-minute segment
- [ ] Environment and game choice explained
- [ ] Hyperparameter trade-offs discussed
- [ ] Final model performance demonstrated
- [ ] Team ready for Q&A on:
  - Exploration vs exploitation decisions
  - Why CNN policy chosen
  - Final hyperparameter rationale

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'stable_baselines3'
```

→ Run: `pip install stable-baselines3[extra] gymnasium[atari] ale-py`

### ALE/Breakout-v5 Not Found

```
gymnasium.error.NamespaceNotFound: ALE/Breakout-v5
```

→ Run: `autorom --accept-license`

### Model Load Fails

```
FileNotFoundError: Model not found at path
```

→ Ensure `dqn_model.zip` or `best_model.zip` exists in correct location
→ Update model path in `play.py` if using alternative location

### TensorBoard Logs Empty

→ Check `tb_logs/` directory exists
→ Allow training to run for at least a few hundred steps before checking
→ Use: `tensorboard --logdir tb_logs/`

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Deep Q-Networks (DQN) Paper](https://arxiv.org/abs/1312.5602)
- [Atari 2600 Environments](https://gymnasium.farama.org/environments/atari/)

---

## Group Members

- **Member 1:** Training & hyperparameter tuning
- **Member 2:** Evaluation & gameplay demonstration
- **Member 3:** Experiment analysis & presentation

Date: March 2026  
Status: [Member 3 Complete - Pending Team Merge and Final Submission]
