# highway-rl-study

How does the collision penalty in the reward function affect the behaviour of a
reinforcement-learning driving agent?

## Research question

> How does the collision penalty weight (`collision_reward`) affect the collision
> rate (`collision_rate`) and the mean speed (`mean_speed`) of an autonomous
> driving agent in [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv)?

**Hypothesis.** A small penalty (`-1`) lets PPO trade safety for speed (higher
speed, more collisions); a large penalty (`-20`) yields more cautious driving
(fewer collisions, possibly lower speed). This is a hypothesis, not a finding.

## Experimental design

| Config | `collision_reward` | Role |
|---|---|---|
| `configs/ppo_low.yaml` | -1.0 | light penalty |
| `configs/ppo_mid.yaml` | -5.0 | reference |
| `configs/ppo_high.yaml` | -20.0 | heavy penalty |
| `configs/baseline.yaml` | -5.0 | env spec for baselines (same as `ppo_mid`) |

- **Single independent variable**: `collision_reward`. All other reward terms
  (`high_speed_reward=0.4`, `right_lane_reward=0.1`, `lane_change_reward=0.0`),
  PPO hyperparameters and the environment are identical across configs.
- **`normalize_reward: false`**. In highway-env v1.12.1 the default reward
  normalisation is `(r - collision_reward) / (max_r - collision_reward)`. With
  `collision_reward=-20`, all non-collision rewards are squeezed into roughly
  `[0.976, 1.0]`, which would distort the comparison. `make_env` therefore
  always disables it.
- **Training**: PPO (Stable-Baselines3, `MlpPolicy`), 100,000 steps, training
  seeds `0, 1, 2`.
- **Evaluation**: 50 fixed episodes per trained model, episode `i` uses seed
  `100 + i` (disjoint from training seeds), deterministic policy.
- **Baselines**: a uniformly random agent (lower bound) and a rule-based
  "defensive driver" that reads the simulator state directly (keeps a 40 m gap,
  changes lane when blocked, otherwise drifts to the right lane). Neither uses
  the reward, so their behaviour does not depend on `collision_reward`.

### Metrics

| Metric | Definition | Cross-config comparison |
|---|---|---|
| `collision_rate` | fraction of episodes with a crash | yes |
| `mean_speed` | per-episode mean longitudinal speed, averaged over episodes | yes |
| `mean_episode_length` | steps survived (max 40) | yes |
| `mean_reward` | mean cumulative reward | **no** - scales differ per config |

Results are reported as mean ± std over the 3 training seeds.

## Project layout

```
configs/            YAML experiment configs (env, train, eval)
src/env_factory.py  build highway-env from a config (forces normalize_reward=False)
src/baseline.py     RandomAgent, DefensiveDriverAgent
src/train.py        PPO training, saves models/<config>_seed<N>.zip
src/evaluate.py     fixed-seed evaluation, appends to results/eval_summary.csv
src/plot_results.py mean ± std bar charts -> results/comparison.png
tests/              pytest suite
docs/PROJECT_PLAN.md  detailed plan and progress notes (Vietnamese)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch          # CPU build is enough
```

Developed with Python 3.12, gymnasium 1.4, highway-env 1.12.1,
stable-baselines3 2.9.

## Usage

```powershell
# Train one run (add --timesteps 2000 for a smoke test)
python -m src.train --config configs/ppo_mid.yaml --seed 0

# Evaluate
python -m src.evaluate --agent ppo --config configs/ppo_mid.yaml --train-seed 0
python -m src.evaluate --agent random --config configs/baseline.yaml
python -m src.evaluate --agent defensive --config configs/baseline.yaml

# Plot and test
python -m src.plot_results
python -m pytest tests -q
```

## Results

**TODO** - the full 3 configs x 3 seeds experiment has not been run yet. The
table below will be filled from `results/eval_summary.csv`; no numbers are
reported until then. See [EXPERIMENTS.md](EXPERIMENTS.md) for the run log.

| Agent / config | collision_rate | mean_speed [m/s] | mean_episode_length |
|---|---|---|---|
| random | TODO | TODO | TODO |
| defensive | TODO | TODO | TODO |
| ppo_low | TODO | TODO | TODO |
| ppo_mid | TODO | TODO | TODO |
| ppo_high | TODO | TODO | TODO |

## Limitations

- **Compute**: the simulator runs at roughly 0.3 s per step on the development
  laptop CPU, i.e. about 8 hours per 100k-step run. This limits the number of
  seeds and the training budget.
- **Few seeds**: 3 training seeds give only a rough estimate of variance; no
  significance tests are performed.
- **One environment and one algorithm**: results are for `highway-v0` with
  PPO and the hyperparameters in the configs; they may not transfer.
- **Reward scale**: changing `collision_reward` also changes the overall reward
  scale seen by PPO, so the effect is not purely "preference for safety".
- **Collision rate** counts an episode as a collision if the ego vehicle
  crashed at any step.
