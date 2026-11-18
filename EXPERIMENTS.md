# Experiment log

One row per run. Only record numbers that were actually produced by the
scripts (`results/eval_summary.csv`); never fill cells by hand from memory.
`mean_reward` is comparable only within the same config.

## Pipeline checks (not experimental results)

| Date | What | Config | Seed | Steps / episodes | Outcome |
|---|---|---|---|---|---|
| 2026-11-17 | Training smoke test | ppo_mid | 0 | 512 steps | ran in 143 s (~0.28 s/step); model and Monitor log written |
| 2026-11-17 | Evaluation smoke test | ppo_mid | 0 | 2 episodes | CSV schema correct (model was barely trained; numbers not meaningful) |
| 2026-11-17 | Baseline sanity check | baseline | - | 4 episodes each | defensive collided less than random; too few episodes to report |
| 2026-11-17 | `pytest tests` | - | - | - | 9 passed |

## Training runs

| Date | Config | Seed | Steps | Wall time | Notes |
|---|---|---|---|---|---|
| 2026-11-17 22:45 -> 2026-11-18 06:09 | ppo_mid | 0 | 100,096 (390 x n_steps=256 rounds up from 100,000) | 7 h 24 min (~0.27 s/step) | pilot run; no errors; checkpoints every 10k steps. Last training `ep_rew_mean` 4.77, `explained_variance` 0.61 |

## Evaluation runs (50 episodes, seeds 100-149)

| Date | Agent | Config | Train seed | collision_rate | mean_speed | mean_episode_length | mean_reward | Notes |
|---|---|---|---|---|---|---|---|---|
| TODO | random | baseline | - | TODO | TODO | TODO | TODO | |
| TODO | defensive | baseline | - | TODO | TODO | TODO | TODO | |
| TODO | ppo | ppo_low | 0 / 1 / 2 | TODO | TODO | TODO | TODO | |
| 2026-11-18 | ppo | ppo_mid | 0 | 0.10 | 20.48 | 36.92 | 3.74 | pilot run, single seed; `mean_reward` only comparable within ppo_mid |
| TODO | ppo | ppo_mid | 1 / 2 | TODO | TODO | TODO | TODO | |
| TODO | ppo | ppo_high | 0 / 1 / 2 | TODO | TODO | TODO | TODO | |

## Observations

- **Pilot (ppo_mid, seed 0, single seed - no conclusions yet).** The agent
  collided in 10% of the 50 evaluation episodes, but its mean speed (20.48 m/s)
  is close to the bottom of `reward_speed_range` [20, 30], where the speed
  bonus is zero. It may have learned to drive slowly to stay safe. This is a
  single run, so it must be checked against seeds 1 and 2 and against the
  low / high penalty configs before drawing any conclusion.
- Compute: ~7.5 h per 100k-step run on this CPU, so the remaining 8 runs need
  parallel processes or a reduced budget.

TODO - fill in the rest after the full runs.
