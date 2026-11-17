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
| TODO | ppo_mid | 0 | 100,000 | TODO | planned pilot run: measure time and inspect the learning curve first |

## Evaluation runs (50 episodes, seeds 100-149)

| Date | Agent | Config | Train seed | collision_rate | mean_speed | mean_episode_length | mean_reward | Notes |
|---|---|---|---|---|---|---|---|---|
| TODO | random | baseline | - | TODO | TODO | TODO | TODO | |
| TODO | defensive | baseline | - | TODO | TODO | TODO | TODO | |
| TODO | ppo | ppo_low | 0 / 1 / 2 | TODO | TODO | TODO | TODO | |
| TODO | ppo | ppo_mid | 0 / 1 / 2 | TODO | TODO | TODO | TODO | |
| TODO | ppo | ppo_high | 0 / 1 / 2 | TODO | TODO | TODO | TODO | |

## Observations

TODO - fill in after the full runs.
