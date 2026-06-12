# Bandit policies on a reward log

`rewards` is a synthetic log of 600 Bernoulli rewards from four checkout
variants with true conversion rates:

| arm | true rate |
|---|---|
| checkout-a | 0.30 |
| checkout-b | 0.22 |
| **checkout-c** | **0.38** |
| checkout-d | 0.10 |

File format: `arm reward`, one observation per line.

`./run.sh` replays the log through all seven policies. Things to look at:

- the score files rank **checkout-c** first under every policy (Thompson's
  occasional `--select` exploration of runner-up arms is the policy working
  as intended);
- `--select 1` answers "which arm should the *next* visitor see?" —
  vary `--seed` and Thompson Sampling occasionally explores a non-best arm;
  greedy never does;
- the state save/resume step shows selection streams surviving restarts:
  the resumed pick equals the uninterrupted one.

## Interpreting the scores

Each policy's `.scores` file lists `arm: value`, but the meaning of that
value differs by policy — it is **not** always the expected reward:

| Policy | Score meaning |
|---|---|
| `greedy` | Empirical mean reward (Σr / n) |
| `epsilon-greedy` | Empirical mean reward (same as greedy) |
| `upper-confidence-bound` | UCB index: mean + √(exploration · ln(t) / n). Higher than the mean; arms with fewer pulls get an optimism bonus. |
| `kl-ucb` | KL-UCB index: the largest p whose KL-divergence from the empirical mean fits within the confidence bound. Higher than the mean. |
| `thompson-beta` | Posterior mean of a Beta(α, β) distribution. Close to the empirical mean but regularised by the prior (default α=β=1 adds one pseudo-success and one pseudo-failure). |
| `thompson-gaussian` | Posterior mean of a Gaussian with prior shrinkage. Pulled toward `prior_mean` (default 0) proportional to `prior_weight` (default 1). |
| `exp3` | Importance-weighted reward estimate under the current mixing distribution. Not a direct mean — it reflects the adversarial-setting exponential weights update. |

For ranking purposes the ordering is what matters; the absolute value only
has a direct probability interpretation for the mean-based policies.

## Provenance

Synthetic, regenerable — this exact Python block produced `rewards`:

```python
import random
rng = random.Random(2)
arms = [("checkout-a", 0.30), ("checkout-b", 0.22),
        ("checkout-c", 0.38), ("checkout-d", 0.10)]
for _ in range(150):
    for arm, p in arms:
        print(arm, 1 if rng.random() < p else 0)
```
