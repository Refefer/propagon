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
- `upper-confidence-bound` and `kl-ucb` report *indices* (mean + bonus),
  not means — arms with little data score optimistically high;
- `--select 1` answers "which arm should the *next* visitor see?" —
  vary `--seed` and Thompson Sampling occasionally explores a non-best arm;
  greedy never does;
- the state save/resume step shows selection streams surviving restarts:
  the resumed pick equals the uninterrupted one.

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
