# Crowdsourced votes with unreliable annotators

`votes` holds 600 synthetic pairwise judgments over six poems with a known
true order (`sonnet > haiku > limerick > ballad > ode > epigram`), cast by
three annotators with very different behavior:

- **alice** — diligent: votes with the true order 92% of the time;
- **bob** — a spammer: flips a coin;
- **mallory** — adversarial: votes *against* the true order 88% of the time.

File format: `annotator winner loser [weight]`, one vote per line.

Run `./run.sh` and look at the second output section: Crowd-BT estimates a
reliability η per annotator alongside the ranking. Alice lands near 0.9,
bob near 0.5, and mallory far below 0.5 — and because the model knows
mallory is inverted, her votes *add* signal instead of poisoning the fit.
Compare `out/naive.scores` (plain Bradley-Terry on the same votes with the
annotator column dropped) to see what ignoring reliability costs.

## Provenance

Synthetic, regenerable — this exact Python block produced `votes`:

```python
import random
rng = random.Random(2024)
items = ["sonnet", "haiku", "limerick", "ballad", "ode", "epigram"]
annotators = [("alice", 0.92), ("bob", 0.50), ("mallory", 0.12)]
for _ in range(200):
    for name, fidelity in annotators:
        i, j = sorted(rng.sample(range(len(items)), 2))
        better, worse = items[i], items[j]
        if rng.random() >= fidelity:
            better, worse = worse, better
        print(name, better, worse)
```
