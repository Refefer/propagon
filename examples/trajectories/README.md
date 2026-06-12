# Trajectories: ranking funnel states by expected value

`sessions` holds 200 synthetic checkout sessions, one episode per
blank-line-separated block of `state reward` steps:

    landing 0
    cart_b 0
    purchase 52.13

Two cart-page variants are in play: `cart_a` converts 35% of the time at a
steady ~$40; `cart_b` converts 45% at a noisier ~$48. Ranking states by
expected discounted return answers "which step of the funnel carries the
value?" — and comparing `cart_a` vs `cart_b` with bootstrap uncertainty
answers "is B actually better, or is the noise fooling us?"

Things to look for in `out/`:

- `monte-carlo` puts `purchase` on top (it banks the reward), `cart_b`
  above `cart_a`, and `bounce`/`abandon` at zero;
- `compare` attaches confidence intervals — the `P(cart_b > cart_a)`
  exceedance line is the decision-grade readout;
- `td` reaches similar values from one-step updates (useful when episodes
  never end); try `--passes 1` vs `--passes 50` to watch it converge;
- `behavior-cloning` ignores rewards entirely: it ranks states by how
  often sessions *visit* them — popularity, not value. The disagreement
  with `monte-carlo` (e.g. `abandon` visited often, worth nothing) is the
  point of having both.

## Provenance

Synthetic, regenerable — the Python block in this directory's history
(seeded with 42) produced `sessions`.
