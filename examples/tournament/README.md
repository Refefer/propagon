# Ranking a season of pairwise outcomes

`baseball.2018` is the full 2018 MLB regular season — 2,431 games as
`winner<TAB>loser<TAB>1` rows (quoted team codes; threshold 1 = a win with unit margin). `baseball.2018.edges`
is the same season with numeric ids (kept because the test suite pins
golden outputs against it). Real data, all 30 teams, ~162 games each: dense
enough that every algorithm has something to say.

`./run.sh` ranks the season fourteen ways. A field guide:

| output | what its scores mean |
|---|---|
| `win-rate` | Wilson upper bound on win probability — the honest counting baseline |
| `elo` | online ratings; order-dependent by design |
| `glicko2` | rating, deviation, and a 95% interval per team |
| `bradley-terry-mm` / `-sgd` | maximum-likelihood strengths (normalized) |
| `bayes-bt` | BT posterior mean with a credible interval per team |
| `luce-spectral-ranking`, `rank-centrality` | spectral one-shot BT estimates |
| `random-utility-model` | (μ, σ) per team — "good but erratic" is visible |
| `kemeny` | consensus order (rank positions, not strengths) |
| `borda-count`, `copeland`, `colley` | counting methods; Colley adds schedule adjustment |
| `keener` | eigenvector rating with automatic strength-of-schedule |
| `massey` | least-squares ratings from margins (this file carries none — demo only) |
| `hodge-rank` | potentials **plus** the inconsistency share printed to stderr |

Worth trying: compare `win-rate` to `bradley-terry-mm` ranks (schedule
strength moves teams several places), and check HodgeRank's inconsistency
number — baseball seasons are noisy, so a meaningful fraction of the win
flow is cyclic and *no* total order can explain it.

The save/load step at the end shows Glicko-2 resuming from a state file —
feed next week's games to `--load-state` instead of refitting the season.

## Provenance

2018 MLB regular-season results, shipped with propagon v1 (compiled from
public box scores).
