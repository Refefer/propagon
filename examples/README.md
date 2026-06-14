# propagon examples

One directory per input shape, each with real or documented-synthetic data,
a `README.md` explaining what to look for, and a `run.sh` that exercises
every relevant algorithm (binary expected at `target/release/propagon`;
override with `PROPAGON_BIN`). Outputs land in each directory's `out/`
(gitignored).

| directory | input shape | file format | algorithms |
|---|---|---|---|
| [`tournament/`](tournament/) | pairwise outcomes (2018 MLB season) | `winner loser [weight]` | win-rate, elo, glicko2, bradley-terry (mm/sgd), bayesian-bt, lsr, rank-centrality, rum, kemeny, borda, copeland, massey, colley, keener, hodge-rank |
| [`rankings/`](rankings/) | full/partial rankings (F1 2024 season) | one ballot per line, best first | plackett-luce, mc4, borda-count, kemeny |
| [`matchups/`](matchups/) | multi-team ranked matches (SC2 2v2 circuit, F1 as FFA) | teams `\|`-separated best-first, `=` ties | weng-lin (openskill) |
| [`crowd/`](crowd/) | annotator-tagged votes | `annotator winner loser [weight]` | crowd-bt |
| [`graph/`](graph/) | directed weighted edges (Wikipedia links) | `src dst [weight]` | page-rank, hits, katz, degree, k-core, birank, components |
| [`bandit/`](bandit/) | reward log | `arm reward` | greedy, ε-greedy, ucb1, kl-ucb, thompson (beta/gaussian), exp3 |
| [`odds/`](odds/) | betting odds / forecasts / trades | blank-line events of `outcome odds`; `source outcome prob`; `outcome shares` | devig (multiplicative/power/shin), opinion-pool, lmsr, kelly, clv, calibrate |

Cross-cutting flags work everywhere: `--threads N`, `--format tsv|jsonl`,
`--save-state FILE`, `--load-state FILE`. See the top-level README for the
algorithm-selection guide and `docs/algorithms.md` for the full survey.
