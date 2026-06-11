# Weng-Lin (OpenSkill) ratings: free-for-alls and teams

Two datasets:

- **`f1-2024.matchups`** — the 2024 F1 season again (see
  [`../rankings/`](../rankings/README.md) for provenance), but read as
  Weng-Lin matchups: each race is a 20-way free-for-all of single-player
  "teams". Where Plackett-Luce fits one batch MLE, Weng-Lin processes
  races *in order* and keeps a running `(μ, σ)` per driver — this is what
  you want for a live leaderboard that updates after every event.
- **`doubles.matchups`** — a hand-written six-match doubles league
  showing real teams (two players each), changing partners, and ties
  (`=`). Players who win with *different* partners (anna) separate from
  players who only win alongside them.

File format: one match per line; `|` separates teams from best to worst;
`=` joins teams that tied; players within a team are whitespace-separated.

`./run.sh` rates both with the Bradley-Terry variant (logistic, the
default) and the Thurstone-Mosteller variant (probit with a draw margin),
prints `mu sigma ordinal` per player (ordinal = μ − 3σ, the conservative
"displayed rating" convention), and demonstrates mid-season state
save/resume — the resumed ratings are byte-identical to the uninterrupted
season.

One thing the F1 output makes visible: with 20-driver matches each race
contributes 19 pairwise updates per driver, so σ shrinks very fast and
late races dominate μ — this is an online filter, not a season-long MLE
(compare `../rankings/` Plackett-Luce for that). `out/f1-tau.scores`
shows the standard remedy: `--tau 0.0833` re-inflates σ before each match
so the ratings stay adaptive.
