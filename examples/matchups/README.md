# Weng-Lin (OpenSkill) ratings: free-for-alls and teams

Three datasets:

- **`uthermal-2v2.matchups`** — the uThermal 2v2 Circuit Main Event
  (StarCraft II, Aug 8–17 2025): 18 pros, 44 series, 117 played maps, one
  line per map (`winnerA winnerB | loserA loserB`). The format's twist is
  that **partners rotate between matches** (players were re-paired round
  by round) — which is precisely the situation player-level team rating
  exists for: a fixed-team Elo can't say anything about *MaxPax* as
  opposed to *whoever MaxPax is queued with*, while Weng-Lin splits every
  team update across the two players and lets the individuals emerge.
  The eventual champions (MaxPax and Spirit, who beat trigger + ShoWTimE
  in the final) should surface near the top of `out/uthermal-bt.scores`.
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
so the ratings stay adaptive. The 2v2 data doesn't have this problem:
two-team matches carry a single pairwise update, so σ decays gently
across the 117 games.

## Provenance (uthermal-2v2.matchups)

Compiled from the wikitext of Liquipedia's
[uThermal 2v2 Circuit Main Event](https://liquipedia.net/starcraft2/UThermal_2v2_Circuit/Main_Event)
page (Liquipedia content is
[CC BY-SA 3.0](https://liquipedia.net/commons/Liquipedia:Copyrights)).
Verified during import: per-map tallies reproduce all 44 published series
scores, recomputed group-table aggregates (series and game win–loss per
player) match the page exactly, and three series were spot-checked
map-by-map against the rendered bracket.

Encoding notes: one line per **played** map, group stage in round order
(rounds 1–9, both groups) followed by the playoff rounds, grand final
last. The grand final was a Bo7 in which MaxPax + Spirit started with a
1-map advantage from the upper bracket — the official 4–1 score therefore
corresponds to the **four** actually-played maps in the file (3 wins +
1 loss); no synthetic game was added for the advantage map. Clem appears
nowhere: he gave up his spot pre-event and was replaced by Nicoract.
