# Aggregating full rankings: the 2024 Formula 1 season

`f1-2024.rankings` holds all 24 races of the 2024 Formula One World
Championship — one line per race in calendar order, each line the
classified finishers best-first as three-letter driver codes
(non-classified drivers excluded, so ballots are *partial*: lines run
15–20 drivers, and mid-season substitutes appear in only some races).

`./run.sh` aggregates the season four ways:

- **plackett-luce** — maximum-likelihood worths under the choice-cascade
  model; every finishing position contributes, not just wins. The honest
  "who was strongest" answer.
- **markov-chain (mc4)** — majority-move random walk; robust to a few
  weird races, generalizes Copeland.
- **borda-count** — positional points (m − rank per race): the same idea
  as the real championship's 25-18-15… table, with a flatter curve.
- **kemeny** — the consensus *order* minimizing pairwise disagreement
  with all 24 races.

Worth checking: the real 2024 championship order was decided by F1's
points table (VER, NOR, LEC, PIA, …). Plackett-Luce, which uses full
finishing orders rather than top-10 points, broadly agrees at the front —
where the methods disagree further down is exactly where points tables
hide information.

Partial appearances matter: substitute drivers (BEA, COL, LAW, DOO) raced
a handful of times; Plackett-Luce handles their short ballots exactly,
no imputation.

## Provenance

Compiled from the World Drivers' Championship results grid of Wikipedia's
"2024 Formula One World Championship" article (verified during import:
per-driver win counts match the published wins table — VER 9, NOR 4,
LEC 3, SAI 2, PIA 2, HAM 2, RUS 2 — and the Monaco, British, and Las Vegas
classifications were checked line-for-line against the race articles).
Sporting results are facts; the compilation here is propagon's.
