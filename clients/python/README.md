# propagon (Python)

Python bindings for [propagon](https://github.com/Refefer/propagon) — ranking
from revealed preferences. Turn match outcomes, pairwise choices, rankings,
interaction graphs, reward events, and trajectories into rankings with a broad
catalog of algorithms: Bradley-Terry, Elo, Glicko-2, Luce spectral ranking,
rank aggregation, centrality, multi-armed bandits, and value estimation.

Built with [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs); ships as
an [abi3](https://docs.python.org/3/c-api/stable.html) wheel (one wheel runs on
CPython 3.9+).

## Install

From a checkout of the repository:

```bash
cd clients/python
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release      # builds the extension into the venv
```

To build a redistributable wheel instead:

```bash
maturin build --release        # wheel lands in target/wheels/
```

## Shape of the API

The API mirrors the Rust crate:

1. Build one of the dataset types (string ids are interned for you).
2. Configure an algorithm — every parameter has a default.
3. `fit` (batch) or `init` + `update` (incremental).
4. Read `sorted_scores`/`top`/`score`, or persist with `save_state` and resume
   later with `load_state`.

```python
import propagon

games = propagon.GamesDataset()
games.push_pair("ARI", "COL")
games.push_pair("ARI", "NYM")
games.push_pair("COL", "NYM")

model = propagon.Glicko2().fit(games)
print(model.sorted_scores())     # [('ARI', ...), ('COL', ...), ('NYM', ...)]
print(model.top(1))              # [('ARI', ...)]
```

### Batch vs. incremental

- **Batch** algorithms (Bradley-Terry, PageRank, Plackett-Luce, …) expose
  `fit(data) -> model` and `fit_warm(data, init) -> model`.
- **Online** algorithms (Elo, Glicko-2, Weng-Lin, bandits, …) expose
  `init() -> model` and `update(model, data)` (in place). They also offer a
  convenience `fit(data)` (one `init` + `update`). For true incremental resume —
  where history is never replayed — use `init`/`update` with
  `save_state`/`load_state` between batches.

```python
elo = propagon.Elo(k=24.0)
model = elo.init()
elo.update(model, week1_games)
state = model.save_state()       # persist between sessions

# ... later ...
model = propagon.EloModel.load(state)
elo.update(model, week2_games)   # continues where it left off
```

## Datasets

| Class | Push | Used by |
|---|---|---|
| `PairwiseDataset` | `push(winner, loser, weight=1.0)` | the win/loss family |
| `GamesDataset` | `push_pair(w, l, weight=1.0)`, `push_game(side1, side2, outcome, weight=1.0)` | Elo, Glicko-2, team BT |
| `GraphDataset` | `push(src, dst, weight=1.0)` | PageRank, HITS, centralities |
| `RewardsDataset` | `push(arm, reward)` | bandits |
| `ContextualRewardsDataset` | `push(arm, reward, x)` | LinUCB |
| `MatchupsDataset` | `push_match(teams, ranks)` | Weng-Lin / OpenSkill |
| `AnnotatedPairsDataset` | `push(annotator, winner, loser, weight=1.0)` | crowd BT |
| `RankingsDataset` | `push_ranking(items)` | Plackett-Luce, Borda, Kemeny |
| `TrajectoriesDataset` | `push_step(state, reward)`, `end_episode()` | value estimation |

`GamesDataset` and `PairwiseDataset` support `new_period()` to mark rating
periods. Game outcomes are built with factory methods:

```python
g = propagon.GamesDataset()
g.push_game(["A", "B"], ["C", "D"], propagon.GameOutcome.side1_win(margin=7.0))
g.push_game(["X"], ["Y"], propagon.GameOutcome.tie())
```

## Algorithm catalog

Every algorithm takes keyword-only parameters. **Unit-valued enum** parameters
are passed as strings (the serde kebab-case spelling); **data-carrying** enum
parameters use small factory classes.

| Dataset | Batch | Online |
|---|---|---|
| pairwise | `BradleyTerryMM`, `BradleyTerryLR`, `BayesianBradleyTerry`, `Colley`, `Massey`, `Copeland`, `Keener`, `HodgeRank`, `Lsr`, `ILsr`, `RankCentrality`, `SerialRank`, `ThurstoneMosteller`, `NashAveraging`, `OffenseDefense`, `BladeChest`, `CovariateBt`, `EsRum`, `RandomWalker`, `Whr`, `Borda`, `Kemeny` | `WinRate`, `DuelingBandit` |
| games | `GeneralizedBt`, `TeamBradleyTerry` | `Elo`, `Glicko2`, `MovElo`, `MElo` |
| graph | `PageRank`, `Hits`, `BiRank`, `Degree`, `Harmonic`, `Katz`, `KCore`, `LeaderRank` | |
| rewards | | `Bandit`, `SlidingWindowUcb` |
| contextual | | `LinUcb` |
| rankings | `PlackettLuce`, `Footrule`, `Mallows`, `Mc4` | |
| matchups | | `WengLin` |
| trajectories | `McValue`, `BehaviorCloning`, `ValueCompare` | `TdValue` |
| annotated | `CrowdBt` | |

```python
# unit enum as a string; data-carrying enum via a factory
pr = propagon.PageRank(damping=0.85, sink="uniform",
                       teleport=propagon.Teleport.seeds([("home", 1.0)]))
bandit = propagon.Bandit(policy=propagon.BanditPolicy.ucb1(exploration=2.0), seed=1)
```

Data-carrying enum classes: `GameOutcome`, `BanditPolicy`, `DuelingPolicy`,
`Granularity`, `SourceBudget`, `KemenyPasses`, `Winsorize`, `Teleport`,
`PairwiseTests`. A few models carry extra accessors: `Glicko2Model.players()`
(rating/RD/volatility) and `WengLinModel.ratings()` (mean/uncertainty).

## State persistence

`save_state()` returns header-line JSONL text; `save_state_bytes()` returns the
same as `bytes` (use it when writing to a file to avoid newline re-encoding).
`load_state(text)` reconstructs the right model class from the file's algorithm
tag; each model class also has `ModelClass.load(text)`, which additionally
enforces the tag matches that class.

```python
text = model.save_state()
same = propagon.load_state(text)         # -> the matching model class
assert same.save_state() == text         # save -> load -> save is byte-identical
```

## Errors

Every failure is a subclass of `propagon.PropagonError`:

```
PropagonError
├── InvalidInputError        bad roster, out-of-range parameter, multi-player side on a 1v1 method
├── EmptyDatasetError        fitting an empty dataset
├── NumericError             no convergence, singular system, disconnected graph
├── StateError               malformed / mis-versioned state file
│   └── AlgorithmMismatchError   loading a state produced by a different algorithm
├── ParamMismatchError       resuming with incompatible parameters
└── IoError                  underlying I/O or JSON failure
```

## Free functions

- `wilson_interval(successes, failures, z=1.96) -> (low, high)`
- `extract_components(graph, min_size=1) -> list[GraphDataset]`
- `load_state(text) -> model`

## Not yet exposed

The `Bootstrap` interval wrapper and the `DifferentialEvolution` optimizer are
available in the Rust core but not yet surfaced in Python: the former needs
generic-over-`Ranker` dispatch and the latter a Python fitness callback. Track
these as follow-ups.

## Tests

```bash
maturin develop && pytest
```

The suite checks correctness against published reference vectors (Agresti's
Bradley-Terry baseball, Newcombe's Wilson intervals, the Tennessee
Borda/Condorcet election, Langville & Meyer's PageRank, the *Who's #1?*
Massey/Colley ratings, Beta-posterior arithmetic), byte-identical state
round-trips, the incremental "split update equals continuous update" guarantee,
the error hierarchy, determinism at fixed seeds, and that every algorithm fits
and round-trips its state.
