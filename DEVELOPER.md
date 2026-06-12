# Developing propagon

How the workspace is built, how to use the library, the full CLI reference,
and the contracts you must not break when adding to it. For *what each
algorithm does and when to reach for it*, see the catalog in
[`README.md`](README.md); the method survey with citations is
[`docs/algorithms.md`](docs/algorithms.md); the product spec is
[`docs/PRD.md`](docs/PRD.md); the non-negotiable coding rules are
[`AGENTS.md`](AGENTS.md).

## Build and install

```bash
git clone https://github.com/Refefer/propagon
cd propagon
cargo build --release        # CLI binary at target/release/propagon
```

MSRV is **Rust 1.88** (the code uses let-chains). To put the CLI on your PATH:

```bash
cargo install --path crates/propagon-cli
```

The repository ships git hooks under `.githooks/` (fmt/clippy/doc on commit,
a Conventional-Commits subject check). Enable them with:

```bash
git config core.hooksPath .githooks
```

## Using propagon as a library

The workspace is two crates: `propagon` (the library — no I/O coupling, WASM-clean)
and `propagon-cli` (the `propagon` binary). Depend on the library directly:

```toml
[dependencies]
propagon = { path = "crates/propagon" }   # or a version / git ref
```

### Feature flags

- **`parallel`** (default) — multi-threaded fitting via rayon. Disable it for
  single-threaded targets; the WASM build runs with it off and the whole suite
  passes either way. Fitting stays bit-stable across thread counts (sequential
  merge, no atomics in the hot loops).
- **`io`** (default) — `save_to_path` / `load_from_path` convenience methods.
  The in-memory forms (`save_jsonl` / `load_jsonl` over any `Write` / `BufRead`)
  are always available, so a no-`io` build still persists models.

```toml
# Minimal, single-threaded, still persists through your own Write/BufRead:
propagon = { path = "crates/propagon", default-features = false, features = ["io"] }
```

### Shape of the API

Data flows **dataset → ranker → model**. Pick the dataset shape that matches
your data, fit a ranker over it, and read the model.

The nine dataset shapes (each owns its `Interner`(s); string ids are interned
to dense `u32` on `push`):

| Dataset | Holds | Feeds |
|---|---|---|
| `PairwiseDataset` | winner/loser/weight + optional rating periods | Bradley-Terry, LSR, Rank Centrality, Massey, Keener, … |
| `GamesDataset` | two-sided games with tie/margin outcomes, repeat counts, periods | Elo, Glicko-2, Generalized/Team BT, mElo, WHR inputs; lowers to pairwise/matchups |
| `RankingsDataset` | CSR-packed ballots (best-first), with `to_pairwise()` rank-breaking | Plackett-Luce, Kemeny, Borda, Mallows, Footrule, MC4 |
| `GraphDataset` | directed weighted edges | PageRank, HITS, Katz, LeaderRank, Harmonic, BiRank, … |
| `RewardsDataset` | arm/reward events | the bandit policies, Sliding-Window UCB |
| `ContextualRewardsDataset` | arm/reward + context features | LinUCB |
| `MatchupsDataset` | multi-team ranked matches (two-level CSR) | Weng-Lin |
| `AnnotatedPairsDataset` | annotator-tagged votes (two interners) | Crowd-BT |
| `TrajectoriesDataset` | CSR episodes of (state, reward) | Monte Carlo / TD values, Value Comparison, Behavior Cloning |

Two fitting tiers and one model trait:

- **`Ranker`** — batch: `fit(&data) -> Model` (plus `fit_warm(&data, &init)` for
  the iterative methods). All batch rankers can be wrapped by `Bootstrap<R>` for
  score/rank confidence intervals.
- **`OnlineRanker`** — incremental: `init() -> Model`, then
  `update(&mut model, &batch)` folds new data into existing state without
  replaying history (Elo, Glicko-2, Weng-Lin, win-rate, bandits, TD, mElo,
  dueling bandits).
- **`RankModel`** — every model is owned and self-contained (carries its own
  names, never borrows the dataset): `scores()`, `sorted_scores()` (descending,
  name-tiebreak), and JSONL persistence.

```rust
use propagon::algos::{BradleyTerryMM, Glicko2};
use propagon::{GamesDataset, OnlineRanker, RankModel, Ranker, TiePolicy};

let mut games = GamesDataset::new();
games.push_pair("alice", "bob", 2.0)?;   // alice beat bob twice
games.push_pair("bob", "carol", 1.0)?;
games.push_pair("alice", "carol", 1.0)?;

// Batch MLE (lower the games to win/loss pairs first):
let bt = BradleyTerryMM::default().fit(&games.to_pairwise(TiePolicy::Error)?)?;
println!("{:?}", bt.sorted_scores());

// Incremental ratings with resumable, human-readable state:
let glicko = Glicko2::default();
let mut ratings = glicko.init();
glicko.update(&mut ratings, &games)?;
ratings.save_to_path("ratings.jsonl")?;  // plain JSONL — head it, grep it
```

Per-algorithm library snippets (construction, the exact accessors, how to read
extras like `predict`, `matchup`, `timeline`, `coefficients`, `gap`) are in the
README catalog, one block per algorithm.

## Algorithm → command reference

Every implemented algorithm by CLI group (survey § references point into
[`docs/algorithms.md`](docs/algorithms.md)).

| Group | Algorithm (survey §) | Command |
|---|---|---|
| `tournament` | Bradley-Terry, MM or logistic SGD (§1.1) | `bradley-terry-model` (`btm`) |
| | Elo (§2.1) | `elo` |
| | Glicko-2 (§2.3) | `glicko2` |
| | Luce Spectral Ranking (§3.2) | `luce-spectral-ranking` (`lsr`) |
| | Rank Centrality (§3.1) | `rank-centrality` |
| | Gaussian RUM via evolution strategies (§1.5) | `random-utility-model` (`rum`) |
| | Keener (§3.3) | `keener` |
| | HodgeRank (§3.6) | `hodge-rank` (`hodge`) |
| | Massey (§5.1) | `massey` |
| | Colley (§5.2) | `colley` |
| | Borda count (§6.1) | `borda-count` (`borda`) |
| | Copeland (§6.2) | `copeland` |
| | Kemeny consensus, heuristic (§6.3) | `kemeny` |
| | Wilson-score win rate (§7.1) | `win-rate` |
| | Bayesian Bradley-Terry (§11.1) | `bayesian-bradley-terry` (`bayes-bt`) |
| | Thurstone-Mosteller Case V (§1.3) | `thurstone-mosteller` (`tm`) |
| | Generalized BT: ties + home advantage (§1.2) | `generalized-bradley-terry` (`gbt`) |
| | Team BT, player strengths from team games (§1.2) | `team-bradley-terry` (`team-bt`) |
| | Margin-of-victory Elo (§2.1) | `elo --margin-of-victory` |
| | I-LSR, exact Plackett-Luce MLE (§3.2) | `i-luce-spectral-ranking` (`ilsr`) |
| | SerialRank (§3.5) | `serial-rank` |
| | Random-walker rankings (§3.4) | `random-walker` |
| | Offense-defense Sinkhorn ratings (§5.3) | `offense-defense` (`od`) |
| | Whole-History Rating (§2.6) | `whole-history-rating` (`whr`) |
| | Multidimensional Elo, mElo (§9.2) | `melo` |
| | Nash averaging (§9.2) | `nash-averaging` (`nash`) |
| | Blade-Chest intransitivity embeddings (§9.1) | `blade-chest` |
| | Covariate BT / conditional logit (§10.1) | `covariate-bradley-terry` (`cbt`) |
| `rankings` | Plackett-Luce (§1.4) | `plackett-luce` (`pl`) |
| | I-LSR on ballots (§3.2) | `i-luce-spectral-ranking` (`ilsr`) |
| | Mallows dispersion φ (§1.7) | `mallows` |
| | Footrule-optimal aggregation (§6.5) | `footrule` |
| | Markov-chain aggregation, MC4 (§6.4) | `markov-chain` (`mc4`) |
| | Borda count (§6.1) | `borda-count` (`borda`) |
| | Kemeny consensus (§6.3) | `kemeny` |
| `crowd` | Crowd-BT, annotator-aware (§11.2) | `bradley-terry` (`crowd-bt`) |
| `matchups` | Weng-Lin / OpenSkill (§2.5) | `weng-lin` (`openskill`) |
| `graph` | PageRank / personalized PageRank (§4.4) | `page-rank` (`--seeds` for PPR/RWR) |
| | LeaderRank (§4.9) | `leader-rank` |
| | Harmonic centrality (§4.10) | `harmonic` |
| | BiRank (§4.7) | `birank` |
| | HITS (§4.5) | `hits` |
| | Katz centrality (§4.3) | `katz-centrality` (`katz`) |
| | Degree/strength (§4.1) | `degree` |
| | k-core decomposition (§4.11) | `k-core` (`kcore`) |
| | Connected components (utility, §14.1) | `components` |
| `bandit` | greedy, ε-greedy, UCB1, KL-UCB, EXP3, Thompson Beta/Gaussian (§8.1) | `greedy`, `epsilon-greedy`, `ucb1`, `kl-ucb`, `exp3`, `ts-beta`, `ts-gaussian` |
| | Sliding-window UCB (§8.1) | `sliding-window-ucb` (`sw-ucb`) |
| | LinUCB, contextual (§8.1) | `linucb` |
| | Dueling bandits: RUCB, Double Thompson Sampling (§8.2) | library API (`DuelingBandit`), no CLI |
| `trajectories` | Monte Carlo state values (§13.1) | `monte-carlo` (`mc`) |
| | TD(0) state values (§13.3) | `td` |
| | Bootstrap value comparison (§13.2) | `compare` |
| | Counting behavior cloning (§13.6) | `behavior-cloning` (`bc`) |
| *(any batch)* | Bootstrap intervals on scores and ranks (§11.4) | `--bootstrap N` on every batch command |

Flags appear only on the commands that use them — each command's `--help` lists
exactly what it accepts, and an irrelevant flag is a parse error, not a runtime
one (every value-bearing flag also shows its `[default: …]`, sourced from the
algorithm's `Default`). Universal: `--threads N`, `--format tsv|jsonl`,
`--save-state PATH`. `--load-state PATH` is on the resumable commands only
(online updaters and the warm-startable batch fits). `--bootstrap N`
(+ `--bootstrap-credible`, `--bootstrap-seed`) is on the batch rankers that can
be wrapped — not the online ones (elo, glicko2, win-rate, melo, whr, td) or the
already-bootstrap ones (compare). `--ties error|discard|half-win` is on the
win/loss-lowering tournament commands; `--margin-ties error|discard|zero` on the
margin ones (massey, keener, offense-defense, hodge-rank); `--groups-are-separate`
on the period-aware ones (glicko2, whr); `--min-count` on the whole tournament
group.

## CLI reference: key flags

The README catalog has a usage block for every command; this is the dense
tuning reference for the flags that change results most.

**`tournament bradley-terry-model`** — `--estimator mm` (exact MM iteration, the default) or `sgd` (logistic gradient descent: streams better, supports `--passes/--alpha/--decay`). MM needs a connected comparison graph with no undefeated entities; mitigations built in: `--remove-total-losers`, `--create-fake-games W`, `--random-subgraph-links N`. Tighten `--tolerance` (default 1e-6) for publication-grade fits.

**`tournament generalized-bradley-terry`** — `--tie-model none|davidson|rao-kupper` (default `davidson`) models draws; `--home-advantage` estimates a multiplicative edge for the side listed **first**. Fitted tie ν/θ and home γ are reported on stderr. (Handles ties natively, so it has its own `--tie-model` and no `--ties`.)

**`tournament team-bradley-terry`** — `--aggregate additive|product` builds team strength from member strengths; tie handling via `--ties`.

**`tournament glicko2`** — `--tau` (0.3–1.2, default 0.5) bounds how fast a player's *volatility* can grow: lower = stabler ratings, higher = faster reaction to form swings. Feed rating periods as blank-line-separated batches with `--groups-are-separate`; each period is one Bayesian update.

**`tournament elo`** — `--k` (default 32) is the whole game: high K tracks fast and stays noisy, low K converges and goes stale. `--scale` (400) sets how many points mean 10:1 odds. `--margin-of-victory` switches to MOV-Elo, scaling K by `(ln(1+margin)/ln 2)^mov_exponent` (`--mov-exponent`, default 1.0); a margin-1 win matches plain Elo. Elo is order-dependent and never converges by design — for frozen entities use Bradley-Terry.

**`tournament whole-history-rating`** — `--w2` (Wiener variance per period; smaller = flatter careers) and `--prior-games` (virtual anchor draws). Periods come from `--groups-are-separate`; `--timeline` emits the full `(period, rating, sd)` curve per player. No `--bootstrap` (row resampling would drop the rating periods).

**`tournament melo`** — `--k` cyclic dimension pairs (model rank 2k; `k=0` ≡ Elo), `--lr-rating`/`--lr-vector` step sizes, `--seed`. Online, so no `--bootstrap`. The leaderboard shows the transitive rating; the cyclic part lives in the library `predict(a,b)`.

**`tournament nash-averaging`** — `--iterations`/`--tolerance` for the multiplicative-weights solve, `--learning-rate` (in (0,1]), `--anneal-every` (temperature halving). The duality gap is printed to stderr as a convergence certificate.

**`tournament blade-chest`** — `--variant inner|dist`, `--dims`, `--epochs`, `--lr`, `--l2`, `--seed`. Leaderboard is the transitive γ; the intransitive part is the library `matchup(a,b)`.

**`tournament covariate-bradley-terry`** — `--features FILE` (required; rows `entity x1 … xd`), `--l2` ridge penalty, `--intercepts` adds per-entity offsets (needs `--l2 > 0`). Fitted coefficients β print to stderr.

**`tournament luce-spectral-ranking`** — `--estimator power` (deterministic, default) or `monte-carlo` (`--steps` walk length, `--seed`). Scores are log-scale and mean-centered; only differences mean anything.

**`tournament kemeny`** — `--algo insertion` (fast, ~97%+ of optimum on hard instances) or `de` (differential evolution, bigger budget via `--passes`). NP-hard exactly: both are principled heuristics.

**`tournament random-utility-model`** — `--passes` evolution-strategy iterations, `--gamma` regularization, `--fixed` pins variances (Thurstone-style). Outputs `μ σ` per entity; values are relative.

**`tournament massey` / `colley` / `keener`** — Massey reads `|threshold|` as the **margin of victory** (margins required); Colley uses game counts and ignores margins by design; Keener reads scores per direction. `--margin-ties error|discard|zero` controls tie rows. Keener's `--no-skew` disables blowout damping, `--no-normalize-games` the unequal-schedule correction.

**`tournament hodge-rank`** — `--flow log-odds|win-rate-delta|mean-margin` picks the pairwise statistic to decompose. Watch stderr: the **inconsistency** number is the share of flow living in cycles — near 0 the ranking is trustworthy, near 1 a total order is fiction.

**`tournament bayesian-bradley-terry`** — `--shape` (1.0 = posterior mode at the MLE; higher shrinks entities together), `--samples`/`--burn-in` for the Gibbs chain, `--credible` interval mass, `--seed`. Handles undefeated/winless entities that break plain BT.

**`rankings plackett-luce`** — Hunter's MM on ballots; partial ballots (top-k, subsets) are likelihood-exact. Items last (or first) in *every* ballot have no finite MLE and get sectioned out.

**`rankings markov-chain`** — `--damping` (0.85 web convention; 0.5–0.6 suits sports data) keeps the majority-move chain ergodic.

**`matchups weng-lin`** — `--variant bradley-terry` (logistic, default) or `thurstone-mosteller` (probit, `--epsilon` draw margin); paper defaults μ=25, σ=25/3, β=25/6. `--tau` (try 0.0833) re-inflates σ each match. Output: `mu sigma ordinal` (ordinal = μ−3σ).

**`crowd bradley-terry`** — `--lambda` (0.5) anchors and regularizes via the virtual node; the second output section is per-annotator reliability η (~1 truthful, ~0.5 spammer, <0.5 adversarial).

**`graph page-rank`** — `--damping-factor` (0.85); `--sink-dispersion reverse|all|uniform|none` picks dangling-node treatment; `--seeds a,b,c` / `--seeds-file` switches to personalized PageRank (random-walk-with-restart); `--matches` reads tournament files as loser-endorses-winner.

**`graph hits` / `katz-centrality` / `degree` / `k-core` / `leader-rank` / `harmonic`** — HITS emits authorities then hubs. Katz needs `--alpha` below `1/λ_max` (diverging runs abort with advice — try 0.05 on weighted graphs). `degree --direction in|out|total`. k-core reports coreness (undirected). LeaderRank is parameter-free. `harmonic --direction`/`--weighted`, with `SourceBudget` sampling for very large graphs.

**`bandit *`** — every policy: `--seed` (selection streams reproduce and resume exactly across `--save-state`/`--load-state`), `--select N` prints the next arms to play. `epsilon-greedy --epsilon`; `upper-confidence-bound --exploration` (2.0 = UCB1); `kl-ucb --c` (0 = practical, ≥3 for theory); `thompson-beta` needs rewards in [0,1] (`--prior-alpha/--prior-beta`); `thompson-gaussian --prior-mean/--prior-weight`; `exp3 --gamma` (offline replay is approximate). `sliding-window-ucb --window N --exploration` (no `--seed` — deterministic argmax). `linucb --alpha --ridge`, decision via `--select-for "x1,x2,…"`.

**`trajectories *`** — `monte-carlo --gamma --visit first|every --aggregate mean|median --winsorize q --min-observations`; `td --alpha --gamma --passes --initial-value` (online, supports `--load-state`); `compare --replicates --method bootstrap|bayesian --credible --pairwise PERMS --seed`; `behavior-cloning --per-state SEP --smoothing α --emit-pairs` (export preference edges as tournament rows).

## State files

Every model (and dataset) saves as JSONL with a self-describing header:

```jsonl
{"propagon":1,"kind":"model","algorithm":"glicko2","params":{"tau":0.5,...},"entities":30}
{"id":"BOS","r":1670.8,"rd":40.5,"sigma":0.06}
```

Rules (PRD FR-4): the schema version is mandatory and checked; unknown fields are
tolerated on read; params travel with the state and a mismatched resume is a typed
error, not a silent reconfiguration. All model floats are `f64`, so
save → load → save is byte-identical — this relies on serde_json's
`float_roundtrip` feature (already on; do not drop it). Incremental algorithms
(`glicko2`, `elo`, `weng-lin`, `win-rate`, all bandits, `td`, `melo`) fold new
batches into loaded state without replaying history; iterative ones
(`bradley-terry-model`, `plackett-luce`, `luce-spectral-ranking`,
`rank-centrality`, `whr`) warm-start from it.

## Workspace layout

```
propagon/
├─ Cargo.toml                  # virtual workspace: shared deps, lints, metadata
├─ .githooks/                  # fmt/clippy/doc + commit-msg hooks (opt in via core.hooksPath)
├─ crates/
│  ├─ propagon/                # the library — no I/O coupling, wasm-clean
│  │  ├─ src/
│  │  │  ├─ dataset/           # nine shapes (pairwise, games, rankings, graph,
│  │  │  │                     #   rewards, contextual, matchups, annotated,
│  │  │  │                     #   trajectories) + resample.rs + JSONL dataset io
│  │  │  ├─ algos/             # one module per algorithm (+ bandits, DE optimizer,
│  │  │  │                     #   bootstrap wrapper)
│  │  │  ├─ interner.rs        # string id ↔ dense u32
│  │  │  ├─ traits.rs          # Ranker, OnlineRanker, RankModel, FitOptions
│  │  │  ├─ state.rs           # header-line JSONL model persistence
│  │  │  ├─ solver.rs          # sparse CG (Massey, Colley, HodgeRank)
│  │  │  ├─ parallel.rs        # rayon shim (sequential under --no-default-features)
│  │  │  ├─ progress.rs        # Progress trait (core never prints)
│  │  │  └─ mathx.rs           # normal pdf/CDF/quantile approximations
│  │  └─ tests/                # parity.rs, state.rs, reference.rs
│  └─ propagon-cli/            # bin "propagon": clap wiring, file parsing, emitters
│     └─ tests/golden.rs + golden/   # captured v1 outputs + harness
├─ docs/                       # algorithms survey, PRD
├─ examples/                   # seven worked demos (one per input shape)
└─ scripts/capture_golden.sh   # provenance of the golden files (v1 binary, run once)
```

## Core contracts

**Datasets are shared and immutable during fitting.** Algorithms never define
private input formats — they consume one of the nine shapes above. Each owns its
`Interner`(s); the invariant that every stored id came from the owning interner
is established at `push`/`push_ids` and relied on everywhere (that is what makes
`Interner::resolve` total). `GamesDataset` is the v2 tournament input: it carries
tie and margin outcomes and **lowers** to the older shapes on demand —
`to_pairwise(TiePolicy)`, `margin_pairs(MarginTies)`, `to_matchups()` — so
existing rankers keep their pairwise/matchups input types.

**Two fitting tiers** (PRD FR-5):

- `Ranker::fit(&data) -> Model` + `fit_warm(&data, &init)` — batch, with
  warm-starting for the iterative methods (BT-MM, BT-SGD, LSR, Plackett-Luce,
  I-LSR, WHR). Contract: `fit_warm` never converges to a worse objective than
  `fit`. Any `Ranker` over a `Resample` dataset can be wrapped by `Bootstrap<R>`.
- `OnlineRanker::init() -> Model` + `update(&mut model, &batch)` — true
  incremental state (Glicko-2, Elo, Weng-Lin, win-rate, bandits, TD, mElo,
  dueling bandits). Contract: `update` never replays history, and split updates
  equal one continuous run (tested in `tests/state.rs` and the CLI golden flow).
  Documented exception: EXP3's replay is order-dependent, so its `merge`
  approximates the concatenated log. Online rankers are **excluded** from
  `Bootstrap` by the type bound — resampling rows changes what an order-dependent
  model means, so their CLI commands don't carry `--bootstrap` at all.

**Resampling**: `Resample` is implemented for every shape, with the natural unit
(rows / ballots / edges / matches / games / episodes). `Bootstrap<R: Ranker>`
fits the inner ranker once for point estimates, then refits on `replicates`
resampled copies; it reports per-entity score **and** rank intervals, skips and
counts replicates that fail to fit, and errors only if fewer than half survive.

**Models are owned and self-contained** — they carry their own names, never
borrow the dataset. `RankModel` provides `scores()`, `sorted_scores()`
(descending, name-tiebreak), and JSONL persistence. A few models expose extra
accessors beyond `scores()` (e.g. WHR `timeline`, mElo `predict`, Blade-Chest
`matchup`, Nash `gap`/`distribution`, Covariate-BT `coefficients`); these are
documented per algorithm in the README catalog.

**Execution options**: `FitOptions { progress: &dyn Progress, threading }`. The
core never prints — the CLI renders `Progress` with indicatif; bindings forward
callbacks. `Threading::Dedicated(pool)` exists so embedders never have their
global rayon pool reconfigured.

**Determinism**: every stochastic algorithm has a `seed` param; fixed seed +
fixed thread count ⇒ identical output. The spectral power methods are bitwise
deterministic at any thread count (transposed accumulation, no atomics). Bandit
and dueling-bandit selection streams persist their draw counter, so
save → load → `select()` is indistinguishable from an uninterrupted run.

## AGENTS.md compliance notes

The workspace denies `clippy::unwrap_used`, `expect_used`, and `panic`. Practical
patterns in use:

- Unit tests are exempt via `#![cfg_attr(test, allow(...))]` in `lib.rs`.
  **Integration tests are not** — they return
  `Result<(), Box<dyn std::error::Error>>` and propagate with `?`.
- Invariant-by-construction lookups go through `Interner::resolve` (placeholder
  on the impossible miss, never a panic).
- The CLI never re-states numeric defaults: it builds `Algo::default()` and
  overrides only the flags the user passed (`set`/`choice` helpers in `main.rs`).
  Param-struct `Default` impls are the single source; canonical bandit-policy
  values live on `BanditPolicy::DEFAULT_*` consts.
- No `0 = auto` sentinels: optional budgets are enums
  (`KemenyPasses::Auto | Fixed(n)`) or derived from a single core source.
- The CLI crate carries its own `[lints.clippy]` (print lints relaxed there only
  — printing is its job).

## Adding an algorithm

1. **Survey first**: add or locate its entry in `docs/algorithms.md` so the
   "when to use it" story exists before the code, and add a catalog block to
   `README.md` (real-world scenario / when to use / when to avoid / CLI &
   library usage / what the numbers mean).
2. Create `crates/propagon/src/algos/<name>.rs`:
   - a params struct with public fields, documented canonical defaults, and
     `Default`; a `seed: u64` field if anything is stochastic;
   - a model type owning an `Interner` + state vectors; implement `RankModel`
     (the `impl_simple_score_model!` macro covers one-f64-per-entity models);
   - implement `Ranker` (override `fit_warm_opts` if iterative) or `OnlineRanker`
     (state must merge without history);
   - keep the file ordered per AGENTS rule 6: type → its impls, helpers adjacent;
     report progress via `opts.progress`, never print.
3. Export from `algos/mod.rs`.
4. Tests, in the module: recover-an-obvious-order fixture, seed determinism if
   stochastic, byte-identical save/load/save round trip.
5. If a published worked example exists, add it to `tests/reference.rs`
   **with the source cited** — never hard-code numbers you cannot point to.
6. Wire the CLI: a leaf under the right group (`tournament`/`rankings`/`crowd`/
   `matchups`/`graph`/`bandit`/`trajectories`). Use `opt_def` (not `opt`) for any
   value-bearing flag, sourcing the default from the algorithm's `Default` so the
   `[default: …]` in `--help` stays single-sourced; read it back with `set`/
   `choice`. Attach only the flags the algorithm uses via the `with_*`
   combinators (`with_path`, `with_load_state`, `with_bootstrap`, `with_ties`,
   `with_margin_ties`, `with_periods`) — e.g. wrap a batch `Ranker` in
   `with_bootstrap` and route it through `maybe_bootstrap`; leave `with_bootstrap`
   off an online ranker so `--bootstrap` is a parse error there. Spell names out
   for laymen with a short `visible_alias`.
7. Run the full gate (below).

## Testing and correctness

```bash
cargo test --workspace                                        # everything
cargo test -p propagon --no-default-features --features io    # sequential build (wasm config)
cargo clippy --workspace --all-targets -- -D warnings         # zero tolerance
cargo fmt --all --check
```

Four layers, by what they catch:

- **Unit tests** (in-module): algorithm behavior, edge cases, round trips.
- **`tests/reference.rs`** — third-party-published vectors: Agresti's 1987
  Bradley-Terry baseball coefficients (also pinning Plackett-Luce reduced to
  pairs and the Bayesian-BT posterior), the *Who's #1?* Massey/Colley/Keener
  worked example, the openskill Weng-Lin vectors (BT-full at 1e-9, TM-full at
  1e-6), Langville & Meyer's PageRank **and** HITS examples, Newcombe's Wilson
  intervals, the Tennessee Borda/Condorcet election, canonical Elo expectations,
  Beta-posterior arithmetic, and brute-force oracles (Kemeny over all orderings
  at ≥95% of optimum; MC4 against a dense stationary solve; Katz against a dense
  linear solve). I-LSR is cross-checked against the exact Plackett-Luce MLE.
- **`tests/parity.rs` + CLI `tests/golden.rs`** — numeric regression against
  outputs captured from the last v1 build (`scripts/capture_golden.sh` documents
  exactly how). Tier T = per-entity tolerance; tier S = rank correlation ≥ 0.95
  (for the stochastic fitters whose RNG streams legitimately changed in the
  v1→v2 port). **These expectations are frozen**; if a refactor moves them, the
  refactor is wrong.
- **`tests/state.rs`** — cross-cutting FR-4/FR-5 guarantees (byte-identical
  persistence, warm-start wins).

Doc-comment examples may use `unwrap()` (rustdoc convention; not linted).

## Releases and roadmap

Versioning is workspace-wide (`2.0.0-alpha.1`); MSRV 1.88 (let-chains). CI
(`.github/workflows/ci.yml`) runs the same gate as above on stable.

Next milestones (PRD §10): **M3** Python bindings (PyO3/maturin, abi3 wheels),
**M4** WASM (wasm-bindgen, single-threaded default — which is why the
`parallel`-off build must always stay green), **M5** mobile via UniFFI. The core
was shaped for these: no filesystem or printing in algorithms, owned
lifetime-free models, u32/f64 API boundaries.
