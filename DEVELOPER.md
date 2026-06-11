# Developing propagon

How the workspace is built, what the contracts are, and how to add to it
without breaking the guarantees. The product spec is `docs/PRD.md`; the
method survey behind every algorithm is `docs/algorithms.md`; the
non-negotiable coding rules are `AGENTS.md`.

## Workspace layout

```
propagon/
├─ Cargo.toml                  # virtual workspace: shared deps, lints, metadata
├─ crates/
│  ├─ propagon/                # the library — no I/O coupling, wasm-clean
│  │  ├─ src/
│  │  │  ├─ dataset/           # PairwiseDataset, RankingsDataset, GraphDataset,
│  │  │  │                     #   RewardsDataset, JSONL dataset io
│  │  │  ├─ algos/             # one module per algorithm (+ bandits, DE optimizer)
│  │  │  ├─ interner.rs        # string id ↔ dense u32
│  │  │  ├─ traits.rs          # Ranker, OnlineRanker, RankModel, FitOptions
│  │  │  ├─ state.rs           # header-line JSONL model persistence
│  │  │  ├─ parallel.rs        # rayon shim (sequential under --no-default-features)
│  │  │  ├─ progress.rs        # Progress trait (core never prints)
│  │  │  └─ mathx.rs           # normal CDF/quantile approximations
│  │  └─ tests/                # parity.rs, state.rs, reference.rs
│  └─ propagon-cli/            # bin "propagon": clap wiring, file parsing, emitters
│     └─ tests/golden.rs + golden/   # captured v1 outputs + harness
├─ docs/                       # algorithms survey, PRD
├─ example/tournament/         # bundled 2018 MLB data + run.sh demo
└─ scripts/capture_golden.sh   # provenance of the golden files (v1 binary, run once)
```

## Core contracts

**Datasets are shared and immutable during fitting.** Algorithms never define
private input formats. The four shapes: `PairwiseDataset` (winner/loser/weight
+ optional rating periods), `RankingsDataset` (CSR-packed ballots),
`GraphDataset` (directed weighted edges), `RewardsDataset` (arm/reward events).
Each owns an `Interner`; the invariant that every stored id came from the
owning interner is established at `push`/`push_ids` and relied on everywhere
(that is what makes `Interner::resolve` total).

**Two fitting tiers** (PRD FR-5):

- `Ranker::fit(&data) -> Model` + `fit_warm(&data, &init)` — batch, with
  warm-starting for the iterative methods (BT-MM, BT-SGD, LSR). Contract:
  `fit_warm` never converges to a worse objective than `fit`.
- `OnlineRanker::init() -> Model` + `update(&mut model, &batch)` — true
  incremental state (Glicko-2, Elo, win-rate, bandits). Contract: `update`
  never replays history, and split updates equal one continuous run
  (tested in `tests/state.rs` and the CLI golden flow).

**Models are owned and self-contained** — they carry their own names, never
borrow the dataset. `RankModel` provides `scores()`, `sorted_scores()`
(descending, name-tiebreak), and JSONL persistence.

**State files** (PRD FR-4): one header line
(`{"propagon":1,"kind":"model","algorithm":...,"params":...,"entities":N}`)
then one JSON object per entity, string-keyed. Rules: schema version is
mandatory and checked; unknown fields are tolerated on read; all model floats
are f64 so save → load → save is byte-identical (this requires serde_json's
`float_roundtrip` feature — already on; do not drop it); loading validates
the algorithm tag and (for `update`) parameter compatibility as typed errors.

**Execution options**: `FitOptions { progress: &dyn Progress, threading }`.
The core never prints — the CLI renders `Progress` with indicatif; bindings
will forward callbacks. `Threading::Dedicated(pool)` exists so embedders
never have their global rayon pool reconfigured.

**Determinism**: every stochastic algorithm has a `seed` param; fixed seed +
fixed thread count ⇒ identical output. The LSR power method is bitwise
deterministic at any thread count (transposed accumulation, no atomics).
Bandit selection streams persist their draw counter, so save → load →
`select()` is indistinguishable from an uninterrupted run.

## AGENTS.md compliance notes

The workspace denies `clippy::unwrap_used`, `expect_used`, and `panic`.
Practical patterns in use:

- Unit tests are exempt via `#![cfg_attr(test, allow(...))]` in `lib.rs`.
  **Integration tests are not** — they return
  `Result<(), Box<dyn std::error::Error>>` and propagate with `?`.
- Invariant-by-construction lookups go through `Interner::resolve`
  (placeholder on the impossible miss, never a panic).
- The CLI never re-states numeric defaults: it builds `Algo::default()` and
  overrides only flags the user passed (`set`/`choice` helpers in
  `main.rs`). Param-struct `Default` impls are the single source; canonical
  bandit-policy values live on `BanditPolicy::DEFAULT_*` consts.
- No `0 = auto` sentinels: optional budgets are enums
  (`KemenyPasses::Auto | Fixed(n)`) or derived from a single core source
  (`Estimator::default_steps`).
- The CLI crate carries its own `[lints.clippy]` (print lints relaxed there
  only — printing is its job).

## Adding an algorithm

1. **Survey first**: add or locate its entry in `docs/algorithms.md` so the
   "when to use it" story exists before the code.
2. Create `crates/propagon/src/algos/<name>.rs`:
   - a params struct with public fields, documented canonical defaults, and
     `Default`; a `seed: u64` field if anything is stochastic;
   - a model type owning an `Interner` + state vectors; implement
     `RankModel` (the `impl_simple_score_model!` macro covers
     one-f64-per-entity models);
   - implement `Ranker` (override `fit_warm_opts` if iterative) or
     `OnlineRanker` (state must merge without history);
   - keep the file ordered per AGENTS rule 6: type → its impls, helpers
     adjacent; report progress via `opts.progress`, never print.
3. Export from `algos/mod.rs`.
4. Tests, in the module: recover-an-obvious-order fixture, seed determinism
   if stochastic, byte-identical save/load/save round trip.
5. If a published worked example exists, add it to `tests/reference.rs`
   **with the source cited** — never hard-code numbers you cannot point to.
6. Wire the CLI: a leaf under the right group (`tournament`/`graph`/
   `bandit`) using the `set`/`choice` pattern; spell names out for laymen
   with a short `visible_alias`.
7. Run the full gate (below).

## Testing

```bash
cargo test --workspace                                   # everything
cargo test -p propagon --no-default-features --features io   # sequential build (wasm config)
cargo clippy --workspace --all-targets -- -D warnings    # zero tolerance
cargo fmt --all --check
```

Four layers, by what they catch:

- **Unit tests** (in-module): algorithm behavior, edge cases, round trips.
- **`tests/reference.rs`** — third-party-published vectors: Agresti's 1987
  Bradley-Terry baseball coefficients, Newcombe's Wilson intervals,
  Langville & Meyer's PageRank example, the Tennessee Borda/Condorcet
  election, canonical Elo expectations, Beta-posterior arithmetic, and a
  brute-force Kemeny oracle (all 720 orderings; the insertion heuristic is
  held to ≥95% of the exhaustive optimum — Kemeny is NP-hard, exactness is
  not promised).
- **`tests/parity.rs` + CLI `tests/golden.rs`** — numeric regression against
  outputs captured from the last v1 build (`scripts/capture_golden.sh`
  documents exactly how). Tier T = per-entity tolerance; tier S = rank
  correlation ≥ 0.95 (for the stochastic fitters whose RNG streams
  legitimately changed in the v1→v2 port). **These expectations are frozen**;
  if a refactor moves them, the refactor is wrong.
- **`tests/state.rs`** — cross-cutting FR-4/FR-5 guarantees (byte-identical
  persistence, warm-start wins).

Doc-comment examples may use `unwrap()` (rustdoc convention; not linted).

## Releases and roadmap

Versioning is workspace-wide (`2.0.0-alpha.1`); MSRV 1.88 (let-chains).
CI (`.github/workflows/ci.yml`) runs the same gate as above on stable.

Next milestones (PRD §10): **M3** Python bindings (PyO3/maturin, abi3
wheels), **M4** WASM (wasm-bindgen, single-threaded default — which is why
the `parallel`-off build must always stay green), **M5** mobile via UniFFI.
The core was shaped for these: no filesystem or printing in algorithms,
owned lifetime-free models, u32/f64 API boundaries.
