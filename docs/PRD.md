# Propagon v2 — Product Requirements Document

**Status**: Draft — use cases under active expansion by owner
**Companion**: [algorithms.md](algorithms.md) (method catalog; §15 decision guide; §16 coverage map)

---

## 1. Summary & Vision

Propagon v2 rebuilds propagon from a research CLI into **the canonical library for ranking entities from revealed preferences** — match outcomes, pairwise choices, multiway rankings, interaction graphs, and (via the sibling mcrl-rs) reward-bearing trajectories.

One Rust core, three first-class surfaces:

| Surface | Delivery | Consumer |
|---|---|---|
| **Rust** | `propagon` crate (crates.io) | native applications, services, the CLI |
| **Python** | PyO3 wheels (PyPI) | data science, notebooks, pipelines |
| **WASM** | npm package (wasm-bindgen) | browsers — Firefox, Safari, Chromium-family alike |

The product promise, in one paragraph: *bring comparison data of any size, in whatever shape you have it; pick an algorithm (or several — they share datasets); run it at full multi-core speed; save everything — datasets, parameters, fitted state — as human-readable JSONL; and when new results arrive next week, update from where you left off instead of reprocessing history.*

The four pillars (detailed as FR-1…FR-5):

1. **Effortless bulk ingestion** into shared, algorithm-agnostic datasets.
2. **Parameter objects with sensible defaults** — zero-config to start, fully tunable when needed.
3. **Multi-threaded execution** on every platform that allows it.
4. **Serializable, resumable state** — human-readable, versioned, incremental.

## 2. Goals & Non-Goals

### Goals

- G1. Library-first architecture: the CLI becomes a thin consumer of the public Rust API.
- G2. Ranking algorithms from [algorithms.md §16](algorithms.md#16-propagon-coverage-map): all nine currently-implemented rankers ported, recommended candidates added incrementally (§8 Rollout).
- G3. Identical data model and state format across Rust, Python, and JS surfaces.
- G4. Large-data ergonomics: streaming and bulk ingestion paths; columnar internals; string IDs handled natively (no external remap step).
- G5. Incremental workflows: online update where the algorithm supports it (Glicko-2, Elo), warm-start refit where it doesn't (BT-MM, LSR, PageRank).
- G6. Deployable on desktop OSes, modern browsers, and (later milestone) Android/iOS.

### Non-Goals

- N1. **Embeddings & clustering** (vec-prop, vec-walk, gcs, hash-embedding, lpa, label-rank, mc-cluster): dropped from v2. They remain in v1 history (master) and could return as a separate crate; they are not part of the ranking library's identity.
- N2. Neural / feature-based learning-to-rank training (LambdaMART, RankNet et al. — see [algorithms.md §10](algorithms.md#10-feature-based-contextual--learning-to-rank)): out of scope as *training* systems; BT-with-covariates is in scope as a future algorithm.
- N3. Serving infrastructure (HTTP services, schedulers, dashboards).
- N4. Binary serialization formats. Human-readable state is a hard requirement, not a default.
- N5. Browser-specific machinery. Web-standards only; anything that works in Chrome must work in Firefox and Safari the same way.

## 3. Personas & Use Cases

Use cases follow a fixed template so requirements trace cleanly. **This section is a scaffold: the owner will expand existing entries and add new ones.**

> **Template** — Actor · Data shape · Algorithms · Surface/platform · Scale · Incremental needs · Flow.

### UC-1: Sports league ratings

- **Actor**: hobbyist / analyst maintaining season-long ratings for a league.
- **Data shape**: pairwise results with dates and margins, arriving weekly.
- **Algorithms**: Glicko-2 ([§2.3](algorithms.md#23-glicko-2-2001)), BT-MM, Massey (margins), Elo.
- **Surface / platform**: CLI or Python on desktop.
- **Scale**: 10²–10⁴ entities, 10³–10⁶ games.
- **Incremental needs**: **core** — append each week's results to saved state; never reprocess the season.
- **Flow**: load `state.jsonl` → `update(new_week)` → save → publish leaderboard with rating deviations.

> TODO(owner): expand.

### UC-2: LLM / model-evaluation leaderboard

- **Actor**: ML team ranking model checkpoints from human or judge preferences.
- **Data shape**: pairwise battles with ties, style covariates, annotator IDs.
- **Algorithms**: BT (MM/LR) with bootstrap CIs; tie-aware extensions; later Crowd-BT, style control ([algorithms.md §12](algorithms.md#12-applied-deep-dive-ranking-llms--model-evaluation)).
- **Surface / platform**: Python in an eval pipeline; WASM for an interactive leaderboard page.
- **Scale**: 10¹–10³ models, 10⁴–10⁷ battles.
- **Incremental needs**: warm-start refit as battles stream in; bootstrap error bars on every publish.
- **Flow**: nightly job folds new battles into the dataset → `fit_warm` from yesterday's model → bootstrap → JSON state powers the web UI.

> TODO(owner): expand.

### UC-3: Game matchmaking service

- **Actor**: game backend rating players for fair matchmaking.
- **Data shape**: continuous stream of match results (teams later).
- **Algorithms**: Glicko-2 now; Weng-Lin/OpenSkill for teams (v2.x).
- **Surface / platform**: Rust crate embedded in the service; mobile clients later read state via FFI.
- **Scale**: 10⁵–10⁷ players.
- **Incremental needs**: **core** — per-rating-period updates against persistent state; conservative estimates (μ − kσ) for matching.

> TODO(owner): expand.

### UC-4: Crowdsourced comparison QA

- **Actor**: researcher collecting "which is better, A or B?" judgments from crowdworkers.
- **Data shape**: (annotator, winner, loser) triples; uneven annotator quality.
- **Algorithms**: BT baseline; Crowd-BT (v2.x); Kemeny consensus for small panels.
- **Surface / platform**: Python.
- **Scale**: 10²–10⁵ items, 10³–10⁶ judgments.
- **Incremental needs**: warm-start refits between collection rounds; active-pair suggestions later.

> TODO(owner): expand.

### UC-5: Interaction-graph importance ranking

- **Actor**: product team ranking items (and users) from click/purchase logs.
- **Data shape**: weighted bipartite interaction edges; also general directed graphs.
- **Algorithms**: BiRank, PageRank/personalized PageRank; Rank Centrality when edges are comparisons.
- **Surface / platform**: Python batch job; WASM for in-browser exploration of small graphs.
- **Scale**: 10⁵–10⁸ edges.
- **Incremental needs**: warm-start power iteration from previous scores after graph updates.

> TODO(owner): expand.

### UC-6: Value-based ranking via mcrl-rs

- **Actor**: experimenter ranking variants/states by expected downstream outcome.
- **Data shape**: trajectories with rewards (mcrl-rs JSONL) → derived preference edges `a ≻ b` weighted by P(V_a > V_b) ([algorithms.md §13](algorithms.md#13-value-function--trajectory-based-ranking)).
- **Algorithms**: mcrl-rs produces the edges; propagon BT/Borda/Kemeny aggregates them.
- **Surface / platform**: CLI pipeline (mcrl-rs → propagon).
- **Incremental needs**: dataset append + warm-start refit as new trajectories accumulate.

> TODO(owner): expand; define the edge-export format contract between the two tools.

### UC-7: Adaptive experimentation (bandit-driven A/B/n)

- **Actor**: growth/experimentation team allocating traffic among variants while the test runs.
- **Data shape**: (variant, reward) events — clicks, conversions, revenue; batched or streaming.
- **Algorithms**: Thompson Sampling, UCB1, ε-greedy ([algorithms.md §8.1](algorithms.md#81-standard-multi-armed-bandits-ε-greedy-ucb-thompson-sampling)); best-arm identification to conclude the experiment.
- **Surface / platform**: Python in an assignment service; WASM for fully in-browser assignment with no backend.
- **Scale**: 2–10³ arms, 10⁴–10⁸ events.
- **Incremental needs**: **core** — `update(batch)` merges sufficient statistics; `select_k(n)` drives the next allocation; state saved between rounds (FR-8).
- **Flow**: load `state.jsonl` → `update(last_hour_events)` → `select_k` for the next traffic split → save; ranking of arms with CIs published continuously.

> TODO(owner): expand.

> TODO(owner): add further use cases here (mobile-first scenarios, browser-only scenarios, etc.).

## 4. Functional Requirements

### FR-1 — Shared datasets & bulk ingestion

Three columnar dataset types cover every algorithm in scope; algorithms never define private input formats.

| Dataset | Contents | Consumers |
|---|---|---|
| `PairwiseDataset` | `winners: Vec<Id>`, `losers: Vec<Id>`, `weights: Vec<f32>`; optional period boundaries (replaces v1's blank-line batches); optional tie flags | BT-MM, BT-LR, Elo, Glicko-2, LSR, Rank Centrality, ES-RUM, Wilson rate, Massey/Colley, HodgeRank, Copeland |
| `RankingsDataset` | CSR-style ragged lists: `items: Vec<Id>`, `offsets: Vec<u64>` | Plackett-Luce / I-LSR, Kemeny, Borda, Mallows |
| `GraphDataset` | `src/dst: Vec<Id>`, `weight: Vec<f32>`; bipartite mode (two ID spaces) | PageRank, BiRank, HITS, Katz, components |
| `RewardsDataset` | `arms: Vec<Id>`, `rewards: Vec<f32>`; optional period boundaries | Bandits (UCB, Thompson Sampling, ε-greedy — FR-8); ingestible from mcrl-rs (entity, return) exports |

Requirements:

- **FR-1.1** Every dataset owns (or shares via `Arc`) an **interner** mapping string IDs ↔ dense `u32`. Users pass strings; the v1 `dehydrate`/`hydrate`/`remap.py` step disappears. Pre-interned u32 input remains supported (identity vocab).
- **FR-1.2** Ingestion paths: single-record `push(winner, loser, weight)` and bulk `push_chunk(&[u32], &[u32], &[f32])`. Python: numpy arrays (zero-copy read). JS: `Uint32Array`/`Float32Array`.
- **FR-1.3** `PairwiseDataset::as_graph()` provides a zero-copy graph view (loser→winner edges) so spectral/centrality code shares one implementation.
- **FR-1.4** Datasets are **immutable during fit**. Algorithm-specific preprocessing (BT-MM's fake games / undefeated removal / subgraph links) operates on internal overlays, never on the shared dataset — one dataset must feed many algorithms unchanged.
- **FR-1.5** Datasets are serializable/loadable per FR-4 and appendable: loading a dataset file and `push`-ing more rows is the normal flow.

*Acceptance*: one `PairwiseDataset` built from a 10M-row file feeds `btm-mm`, `lsr`, and `glicko2` without copies or mutation; Python bulk ingestion of 10M rows completes in seconds, not minutes.

### FR-2 — Parameter objects with defaults

- **FR-2.1** Each algorithm has a plain params struct with public fields and `Default` (e.g. `Glicko2Params { tau: 0.5, rating: 1500.0, rd: 350.0, sigma: 0.06 }`).
- **FR-2.2** Surfaces: Rust struct-update syntax (`Glicko2Params { tau: 0.8, ..Default::default() }`); Python keyword args (`Glicko2(tau=0.8)`); JS options object (`new Glicko2({ tau: 0.8 })`).
- **FR-2.3** Params are serialized into saved state (FR-4) so a resumed run knows exactly how it was configured; param mismatch on `update`/`fit_warm` is an error, not a silent reconfiguration.
- **FR-2.4** Every stochastic algorithm exposes a `seed` param (Kemeny-DE, LSR Monte Carlo, ES-RUM) for reproducibility.

*Acceptance*: `propagon.Glicko2()` with no arguments matches v1 default behavior on the `example/tournament` data.

### FR-3 — Multi-threaded execution

- **FR-3.1** Core parallelism via rayon behind a default-on `parallel` feature; with the feature off, identical sequential code paths compile (shim layer) — required for the default WASM build.
- **FR-3.2** Thread control: dedicated rayon pools passed into fit/update (never reconfigure the global pool); surfaced as CLI `--threads N`, Python `propagon.set_num_threads(n)` / `PROPAGON_NUM_THREADS`, Rust `ThreadPool::new(n)`. Default: all cores.
- **FR-3.3** **Python**: all compute runs inside `py.allow_threads` (GIL released). Inputs are extracted to Rust-owned buffers first; no Python objects are touched from worker threads.
- **FR-3.4** **WASM**: default build is single-threaded and runs in any modern browser with zero special headers. An opt-in parallel build (wasm-bindgen-rayon, Web Workers + SharedArrayBuffer) requires the page to be cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`). This is a web standard supported equally by Firefox, Safari (≥15.2), and Chromium-family browsers — the constraint is *server headers*, never browser choice. Both builds ship in one npm package via subpath exports (`propagon`, `propagon/parallel`).
- **FR-3.5** No `println!`/`eprintln!` in core. Progress reporting via a callback trait (CLI renders indicatif bars; Python gets an optional callable; JS gets an optional function); diagnostics via the `log` facade.

*Acceptance*: BT-MM on a 10M-edge dataset scales near-linearly to 8 cores natively; the same build runs single-threaded in Firefox without any header configuration; a Python caller's other threads make progress during a long fit.

### FR-4 — Human-readable, versioned serialization

Format: **JSON Lines with a self-describing header line** — one format for models and datasets, on every surface. Single-document JSON is rejected for scale reasons (whole-tree materialization, no streaming/append at 10M entities).

Model file:

```jsonl
{"propagon":1,"kind":"model","algorithm":"glicko2","params":{"tau":0.5,"rating":1500.0,"rd":350.0,"sigma":0.06},"entities":12842}
{"id":"alice","r":1464.0506,"rd":151.5165,"sigma":0.059996}
{"id":"bob","r":1398.1436,"rd":31.6703,"sigma":0.059996}
```

Dataset file (embedded vocab; columnar chunk lines, ≤64k rows per line):

```jsonl
{"propagon":1,"kind":"dataset","schema":"pairwise","entities":12842,"rows":5000000}
{"vocab":["alice","bob","carol"]}
{"w":[0,2],"l":[1,0],"x":[1.0,0.5]}
```

Requirements:

- **FR-4.1** Mandatory `propagon` schema-version integer; readers tolerate unknown fields (forward compatibility); version bumps documented.
- **FR-4.2** Model lines use **string IDs** — a model file is self-contained and human-greppable. Dataset edge lines use u32 indices into the embedded vocab (strings not repeated millions of times).
- **FR-4.3** Model state is `f64`; serde_json's shortest-roundtrip float formatting makes save → load **bit-exact**, so resume is deterministic.
- **FR-4.4** Streaming read/write on all surfaces; WASM serializes to/from strings or `Uint8Array` (no filesystem assumption).
- **FR-4.5** No binary format, period. Compression (gzip) is the user's transport concern, not the format's.

*Acceptance*: save → load → save produces byte-identical model files; a 1M-entity Glicko-2 state round-trips in seconds; `head -3 state.jsonl` is meaningful to a human.

### FR-5 — Incremental update & resume

Two capability tiers, encoded as separate traits:

```rust
pub trait Ranker {            // batch with warm start
    type Data; type Model: RankModel;
    fn fit(&self, data: &Self::Data) -> Result<Self::Model>;
    fn fit_warm(&self, data: &Self::Data, init: &Self::Model) -> Result<Self::Model>;
}
pub trait OnlineRanker {      // true incremental
    type Data; type Model: RankModel;
    fn init(&self) -> Self::Model;
    fn update(&self, model: &mut Self::Model, batch: &Self::Data) -> Result<()>;
}
```

Per-algorithm support matrix (v2.0 set):

| Algorithm | `update` (no history replay) | `fit_warm` | Notes |
|---|---|---|---|
| Glicko-2 | ✅ per rating period | — | the flagship incremental case; owned state replaces v1's borrowed `Series<'a>` |
| Elo | ✅ per game/batch | — | order-dependent by definition |
| BT-MM | ❌ | ✅ | warm π converges in few sweeps on appended data |
| BT-LR (SGD) | ◐ (continue SGD) | ✅ | continued training ≈ warm start |
| LSR / Rank Centrality | ❌ | ✅ | warm-start power iteration |
| PageRank / BiRank | ❌ | ✅ | warm-start power iteration |
| ES-RUM | ❌ | ✅ | resume ES from saved (μ, σ) |
| Wilson rate | ✅ (count merge) | — | trivially mergeable tallies |
| Bandits (UCB / TS / ε-greedy) | ✅ (sufficient-statistic merge) | — | counts+sums (UCB) or posterior params (TS) merge exactly; the easiest row in this table |
| Kemeny | ❌ | ◐ | re-run heuristic seeded with previous order |

- **FR-5.1** `update` never reads historical comparisons — state + new batch only.
- **FR-5.2** `fit_warm` must never produce a worse final objective than cold `fit` on the same data (it may converge faster).
- **FR-5.3** The matrix above ships in user docs; calling `update` on a batch-only algorithm is a typed error suggesting `fit_warm`.

*Acceptance*: the UC-1 flow — load January state, `update(february)`, save — produces ratings identical to v1's two-period batch run on the same data; warm BT-MM on appended data converges in strictly fewer MM sweeps than a cold start (linear-rate convergence means warm starting saves the early error decades — the smaller the increment relative to the tolerance, the larger the saving).

### FR-6 — Bindings

- **FR-6.1 Python**: PyO3 (≥0.27) + maturin; `abi3-py310` stable-ABI wheels (one wheel per OS/arch covering CPython ≥3.10); rust-numpy for bulk in/out (`scores` as name list + f64 arrays); hand-maintained `.pyi` stubs; wheel CI for manylinux/musllinux, macOS universal2, Windows.
- **FR-6.2 WASM**: wasm-bindgen + wasm-pack `--target web` (plain ESM, bundler-compatible); getrandom `wasm_js` backend configured; **u32/f64-only API boundary** (no `u64` → BigInt surprises); bulk results as TypedArrays + `idToName(id)` lookup (never serialize 10M-entry objects across the boundary); `console_error_panic_hook` in debug; docs recommend running fits inside a Web Worker regardless of threading.
- **FR-6.3 Mobile (post-v2.0 milestone)**: `propagon-ffi` crate using **UniFFI** → Kotlin/Swift bindings; Android via cargo-ndk (four ABIs → AAR), iOS via XCFramework. Chosen over cbindgen (UniFFI generates idiomatic string/error/collection handling matching our API shape) and over wasm-in-webview (JS-bridge and memory overhead). The core's owned-state, callback-driven, lifetime-free design is the enabler; the FFI crate is mostly annotation.
- **FR-6.4** Errors are typed `Result`s in core (`thiserror`) and surface as Python exceptions / JS exceptions. **No panics across any FFI boundary.**

*Acceptance*: `pip install propagon && python -c "import propagon"` works on the three desktop OSes; `npm i propagon` + 10-line ESM snippet ranks a dataset in Firefox; identical state files interchange between all surfaces.

### FR-7 — CLI (grouped by data shape; no v1 surface compatibility)

The v2 CLI makes a clean break from the v1 surface (owner decision: zero
backward-compatibility requirements). Subcommands are grouped by input shape:

- **FR-7.1** `propagon tournament <algo> <path>` — pairwise rankers over
  `winner loser [weight]` rows: `win-rate`, `elo`, `glicko2`,
  `bradley-terry-model` (one command; `--estimator mm|sgd` selects the
  fitting method), `luce-spectral-ranking`, `rank-centrality`,
  `random-utility-model`, `kemeny`, `borda-count`, `copeland`. Names are
  spelled out for readability; short visible aliases exist (`rate`, `btm`,
  `lsr`, `rum`, `borda`). Group flags: `--min-count`,
  `--groups-are-separate` (rating periods).
- **FR-7.2** `propagon graph <algo> <path>` — node importance and utilities
  over `src dst [weight]` edges: `page-rank` (with `--matches` for the
  loser-endorses-winner orientation of tournament files), `birank`,
  `components`.
- **FR-7.3** `propagon bandit <policy> <path>` — `greedy`, `epsilon-greedy`,
  `upper-confidence-bound` (alias `ucb1`), `thompson-beta` (alias `ts-beta`),
  `thompson-gaussian` (alias `ts-gaussian`) over `arm reward` rows, each with
  `--select N` (print the next arms to play) and `--seed`.
- **FR-7.4** Cross-cutting flags on every leaf: `--threads N`,
  `--save-state PATH`, `--load-state PATH`, `--format tsv|jsonl`.
- **FR-7.5** String ids are read natively (interner); the v1
  `dehydrate`/`hydrate`/`remap.py` pipeline is **removed**, not deprecated.
- **FR-7.6** The golden suite captured from v1 (`tests/golden/`) is retained
  as a **numerical regression baseline**: v2 must reproduce v1's numbers
  (tolerance/rank-correlation tiers) through the new command surface.

### FR-8 — Bandit policies (rank arms *and* pick the next one)

Standard multi-armed bandits ([algorithms.md §8.1](algorithms.md#81-standard-multi-armed-bandits-ε-greedy-ucb-thompson-sampling)) are both rankers and decision policies; the library exposes both faces.

- **FR-8.1** Bandit models implement `OnlineRanker` (FR-5) over `RewardsDataset`: `update` is an exact sufficient-statistic merge (counts/sums for UCB and ε-greedy; Beta or Gaussian posterior parameters for Thompson Sampling); `scores()` ranks arms by the policy's estimate (posterior mean / UCB index), with uncertainty fields per arm.
- **FR-8.2** Policy surface beyond ranking: `select() -> Id` and `select_k(n) -> Vec<Id>` implement the exploration rule (UCB argmax, posterior sampling, ε-greedy). Stochastic policies (TS, ε-greedy) are deterministic given the `seed` param and current state.
- **FR-8.3** Dual mode: **offline** — rank arms from logged (arm, reward) events; **online** — the model lives in the application loop: `update(batch)` → `select_k(n)` → serve → repeat, with state persisted per FR-4 between rounds.
- **FR-8.4** Available on all three surfaces; the WASM build makes in-browser adaptive assignment possible with no backend.
- **FR-8.5** v2.0 algorithms: greedy/ε-greedy, UCB1, Thompson Sampling (Beta-Bernoulli + Gaussian). v2.x: KL-UCB, EXP3, sliding-window/discounted UCB, LinUCB (linear contextual only — deeper contextual modeling stays out of scope per N2).

*Acceptance*: UCB1 `select()` matches a hand-computed argmax on a fixture; seeded Thompson Sampling reproduces identical selection sequences; save → load → `select()` is indistinguishable from an uninterrupted run; merging two state files equals processing the concatenated logs.

## 5. Architecture

### 5.1 Workspace layout

```
propagon/
├─ Cargo.toml            # [workspace]; shared deps & lints
├─ crates/
│  ├─ propagon/          # core: datasets, interner, algorithms, traits, serde state
│  ├─ propagon-cli/      # bin "propagon": clap 4, file I/O, progress bars
│  ├─ propagon-py/       # cdylib: pyo3 + maturin + numpy
│  ├─ propagon-wasm/     # cdylib: wasm-bindgen (+ optional wasm-bindgen-rayon)
│  └─ propagon-ffi/      # (M5) uniffi → Kotlin/Swift
├─ docs/  example/  scripts/
```

Core feature flags: `parallel` (default on; off = sequential shims for wasm), `io` (default on; gates `std::fs` conveniences only — in-memory readers/writers always available). serde/serde_json are unconditional (FR-4 is not optional).

### 5.2 Toolchain & dependency modernization

Edition **2024**, workspace MSRV **1.85** (CI-checked; binding crates may ratchet independently).

| v1 | v2 | Why |
|---|---|---|
| clap 2.26 | clap 4.5 (derive), CLI crate only | maintained; derive API |
| rand 0.7 / rand_distr 0.2 / rand_xorshift | rand 0.9 / rand_distr 0.5 / rand_xoshiro | API churn absorbed once; seeded Xoshiro everywhere |
| `random` 0.12 | **removed** | unmaintained; seeding moves to seeded Xoshiro |
| hashbrown 0.9 (+rayon) | hashbrown 0.15+ | keep; foldhash default replaces ahash |
| ahash, float-ord | **removed** | foldhash / `f64::total_cmp` |
| indicatif 0.15 | indicatif 0.18, CLI-only | core is callback-driven (FR-3.5) |
| statrs 0.15 | statrs 0.18+ | pure Rust, wasm-safe |
| atomic_float 0.1 | atomic_float 1.x / portable-atomic | `lsr.rs` hot path |
| — | serde, serde_json, thiserror, log | core requirements |
| — | pyo3 + numpy; wasm-bindgen + js-sys + serde-wasm-bindgen + getrandom(wasm_js) | bindings |

### 5.3 Illustrative API sketches (non-normative)

Build once, fit many (FR-1):

```rust
let mut ds = PairwiseDataset::new();
ds.push("ARI", "COL", 1.0);                          // strings interned inline
ds.push_chunk(&winners, &losers, &weights)?;         // bulk u32 slices
let bt  = BradleyTerryMM::default().fit(&ds)?;
let lsr = Lsr { seed: Some(42), ..Default::default() }.fit(&ds)?;   // same dataset
```

Incremental Glicko-2 with owned, serializable state (FR-2/4/5 — replaces v1 `g2.rs`'s `Series<'a, ID>` borrow):

```rust
let g2 = Glicko2 { params: Glicko2Params { tau: 0.5, ..Default::default() } };
let mut model = g2.init();
g2.update(&mut model, &january)?;
g2.update(&mut model, &february)?;                   // no history replay
model.save_jsonl(File::create("state.jsonl")?)?;
let mut model = Glicko2Model::load_jsonl(BufReader::new(File::open("state.jsonl")?))?;
```

Python (kwargs → params; GIL released during compute):

```python
ds = propagon.PairwiseDataset()
ds.add("ARI", "COL")
ds.add_arrays(winners, losers, weights)              # numpy u32/u32/f32
model = propagon.Glicko2(tau=0.5).init()
model.update(ds)
model.save("state.jsonl")
names, ratings, rd = model.to_arrays()               # list[str], np.f64, np.f64
```

JS (options object; TypedArrays at the boundary):

```js
import init, { PairwiseDataset, Glicko2 } from "propagon";
await init();
const ds = new PairwiseDataset();
ds.addChunk(winnersU32, losersU32, weightsF32);
const model = new Glicko2({ tau: 0.5 }).init();
model.update(ds);
const text = model.saveToString();                   // same JSONL format
const ratings = model.ratings();                     // Float64Array
```

### 5.4 Core refactor keystones (from v1 code)

1. `src/g2.rs` — `Series<'a, ID>` borrows `&'a Env`: redesign as owned `Glicko2Model` (template for all owned state).
2. `src/mm.rs` — mutates its input (fake games, loser removal) and prints inline: move mitigations to internal overlays; evict output (FR-1.4, FR-3.5).
3. `src/reader.rs` — blank-line batch semantics map to `PairwiseDataset` periods; file parsing moves to the CLI crate.
4. `src/main.rs` — `emit_scores`/`emit_dense_embs` become CLI-side formatters over `RankModel::scores()`.

## 6. Platform Support Matrix

| Platform | Tier | Surface | Notes |
|---|---|---|---|
| Linux / macOS / Windows | 1 | Rust, Python, CLI | rayon full speed |
| Firefox, Safari, Chromium-family (current) | 1 | WASM (single-thread default) | zero special requirements |
| Same browsers, cross-origin-isolated pages | 1 (opt-in) | WASM parallel build | COOP/COEP headers; web standard, not vendor-specific |
| Android / iOS | 2 (M5) | UniFFI Kotlin/Swift | rayon works on both |
| wasm32 memory ceiling | — | — | 4 GB hard cap ⇒ ~50–100M columnar edges practical; builders must fail gracefully with a typed error, not an OOM trap |

## 7. Algorithm Rollout

- **v2.0 (port + trivial adds)**: BT-MM, BT-LR, Glicko-2, LSR, ES-RUM, Kemeny, Wilson rate, PageRank, BiRank, components — plus Elo, Borda, Copeland, Rank Centrality, and the core bandits (greedy/ε-greedy, UCB1, Thompson Sampling Beta + Gaussian over `RewardsDataset`, FR-8) — all near-free on the new core.
- **v2.x (per [algorithms.md §16](algorithms.md#16-propagon-coverage-map) recommended candidates)**: BT extensions (ties/home/covariates), HodgeRank rankability audit, Massey + Colley, Bayesian BT, Crowd-BT, Weng-Lin/OpenSkill (teams), library-wide `--bootstrap N`, I-LSR + native multiway input, KL-UCB / EXP3 / sliding-window UCB / LinUCB.
- Lower priority / out of scope: see §16's remaining lists.

## 8. Non-Functional Requirements

- **NFR-1 Performance**: ≥ v1 throughput on `example/tournament` and on a synthetic 10M-edge benchmark (CI-tracked).
- **NFR-2 Determinism**: fixed seed + fixed thread count ⇒ identical output; thread-count variation may differ in last-ulp parallel reductions (documented; deterministic chunked reductions used where cheap).
- **NFR-3 Memory**: model state f64; edge weights f32; columnar datasets — 10M pairwise rows ≈ 120 MB resident.
- **NFR-4 Robustness**: no panics across FFI; all fallible APIs return typed errors; malformed state files produce line-numbered errors.
- **NFR-5 Compatibility**: state schema versioned from day one; v2.x readers accept v2.0 files.

## 9. Risks

| Risk | Mitigation |
|---|---|
| wasm threads still need pinned nightly + build-std (wasm-bindgen-rayon) | parallel wasm artifact is "advanced mode"; default artifact ships from stable, single-threaded |
| COOP/COEP hosting friction for parallel wasm | single-thread default; document header setup; never required for correctness |
| clap 2→4 behavior drift breaks scripts | golden-output suite frozen before port (FR-7.3) |
| rand 0.7→0.9 churn across stochastic modules | mechanical; absorbed in M1 with seeded-Xoshiro standardization |
| PyO3 GIL traps (deadlocks, accidental serialization) | "extract → allow_threads → compute" pattern enforced in review; no Py objects in rayon closures |
| JSONL at 10M+ entities tempts a binary shortcut | rejected by requirement (N4); chunked columnar lines keep it fast enough |
| u32 ID cap (4.29B entities) | accepted & documented; unreachable on wasm32 anyway |
| f32→f64 score change shifts outputs vs v1 | golden tests compare within tolerance; documented as v2 behavior change |

## 10. Milestones

1. **M1 — Core extraction**: workspace split; interner + three datasets; owned-state Glicko-2; BT-MM immutability; log/progress eviction; dependency modernization; all nine algorithms compiling against the trait layer.
2. **M2 — State & CLI**: JSONL save/load; `update`/`fit_warm` APIs + support matrix; clap-4 CLI with golden tests; `--save-state`/`--load-state`.
3. **M3 — Python**: propagon-py, abi3 wheels, numpy ingestion, stubs, wheel CI.
4. **M4 — WASM**: single-thread npm package (`--target web`); then optional `propagon/parallel` build.
5. **M5 — Mobile**: propagon-ffi via UniFFI; Android AAR + iOS XCFramework.

## 11. Open Questions

1. Package names: is `propagon` free on PyPI and npm? (Check before M3/M4; fallbacks: `propagon-rank`.)
2. Do any v1 score-file consumers need a migration tool, or is `--format tsv` compatibility sufficient?
3. Free-threaded CPython (3.13t/3.14t) wheels: demand-driven; abi3 wheels don't load there.
4. Default bootstrap sample counts per algorithm when `--bootstrap` lands.
5. Should `RankingsDataset` support weighted/duplicated ballots natively (vote multiplicity) at v2.0 or v2.x?

## 12. Requirements Traceability

| User requirement (from kickoff) | Covered by |
|---|---|
| Rust / Python (PyO3) / WASM bindings | FR-6, §5.1, §6 |
| Browser support incl. Firefox, nothing Chrome-specific | FR-3.4, FR-6.2, §6, N5 |
| Easy very-large-data ingestion, shared datasets | FR-1, NFR-3 |
| Params struct/class with reasonable defaults | FR-2 |
| Multi-threaded performance | FR-3, NFR-1 |
| Serializable results & datasets, human-readable, no binary | FR-4, N4 |
| Updateable state (Glicko-2 continue), resume without reprocessing | FR-5 |
| Desktop / Android / iOS / browsers | §6, FR-6.3, M5 |
| Standard bandits (UCB, TS, greedy, …) | FR-1 (`RewardsDataset`), FR-5, FR-8, UC-7; survey §8.1 |
