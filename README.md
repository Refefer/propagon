# propagon

**Ranking from revealed preferences** — turn match results, pairwise choices, ballots, interaction graphs, and reward logs into rankings.

Propagon is a Rust library (`propagon`) and CLI (`propagon-cli`) implementing the classical and modern ranking algorithms: Bradley-Terry, Elo, Glicko-2, Luce spectral ranking, Rank Centrality, Kemeny consensus, Borda, Copeland, PageRank, BiRank, and multi-armed bandits. One shared data model feeds every algorithm; every fitted model serializes to human-readable JSONL and resumes incrementally where the math allows.

- **Survey**: [`docs/algorithms.md`](docs/algorithms.md) — what each method assumes, when to use which, with citations.
- **Roadmap**: [`docs/PRD.md`](docs/PRD.md) — Python and WASM bindings are the next milestones.
- **Contributing**: [`DEVELOPER.md`](DEVELOPER.md).

Licensed under MIT OR Apache-2.0.

## Install

```bash
git clone https://github.com/Refefer/propagon
cd propagon
cargo build --release          # binary at target/release/propagon
cargo test --workspace         # 90+ tests incl. published reference vectors
```

Requires Rust 1.88+.

## Sixty-second tour (CLI)

Input files are plain text. Tournament rows are `winner loser [weight]`; graph rows are `src dst [weight]`; bandit rows are `arm reward`. IDs are arbitrary strings — team names, model checkpoints, URLs.

```bash
# Rank the 2018 MLB season three ways (data ships in the repo):
cd example/tournament
propagon tournament glicko2             baseball.2018   # ratings + uncertainty
propagon tournament bradley-terry-model baseball.2018   # maximum-likelihood strengths
propagon tournament elo                 baseball.2018   # classic online ratings

# Continue from saved state when next week's games arrive:
propagon tournament glicko2 --save-state ratings.jsonl  week1.txt
propagon tournament glicko2 --load-state ratings.jsonl  week2.txt

# Graph importance and bandits use the same shapes:
propagon graph page-rank links.txt
propagon bandit thompson-beta --select 1 rewards.txt    # which arm to play next
```

`./example/tournament/run.sh` runs every tournament algorithm against the bundled season. Cross-cutting flags everywhere: `--threads N`, `--format tsv|jsonl`, `--save-state`, `--load-state`.

## Sixty-second tour (library)

```rust
use propagon::algos::{BradleyTerryMM, Glicko2};
use propagon::{OnlineRanker, PairwiseDataset, RankModel, Ranker};

fn rank() -> propagon::Result<()> {
    // One dataset, string ids interned for you, feeds many algorithms.
    // (Bradley-Terry needs everyone to have at least one win and one loss —
    //  undefeated/winless entities get sectioned out; see the survey §0.2.)
    let mut games = PairwiseDataset::new();
    games.push("alice", "bob", 2.0); // alice beat bob twice
    games.push("bob", "carol", 2.0);
    games.push("alice", "carol", 1.0);
    games.push("carol", "alice", 1.0);
    games.push("bob", "alice", 1.0);
    games.push("carol", "bob", 1.0);

    // Batch maximum likelihood:
    let bt = BradleyTerryMM::default().fit(&games)?;
    println!("{:?}", bt.sorted_scores());

    // Incremental ratings with resumable, human-readable state:
    let glicko = Glicko2::default();
    let mut ratings = glicko.init();
    glicko.update(&mut ratings, &games)?;
    ratings.save_to_path("ratings.jsonl")?; // plain JSONL — `head` it, grep it
    Ok(())
}
```

## Choosing an algorithm

The full decision guide lives in [`docs/algorithms.md` §15](docs/algorithms.md#15-method-selection-decision-guide). The short version:

| Your situation | Reach for | Why |
|---|---|---|
| Pairwise outcomes, skill is static (leaderboards, LLM evals) | `bradley-terry-model` | The maximum-likelihood standard; calibrated win probabilities |
| Skill drifts / data streams in | `glicko2` (or `elo` for simplicity) | Online updates with (Glicko-2) honest uncertainty |
| Millions of comparisons, need speed | `luce-spectral-ranking` / `rank-centrality` | Spectral one-shot estimates of the same models |
| Multiple judges' full rankings | `kemeny` | Optimal consensus (MLE of the Condorcet noise model) |
| Quick, robust baseline | `win-rate` / `borda-count` / `copeland` | Counting methods — near-optimal more often than you'd think |
| Win rates feel noisy and erratic per entity | `random-utility-model` | Fits a per-entity (μ, σ): "good but inconsistent" is visible |
| A graph instead of matches (links, follows, purchases) | `graph page-rank` / `graph birank` | Importance from structure; birank for user↔item data |
| Choosing what to try next under live traffic | `bandit thompson-beta` / `upper-confidence-bound` | Rank arms *and* allocate the next trial |

### Critical parameters per algorithm

**`tournament bradley-terry-model`** — `--estimator mm` (exact MM iteration, the default) or `sgd` (logistic gradient descent: streams better, supports `--passes/--alpha/--decay`). MM needs a connected comparison graph with no undefeated entities; mitigations built in: `--remove-total-losers`, `--create-fake-games W`, `--random-subgraph-links N`. Tighten `--tolerance` (default 1e-6) for publication-grade fits.

**`tournament glicko2`** — `--tau` (0.3–1.2, default 0.5) bounds how fast a player's *volatility* can grow: lower = stabler ratings, higher = faster reaction to form swings. Feed rating periods as blank-line-separated batches with `--groups-are-separate`; each period is one Bayesian update. Output: rating, deviation, and a 95% interval per entity.

**`tournament elo`** — `--k` (default 32) is the whole game: high K tracks fast and stays noisy, low K converges and goes stale. `--scale` (400) sets how many points mean 10:1 odds. Remember Elo is order-dependent and never converges by design — for frozen entities use Bradley-Terry instead.

**`tournament luce-spectral-ranking`** — `--estimator power` (deterministic, default) or `monte-carlo` (`--steps` walk length, `--seed`). Scores are log-scale and mean-centered; only differences mean anything.

**`tournament kemeny`** — `--algo insertion` (fast, ~97%+ of optimum on hard instances) or `de` (differential evolution, bigger budget via `--passes`). NP-hard exactly: both are principled heuristics.

**`tournament random-utility-model`** — `--passes` evolution-strategy iterations, `--gamma` regularization, `--fixed` pins variances (Thurstone-style). Outputs `μ σ` per entity; values are relative (the model is only identified up to location/scale).

**`graph page-rank`** — `--damping-factor` (0.85) splits "follow links" vs "teleport"; `--sink-dispersion {reverse,all,uniform,none}` picks the dangling-node treatment (`uniform` is the textbook one); `--matches` reads tournament files as loser-endorses-winner.

**`bandit *`** — every policy: `--seed` (selection streams are reproducible and resume exactly across `--save-state`/`--load-state`), `--select N` prints the next arms to play instead of scores. `epsilon-greedy --epsilon` sets the exploration rate; `upper-confidence-bound --exploration` (2.0 = classic UCB1); `thompson-beta` needs rewards in [0,1] (`--prior-alpha/--prior-beta`); `thompson-gaussian` takes `--prior-mean/--prior-weight`.

## State files

Every model (and dataset) saves as JSONL with a self-describing header:

```jsonl
{"propagon":1,"kind":"model","algorithm":"glicko2","params":{"tau":0.5,...},"entities":30}
{"id":"BOS","r":1670.8,"rd":40.5,"sigma":0.06}
```

Guarantees: save → load → save is byte-identical; params travel with the state (mismatched resumes are typed errors, not silent reconfigurations); incremental algorithms (`glicko2`, `elo`, `win-rate`, all bandits) fold new batches into loaded state without replaying history; iterative ones (`bradley-terry-model`, `luce-spectral-ranking`, `rank-centrality`) warm-start from it.

## Correctness

Three independent test layers (see [`DEVELOPER.md`](DEVELOPER.md)):

- **Published reference vectors** — Bradley-Terry reproduces Agresti's 1987 baseball coefficients to 4 decimals; Wilson intervals match Newcombe (1998) Table I; PageRank matches Langville & Meyer's worked example; Borda/Copeland match the classic Tennessee election; Elo expectations match the canonical table.
- **v1 golden parity** — outputs match the original research implementation on the bundled season data.
- **Exhaustive oracles** — Kemeny is checked against brute-force enumeration of the full permutation space.

## Library features

- `parallel` (default) — multi-threaded fitting via rayon; disable for single-threaded targets (the WASM milestone builds with it off, and the whole suite passes either way).
- `io` (default) — `save_to_path`/`load_from_path` conveniences; in-memory readers/writers always available.
