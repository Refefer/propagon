# propagon

**Ranking from revealed preferences** — turn match results, pairwise choices, ballots, interaction graphs, and reward logs into rankings.

Propagon is a Rust library (`propagon`) and CLI (`propagon-cli`) implementing ~50 classical and modern ranking algorithms: the Bradley-Terry family (MM, SGD, Bayesian, crowd-annotated, tie/home-aware, team, covariate), Elo (plain, margin-of-victory, multidimensional), Glicko-2, Whole-History Rating, Weng-Lin/OpenSkill team ratings, Plackett-Luce and I-LSR, Thurstone-Mosteller, Rank Centrality, SerialRank, random-walker rankings, Massey, Colley, Keener, offense-defense, HodgeRank, Nash averaging, Blade-Chest, Kemeny consensus, Mallows, footrule, Borda, Copeland, MC4, PageRank with personalized teleport, LeaderRank, HITS, Katz, harmonic centrality, BiRank, k-core, Monte Carlo and TD state values with bootstrap comparison, behavior cloning, nine multi-armed-bandit policies (plus dueling bandits as a library API), and a generic `--bootstrap` wrapper that puts score and rank intervals on any batch ranker. Nine shared dataset shapes feed every algorithm; every fitted model serializes to human-readable JSONL and resumes incrementally where the math allows.

- **Survey**: [`docs/algorithms.md`](docs/algorithms.md) — what each method assumes, when to use which, with citations.
- **Roadmap**: [`docs/PRD.md`](docs/PRD.md) — Python and WASM bindings are the next milestones.
- **Contributing**: [`DEVELOPER.md`](DEVELOPER.md).

Licensed under MIT OR Apache-2.0.

## Install

```bash
git clone https://github.com/Refefer/propagon
cd propagon
cargo build --release          # binary at target/release/propagon
cargo test --workspace         # 140+ tests incl. published reference vectors
```

Requires Rust 1.88+.

## Sixty-second tour (CLI)

Input files are plain text, grouped by shape: tournament rows are `side1<TAB>side2<TAB>threshold[<TAB>count]` — rosters space-separated within a side, signed threshold (`3` = side 1 wins by 3, `-2` = side 2 wins by 2, `0` = tie), optional repeat count; rankings files hold one ballot per line (best first); matchups hold one match per line (teams `|`-separated, `=` for ties); crowd votes are `annotator winner loser`; graph rows are `src dst [weight]`; bandit rows are `arm reward` (LinUCB: `arm reward x1 ... xd`); trajectories are `state reward` rows with a blank line ending each episode. IDs are arbitrary strings — team names, model checkpoints, URLs.

```bash
# Rank the 2018 MLB season three ways (data ships in the repo):
cd examples/tournament
propagon tournament glicko2             baseball.2018   # ratings + uncertainty
propagon tournament bradley-terry-model baseball.2018   # maximum-likelihood strengths
propagon tournament elo                 baseball.2018   # classic online ratings

# Continue from saved state when next week's games arrive:
propagon tournament glicko2 --save-state ratings.jsonl  week1.txt
propagon tournament glicko2 --load-state ratings.jsonl  week2.txt

# Aggregate a season of race results (ballots), rate team matches:
propagon rankings plackett-luce f1-2024.rankings        # full-order MLE
propagon matchups weng-lin games.matchups               # OpenSkill-style (mu, sigma)

# Annotator-aware ranking, graph importance, bandits:
propagon crowd bradley-terry votes.txt                  # + per-annotator reliability
propagon graph page-rank links.txt
propagon bandit thompson-beta --select 1 rewards.txt    # which arm to play next
```

[`examples/`](examples/) ships seven worked demos with real data (2018 MLB season, 2024 F1 season, a Wikipedia link graph, synthetic funnels) — each directory has a README and a `run.sh` exercising every relevant algorithm. Cross-cutting flags everywhere: `--threads N`, `--format tsv|jsonl`, `--save-state`, `--load-state`; every batch command also takes `--bootstrap N` for score and rank intervals.

## Sixty-second tour (library)

```rust
use propagon::algos::{BradleyTerryMM, Glicko2};
use propagon::{GamesDataset, OnlineRanker, RankModel, Ranker, TiePolicy};

fn rank() -> propagon::Result<()> {
    // One dataset, string ids interned for you, feeds many algorithms.
    // (Bradley-Terry needs everyone to have at least one win and one loss —
    //  undefeated/winless entities get sectioned out; see the survey §0.2.)
    let mut games = GamesDataset::new();
    games.push_pair("alice", "bob", 2.0)?; // alice beat bob twice
    games.push_pair("bob", "carol", 2.0)?;
    games.push_pair("alice", "carol", 1.0)?;
    games.push_pair("carol", "alice", 1.0)?;
    games.push_pair("bob", "alice", 1.0)?;
    games.push_pair("carol", "bob", 1.0)?;

    // Batch maximum likelihood (lower the games to win/loss pairs):
    let bt = BradleyTerryMM::default().fit(&games.to_pairwise(TiePolicy::Error)?)?;
    println!("{:?}", bt.sorted_scores());

    // Incremental ratings with resumable, human-readable state:
    let glicko = Glicko2::default();
    let mut ratings = glicko.init();
    glicko.update(&mut ratings, &games)?;
    ratings.save_to_path("ratings.jsonl")?; // plain JSONL — `head` it, grep it
    Ok(())
}
```

## Algorithm reference

Every implemented algorithm, by input shape (survey § references point into [`docs/algorithms.md`](docs/algorithms.md)):

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
| | Dueling bandits: RUCB, Double Thompson Sampling (§8.2) | library API (`DuelingBandit`), no CLI yet |
| `trajectories` | Monte Carlo state values (§13.1) | `monte-carlo` (`mc`) |
| | TD(0) state values (§13.3) | `td` |
| | Bootstrap value comparison (§13.2) | `compare` |
| | Counting behavior cloning (§13.6) | `behavior-cloning` (`bc`) |
| *(any batch)* | Bootstrap intervals on scores and ranks (§11.4) | `--bootstrap N` on every batch command |

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
| Full finishing orders / ballots, not just pairs | `rankings plackett-luce` (or `mc4` for robustness) | Every position informs the fit; partial ballots are exact |
| Team or multiplayer matches, live leaderboard | `matchups weng-lin` | TrueSkill-class (μ, σ) updates in closed form |
| Margins of victory available | `massey` (or `keener`) | Least-squares on margins; Keener adds strength-of-schedule |
| Win/loss only, want schedule adjustment without a model | `colley` | "Bias-free" Laplace-smoothed ratings |
| Crowdsourced votes with spammers/trolls | `crowd bradley-terry` | Joint ranking + per-annotator reliability |
| Need intervals, sparse or lopsided data | `bayesian-bradley-terry` | Posterior credible intervals; no connectivity requirements |
| "Is this data even rankable?" | `hodge-rank` | Reports the cyclic share *no* ranking can explain |
| A graph instead of matches (links, follows, purchases) | `graph page-rank` / `hits` / `birank` | Importance from structure; hits for dual roles, birank for user↔item |
| Choosing what to try next under live traffic | `bandit thompson-beta` / `kl-ucb` | Rank arms *and* allocate the next trial |

### Critical parameters per algorithm

**`tournament bradley-terry-model`** — `--estimator mm` (exact MM iteration, the default) or `sgd` (logistic gradient descent: streams better, supports `--passes/--alpha/--decay`). MM needs a connected comparison graph with no undefeated entities; mitigations built in: `--remove-total-losers`, `--create-fake-games W`, `--random-subgraph-links N`. Tighten `--tolerance` (default 1e-6) for publication-grade fits.

**`tournament glicko2`** — `--tau` (0.3–1.2, default 0.5) bounds how fast a player's *volatility* can grow: lower = stabler ratings, higher = faster reaction to form swings. Feed rating periods as blank-line-separated batches with `--groups-are-separate`; each period is one Bayesian update. Output: rating, deviation, and a 95% interval per entity.

**`tournament elo`** — `--k` (default 32) is the whole game: high K tracks fast and stays noisy, low K converges and goes stale. `--scale` (400) sets how many points mean 10:1 odds. Remember Elo is order-dependent and never converges by design — for frozen entities use Bradley-Terry instead.

**`tournament luce-spectral-ranking`** — `--estimator power` (deterministic, default) or `monte-carlo` (`--steps` walk length, `--seed`). Scores are log-scale and mean-centered; only differences mean anything.

**`tournament kemeny`** — `--algo insertion` (fast, ~97%+ of optimum on hard instances) or `de` (differential evolution, bigger budget via `--passes`). NP-hard exactly: both are principled heuristics.

**`tournament random-utility-model`** — `--passes` evolution-strategy iterations, `--gamma` regularization, `--fixed` pins variances (Thurstone-style). Outputs `μ σ` per entity; values are relative (the model is only identified up to location/scale).

**`tournament massey` / `colley` / `keener`** — Massey reads the weight column as the **margin of victory** (margins required); Colley reads it as game counts and ignores margins by design; Keener reads rows as `(scorer, opponent, amount)` — push both directions of each game for score data. Keener's `--no-skew` disables blowout damping, `--no-normalize-games` the unequal-schedule correction.

**`tournament hodge-rank`** — `--flow {log-odds,win-rate-delta,mean-margin}` picks the pairwise statistic to decompose. Watch stderr: the **inconsistency** number is the share of flow living in cycles — near 0 the ranking is trustworthy, near 1 a total order is fiction. Check it before believing any other algorithm on the same data.

**`tournament bayesian-bradley-terry`** — `--shape` (1.0 = posterior mode at the MLE; higher shrinks entities together), `--samples`/`--burn-in` for the Gibbs chain, `--credible` interval mass, `--seed`. Output: `mean lo hi` per entity. Handles undefeated/winless entities that break plain BT.

**`rankings plackett-luce`** — Hunter's MM on ballots; partial ballots (top-k, subsets) are likelihood-exact. Items last (or first) in *every* ballot have no finite MLE and get sectioned out, like BT's undefeated handling.

**`rankings markov-chain`** — `--damping` (0.85 web convention; 0.5–0.6 suits sports data) keeps the majority-move chain ergodic.

**`matchups weng-lin`** — `--variant bradley-terry` (logistic, default) or `thurstone-mosteller` (probit, `--epsilon` draw margin); paper defaults μ=25, σ=25/3, β=25/6. `--tau` (try 0.0833) re-inflates σ each match — important for many-team matches where σ otherwise collapses quickly. Output: `mu sigma ordinal` (ordinal = μ−3σ, the conservative display rating).

**`crowd bradley-terry`** — `--lambda` (0.5) anchors and regularizes via the virtual node; the second output section is per-annotator reliability η (~1 truthful, ~0.5 spammer, <0.5 adversarial — adversaries *add* signal once detected).

**`graph page-rank`** — `--damping-factor` (0.85) splits "follow links" vs "teleport"; `--sink-dispersion {reverse,all,uniform,none}` picks the dangling-node treatment (`uniform` is the textbook one); `--matches` reads tournament files as loser-endorses-winner.

**`graph hits` / `katz-centrality` / `degree` / `k-core`** — HITS emits authorities then hubs (two sections). Katz needs `--alpha` below `1/λ_max` (diverging runs abort with advice — try 0.05 on weighted graphs). `degree --direction {in,out,total}` is the baseline every fancier method should beat. k-core reports coreness (undirected reading).

**`bandit *`** — every policy: `--seed` (selection streams are reproducible and resume exactly across `--save-state`/`--load-state`), `--select N` prints the next arms to play instead of scores. `epsilon-greedy --epsilon` sets the exploration rate; `upper-confidence-bound --exploration` (2.0 = classic UCB1); `kl-ucb --c` (0 = the paper's practical choice, ≥3 for the theory) gives uniformly tighter indices for [0,1] rewards; `thompson-beta` needs rewards in [0,1] (`--prior-alpha/--prior-beta`); `thompson-gaussian` takes `--prior-mean/--prior-weight`; `exp3 --gamma` is the adversarial-setting option (offline replay is approximate — see its docs).

## State files

Every model (and dataset) saves as JSONL with a self-describing header:

```jsonl
{"propagon":1,"kind":"model","algorithm":"glicko2","params":{"tau":0.5,...},"entities":30}
{"id":"BOS","r":1670.8,"rd":40.5,"sigma":0.06}
```

Guarantees: save → load → save is byte-identical; params travel with the state (mismatched resumes are typed errors, not silent reconfigurations); incremental algorithms (`glicko2`, `elo`, `weng-lin`, `win-rate`, all bandits) fold new batches into loaded state without replaying history; iterative ones (`bradley-terry-model`, `plackett-luce`, `luce-spectral-ranking`, `rank-centrality`) warm-start from it.

## Correctness

Three independent test layers (see [`DEVELOPER.md`](DEVELOPER.md)):

- **Published reference vectors** — Bradley-Terry (and Plackett-Luce reduced to pairs, and the Bayesian BT posterior) reproduce Agresti's 1987 baseball coefficients; Massey, Colley, and Keener match the *Who's #1?* worked example; Weng-Lin matches the openskill test vectors to 9 decimals; PageRank and HITS match Langville & Meyer's published examples; Wilson intervals match Newcombe (1998) Table I; Borda/Copeland match the classic Tennessee election; Elo expectations match the canonical table.
- **v1 golden parity** — outputs match the original research implementation on the bundled season data.
- **Exhaustive oracles** — Kemeny is checked against brute-force enumeration of the full permutation space.

## Library features

- `parallel` (default) — multi-threaded fitting via rayon; disable for single-threaded targets (the WASM milestone builds with it off, and the whole suite passes either way).
- `io` (default) — `save_to_path`/`load_from_path` conveniences; in-memory readers/writers always available.
