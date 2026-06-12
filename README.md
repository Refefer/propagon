# propagon

Rank anything from comparisons — turn match results, ballots, graphs, and reward logs into clear, defensible rankings.

## What is propagon?

Propagon is a Rust library and command-line tool that turns messy comparison data into rankings. Feed it head-to-head match results, judge ballots, web links, or A/B test rewards, and it gives you a leaderboard — with methods ranging from simple win rates to Bayesian models that quantify uncertainty. Whether you're ranking chess players, sorting search results, or finding the most influential nodes in a network, propagon has a well-studied algorithm for the job.

## Quick start

```bash
git clone https://github.com/Refefer/propagon
cd propagon
cargo build --release          # binary at target/release/propagon
```

Requires Rust 1.88+. Building from source, the crate API, the full CLI reference, and the test strategy live in [`DEVELOPER.md`](DEVELOPER.md).

## Which algorithm should I use?

Use this guide to find the right category for your data:

| My data looks like... | Try these algorithms |
|---|---|
| Head-to-head match results (winner/loser pairs) | **Tournament algorithms** — Win Rate, Bradley-Terry, Elo, Glicko-2, Rank Centrality, and more |
| Matches with ties, margins of victory, or a home-field edge | **Tie/margin-aware** — Generalized Bradley-Terry, Massey, margin-of-victory Elo, Offense-Defense |
| Team or multiplayer matches, but you want per-player ratings | **Team models** — Team Bradley-Terry, Weng-Lin |
| Ratings that need to update as new games arrive | **Online algorithms** — Elo, Glicko-2 (incremental, no reprocessing); Whole-History Rating for full historical curves |
| Match-ups that form cycles (A beats B beats C beats A) | **Intransitivity-aware** — Blade-Chest, mElo, Nash averaging |
| Judges or voters ranking items from best to worst | **Rank aggregation** — Borda Count, Kemeny, Plackett-Luce, Mallows Model, Footrule, MC4 |
| A network of links, citations, or interactions | **Graph centrality** — PageRank, HITS, Katz, Degree, k-Core, LeaderRank, Harmonic, Connected Components |
| Each entity has features and you want strengths to depend on them | **Covariate models** — Covariate Bradley-Terry |
| Choosing between options with uncertain rewards (A/B testing, ad selection) | **Bandits** — Epsilon-Greedy, UCB, KL-UCB, Thompson Sampling, EXP3; Sliding-Window UCB (drift), LinUCB (with context), dueling bandits (preference feedback) |
| Episodes of states and rewards (sessions, rollouts, demonstrations) | **Trajectory values** — Monte Carlo, TD(0), Value Comparison, Behavior Cloning |
| You need confidence intervals on any leaderboard | **Bootstrap intervals** — the `--bootstrap N` wrapper on any batch ranker |

## Algorithm catalog

### Bayesian Bradley-Terry

**Real-world scenario**

You are running a chess tournament with 200 players and want a leaderboard that shows not just ratings but how confident you can be in each player's true skill — especially for players who only played 2 or 3 games and whose ratings would normally be wildly unstable. Unlike standard ranking methods that give you a single number and pretend they're sure, this approach embraces uncertainty and tells you exactly how much you can trust each player's rating.

**When to use**

- You need uncertainty quantification (credible intervals, rank probabilities) alongside rankings, not just point estimates.
- Your data is sparse or contains undefeated/winless entities that would cause standard Bradley-Terry to diverge — the Gamma priors regularize gracefully.
- You have a small-to-medium leaderboard (hundreds of entities) where honest uncertainty drives decisions, such as official rankings or seeding.

**When to avoid**

- You need real-time or streaming updates — the Gibbs sampler is batch-only and slower than iterative MM or online Elo.
- Your dataset is massive (millions of comparisons) — MCMC bookkeeping (convergence checks, mixing) adds overhead that simpler methods avoid.

**CLI usage**

```bash
propagon tournament bayesian-bradley-terry --shape 2.0 --samples 5000 --burn-in 1000 --credible 0.95 --seed 42 baseball.2018
```

**Library usage**

```rust
use propagon::algos::BayesianBradleyTerry;
use propagon::{GamesDataset, Ranker};

let mut games = GamesDataset::new();
games.push_pair("ARI", "COL", 1.0).unwrap();
games.push_pair("ARI", "NYM", 1.0).unwrap();
games.push_pair("COL", "NYM", 1.0).unwrap();

let bb = BayesianBradleyTerry::default();
let model = bb.fit(&games).unwrap();
for (name, mean, lo, hi) in model.posteriors() {
    println!("{name}: mean={mean:.3} [{lo:.3}, {hi:.3}]");
}
```

**What the numbers mean**

Each entity gets a posterior mean strength and a 95% credible interval (the lo/hi columns). The mean is the best-guess skill level on a log scale — higher means stronger. The interval width tells you how certain the model is: a narrow interval means the data strongly constrains the estimate, while a wide interval (common for sparse players) means the true skill could plausibly be much higher or lower. Unlike plain Bradley-Terry which gives a single number, you can ask "what is the probability team A is actually better than team B?" by comparing their posterior distributions.

### Behavior Cloning

**Real-world scenario**

You have logs of an expert at work — a seasoned dispatcher routing trucks, a champion player's recorded games, a deployed policy's decisions — and you want to turn "what the expert did" into "what to prefer." Behavior cloning counts how often each action was taken in each situation and treats that frequency as a revealed preference: the moves the expert made most are the moves worth imitating. It needs no rewards and no labels, just the demonstrations themselves.

**When to use**

- You have demonstration or interaction logs (state→action tokens) and want a fast imitation baseline or a revealed-preference ranking of actions.
- You want to convert expert behavior into pairwise preference edges you can feed to Bradley-Terry, Kemeny, or LSR — the `--emit-pairs` / `implied_pairs()` export does exactly that.
- You need per-state action distributions (which action dominates in each situation), optionally Laplace-smoothed so unseen actions don't sit at exactly zero.

**When to avoid**

- Your expert is suboptimal or inconsistent — behavior cloning imitates whatever was demonstrated, mistakes and all; it has no notion of reward or outcome quality. If you have rewards, rank states with Monte Carlo or TD value estimation instead.
- You need to generalize across situations that share structure but not exact labels — the counting is purely tabular, so a never-seen state gets only the smoothing prior.

**CLI usage**

```bash
# Rank actions by how often the expert took them (rewards in the file are ignored):
propagon trajectories behavior-cloning --per-state : --smoothing 1.0 expert-sessions.txt

# Or export the implied preference edges as tournament rows for a downstream ranker:
propagon trajectories behavior-cloning --per-state : --emit-pairs expert-sessions.txt > prefs.tsv
propagon tournament bradley-terry-model prefs.tsv
```

**Library usage**

```rust
use propagon::algos::{BehaviorCloning, Granularity, BradleyTerryMM};
use propagon::{Ranker, RankModel, TrajectoriesDataset};

let mut log = TrajectoriesDataset::new();
log.push_step("junction:turn-left", 0.0)?;
log.push_step("junction:go-straight", 0.0)?;
log.end_episode();                       // one episode of expert steps

let bc = BehaviorCloning {
    granularity: Granularity::PerState { separator: ':' },
    smoothing: 1.0,
};
let model = bc.fit(&log)?;
for (token, freq) in model.sorted_scores() {
    println!("{token}: {freq:.3}");      // P(action | state)
}

// Net preferences → feed any pairwise ranker:
let prefs = model.implied_pairs();
let ranking = BradleyTerryMM::default().fit(&prefs)?;
```

**What the numbers mean**

With `--per-state`, every `state:action` token gets a frequency between 0 and 1 that sums to 1 within its state — the empirical probability the expert took that action there. Without `--per-state`, each token is scored by its global share across all demonstrations. Laplace smoothing (`--smoothing α`) keeps never-seen actions off exactly zero: a score is `(count + α) / (N + α·K)` over the K actions in the group. With `--emit-pairs` you instead get preference rows `winner⇥loser⇥1⇥weight`, where the weight is how many more times the winning action was chosen than the losing one — the raw material for a full Bradley-Terry or Kemeny fit.

### BiRank

**Real-world scenario:** Imagine you're running a movie streaming platform with 10,000 users and 5,000 films. You want to identify both your most influential viewers — those who interact with highly-rated content — and your most important films — those engaged by influential viewers — all from your click and purchase logs. BiRank does exactly that: it co-ranks users and items simultaneously, so each side reinforces the other.

**When to use:**

- You have a natural bipartite interaction dataset (users↔items, authors↔venues, queries↔documents) and need importance scores for both sides.
- You want to go beyond raw popularity (degree counts) by letting the two sides reinforce each other — active users boost items, popular items boost users.
- You need a single method that replaces separate popularity counts on each side with a unified co-ranking.

**When to avoid:**

- You need preference strengths or matchup semantics (e.g., "user A prefers item X over Y") — BiRank scores are importance, not preference.
- Your graph is not naturally bipartite (e.g., a social network where everyone is the same node type) — use PageRank or HITS instead.

**CLI usage:**

```bash
propagon graph birank --alpha 0.85 --beta 0.85 --iterations 20 interactions.txt
```

**Library usage:**

```rust
use propagon::algos::{BiRank, BiRankModel};
use propagon::{GraphDataset, Ranker};

let mut g = GraphDataset::new();
g.push("user1", "item_A", 3.0);
g.push("user1", "item_B", 1.0);
g.push("user2", "item_B", 1.0);

let model = BiRank::default().fit(&g).unwrap();
for (name, score) in model.dst_scores() {
    println!("item {}: {:.4}", name, score);
}
```

**What the numbers mean:**

BiRank outputs two sorted lists: one for each side of the bipartite graph. Each node gets a non-negative importance score — higher means the node is connected to other high-scoring nodes on the opposite side. Scores are relative within each side (not directly comparable across sides). A user with a high score interacts with items that are themselves popular among influential users; an item with a high score is interacted with by influential users.

### Blade-Chest

**Real-world scenario**

Rock beats scissors, scissors beats paper, paper beats rock — no single "strength" number can explain a cycle like that. Blade-Chest learns a small vector per entity (a *blade* for how it attacks and a *chest* for how it is attacked) so it can model intransitive matchups: A reliably beats B, B beats C, yet C beats A. It's built for games, character match-ups, and preference data where a flat leaderboard would lie about who really beats whom.

**When to use**

- Your data has intransitive cycles — fighting-game character match-ups, strategy counters, LLM-vs-LLM evaluations where some models specifically exploit others — that Bradley-Terry or Elo cannot represent.
- You want both a transitive "overall strength" and a per-pair prediction that accounts for the cyclic part.
- You want to compare the inner-product and squared-distance ways of parameterizing intransitivity.

**When to avoid**

- Your data is essentially transitive — the extra embedding dimensions only add variance; Bradley-Terry is simpler and estimates the same order more tightly.
- You need a single interpretable score per entity — the printed γ is only the transitive slice; the interesting signal lives in the embeddings, which you read through `matchup(a, b)`, not off a leaderboard.

**CLI usage**

```bash
propagon tournament blade-chest --variant inner --dims 8 --epochs 50 --seed 2016 matchups.tsv
```

**Library usage**

```rust
use propagon::algos::{BladeChest, BladeChestVariant};
use propagon::{PairwiseDataset, Ranker, RankModel};

let mut data = PairwiseDataset::new();
data.push("rock", "scissors", 1.0);
data.push("scissors", "paper", 1.0);
data.push("paper", "rock", 1.0);

let model = BladeChest { variant: BladeChestVariant::Inner, ..BladeChest::default() }.fit(&data)?;
for (name, gamma) in model.sorted_scores() {
    println!("{name}: {gamma:.4}");                       // transitive part only
}
println!("P(rock ≻ paper) = {:?}", model.matchup("rock", "paper")); // full, cyclic-aware
```

**What the numbers mean**

The printed score per entity is only the transitive strength γ. For a pure cycle like rock-paper-scissors these come out essentially equal — which is the honest answer: no entity is globally stronger. The real content is the pairwise prediction `matchup(a, b) = P(a beats b)`, which combines γ with the blade and chest vectors and can sit far from 50% even when γ_a equals γ_b. `--variant inner` scores match-ups with vector inner products; `--variant dist` uses squared distances, a different geometry for the same idea. More `--dims` and `--epochs` fit richer cycles, at the usual risk of overfitting data that was transitive all along.

### Bootstrap Intervals

**Real-world scenario**

You fit a Bradley-Terry leaderboard and report that model A is #1 — but with only a few hundred comparisons, would a slightly different sample have put model B on top? Bootstrap intervals answer that. They refit *your chosen ranker* on many resampled copies of the data and report, for every entity, a score interval **and** a rank interval — so "#1, but the 95% rank interval is [1, 4]" tells the real story instead of a false certainty. It is not an algorithm of its own; it is a wrapper that puts honest error bars on any batch ranker.

**When to use**

- You're reporting a leaderboard from limited or noisy data and need honest uncertainty on both scores and ranks — "is this lead real, or sampling luck?"
- You want uncertainty on *any* batch ranker (Bradley-Terry, Massey, Kemeny, PageRank, Monte Carlo values, …) without each one needing its own Bayesian variant.
- You need defensible error bars for a report, a paper, or a decision such as seeding, promotion, or model selection.

**When to avoid**

- Your ranker is online and order-dependent (Elo, Glicko-2, the bandits, TD, mElo) — resampling rows changes what those models even mean, so `--bootstrap` is rejected on them by design. Use a model with native uncertainty (Glicko-2's rating deviation) instead.
- Your data is so sparse that many resamples disconnect the comparison graph — failed replicates are skipped and counted, and the run errors out if fewer than half survive. That failure is itself the signal: you don't have enough data for stable intervals.

**CLI usage**

```bash
# A group-global flag on every batch command — add 95% score + rank intervals:
propagon tournament bradley-terry-model --bootstrap 500 --bootstrap-credible 0.95 baseball.2018
propagon trajectories monte-carlo --bootstrap 1000 sessions.txt
```

**Library usage**

```rust
use propagon::algos::{Bootstrap, BradleyTerryMM};
use propagon::{PairwiseDataset, Ranker};

let mut data = PairwiseDataset::new();
data.push("ARI", "COL", 1.0);
data.push("COL", "NYM", 1.0);
data.push("ARI", "NYM", 1.0);

// Wrap any batch ranker; defaults are 200 replicates at 0.95:
let bs = Bootstrap { replicates: 500, ..Bootstrap::new(BradleyTerryMM::default()) };
let model = bs.fit(&data)?;
for (name, score, lo, hi) in model.intervals() {
    println!("{name}: {score:.3} [{lo:.3}, {hi:.3}]");
}
for (name, rank, rlo, rhi) in model.rank_intervals() {
    println!("{name}: rank {rank:.0} [{rlo:.0}, {rhi:.0}]");
}
```

**What the numbers mean**

Each entity gets its point estimate plus two intervals: a **score interval** (the central `--bootstrap-credible` mass of the refitted scores, default 95%) and a **rank interval** (the same band on its 1-based leaderboard position, where 1 is best). A narrow rank interval pinned at [1, 1] means the top spot is robust; a wide one like [1, 6] means the ordering near that entity is not yet decided by the data. If some replicates failed to fit, a `replicates ok: k/N` note on stderr tells you how many of the N resamples actually contributed.

### Borda Count

**Real-world scenario**

Imagine you've got 50 judges each ranking 10 essays and you need a single consensus ranking out of the mess. Borda Count does the heavy lifting: it gives each essay a score equal to the number of items ranked below it on each ballot, sums those scores across all judges, and ranks by the total. No fancy modeling — just straightforward point tallies that are easy to explain to stakeholders.

Borda Count works with both pairwise comparison data and full rankings: with pairwise data it reduces to weighted win totals; with full ballots it assigns positional points (N-1 for first, N-2 for second, …, 0 for last) and sums across all ballots.

**When to use**

- You need a fast, transparent baseline ranking with no model assumptions — Borda runs in linear time and the scoring is easy to explain.
- Your comparison schedule is roughly uniform (each item compared against similar opponents), where Borda is provably near-optimal for rank recovery.
- You want to aggregate heterogeneous rankings from multiple sources (e.g., combining results from different search engines, judges, or model outputs).

**When to avoid**

- Your comparison schedule is highly non-uniform (e.g., a few strong items dominate most matchups) — Borda inherits schedule bias with no correction; use Bradley-Terry instead.
- You need Condorcet consistency (a candidate who beats every other head-to-head should always win) — Borda can rank a Condorcet winner below others due to positional scoring.
- You need win-probability predictions or uncertainty estimates — Borda only produces scores, not probabilistic forecasts.

**CLI usage (pairwise data)**

```bash
propagon tournament borda-count examples/tournament/baseball.2018
```

**CLI usage (rankings data)**

```bash
propagon rankings borda-count f1-2024.rankings
```

**Library usage (pairwise data)**

```rust
use propagon::algos::Borda;
use propagon::{PairwiseDataset, Ranker, RankModel};

let mut data = PairwiseDataset::new();
data.push("ARI", "COL", 1.0);
data.push("ARI", "NYM", 1.0);
data.push("COL", "NYM", 1.0);

let model = Borda::default().fit(&data)?;
for (name, score) in model.sorted_scores() {
    println!("{name}: {score}");
}
```

**Library usage (rankings data)**

```rust
use propagon::algos::Borda;
use propagon::{RankingsDataset, RankModel};

let mut data = RankingsDataset::new();
data.push_ranking(["VER", "NOR", "LEC", "PIA"]).unwrap();
data.push_ranking(["NOR", "LEC", "VER", "PIA"]).unwrap();
let model = Borda::default().fit_rankings(&data)?;
for (id, score) in model.sorted_scores() {
    println!("{id}: {score}");
}
```

**What the numbers mean**

With pairwise data: each entity receives a score equal to the sum of weights on its wins. For example, if ARI beats COL (weight 1) and NYM (weight 1), ARI scores 2.0; if COL only beats NYM (weight 1), COL scores 1.0; NYM scores 0.0. Higher scores mean more weighted wins — the ranking is simply entities sorted by this total.

With rankings data: each item's score is the sum of positional points across all ballots. In a ballot of 4 items, a first-place finish earns 3 points, second earns 2, third earns 1, and last earns 0. These points add up across every ballot the item appears in. Higher total scores mean the item was consistently ranked higher across all ballots.

In both cases, scores are not on a standardized scale — they depend on how many comparisons or ballots were cast and how many items each contained.

### Bradley-Terry Model

**Real-world scenario**

Imagine you're running a chess tournament with 16 players and need a final ranking from head-to-head match results. The Bradley-Terry model converts every win and loss into a strength score that predicts how likely any two players are to beat each other in a future game — even if they never actually played. It's the go-to choice when you have pairwise outcomes and want calibrated, comparable scores rather than a simple win count.

**When to use**

- You have pairwise outcomes (win/loss) between entities and want calibrated strength scores, not just a simple win count.
- Skill is static during the observation period — for example, a single season or a frozen set of model checkpoints in an LLM arena.
- You need predictions for unseen pairs: the model interpolates win probabilities between any two entities even if they never played each other.

**When to avoid**

- Your comparison graph is disconnected (e.g., two separate pools that never played each other) — the model is undefined without bridging edges or regularization.
- Ties are frequent and meaningful — the base model handles only wins and losses; use the tie-aware extension or a different model instead.

**CLI usage**

```bash
propagon tournament bradley-terry-model --estimator mm --tolerance 1e-8 chess-tournament.csv
```

**Library usage**

```rust
use propagon::algos::BradleyTerryMM;
use propagon::{GamesDataset, Ranker};

let mut games = GamesDataset::new();
games.push_pair("Alice", "Bob", 1.0).unwrap();
games.push_pair("Bob", "Charlie", 1.0).unwrap();
games.push_pair("Alice", "Charlie", 1.0).unwrap();

let model = BradleyTerryMM::default().fit(&games).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {:.4}", id, score);
}
```

**What the numbers mean**

Each entity receives a strength score on a log scale: higher scores mean stronger entities. The difference between two scores translates directly into win probability via the logistic function — a 1.0-point gap means the stronger entity wins about 73% of the time, a 2.0-point gap means about 88%, and equal scores predict a 50/50 split. Only differences between scores are meaningful; the absolute level is arbitrary (identified only up to a constant shift).

### Colley

**Real-world scenario**

You are running a 12-team college football conference and want a fair, defensible power ranking that accounts for strength of schedule without being influenced by blowout scores or preseason biases. Colley is built exactly for this: it converts win/loss records into schedule-aware ratings by solving a single linear system that adjusts each team's effective wins based on the strength of their opponents — deliberately ignoring margins of victory to stay bias-free.

**When to use**

- You need rankings that are transparent, deterministic, and easy to audit (e.g., official standings or seeding).
- Your data is win/loss only with no meaningful margin information.
- You want a method that handles teams with easy or hard schedules fairly, and gracefully smooths undefeated or winless records.

**When to avoid**

- You care about blowout margins — Colley deliberately discards margin-of-victory signal, so Massey is a better fit if score differentials matter.
- You need probabilistic outputs (e.g., win probability estimates) — Colley produces point ratings only, unlike Bradley-Terry or Bayesian methods.

**CLI usage**

```bash
propagon tournament colley --iterations 500 --tolerance 1e-8 conference-games.txt
```

**Library usage**

```rust
use propagon::algos::{Colley, RankModel};
use propagon::dataset::PairwiseDataset;

let mut data = PairwiseDataset::new();
data.push("A", "B", 1.0);
data.push("B", "C", 1.0);
data.push("A", "C", 1.0);

let model = Colley::default().fit(&data)?;
for (name, score) in model.scores() {
    println!("{}: {:.4}", name, score);
}
```

**What the numbers mean**

Ratings are centered at 0.5, so a score of 0.5 means exactly average. A team with 0.625 is above average (it won one game against an average opponent), while 0.375 is below. The ratings always sum to n/2 across all n teams, and the gap between any two scores reflects their relative strength adjusted for schedule difficulty. Higher is always better.

### Connected Components

**Real-world scenario**

You have a social network of 10,000 users from three separate communities that never interacted and want to know how many isolated clusters exist before computing PageRank — running PageRank on a disconnected graph gives meaningless scores because users in different clusters can't be compared on a common scale.

**When to use**

- Before fitting any ranking algorithm that requires graph connectivity (Bradley-Terry, Elo, spectral methods) to detect whether your data satisfies the Ford condition.
- When you suspect your graph data comes from multiple independent sources or time periods that may not overlap.
- As a diagnostic step: if you get many small components, you need to bridge them with virtual edges or priors before ranking.

**When to avoid**

- When you need directed strongly-connected components (this treats edges as undirected; a one-way link still connects two nodes).
- When the graph is already known to be connected — it adds overhead with no benefit.

**CLI usage**

```bash
propagon graph components --min-graph-size 5 network_edges.txt
```

**Library usage**

```rust
use propagon::algos::extract_components;
use propagon::dataset::GraphDataset;

let mut g = GraphDataset::new();
g.push("alice", "bob", 1.0).unwrap();
g.push("charlie", "dave", 1.0).unwrap();
// alice-bob and charlie-dave are separate components
let comps = extract_components(g.view(), 1);
assert_eq!(comps.len(), 2); // two disconnected components
```

**What the numbers mean**

The tool writes one output file per component (named input.0, input.1, etc.), with the largest component first. Each file contains only the edges belonging to that component. If the original graph had 3 components and you used --min-graph-size 5, only components with 5+ nodes are written — smaller ones are silently dropped. A single output file means your graph is fully connected; multiple files means you have isolated groups that cannot be ranked against each other.

### Copeland

**Real-world scenario**

You have 8 candidates in a knockout tournament and want a simple, transparent ranking that rewards beating the most opponents head-to-head. For example, imagine you need to rank 12 debate team captains after a round-robin where each pair faced off once — Copeland gives you a straightforward answer by counting who beat who.

**When to use**

- You need a Condorcet-consistent method: if one entity beats every other entity head-to-head, it will always rank first.
- You want a fast, parameter-free ranking from pairwise outcomes with no tuning or convergence concerns (O(n²) majority counts).
- You need an interpretable score that a non-expert can verify by hand — each point means "beat one opponent."

**When to avoid**

- Your data is sparse (many pairs never observed) — Copeland needs most pairs to have played for meaningful discrimination.
- You need fine-grained separation near the middle of the table — Copeland scores are coarse integers (or half-integers), so many entities tie at the same score.

**CLI usage**

```bash
propagon tournament copeland --format tsv debate_results.txt
```

**Library usage**

```rust
use propagon::algos::Copeland;
use propagon::{PairwiseDataset, RankModel, Ranker};

let mut data = PairwiseDataset::new();
data.push("Alice", "Bob", 1.0);
data.push("Alice", "Carol", 1.0);
data.push("Bob", "Carol", 1.0);

let model = Copeland::default().fit(&data).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {}", id, score);
}
```

**What the numbers mean**

Each entity's score is the number of pairwise opponents it holds a majority over, with ties worth 0.5 points. A score of 2.0 means the entity beat 2 opponents head-to-head; a score of 0.5 means it split one matchup evenly and lost the rest. Higher scores rank first. Unlike Elo or Bradley-Terry, the score has no probabilistic interpretation — it is a pure head-count of pairwise victories.

### Covariate Bradley-Terry

**Real-world scenario**

You're ranking 200 chess players, but you also know each player's published rating, age, and hours practiced — and you'd like the leaderboard to *explain* itself in terms of those traits, not just fit a free strength per name. Covariate Bradley-Terry models each entity's strength as a weighted sum of its features (`s_i = β·x_i`), so it both ranks the players and tells you how much each feature is worth. Crucially, it can then score a brand-new entity you've never seen play, straight from its features.

**When to use**

- You have per-entity features and want strengths driven by them — useful for cold-start (rate a new item from its attributes alone) and for interpretability (which features predict winning?).
- Your comparison graph is sparse or has newcomers — sharing strength through features regularizes where a free-per-entity Bradley-Terry would be unstable.
- You want a conditional-logit model: the fitted coefficients β are the log-odds contribution of each feature.

**When to avoid**

- You have no meaningful features, or features that don't actually predict outcomes — plain Bradley-Terry is simpler and won't pretend the covariates matter.
- Strength is genuinely idiosyncratic per entity (not a function of measured traits) — force-fitting it through features underfits; use the free-parameter Bradley-Terry, optionally with per-entity intercepts (`--intercepts`, which needs `--l2 > 0`).

**CLI usage**

```bash
# features.txt rows are: entity x1 x2 ... xd  (whitespace-separated, same width)
propagon tournament covariate-bradley-terry --features features.txt --l2 0.01 matches.tsv
```

**Library usage**

```rust
use propagon::algos::CovariateBt;
use propagon::{PairwiseDataset, Ranker, RankModel};

let mut data = PairwiseDataset::new();
data.push("a", "b", 6.0);
data.push("b", "a", 2.0);
data.push("b", "c", 5.0);

let features = vec![
    ("a".to_string(), vec![1.0, -0.5]),
    ("b".to_string(), vec![0.0,  0.5]),
    ("c".to_string(), vec![-1.0, 1.0]),
];
let model = CovariateBt::new(features).fit(&data)?;   // l2 = 1e-4, no intercepts
println!("coefficients β = {:?}", model.coefficients());
for (name, s) in model.sorted_scores() {
    println!("{name}: {s:.4}");
}
let cold_start = model.score(&[0.25, -1.5])?;          // strength of an unseen entity
```

**What the numbers mean**

Each entity's score is `s_i = β·x_i` (plus a per-entity intercept if `--intercepts` is on) — a log-strength, exactly like plain Bradley-Terry, so differences are log-odds: a 1.0 gap means the stronger entity wins about 73% of the time. The extra payoff is the coefficient vector β (printed to stderr by the CLI, or `model.coefficients()` in the library): a positive βₖ means feature k pushes strength up, and its magnitude is how many log-odds points one unit of that feature buys. Because strength is a function of features, you can score any new entity from its feature vector alone with `model.score(x)`.

### Crowd Bradley-Terry

**Real-world scenario:** Imagine you have 200 crowdworkers ranking 15 product designs in pairwise comparisons and you want a trustworthy leaderboard. The catch? About 15 of those workers are randomly clicking or intentionally voting opposite to quality. Crowd Bradley-Terry handles this gracefully — it learns both item rankings and annotator reliability at the same time, so unreliable or adversarial voters get automatically down-weighted without you needing to identify them manually.

**When to use:**
- You have pairwise comparisons collected from multiple annotators of unknown or varying quality (e.g., paid crowdsourcing, peer review, multi-judge contests).
- You suspect some annotators may be spamming (random votes) or adversarial (systematically wrong) and want them detected automatically.
- You need a single ranking that accounts for who voted, not just what was voted — annotator overlap on common pairs is available.

**When to avoid:**
- Every annotator has voted on completely disjoint pairs with no overlap — reliability cannot be identified without shared comparisons.
- You have only one annotator or all annotators are known to be equally reliable — vanilla Bradley-Terry is simpler and faster.

**CLI usage:**
```bash
propagon crowd bradley-terry --lambda 0.5 --eta-prior-alpha 10 --eta-prior-beta 1 --iterations 50 votes.txt
```

**Library usage:**
```rust
use propagon::algos::CrowdBt;
use propagon::{AnnotatedPairsDataset, Ranker, RankModel};

let mut data = AnnotatedPairsDataset::new();
data.push_annotated("alice", "model_a", "model_b", 1.0).unwrap();
data.push_annotated("bob", "model_a", "model_c", 1.0).unwrap();

let model = CrowdBt::default().fit(&data).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {:.3}", id, score);
}
```

**What the numbers mean:** Each item gets a score (higher is better, anchored so the virtual node is 0 — only differences matter). Each annotator gets a reliability score η between 0 and 1: η near 1.0 means the annotator is truthful and their votes are heavily weighted; η near 0.5 means they are a spammer (essentially random) and their votes barely affect the ranking; η below 0.5 means they are adversarial (systematically wrong) — once detected, their votes are inverted to add signal rather than noise.

### Degree

**Real-world scenario**

Imagine you have a citation network of 2,000 research papers and want to quickly identify which papers are cited the most. With Degree centrality, the papers with the highest in-degree are your most referenced — no complicated algorithms needed, just a straightforward count of incoming edges.

**When to use**

- You need a fast, interpretable baseline before running more complex centrality measures like PageRank or HITS.
- Your graph is small to medium and raw endorsement counts are a meaningful signal (e.g., citation counts, follower counts).
- You want to understand which nodes are structurally popular without modeling the quality of their connections.

**When to avoid**

- The quality of endorsements matters more than quantity (e.g., a citation from a top journal vs. an obscure blog) — use PageRank or Eigenvector centrality instead.
- Your graph is easily gameable through reciprocal linking or bot accounts, since degree gives equal weight to every edge regardless of source.

**CLI usage**

```bash
propagon graph degree --direction in citations.tsv
```

**Library usage**

```rust
use propagon::algos::{Degree, Direction, Ranker};
use propagon::dataset::GraphDataset;

let mut graph = GraphDataset::new();
graph.push("paper_a", "paper_b", 1.0);
graph.push("paper_c", "paper_b", 1.0);

let model = Degree { direction: Direction::In }.fit(&graph)?;
for (name, score) in model.scores() {
    println!("{}: {}", name, score);
}
```

**What the numbers mean**

Each node gets a score equal to the sum of weights on edges matching the chosen direction. With `--direction in` (the default), a score of 5 means five incoming endorsements (or weighted endorsements totaling 5). With `--direction out`, it counts outgoing links. Higher scores mean more connected, but the score is a raw count — it does not distinguish between endorsements from important vs. trivial sources. If you need that nuance, look at PageRank or Eigenvector centrality.

### Dueling Bandits

**Real-world scenario**

You want to find the best of 10 chatbot configurations, but no one can score a single reply on an absolute scale — people can only say "this reply is better than that one." Dueling bandits run exactly that kind of experiment online: at each step the algorithm proposes a pair to compare, you report which won, and it uses the verdict to decide the next, smarter comparison — homing in on the best option while wasting as few duels as possible. This is a **library-only** tool: it drives a live comparison loop, so there's no batch CLI command for it.

**When to use**

- Your feedback is inherently relative ("A beat B"), not a numeric reward — human preference judgments, side-by-side A/B duels, taste tests.
- You're running the experiment live and want the algorithm to *choose the next pair to compare* rather than analyzing a fixed log.
- You want strong regret guarantees from a method built for preferences (RUCB or Double Thompson Sampling) rather than bolting comparisons onto a scalar bandit.

**When to avoid**

- You have absolute scalar rewards (clicks, conversions) — a standard bandit (UCB, Thompson, ε-greedy) is the right tool and converges faster.
- You only have a static, already-collected set of comparisons and don't control which pairs get played — fit a batch model (Bradley-Terry, Rank Centrality) instead; the value here is the active pair selection.

**Library usage** (it drives the loop — no CLI)

```rust
use propagon::algos::{DuelingBandit, DuelingPolicy};
use propagon::{OnlineRanker, PairwiseDataset, RankModel};

let algo = DuelingBandit { policy: DuelingPolicy::Rucb { alpha: 0.51 }, seed: 7 };
let mut model = algo.init();
model.add_arm("a");                       // register every option up front
model.add_arm("b");
model.add_arm("c");

for _ in 0..500 {
    let (x, y) = model.select_pair()?;    // the algorithm proposes the next duel (names)
    let (winner, loser) = run_duel(&x, &y); // your environment decides the outcome
    let mut batch = PairwiseDataset::new();
    batch.push(winner, loser, 1.0);
    algo.update(&mut model, &batch)?;     // fold the result back in
}
for (arm, copeland) in model.sorted_scores() {
    println!("{arm}: {copeland:.3}");     // Copeland fraction
}
```

**What the numbers mean**

`select_pair()` hands you the next `(champion, challenger)` to compare — that's the active part of the loop, and it advances a persisted RNG counter so a saved-and-reloaded run keeps proposing the same sequence. The leaderboard score is each arm's **Copeland fraction**: the share of the opponents it has actually faced that it beats by a strict majority of duel weight, so 1.0 means "beats everyone it has met" and 0.0 means "beats no one yet." `DuelingPolicy::Rucb` uses optimistic upper-confidence bounds on the preference matrix; `DoubleThompson` samples two challengers from the posterior — both default to `alpha = 0.51`, the value both papers recommend.

### Elo

**Real-world scenario**

You are running a chess tournament with 200 players and want to maintain a live leaderboard that updates after every round, so spectators can see who is rising and falling in real time. Elo does exactly that: after each game it nudges both players' ratings toward what the result implied, by a step proportional to how surprising the outcome was. No reprocessing, no batch jobs — just feed it the new results and the rankings update instantly.

**When to use**

- You need a live leaderboard that updates incrementally as new results arrive, without reprocessing all history.
- Simplicity and low latency matter more than statistical nuance — Elo is O(1) per game and trivially transparent.
- Your entities' true skill genuinely drifts over time (e.g., players improving or declining), and you want ratings to track that drift naturally.

**When to avoid**

- You need uncertainty estimates (e.g., how confident are we in a newcomer's rating?) — Elo treats a 3-game rookie and a 3000-game veteran identically; use Glicko-2 instead.
- Skills are static (e.g., frozen LLM checkpoints) — Elo's constant learning rate makes it a noisy estimator for static skills; an offline Bradley-Terry fit is better.

**CLI usage**

```bash
propagon tournament elo --k 32 --initial-rating 1500 --scale 400 matches.txt
```

**Library usage**

```rust
use propagon::algos::{Elo, EloModel, OnlineRanker};
use propagon::dataset::GamesDataset;

let elo = Elo::default(); // k=32, initial_rating=1500, scale=400
let mut model: EloModel = elo.init();
let dataset = GamesDataset::read("matches.txt").unwrap();
elo.update(&mut model, &dataset).unwrap();
for (name, rating) in model.scores() {
    println!("{name}: {rating:.0}");
}
```

**What the numbers mean**

Ratings are on a scale where the default initial rating of 1500 represents an average player. The difference between two ratings predicts the expected win probability: with the standard 400-point scale, a 200-point gap means the higher-rated player wins about 76% of games, and a 400-point gap means about 91%. A player who wins gains points (more if the win was unexpected) and the loser loses an equal amount; draws split the expected score (each gets 0.5).


### Epsilon-Greedy

**Real-world scenario**

Imagine you're running an online store with five product banner designs and you want to maximize click-through rate. Instead of sticking with one banner forever or randomly shuffling them, Epsilon-Greedy shows the best-performing banner 90% of the time but rotates the other four banners the remaining 10% — keeping the data flowing so a newer design can still prove itself against the current leader.

**When to use**

- You need a simple, well-understood baseline for adaptive allocation (A/B/n testing, ad selection, recommendation slots).
- You have a fixed budget for exploration and want to control it directly via the epsilon parameter (e.g., always spend exactly 5% of traffic on exploration).
- Your arms produce scalar rewards (clicks, conversions, revenue) rather than pairwise comparisons.

**When to avoid**

- You need provably optimal regret guarantees — a constant epsilon yields linear regret over time; use UCB1 or Thompson Sampling instead for logarithmic regret.
- Your arms are related through a comparison graph (beating a strong opponent teaches a lot) — bandits treat arms independently; use Bradley-Terry or Elo for that structure.

**CLI usage**

```bash
propagon bandit epsilon-greedy --epsilon 0.1 --seed 42 ad-clicks.txt
```

**Library usage**

```rust
use propagon::algos::{Bandit, BanditModel, BanditPolicy};
use propagon::{OnlineRanker, RankModel, RewardsDataset};

let mut rewards = RewardsDataset::new();
rewards.push("banner_a", 1.0);
rewards.push("banner_b", 0.0);

let mut model = Bandit {
    policy: BanditPolicy::EpsilonGreedy { epsilon: 0.1 },
    ..Default::default()
}.init();
model.update(&rewards).unwrap();
println!("{:?}", model.sorted_scores());
```

**What the numbers mean**

Each arm gets a score equal to its empirical mean reward — total rewards divided by number of pulls. A score of 0.15 means that banner was clicked 15% of the time it was shown. Arms are ranked from highest to lowest mean reward. The epsilon parameter (e.g., 0.1) controls how often a random arm is picked instead of the current best — higher epsilon means more exploration but slower convergence to the true ranking.

### EXP3

**Real-world scenario**

Imagine you run a news website that shows five different headline variants and want to maximize clicks. Competing sites adapt their content daily, so click patterns shift unpredictably — yesterday's best headline might be today's worst. EXP3 is built for exactly this kind of adversarial setting, where the environment doesn't play fair and reward distributions change in response to your choices.

**When to use**

- Your rewards are not i.i.d. per arm: the environment is strategic or adversarial (e.g., A/B testing ads against competitors who adapt, market-making, or any setting where reward distributions shift in response to your policy).
- You need provable regret guarantees without assuming stationarity: EXP3 achieves O(√(TK log K)) regret even under worst-case reward sequences.
- Rewards are bounded in [0, 1] and you want an algorithm that requires no distributional assumptions.

**When to avoid**

- Your data is truly i.i.d. per arm (e.g., a static product catalog): Thompson Sampling or UCB1 will give much tighter performance with O(log T) regret.
- You need exact reproducibility from offline logs: EXP3's replay importance-weights each row by the current policy's selection probability, making offline analysis approximate and order-dependent.

**CLI usage**

```bash
propagon bandit exp3 --gamma 0.1 --select 1 --seed 42 ad-clicks.txt
```

**Library usage**

```rust
use propagon::algos::{Bandit, BanditPolicy};
use propagon::{OnlineRanker, RankModel, RewardsDataset};

let mut rewards = RewardsDataset::new();
rewards.push("headline_a", 1.0)?;
rewards.push("headline_b", 0.0)?;

let bandit = Bandit { policy: BanditPolicy::Exp3 { gamma: 0.1 } };
let mut model = bandit.init();
bandit.update(&mut model, &rewards)?;
println!("{:?}", model.sorted_scores());
```

**What the numbers mean**

Each arm receives a log-weight score derived from exponentially weighted rewards, with the exploration mix (gamma, default 0.1) controlling how much uniform random exploration is blended in. Higher scores indicate arms that have accumulated more reward-weighted evidence. The scores are relative — only differences matter, not absolute values. The default gamma of 0.1 means 10% of each selection is pure exploration and 90% follows the exponential-weight ranking.

### Footrule

**Real-world scenario**
You have 15 judges each ranking 8 research proposals from best to worst and want to produce a single fair consensus ranking. Footrule gives you the exact optimal order in seconds, whereas the gold-standard Kemeny method would take hours.

**When to use**
- You need an exact optimal consensus ranking but Kemeny is too slow (Footrule is a guaranteed 2-approximation to Kemeny).
- You have complete or partial ballots from multiple rankers and want a principled aggregation.
- You want a ranking method with provable guarantees rather than heuristic approximations.

**When to avoid**
- Your dataset has more than a few thousand items — the algorithm uses O(n²) memory and O(n³) time, so it doesn't scale to large catalogs.
- You only have pairwise comparisons (not full rankings) — Footrule needs at least partial ordered ballots, not head-to-head votes.

**CLI usage**
```bash
propagon rankings footrule proposal-rankings.txt
```

**Library usage**
```rust
use propagon::algos::Footrule;
use propagon::Ranker;

let dataset = RankingsDataset::from_file("proposal-rankings.txt")?;
let model = Footrule::default().fit(&dataset)?;
println!("total footrule distance: {}", model.cost());
for item in model.order() {
    println!("  {}", item);
}
```

**What the numbers mean**
The output is an ordered list of items from best to worst consensus rank, plus a total footrule distance printed to stderr. The footrule distance is the sum of absolute rank displacements — a lower number means the consensus ranking agrees more closely with the input ballots. For example, if an item was ranked 1st in most ballots but placed 3rd in the consensus, it contributes 2 to the total distance. A distance of 0 means perfect agreement among all ballots.

### Generalized Bradley-Terry

**Real-world scenario**

You're ranking a soccer league where draws are common and home teams have a real edge — two things plain Bradley-Terry simply can't see, because it only knows wins and losses and treats every game as neutral-site. Generalized Bradley-Terry adds both: a tie model so a draw is its own outcome (not a discarded or half-counted win), and an estimated home advantage so the leaderboard isn't fooled by who hosted whom. You get strengths plus two interpretable extras: how often games tie and how big the home edge is.

**When to use**

- Ties are frequent and meaningful (soccer, chess, many board games) and you want them modeled, not thrown away — choose Davidson or Rao-Kupper as the tie likelihood.
- One side has a structural advantage (home field, white pieces, first move) and you want it estimated and removed from the strengths — list the advantaged side first and pass `--home-advantage`.
- You want the familiar Bradley-Terry strengths but with these real-world wrinkles handled inside the model rather than pre-processed away.

**When to avoid**

- Your data is clean win/loss with no ties and no venue effect — plain Bradley-Terry is simpler and identical in that case.
- Your comparison graph is disconnected or has entities that only ever win or only ever lose — like all maximum-likelihood Bradley-Terry variants, the fit needs connectivity (the error names the offenders); use Bayesian Bradley-Terry for sparse, separated data.

**CLI usage**

```bash
# side 1 (listed first) is treated as home when --home-advantage is set:
propagon tournament generalized-bradley-terry --tie-model davidson --home-advantage soccer.tsv
```

**Library usage**

```rust
use propagon::algos::{GeneralizedBt, TieModel, HomeAdvantage};
use propagon::{Ranker, RankModel, GamesDataset, GameOutcome};

let mut d = GamesDataset::new();
d.push_game(&["arsenal"], &["chelsea"], GameOutcome::Side1Win(1.0), 5.0)?; // home win
d.push_game(&["arsenal"], &["chelsea"], GameOutcome::Side2Win(1.0), 3.0)?; // away win
d.push_game(&["arsenal"], &["chelsea"], GameOutcome::Tie, 4.0)?;           // draw

let algo = GeneralizedBt {
    ties: TieModel::Davidson,
    home: HomeAdvantage::Estimate,
    ..GeneralizedBt::default()
};
let model = algo.fit(&d)?;
for (name, pi) in model.sorted_scores() {
    println!("{name}: {pi:.4}");
}
println!("tie ν = {:.3}, home γ = {:.3}", model.tie_parameter(), model.home_advantage());
```

**What the numbers mean**

Each entity gets a strength π (normalized to sum to 1, so larger is stronger and ratios are odds), plus two fitted extras printed to stderr by the CLI (and exposed as `tie_parameter()` / `home_advantage()` in the library). The **tie parameter** controls how readily evenly-matched entities draw — larger means draws are more common at equal strength. The **home advantage γ** is a multiplicative boost applied to whichever side was listed first: γ = 1 means no edge, γ = 1.3 means the home side plays as if about 30% stronger. With those removed, the strengths reflect true ability rather than schedule luck.

### Glicko-2

**Real-world scenario**

You run an online chess server with 10,000 players of varying activity levels and want to display not just each player's skill rating but also a confidence band showing how well-established that rating is, so new players get matched fairly until the system has enough data. Glicko-2 extends the classic Glicko system by tracking per-player volatility, so it learns who is steady and who is streaky — surprising results move erratic players' ratings more than established ones.

**When to use**

- Your entities' true skill drifts over time and data arrives as a stream (ongoing leagues, matchmaking servers, weekly tournaments).
- You need to display honest uncertainty alongside ratings — e.g. 'this player's rating could be ±50 points' — for matching or leaderboards.
- Participation is intermittent: some players compete every round, others sporadically, and you want the system to handle missing data gracefully.

**When to avoid**

- Your entities are static (frozen LLM checkpoints, published products) — online updating is the wrong tool; use a static batch method like Bradley-Terry instead.
- You need to handle team matches or score margins — Glicko-2 is pairwise-only with no team or margin support.

**CLI usage**

```bash
propagon tournament glicko2 --tau 0.5 --save-state ratings.jsonl week1.txt
```

**Library usage**

```rust
use propagon::algos::Glicko2;
use propagon::{GamesDataset, OnlineRanker, RankModel};

let mut games = GamesDataset::new();
games.push_pair("alice", "bob", 1.0).unwrap();
games.push_pair("bob", "carol", 0.5).unwrap(); // tie

let algo = Glicko2::default();
let mut ratings = algo.init();
algo.update(&mut ratings, &games).unwrap();
for (name, state) in ratings.players() {
    println!("{}: r={:.1} rd={:.1} σ={:.4}", name, state.r, state.rd, state.sigma);
}
```

**What the numbers mean**

Each entity gets three numbers: a rating (r, default center 1500), a rating deviation (RD), and a volatility (σ). The rating works like Elo — a higher number means stronger play, and a 173.89-point gap means the higher-rated entity wins roughly 75% of games. The RD quantifies uncertainty: the 95% credible interval is [r − 1.96×RD, r + 1.96×RD]. A new player starts at RD=350 (wide uncertainty); RD shrinks with more games. The volatility σ (typically 0.01–0.15) governs how fast RD regrows between rating periods — a high σ means the player's form is unstable and ratings should react faster to new results.

### Greedy

**Real-world scenario**
You are running an A/B/n test on 5 email subject lines and want to quickly route traffic to whichever subject line has the highest click-through rate so far, without wasting sends on underperforming variants. Greedy does exactly that: it always picks the arm with the highest observed average reward and never deliberately explores the rest.

**When to use**

- You have enough historical reward data that the empirical means are already well-estimated and you trust the current leader.
- You want a fast, parameter-free baseline to compare against more sophisticated bandit policies like UCB1 or Thompson Sampling.
- You are ranking arms from logged (arm, reward) data offline and just need a simple ordering by average reward.

**When to avoid**

- You have few observations per arm or arms have very unequal sample counts — the greedy policy will lock onto early noise and starve under-sampled arms of data.
- The true best arm is unknown and you need guarantees about regret — greedy has no theoretical regret bounds once its initial optimism (if any) is spent.

**CLI usage**

```bash
propagon bandit greedy --seed 42 campaign-clicks.txt
```

**Library usage**

```rust
use propagon::algos::{Bandit, BanditPolicy};
use propagon::{OnlineRanker, RankModel, RewardsDataset};

let algo = Bandit {
    policy: BanditPolicy::Greedy,
    ..Default::default()
};
let mut model = algo.init();
let mut rewards = RewardsDataset::new();
rewards.push("subject_a", 0.12).unwrap();
rewards.push("subject_b", 0.08).unwrap();
algo.update(&mut model, &rewards).unwrap();
let scores = model.scores();
```

**What the numbers mean**

Each arm receives a score equal to its empirical mean reward (total reward divided by number of observations). A higher score means that arm has historically delivered better rewards on average. An arm with score 0.15 has earned 15% average reward across all its pulls; an arm with score 0.08 earned 8%. Arms with no observations score 0.0 and rank last.

### Harmonic Centrality

**Real-world scenario**

Imagine you're running a Wikipedia-style knowledge base with 10,000 articles linked by hyperlinks, and you want to identify the most important (well-connected) articles. The catch? Some clusters of articles are completely disconnected from the rest of the network. Harmonic centrality handles this gracefully — it ranks nodes by summing the inverse of shortest-path distances to all other reachable nodes, so unreachable nodes simply contribute zero rather than breaking the calculation entirely.

**When to use**

- Your graph may be disconnected or have isolated clusters — unlike closeness centrality, harmonic centrality handles unreachable nodes gracefully by treating them as contributing zero.
- You need a principled, axiom-compliant default for "generic importance" — harmonic centrality is the only centrality measure satisfying all of Boldi-Vigna's axioms (size, density, score monotonicity).
- You care about reachability and information flow rather than raw endorsement quality — for example, identifying spreaders in a social network or key pages in a link graph.

**When to avoid**

- Your edges represent endorsements of quality rather than structural connectivity (e.g., citations meaning "this paper is good") — use PageRank or eigenvector centrality instead.
- Your graph is very large and dense — the O(V·E) time complexity means it can be slow on big graphs without sampling.

**CLI usage**

```bash
propagon graph harmonic --direction in --weighted citation-network.txt
```

**Library usage**

```rust
use propagon::algos::Harmonic;
use propagon::dataset::GraphDataset;

let mut graph = GraphDataset::new();
graph.push("page_a", "page_b", 1.0);
graph.push("page_b", "page_c", 1.0);
graph.push("page_c", "page_a", 1.0);

let model = Harmonic::default().fit(&graph)?;
for score in model.iter_scores() {
    println!("{}: {:.4}", score.entity, score.score);
}
```

**What the numbers mean**

Each node gets a non-negative score equal to the sum of 1/distance to every other reachable node. A score of 0 means the node is completely isolated. Higher scores mean the node is, on average, closer to more nodes in the graph. For a fully connected graph of N nodes where all distances are 1, every node scores N-1. Unlike closeness centrality, scores are comparable across disconnected graphs because unreachable pairs simply contribute 0.

### HITS

**Real-world scenario**
You are building a research paper recommender and have a citation network of 10,000 papers — you want to identify both the most influential surveys (authorities) and the most useful review journals or meta-analyses that point to many important papers (hubs). Unlike PageRank, which gives every node a single importance score, HITS gives you two complementary lenses: one for "this node is great content" and one for "this node is a great curator."

**When to use**
- You have a directed endorsement graph (citations, retweets, link directories) where nodes naturally fall into two roles: curators pointing to content, and content being pointed to.
- You want to separate "good linkers" from "good content" rather than collapsing everything into a single importance score like PageRank.
- The graph is query-focused or domain-narrowed, where the hub/authority split is genuinely meaningful (e.g., a subgraph of papers about one topic).

**When to avoid**
- Your graph contains tightly-knit communities of nodes that link to each other (e.g., link farms, reciprocal citation rings) — these communities capture the principal eigenvector and distort all scores; use SALSA or BiRank instead.
- You need a single per-node importance score — HITS produces two scores which complicates downstream ranking.

**CLI usage**
```bash
propagon graph hits --iterations 100 --tolerance 1e-12 citations.txt
```

**Library usage**
```rust
use propagon::{Hits, RankModel};

let mut hits = Hits::default();
let model = hits.fit(&graph)?;

for (id, score) in model.authority_scores() {
    println!("{}: {:.4}", id, score);
}
```

**What the numbers mean**
Each node gets two L1-normalized scores between 0 and 1. Authority scores sum to 1 across all nodes — a score of 0.3 means that node accounts for 30% of the total authority mass in the graph. Hub scores also sum to 1 independently. In a star graph where three nodes all point to one center, the center gets authority 1.0 and each spoke gets hub 0.333. Nodes outside the dominant connected component can legitimately score 0. Only relative rankings within each score type are meaningful, not cross-score comparisons.

### HodgeRank

**Real-world scenario:** You have 50 judges ranking 10 essays and want to know not just the final ranking but whether the judges' preferences are actually consistent enough to produce a meaningful leaderboard — a high inconsistency score would tell you the essays are too stylistically diverse for a single linear ranking.

**When to use:**

- You need both a ranking and an audit of whether the data supports a total ordering (the inconsistency score quantifies cyclic noise).
- Your data has margins or weights (point differentials, win counts) and you want a least-squares approach that naturally incorporates them.
- You want a fast, single linear-solve method that handles imbalanced or incomplete comparison graphs without special case handling.

**When to avoid:**

- You need calibrated win probabilities (HodgeRank outputs least-squares potentials, not a generative model with probability outputs).
- Your comparison graph is disconnected (the solver requires connectivity to share a common scale across entities).

**CLI usage:**

```bash
propagon tournament hodge-rank --flow log-odds baseball.2018
```

**Library usage:**

```rust
use propagon::algos::{HodgeRank, HodgeFlow};
use propagon::{PairwiseDataset, Ranker, RankModel};

let mut data = PairwiseDataset::new();
data.push("TeamA", "TeamB", 5.0);
data.push("TeamB", "TeamC", 3.0);
data.push("TeamA", "TeamC", 7.0);

let model = HodgeRank { flow: HodgeFlow::LogOdds, ..Default::default() }.fit(&data)?;
println!("inconsistency: {:.4}", model.inconsistency());
```

**What the numbers mean:** Each entity gets a potential score (mean-centered, so only differences matter — like log-strengths). A higher score means the entity is ranked higher. The inconsistency value is the key diagnostic: it ranges from 0 to 1 and represents the fraction of the observed pairwise flow that lives in cycles and cannot be explained by any ranking. An inconsistency near 0 means the data is highly consistent and the ranking is trustworthy; near 1 means the data is dominated by rock-paper-scissors cycles and no total ordering is meaningful.


### I-LSR Spectral Ranking

**Real-world scenario**

You have 50 judges each ranking 10 essays from best to worst and want a single, statistically principled overall ranking that accounts for all position information — not just pairwise head-to-heads. I-LSR is built exactly for this: it takes full or partial ballots and produces a ranking that uses every piece of positional data, using the speed of spectral methods and the statistical rigor of maximum-likelihood estimation.

**When to use**

- You need Plackett-Luce strength estimates (the gold standard for multiway choices) but iterative optimization like MM is too slow on large datasets.
- Your data includes full rankings, partial rankings, choice-from-set events, or pairwise outcomes — I-LSR handles all three without decomposing to pairs.
- You want the exact Plackett-Luce MLE without the slow convergence of Hunter's MM or SGD — I-LSR reaches the same fixed point in a handful of spectral iterations.
- You want the statistical guarantees of MLE (unbiased, efficient) with the speed of a spectral method (power iteration, O(events) per pass).

**When to avoid**

- Your data has entities that never win or never lose (Ford-connectivity violation) — the MLE does not exist and I-LSR will error; use Bayesian Bradley-Terry instead.
- Your data contains ties or uncertainty — I-LSR does not handle tied positions or probabilistic preference annotations.
- You need uncertainty estimates or credible intervals — I-LSR produces point estimates only; pair it with bootstrap or switch to Bayesian methods.

**CLI usage (pairwise data)**

```bash
propagon tournament i-luce-spectral-ranking --outer 50 --inner-steps 30 --tolerance 1e-8 search-rankings.txt
```

**CLI usage (rankings data)**

```bash
propagon rankings i-luce-spectral-ranking --outer 200 --inner-steps 100 f1-2024.rankings
```

**Library usage (pairwise data)**

```rust
use propagon::algos::ILsr;
use propagon::{PairwiseDataset, Ranker, RankModel};

let mut data = PairwiseDataset::new();
data.push("model_a", "model_b", 1.0).unwrap();
data.push("model_b", "model_c", 1.0).unwrap();

let algo = ILsr { outer: 50, inner_steps: 30, tolerance: 1e-8 };
let model = algo.fit(&data).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {:.4}", id, score);
}
```

**Library usage (rankings data)**

```rust
use propagon::algos::ILsr;
use propagon::{Ranker, RankModel, RankingsDataset};

let mut ballots = RankingsDataset::new();
ballots.push_ranking(&["A", "B", "C", "D"])?;
ballots.push_ranking(&["B", "A", "D", "C"])?;

let model = ILsr::default().fit(&ballots)?;
for (id, score) in model.sorted_scores() {
    println!("{}: {:.4}", id, score);
}
```

**What the numbers mean**

Scores are Plackett-Luce strength parameters on a log scale, mean-centered so only differences matter. A difference of 1.0 between two entities means the higher one is roughly 2.7× as likely to be chosen (e^1.0). A difference of about 0.69 (ln 2) means roughly twice as likely; a gap of 2.3 (ln 10) means about 10× as likely to win a head-to-head. The ranking order is the same as the underlying Plackett-Luce model: higher scores indicate stronger entities. Because scores are mean-centered, an entity with score 0.0 is average; positive scores are above average, negative below.

### k-Core Decomposition

**Real-world scenario:** You are analyzing a social network of 10,000 users and want to identify the most influential spreaders for a marketing campaign — not just the people with the most followers, but those embedded in tightly-knit communities where information cascades fastest.

**When to use:**

- You need a fast, robust measure of node importance on a large network (runs in O(edges) time).
- You care about spreading power or influence cascades — coreness predicts epidemic-style spread better than raw degree or betweenness centrality.
- You want a coarse filter to narrow down candidates before applying a finer-grained (but expensive) ranking method within the top cores.

**When to avoid:**

- You need fine-grained differentiation: coreness produces many ties (integer values), so it cannot rank nodes within the same shell.
- Your graph is inherently directed or weighted and those semantics matter — k-core treats the graph as undirected and unweighted, ignoring edge direction and weights.

**CLI usage:**

```bash
propagon graph k-core collaboration_network.txt
```

**Library usage:**

```rust
use propagon::algos::KCore;
use propagon::dataset::GraphDataset;

let mut graph = GraphDataset::new();
graph.push("alice", "bob", 1.0);
graph.push("bob", "carol", 1.0);
graph.push("carol", "alice", 1.0);

let model = KCore::default().fit(&graph)?;
for (name, score) in model.scores() {
    println!("{}: coreness {}", name, score);
}
```

**What the numbers mean:**

Each node receives an integer coreness score. A score of 1 means the node is a leaf or pendant (only connected to the rest of the network through one edge). A score of 2 means the node survives in the 2-core — it sits inside at least one cycle. Higher scores indicate deeper embedding in dense subgraphs: a node with coreness 5 belongs to a subgraph where every node has at least 5 neighbors within that subgraph. Nodes with the highest coreness are the best candidates for seeding information cascades, as being embedded in a dense core is more predictive of spreading power than simply having many peripheral connections.

### Katz Centrality

**Real-world scenario**

Imagine you have a citation network of 200 research papers and you want to find out which ones are truly influential. You don't just want the papers with the most citations — you want the papers cited by other well-cited papers. Katz centrality gives you that: it counts every incoming walk to each node, but discounts longer walks geometrically so direct citations matter most while indirect endorsement chains still contribute. And unlike eigenvector centrality, it works perfectly fine on DAGs (like citation graphs) where there are no cycles.

**When to use**

- You have a directed graph (e.g., citations, endorsements, follower links) and want importance scores that reward being pointed to by important nodes.
- Your graph is a DAG or weakly connected where eigenvector centrality would collapse to zeros — Katz handles these gracefully.
- You want a tunable trade-off between degree centrality (low alpha) and eigenvector centrality (high alpha) via the alpha parameter.

**When to avoid**

- Your graph has extreme degree hubs that should be normalized out — use PageRank instead, which normalizes by out-degree and adds teleportation.
- You need a purely statistical model grounded in comparison data rather than graph structure — consider Rank Centrality or Bradley-Terry.

**CLI usage**

```bash
propagon graph katz-centrality --alpha 0.1 citations.tsv
```

**Library usage**

```rust
use propagon::algos::Katz;
use propagon::dataset::GraphDataset;

let mut g = GraphDataset::new();
g.push("paper_a", "paper_b", 1.0);
g.push("paper_b", "paper_c", 1.0);
g.push("paper_a", "paper_c", 1.0);

let model = Katz::default().fit(&g)?;
println!("{:?}", model.sorted_scores());
```

**What the numbers mean**

Each node receives a score of at least 1.0 (the baseline self-contribution). A score of 3.5 means the node is reached by roughly 2.5 times as many discounted walks as an average node. The scores are proportional to influence: if node A scores 10 and node B scores 5, A is approximately twice as central in the endorsement graph. Higher alpha values amplify differences between highly-connected and peripheral nodes.

### Keener

**Real-world scenario**

You have the final score from every game in a 12-team college football conference and want a single ranking that rewards teams for beating strong opponents by large margins, not just for winning against weak teams. Keener builds a score matrix from your pairwise results and computes its leading eigenvector — the idea being that a win over a top team should boost your rating more than a win over a bottom-feeder.

**When to use**

- You have pairwise results with scores or margins (e.g. points, goals, run-differentials) and want a static season-end ranking.
- Strength of schedule matters: you want points scored against strong opponents to count more than points against weak ones.
- You need a simple, fast ranking with minimal hyperparameters — the defaults (skew on, row-normalization on) work well out of the box for most sports leagues.

**When to avoid**

- Your schedule is disconnected (e.g. separate divisions with no cross-play) — the eigenvector will concentrate on the largest component and silently ignore the rest.
- You need probabilistic semantics like win probabilities or confidence intervals; Keener outputs eigenvector mass, not calibrated probabilities.

**CLI usage**

```bash
propagon tournament keener --iterations 500 --tolerance 1e-10 football.scores
```

**Library usage**

```rust
use propagon::{PairwiseDataset, Ranker};
use propagon::algos::Keener;

let mut data = PairwiseDataset::new();
data.push("Chiefs", "Bills", 27.0);
data.push("Bills", "Chiefs", 24.0);
data.push("Chiefs", "Eagles", 31.0);
data.push("Eagles", "Chiefs", 7.0);

let model = Keener::default().fit(&data).unwrap();
for (name, score) in model.sorted_scores() {
    println!("{}: {:.4}", name, score);
}
```

**What the numbers mean**

Each entity receives a nonnegative score that sums to 1 across all entities. A higher score means a stronger team: for example, a score of 0.35 means the team accounts for 35% of the total eigenvector mass, indicating it is the top-ranked entity. The absolute values only matter relative to each other — they represent each team's share of the total 'strength' in the system. Points earned against high-scoring opponents contribute more to your rating because the eigenvector computation propagates strength through the network of games.

### Kemeny

**Real-world scenario**

You have 50 judges each ranking 10 essays from best to worst and want to produce one definitive contest ranking that disagrees with the judges' opinions as little as possible. Kemeny finds the single consensus order that minimizes the total pairwise disagreement (Kendall tau distance) across every ballot — the mathematically principled "best compromise" when multiple sources each provide their own ranking.

**When to use**

- You have multiple complete or near-complete rankings from different judges, voters, or benchmark sources and need a single consensus order.
- You need a theoretically grounded result: Kemeny is the maximum-likelihood estimate under the Condorcet noise model and the unique rule satisfying neutrality, consistency, and the Condorcet criterion.
- You have a small-to-moderate number of items (up to a few hundred) where the NP-hard optimization is tractable with heuristics like insertion or differential evolution.

**When to avoid**

- You have partial, sparse, or heavily overlapping top-k lists — Kemeny is ill-posed on incomplete rankings; use MC4 or Borda instead.
- Your input consists of sparse pairwise comparisons rather than full rankings — use Bradley-Terry or Rank Centrality instead, which estimate per-entity strengths.
- You have thousands of items — the problem is NP-hard, and even the best heuristics become slow and provide no certificate of optimality.
- You need calibrated strength scores or win-probability predictions — Kemeny outputs only an order, not per-entity ratings or uncertainties.

**CLI usage (pairwise data)**

```bash
propagon tournament kemeny --algo insertion --passes 3 judges-consensus.txt
```

**CLI usage (rankings data)**

```bash
propagon rankings kemeny judge_ballots.tsv --algo insertion --passes 5 --min-obs 3
```

**Library usage (pairwise data)**

```rust
use propagon::algos::{Kemeny, KemenyPasses};
use propagon::{PairwiseDataset, Ranker};

let mut data = PairwiseDataset::new();
data.push("alice", "bob", 1.0).unwrap();
data.push("bob", "carol", 1.0).unwrap();
data.push("alice", "carol", 1.0).unwrap();

let model = Kemeny { passes: KemenyPasses::Fixed(5), ..Default::default() }
    .fit(&data).unwrap();
for name in model.order() {
    println!("{name}");
}
```

**Library usage (rankings data)**

```rust
use propagon::algos::{Kemeny, KemenyPasses};
use propagon::Ranker;

let mut dataset = RankingsDataset::new();
// load ballots...
let model = Kemeny {
    passes: KemenyPasses::Fixed(5),
    ..Default::default()
}.fit(&dataset).unwrap();
for (rank, (id, _)) in model.sorted_scores().iter().enumerate() {
    println!("{}. {}", rank + 1, id);
}
```

**What the numbers mean**

The output is a total ordering of all entities from most-preferred to least-preferred. The ranking minimizes the total Kendall tau distance (number of pairwise disagreements) across all input ballots — a lower objective value means the consensus agrees more closely with the judges. Each entity receives a score equal to its rank position: the best entity gets a score of N (the total number of entities), the next gets N-1, down to 1 for the last. Unlike Elo or Bradley-Terry scores, there are no per-entity ratings or win probabilities: the result is purely an order. If you need per-entity strength estimates you can compare across different contests, pick Elo or Bradley-Terry instead; if you just want the fairest single ranking that best represents everyone's opinions, Kemeny is the way to go.


### KL-UCB

**Real-world scenario**

You are running an e-commerce site with 8 product banner designs and want to maximize conversions over 10,000 visitors. KL-UCB automatically shows promising banners more often while still exploring less-tested ones, converging faster to the best banner than plain UCB1.

**When to use**

- You have a set of independent options (ads, headlines, treatments) producing [0,1] rewards and need to both rank them and decide what to show next.
- Your rewards are Bernoulli (binary outcomes like clicks, conversions, wins) — KL-UCB's KL divergence bound is specifically optimized for this case and exactly matches the theoretical Lai-Robbins regret lower bound.
- You want tighter exploration bounds than UCB1 without the Bayesian machinery of Thompson Sampling.

**When to avoid**

- Your rewards are unbounded or not in [0,1] — KL-UCB requires bounded rewards and will reject out-of-range values.
- You need a full leaderboard with precise rankings for rarely-tested arms — the policy intentionally starves low-performing arms of data, so their scores remain uncertain by design.

**CLI usage**

```bash
propagon bandit kl-ucb --c 0 --seed 42 ad-conversions.txt
```

**Library usage**

```rust
use propagon::algos::{Bandit, BanditPolicy};

let algo = Bandit {
    policy: BanditPolicy::KlUcb { c: 0.0 },
    seed: 42,
};
let mut model = algo.init();
model.update(&algo, &dataset)?;
let scores = model.scores();
```

**What the numbers mean**

Each arm gets a KL-UCB index score: the highest plausible mean reward consistent with the data, found by solving `max{q in [mean, 1) : n·KL(mean, q) ≤ ln(t) + c·ln(ln(t))}`. A higher index means either a high observed reward rate or high uncertainty (few observations). Unpulled arms get infinity, so they are always tried first. The score is not a probability — it is an upper confidence bound; arms with similar observed rates but fewer pulls will rank higher due to their wider uncertainty interval.

### LeaderRank

**Real-world scenario**

Imagine you have a Wikipedia link graph of 10,000 articles — some isolated clusters, a bunch of dead-end pages that link nowhere, and you need to surface the most important articles. With LeaderRank, you don't have to manually configure damping factors or decide how to handle sink nodes. It just works. You feed it the graph, and it gives you importance scores that account for the messy realities of real-world networks.

**When to use**

- You need a parameter-free importance score for nodes in an endorsement graph (links, citations, follows)
- Your graph may be disconnected or have dangling nodes (nodes with no outgoing edges) — LeaderRank handles both automatically
- You want a simpler, more robust alternative to PageRank that adapts teleportation mass to local degree rather than using a fixed damping factor

**When to avoid**

- You need a probabilistic interpretation tied to a fixed teleportation probability (e.g., modeling actual user click behavior with a known bounce rate)
- Your graph has meaningful edge weights that should influence importance — LeaderRank treats the graph as unweighted, ignoring edge weights

**CLI usage**

```bash
propagon graph leader-rank --iterations 500 --tolerance 1e-10 wiki_links.txt
```

**Library usage**

```rust
use propagon::algos::LeaderRank;
use propagon::{GraphDataset, Ranker};

let mut graph = GraphDataset::new();
graph.push("page_a", "page_b", 1.0).unwrap();
graph.push("page_b", "page_c", 1.0).unwrap();

let model = LeaderRank::default().fit(&graph).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {:.4}", id.to_string(), score);
}
```

**What the numbers mean**

Scores are normalized to sum to 1 across all nodes. A score of 0.1 means that node captures 10% of the random walk's stationary mass — roughly one-tenth of all "attention" in the graph. Higher scores indicate nodes that are both well-linked and linked from other important nodes. Unlike PageRank's damping factor, LeaderRank's ground node redistributes mass evenly after convergence, so scores directly reflect structural importance without tuning artifacts. If you're picking between LeaderRank and PageRank, LeaderRank is the choice when you want a set-and-forget algorithm that doesn't require you to reason about what damping factor makes sense for your data.

### LinUCB

**Real-world scenario**

You're choosing which of 5 news articles to show each visitor, and you know things about the visitor — their device, the hour, their topic history. A plain bandit ignores all that and learns one global "best article." LinUCB instead learns how article rewards depend on the *context*: it fits a small linear model per arm and, for each incoming visitor, picks the arm with the best optimistic estimate *for that visitor*. Sports fans on mobile at 8am and finance readers on desktop at noon can get different, individually-justified picks.

**When to use**

- Each decision comes with a feature vector (user/session/item context) and rewards depend on it — personalized recommendation, contextual ad selection, adaptive UI.
- You want the explore-exploit behavior of UCB but conditioned on context, so cold-start arms are still tried while the model learns the per-arm linear fit.
- A fast, closed-form contextual bandit is enough — LinUCB updates each arm's inverse covariance with a rank-one (Sherman-Morrison) step, no heavy inference.

**When to avoid**

- You have no per-decision context, or the reward isn't roughly linear in the features — a context-free bandit (UCB, Thompson) is simpler and won't overfit noise.
- The true reward surface is strongly nonlinear and you can't engineer features to linearize it — LinUCB's linear-per-arm assumption will mislead.

**CLI usage**

```bash
# rows are: arm reward x1 x2 ... xd   (dimensionality fixed by the first row)
propagon bandit linucb --alpha 1.0 --ridge 1.0 contextual-rewards.txt

# ask which arm to play for a specific context:
propagon bandit linucb --alpha 1.0 --select-for "0.3,0.7" contextual-rewards.txt
```

**Library usage**

```rust
use propagon::algos::LinUcb;
use propagon::{OnlineRanker, ContextualRewardsDataset, RankModel};

let algo = LinUcb { alpha: 2.0, ridge: 1.0 };
let mut model = algo.init();

let mut data = ContextualRewardsDataset::new();
data.push("sports",  1.0, &[1.0, 0.0])?;   // first push fixes the dimension d = 2
data.push("finance", 0.0, &[0.0, 1.0])?;
data.push("sports",  1.0, &[1.0, 0.0])?;
algo.update(&mut model, &data)?;

let arm = model.select_for(&[1.0, 0.0])?;  // best arm for this context
println!("play: {arm}");
for (arm, s) in model.sorted_scores() {     // θ·x̄ at the mean observed context
    println!("{arm}: {s:.3}");
}
```

**What the numbers mean**

The real output is `select_for(x)` (CLI: `--select-for`): for a given context vector it returns the arm maximizing `θ_aᵀx + alpha·sqrt(xᵀA_a⁻¹x)` — the predicted reward plus an exploration bonus that's large when this arm has seen little data near this context. `alpha` is the width of that bonus: higher explores more. The leaderboard score (`scores()`) is each arm's predicted reward `θ·x̄` at the *mean* observed context — a single summary number for ranking, since without a query context there's no one "right" score. The dimension d is set by the first row and every later row and query must match it.

### Luce Spectral Ranking

**Real-world scenario**

You have 200 pairwise A/B test results from your website — for instance, visitors chose layout A over layout B — and you want to rank all 30 tested layouts by their relative attractiveness without running an expensive iterative optimizer. Luce Spectral Ranking (LSR) gives you Plackett-Luce-style scores by computing the stationary distribution of a Markov chain built from your win/loss outcomes. It's fast, principled, and scales to massive comparison graphs where iterative maximum-likelihood methods would grind to a halt.

**When to use**

- You need a fast, one-shot ranking from pairwise comparison data and want Plackett-Luce-style scores.
- Your comparison graph is large (thousands of entities, millions of outcomes) where iterative MLE methods like MM would be slow.
- You want a principled spectral estimate that generalizes Rank Centrality and can serve as a warm-start for iterative refinement (I-LSR).

**When to avoid**

- Your data has many ties or uncertain outcomes — LSR does not natively handle ties and assumes clear winner/loser pairs.
- The comparison graph is disconnected or has isolated entities — the Markov chain needs connectivity for a meaningful stationary distribution.

**CLI usage**

```bash
propagon tournament luce-spectral-ranking --estimator power --steps 50 baseball.2018
```

**Library usage**

```rust
use propagon::algos::{Estimator, Lsr};
use propagon::PairwiseDataset;

let mut games = PairwiseDataset::new();
games.push("ARI", "COL", 1.0).unwrap();
games.push("ARI", "NYM", 1.0).unwrap();
games.push("COL", "NYM", 1.0).unwrap();

let model = Lsr { estimator: Estimator::PowerMethod, steps: 50, ..Default::default() }.fit(&games).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {:.4}", id, score);
}
```

**What the numbers mean**

Scores are log-scale and mean-centered, so only differences between scores are meaningful. A larger score means a stronger entity. The score difference between two entities approximates the log-odds that the higher-ranked entity beats the lower-ranked one under the Plackett-Luce model — for example, a difference of about 1.0 means the stronger entity wins roughly 73% of the time, while a difference of 2.0 means about 88%.

### Mallows Model

**Real-world scenario:** Imagine you have 12 food critics each ranking 8 restaurant dishes from best to worst, and you want to find the one ranking that best represents the group's collective taste — plus a measure of how much the critics agree or disagree. The Mallows Model does exactly that: it treats each critic's ballot as a noisy version of one true underlying order, and figures out what that order is.

**When to use:**
- You have a small set of complete rankings (every voter ranks every item) and want a single consensus order.
- You need a principled noise model for how individual judgments deviate from group consensus, not just a combinatorial aggregation.
- The data comes from panels, juries, or expert reviewers where a single underlying truth is plausible.

**When to avoid:**
- Your ballots are partial (some voters only rank a subset of items) — use Plackett-Luce instead, which handles partial data gracefully.
- You need per-item strength scores or predictions for unseen pairs — Mallows produces an ordering and a dispersion parameter, not item-level ratings.

**CLI usage:**

```bash
propagon rankings mallows --passes 200 --seed 42 critics-restaurants.rankings
```

**Library usage:**

```rust
use propagon::{algos::Mallows, dataset::RankingsDataset, traits::Ranker};

let data = RankingsDataset::read_path("critics-restaurants.rankings")?;
let model = Mallows::default().fit(&data)?;
eprintln!("phi (dispersion): {:.4}", model.phi());
for name in model.order() {
    println!("{}", name);
}
```

**What the numbers mean:**

The output gives two key numbers: **phi (dispersion)**, which ranges from 0 to 1 — a phi near 0 means all ballots are nearly identical (strong consensus), while a phi approaching 1 means the ballots are essentially random noise with no agreement. The **mean Kendall distance** tells you the average number of pairwise swaps needed to transform each ballot into the consensus ranking. For example, with 8 items, the maximum possible distance is 28; a mean distance of 3 means ballots are very close to consensus, while a mean of 14 would mean they're about as far as random.

### Margin-of-Victory Elo

**Real-world scenario**

Plain Elo treats a 1-run squeaker and a 15-run blowout as the same single win, which throws away a strong signal about how much better the winner really was. Margin-of-victory Elo keeps Elo's fast, live-updating ratings but scales each update by the score margin — a thrashing moves the ratings more than a nail-biter — using the log-margin form that keeps blowouts from running away. It's the practical choice when you have final scores, not just results, and want them to count.

**When to use**

- Your games carry a margin (points, runs, goals) and you want a *live* rating that reflects how decisively teams win, not just whether they won.
- You like Elo's simplicity and streaming, order-of-arrival updates but want more information per game than win/loss alone.
- You want a single knob (`--mov-exponent`) to dial how aggressively margins matter, from "ignore them" up to "weight them heavily."

**When to avoid**

- You only have win/loss outcomes — there's no margin to use, so plain `elo` is the honest choice (a margin-1 win here reduces to exactly plain Elo anyway).
- Skill is static and you want the most accurate offline fit — like all Elo variants this is an online, order-dependent estimator; a batch margin method (Massey) or Bradley-Terry fit is steadier.

**CLI usage**

```bash
# margin-of-victory is a mode of the elo command, not a separate subcommand:
propagon tournament elo --margin-of-victory --mov-exponent 1.0 --k 32 scores.tsv
```

**Library usage**

```rust
use propagon::algos::MovElo;
use propagon::{OnlineRanker, RankModel, GamesDataset, GameOutcome};

let mut d = GamesDataset::new();
d.push_game(&["a"], &["b"], GameOutcome::Side1Win(12.0), 1.0)?; // a won by 12
d.push_game(&["b"], &["c"], GameOutcome::Side2Win(1.0), 1.0)?;  // c won by 1

let algo = MovElo::default();          // k = 32, scale = 400, mov_exponent = 1.0
let mut model = algo.init();
algo.update(&mut model, &d)?;
for (name, rating) in model.sorted_scores() {
    println!("{name}: {rating:.0}");
}
```

**What the numbers mean**

Ratings are on the usual Elo scale (default start 1500), and a rating gap predicts win probability exactly as in plain Elo — a 200-point edge is about 76%. The difference is in how each game *moves* them: the K step is multiplied by `(ln(1 + margin) / ln 2)^mov_exponent`, so a margin-1 win uses a multiplier of 1 (identical to plain Elo) while larger margins push harder, with diminishing returns so a 30-point rout isn't 30× a 1-point win. `--mov-exponent 0` disables margin scaling entirely; values above 1 make margins matter more steeply.

### Markov Chain (MC4)

**Real-world scenario**

Imagine you have 10 search engines each returning a top-20 results list for the same query, and you want to merge them into one unified ranking that respects the majority opinion across engines. MC4 treats each engine's list as a "ballot" and builds a random walk over the items — essentially applying PageRank to rankings. The result is a single consensus ordering that captures what most engines agree on.

**When to use**

- You have partial, overlapping, or differently-sized rankings (e.g., top-k lists from different recommenders) — MC4 handles them natively, unlike Kemeny or Borda.
- You want a spam-resistant aggregation where outliers and inconsistent ballots have limited influence on the final ranking.
- You need a lightweight alternative to Kemeny that runs in polynomial time via power iteration, suitable for leaderboard or metasearch sizes.

**When to avoid**

- You need calibrated preference-strength semantics (e.g., predicting exact win probabilities) — MC4 stationary scores are relative ranks, not calibrated probabilities.
- Your dataset has tens of thousands of items — the majority tally is a dense n×n matrix, so MC4 becomes impractical past roughly 10,000 items.

**CLI usage**

```bash
propagon rankings markov-chain --damping 0.6 --iterations 300 engine-results.rankings
```

**Library usage**

```rust
use propagon::algos::Mc4;
use propagon::{Ranker, RankingsDataset};

let mut data = RankingsDataset::new();
data.push_ranking(&["engine-a", "engine-b", "engine-c"]).unwrap();
data.push_ranking(&["engine-b", "engine-a", "engine-d"]).unwrap();

let model = Mc4::default().fit(&data)?;
for (name, score) in model.sorted_scores() {
    println!("{}: {:.4}", name, score);
}
```

**What the numbers mean**

Each item receives a stationary score between 0 and 1 that sums to 1 across all items. A higher score means the item is ranked above more others by the majority of input ballots. The scores represent the long-run probability of landing on each item in the random walk — the top-ranked item is the one the majority most consistently places first. Unlike Elo or Bradley-Terry scores, these are relative proportions, not absolute skill levels.


### Massey

**Real-world scenario**

Imagine you're running a high school basketball league with 12 teams and you want a strength-of-schedule-adjusted ranking that accounts for how many points each team won or lost by, not just who beat whom. Massey gives you exactly that: it treats every game's point differential as a clue about the underlying strength gap between the two teams, then solves for ratings that make those clues consistent across the whole league.

**Why pick Massey over the others?** It's the simplest margin-based method available. Unlike Elo or Glicko-2, which process games one at a time and can depend on the order of results, Massey looks at the entire season at once and produces a single, deterministic answer. And unlike Keener, it doesn't require you to think about blowout-damping parameters — it just works with the raw margins.

**When to use**

- You have margin-of-victory data (point differentials) for every game, not just win/loss results.
- You need a fast, deterministic, audit-friendly ranking — the method solves a single sparse linear system and is easy to explain to non-technical stakeholders.
- You want automatic strength-of-schedule adjustment: a team's rating naturally reflects the quality of its opponents because the system is solved jointly.

**When to avoid**

- You only have win/loss data with no margins; Massey degrades to near-uselessness without point differentials (use Colley instead).
- Your data contains extreme blowouts that should not disproportionately influence ratings; the Gaussian-margin assumption rewards large margins linearly unless you cap or transform them.

**CLI usage**

```bash
propagon tournament massey --iterations 500 --tolerance 1e-8 ncaa-basketball.margins
```

**Library usage**

```rust
use propagon::algos::{Massey, MasseyModel};
use propagon::{PairwiseDataset, Ranker};

let mut games = PairwiseDataset::new();
games.push("Lakers", "Celtics", 12.0);
games.push("Warriors", "Celtics", 5.0);
games.push("Lakers", "Warriors", 8.0);

let model: MasseyModel = Massey::default().fit(&games).unwrap();
for (id, score) in model.sorted_scores() {
    println!("{}: {:.2}", id, score);
}
```

**What the numbers mean**

Ratings are on the point-margin scale and sum to zero (mean-centered). A rating of +5 means the team outperforms the league average by roughly 5 points per game; a gap of 10 between two teams means the higher-rated team won their games by about 10 more points on average. The sign of the difference predicts which team was stronger, and the magnitude approximates the expected point differential.

### mElo (Multidimensional Elo)

**Real-world scenario**

You're rating game agents where some strategies hard-counter others — agent A beats B, B beats C, C beats A — so a single Elo number can never predict those match-ups, no matter how much data you feed it. mElo keeps an ordinary Elo rating for overall strength but adds a few low-dimensional "style" vectors that capture these rock-paper-scissors cycles, updating both online as results stream in. You get a normal leaderboard *and* a matchup predictor that knows A will lose to C even though A ranks higher.

**When to use**

- Your match-ups are partly intransitive (cyclic counters) and you want online, streaming updates rather than a batch fit — agent leagues, evolving metagames, self-play evaluation.
- You want a single model that gives both a transitive ranking and calibrated per-pair predictions including the cyclic part.
- You want a smooth dial between "plain Elo" (`--k 0`) and "Elo plus k cyclic dimensions."

**When to avoid**

- Your data is essentially transitive — the cyclic vectors add noise; plain Elo or Glicko-2 is tighter and simpler.
- You need a frozen, reproducible batch estimate — mElo is online and order-dependent; for static analysis of intransitivity, Blade-Chest (batch) or Nash averaging may fit better.

**CLI usage**

```bash
propagon tournament melo --k 1 --lr-rating 16.0 --lr-vector 1.0 --seed 2018 agent-matches.tsv
```

**Library usage**

```rust
use propagon::algos::MElo;
use propagon::{OnlineRanker, RankModel, GamesDataset};

let mut data = GamesDataset::new();
data.push_pair("a", "b", 1.0)?;   // a beat b
data.push_pair("b", "c", 1.0)?;
data.push_pair("c", "a", 1.0)?;

let algo = MElo::default();        // k = 1 (a 2-D cyclic block)
let mut model = algo.init();
algo.update(&mut model, &data)?;
for (name, r) in model.sorted_scores() {
    println!("{name}: {r:.3}");           // transitive rating only
}
println!("P(a ≻ c) = {:?}", model.predict("a", "c")); // full, cyclic-aware
```

**What the numbers mean**

The printed score is the **transitive rating r only** — the part that behaves like ordinary Elo, so a cycle's members come out near-equal. The cyclic information lives in `predict(a, b)`, the full match-up probability `σ(r_a − r_b + c_aᵀΩc_b)`, which can disagree with the leaderboard: a higher-ranked agent can still be predicted to lose a specific match-up it's countered in. `--k` sets the number of 2-D cyclic blocks (model rank 2k); `k = 0` collapses to plain Elo. `--lr-rating` and `--lr-vector` are the step sizes for the transitive and cyclic parts.

### Monte Carlo Value

**Real-world scenario**

You have logs of thousands of user sessions through your website — each a sequence of pages with a reward at the end (a purchase, a signup) — and you want to know which pages are actually worth landing on. Monte Carlo value estimation answers that directly from the logs: for each state it averages the *return* (the total discounted reward that followed every time that state was visited), giving you a data-driven "what is this state worth?" with no model of the dynamics required.

**When to use**

- You have episodic trajectories with rewards (sessions, games, rollouts) and want a per-state value with no transition model — just averaged outcomes.
- You want a robust, tunable estimate: first- or every-visit counting, mean or median aggregation, and winsorization to tame heavy-tailed returns.
- You need the simplest unbiased value estimate as a baseline before reaching for bootstrapping (TD) methods.

**When to avoid**

- Your data is a single long non-episodic stream, or rewards are extremely sparse — returns are high-variance and Monte Carlo will be noisy; TD(0) propagates sparse signal more smoothly.
- You need uncertainty intervals or to compare two states rigorously — use Value Comparison (the `compare` command), which bootstraps over episodes.

**CLI usage**

```bash
# input rows are: state reward   (a blank line ends each episode)
propagon trajectories monte-carlo --gamma 0.9 --visit first --aggregate mean sessions.txt
```

**Library usage**

```rust
use propagon::algos::{McValue, Visit};
use propagon::{Ranker, RankModel, TrajectoriesDataset};

let mut d = TrajectoriesDataset::new();
d.push_step("landing", 0.0)?;
d.push_step("product", 0.0)?;
d.push_step("checkout", 1.0)?;
d.end_episode();

let mc = McValue { gamma: 0.9, visit: Visit::First, ..Default::default() };
let model = mc.fit(&d)?;
for (state, v) in model.sorted_scores() {
    println!("{state}: {v:.3}");
}
```

**What the numbers mean**

Each state's score V(s) is the average discounted return observed after visiting it — concretely, the mean (or median) of `r_t + γ·r_{t+1} + γ²·r_{t+2} + …` over the visits counted. Higher means "reaching this state tends to be followed by more reward." `--gamma` sets how much future reward counts (1.0 = full episode, smaller = myopic); `--visit first` counts one return per episode per state while `every` counts each occurrence; `--winsorize q` clamps each state's returns into its `[q, 1−q]` quantiles before averaging, and `--min-observations` drops states seen too few times to trust (they're omitted entirely, not scored zero).

### Nash Averaging

**Real-world scenario**

You're comparing AI agents on a suite of tasks, but the result depends heavily on *which* agents and tasks you happened to include — pack the field with easy opponents and a mediocre agent looks great. Nash averaging removes that bias: instead of a flat win-rate average, it finds the equilibrium weighting of opponents (the maximum-entropy Nash distribution of the comparison game) under which no agent can pad its score by cherry-picking. The result is an evaluation that's invariant to redundant or stacked opponents.

**When to use**

- You're worried your leaderboard is gameable by the *choice* of opponents/tasks (agent evaluation, benchmark suites) and want a result robust to that selection bias.
- Your data may be intransitive and you want an evaluation that doesn't pretend a total order exists — Nash averaging reports a meaningful "everyone is even" when the game is cyclic.
- You want a self-certifying answer: the method reports a duality gap that bounds how close to the true equilibrium it got.

**When to avoid**

- You specifically *want* a simple, transitive leaderboard with separation between entities — on cyclic data Nash averaging can flatten scores toward zero, which is correct but not what a ranking-hungry stakeholder expects.
- Your comparison data is dense, clean, and transitive — a plain Bradley-Terry or win-rate ranking is simpler and just as defensible.

**CLI usage**

```bash
propagon tournament nash-averaging --iterations 200000 --tolerance 1e-6 agent-vs-agent.tsv
```

**Library usage**

```rust
use propagon::algos::NashAveraging;
use propagon::{PairwiseDataset, Ranker, RankModel};

let mut data = PairwiseDataset::new();
data.push("a", "b", 1.0);   // a fully cyclic game (rock-paper-scissors)
data.push("b", "c", 1.0);
data.push("c", "a", 1.0);

let model = NashAveraging::default().fit(&data)?;
for (name, skill) in model.sorted_scores() {
    println!("{name}: {skill:.4}");                 // Nash-averaged skill
}
let support: Vec<_> = model.distribution().collect(); // the maxent Nash weighting
println!("duality gap = {:.2e}", model.gap());        // closeness-to-equilibrium certificate
```

**What the numbers mean**

Each entity's score is its **Nash-averaged skill** — its expected advantage when opponents are drawn from the equilibrium distribution rather than uniformly. On a fully cyclic game like rock-paper-scissors every skill collapses to 0 and the distribution is uniform: the honest verdict that no agent dominates. When one entity truly dominates, the distribution concentrates on it. The CLI prints the **duality gap** to stderr (`model.gap()` in the library): it bounds each score's distance from its exact-equilibrium value, so a gap of 1e-6 means the approximation has effectively converged.

### Offense-Defense

**Real-world scenario**

You are running a basketball league with 12 teams and want to know whether a dominant team like the Lakers is great because of their offense (LeBron scoring 30) or their defense (holding opponents to 90). Feeding both directions of every game's point totals gives you two ratings per team instead of one — so you can tell if a team wins by outscoring opponents or by shutting them down.

**When to use**

- You have score data (points, goals, runs) from both sides of each game and want to separate offensive strength from defensive strength.
- You need interpretable, decomposed ratings — for example, to identify a strong-offense/weak-defense team that single-scalar methods would mischaracterize.
- Your data forms a connected graph where every team both scores and concedes (the algorithm requires full support).

**When to avoid**

- Teams can have shutout records (never score or never concede) — this breaks the Sinkhorn balancing and produces a numeric error.
- You only have win/loss outcomes without score margins — the algorithm needs actual point/amount data, not binary results.

**CLI usage**

```bash
propagon tournament offense-defense --iterations 500 --tolerance 1e-9 nba-scores.txt
```

**Library usage**

```rust
use propagon::algos::OffenseDefense;
use propagon::{PairwiseDataset, Ranker};

let mut data = PairwiseDataset::new();
data.push("LAL", "GSW", 112.0); // LAL scored 112 on GSW
data.push("GSW", "LAL", 108.0); // GSW scored 108 on LAL

let model = OffenseDefense::default().fit(&data)?;
for (name, score) in model.sorted_scores() {
    println!("{}: {:.4}", name, score);
}
```

**What the numbers mean**

Each team gets three numbers: offensive rating (o), defensive rating (d), and aggregate score (s = o/d). A higher offensive rating means the team scores more points relative to the league average; a lower defensive rating means the team allows fewer points (it scales what opponents manage to score against you). The aggregate score is the ratio — a team with o=1.2 and d=0.8 has a score of 1.5, meaning their offense outperforms their defense by 50%. Only relative differences matter; the ratings are identified up to a multiplicative constant.

### PageRank

**Real-world scenario**

Imagine you're analyzing a Wikipedia link graph of 10,000 articles and want to identify the 50 most important pages — not just by raw link count, but by who links to them. Maybe you're curating a featured collection or seeding a recommendation engine. PageRank does exactly that: it treats each link as an endorsement and figures out which pages are endorsed by other important pages, giving you a quality-aware importance ranking.

**When to use**

- You have a directed endorsement graph (A → B means A endorses B) and need a global importance ranking that accounts for link quality, not just link quantity.
- You want to find nodes that are important from the perspective of specific seeds (personalized PageRank / random walk with restart), such as most relevant pages to this user or most trustworthy items reachable from vetted sources.
- Your graph has cycles, dangling nodes, or is otherwise messy — PageRank's teleportation guarantees convergence where raw eigenvector centrality would fail.

**When to avoid**

- Your data is pairwise comparison data (A beat B) — use Rank Centrality or Bradley-Terry instead, which are statistically grounded for that setting.
- You need uncertainty estimates or confidence intervals — PageRank is a point estimate with no error bars.

**CLI usage**

```bash
propagon graph page-rank --damping-factor 0.85 --sink-dispersion uniform --iterations 100 wikipedia-links.txt
```

**Library usage**

```rust
use propagon::algos::{PageRank, Sink};
use propagon::{GraphDataset, Ranker};

let mut graph = GraphDataset::new();
graph.push("A", "B", 1.0);
graph.push("B", "C", 1.0);
graph.push("A", "C", 1.0);

let pr = PageRank { damping: 0.85, sink: Sink::Uniform, ..Default::default() };
let model = pr.fit(&graph)?;
for (id, score) in model.sorted_scores() {
    println!("{}: {:.4}", id.to_string(), score);
}
```

**What the numbers mean**

Each node receives a score between 0 and 1 that sums to 1 across all nodes — think of it as the long-run fraction of time a random surfer lands on that page. A score of 0.10 means the surfer visits that node 10 percent of the time, indicating high importance (many endorsements from other important nodes). Scores are relative: a node's score shifts if the graph changes, so you compare nodes within the same graph, not across different graphs.

### Plackett-Luce

**Real-world scenario**

Imagine you have 20 judges each ranking 5 AI-generated responses from best to worst, and you want a single strength score per model that accounts for all the ranking positions, not just pairwise wins. That's exactly what Plackett-Luce does — it takes full or partial rankings (ballots) and estimates a strength score for every item that reflects how consistently it appears near the top.

**When to use**

- You have full or partial rankings (e.g., race finishing orders, top-k results) rather than just win/loss pairs — a single 10-item ranking encodes 45 pairwise constraints coherently.
- You need a probabilistic model of choice from a slate (e.g., search result clicks, A/B/n test selections) where probability is proportional to strength.
- Your data comes as ballots or ordered lists from multiple judges or users and you want a maximum-likelihood estimate of item strengths.

**When to avoid**

- Your data violates the Independence of Irrelevant Alternatives (IIA) assumption — e.g., choices exhibit strong substitution effects like the red-bus/blue-bus problem, where adding a similar option disproportionately shifts probability between close competitors.
- You only need a consensus ranking without strength scores — simpler methods like Borda count or Kemeny may suffice.

**CLI usage**

```bash
propagon rankings plackett-luce --iterations 100 --tolerance 1e-8 f1-2024.rankings
```

**Library usage**

```rust
use propagon::algos::PlackettLuce;
use propagon::{Ranker, RankingsDataset};

let mut data = RankingsDataset::new();
data.push_ranking(&["A", "B", "C"])?;
data.push_ranking(&["B", "C", "A"])?;

let mut pl = PlackettLuce::default();
let model = pl.fit_rankings(&data)?;
for (id, score) in model.sorted_scores() {
    println!("{} {:.4}", id, score);
}
```

**What the numbers mean**

Each item receives a positive strength score (π). The probability that item i is chosen over item j is π_i / (π_i + π_j), so a score of 2.0 vs 1.0 means the higher item wins roughly 67% of head-to-head matchups. Scores are relative — only ratios matter, not absolute magnitudes. Higher scores indicate items consistently ranked near the top across ballots.

You'd pick Plackett-Luce over a simple Borda count when you care about *why* an item wins, not just that it does. Borda gives you a ranking; Plackett-Luce gives you a calibrated strength model you can use to predict outcomes on new slates or compare items that never appeared on the same ballot.

### Random Utility Model

**Real-world scenario:** Imagine you're running a chess tournament with 20 players and want to identify not just who the strongest players are, but also who plays inconsistently. A grandmaster who occasionally loses to novices should get a high μ but also a high σ, separating them from a reliably strong player. The Random Utility Model captures exactly this nuance — it assigns each entity a Gaussian utility distribution so that an entity can be both strong and inconsistent.

**When to use:**
- You need per-entity uncertainty estimates (σ) alongside strength scores (μ), not just a single rating.
- Some entities are genuinely erratic or volatile — the model captures 'good but inconsistent' behavior that Bradley-Terry or Elo cannot.
- You want a Thurstone-style probit model (Gaussian noise) rather than the logistic assumption of Bradley-Terry, which matters when pairwise win probabilities are extreme.

**When to avoid:**
- You need absolute, comparable scores across different datasets — the model is only identified up to location and scale, so raw μ values are relative within each fit.
- Your dataset is very small (fewer than ~5 comparisons per entity) — the per-entity σ parameter is hard to identify with sparse data; use Bradley-Terry instead.

**CLI usage:**
```bash
propagon tournament random-utility-model --passes 200 --gamma 0.001 --seed 42 chess-tournament.txt
```

**Library usage:**
```rust
use propagon::algos::{EsRum, RumDistribution};
use propagon::Ranker;

let mut data = propagon::PairwiseDataset::new();
data.push_pair("Alice", "Bob", 1.0).unwrap();
data.push_pair("Alice", "Charlie", 1.0).unwrap();
data.push_pair("Bob", "Charlie", 1.0).unwrap();

let rum = EsRum { passes: 200, ..Default::default() };
let model = rum.fit(&data)?;
for (name, mu, sigma) in model.distributions() {
    println!("{name}: μ={mu:.3} σ={sigma:.3}");
}
```

**What the numbers mean:**
Each entity gets two numbers: μ (mean utility, higher = stronger) and σ (standard deviation of utility, higher = more volatile). A high μ with low σ means consistently strong; high μ with high σ means strong on average but unpredictable — the 'good but erratic' profile. Because the model is identified only up to location and scale, only differences between entities matter: if Alice has μ=1.5 and Bob μ=0.8, Alice is stronger, but the absolute values have no meaning outside this dataset. The win probability between any two entities is computed from the Gaussian CDF of (μ_i − μ_j) / sqrt(σ_i² + σ_j²).

### Random Walker

**Real-world scenario**

Imagine you're ranking 120 college football teams at the end of a season. Some teams only play within their conference, the schedule is sparse, and a simple win-percentage ranking feels too blunt. Random Walker simulates a population of fans who gradually switch allegiance to the winner of any matchup they watch, with a tunable bias parameter controlling how aggressively they chase winning teams. The final ranking emerges from where those fans settle — balancing raw win rates against the strength of opponents each team faced.

**When to use**

- You have pairwise match results (wins/losses) and want a ranking that interpolates between pure win-percentage and schedule-strength smoothing.
- You want a single tunable knob (the bias `p`) to control how aggressively the ranking rewards winning versus how much it values facing strong opponents.
- You want to sweep `p` across values (e.g., 0.55 to 0.95) to check ranking stability — rankings that flip dramatically as `p` changes signal that the data doesn't support a clear ordering.

**When to avoid**

- Your comparison graph is disconnected (e.g., teams only play within isolated groups): the stationary mass splits across components by internal structure, making cross-component score ratios meaningless.
- You need calibrated win-probability predictions between specific pairs: this method produces importance scores, not matchup probabilities.

**CLI usage**

```bash
propagon tournament random-walker --bias 0.75 ncaa-football.2024.txt
```

**Library usage**

```rust
use propagon::algos::RandomWalker;
use propagon::traits::Ranker;

let algo = RandomWalker { p: 0.75, ..Default::default() };
let model = algo.fit(&pairwise_data)?;
for (name, score) in model.scores() {
    println!("{name}: {score:.4f}");
}
```

**What the numbers mean**

Each entity receives a score between 0 and 1 representing its share of the total fan population in the stationary distribution; scores sum to 1 across all entities. A score of 0.15 means 15% of fans ultimately settled on that entity. Higher scores indicate stronger entities. The absolute values depend on the number of entities and the chosen bias: with `p` near 0.5 scores are more evenly distributed (reflecting schedule structure), while `p` near 1 concentrates mass on entities with high win-percentage.

### Rank Centrality

**Real-world scenario**

You have 200 chess players who've played a sparse round-robin and want a reliable leaderboard without tuning hyperparameters. Rank Centrality builds a win-fraction matrix from your match data and converges in just a handful of power-iteration sweeps — no iterative likelihood optimization needed. It's the "just give me a ranking" choice when you want something fast with solid statistical backing.

**When to use**

- You need a fast, parameter-free ranking from pairwise comparisons and want statistical guarantees close to maximum-likelihood estimation.
- Your dataset is large (millions of items or comparisons) where iterative Newton/MM optimization is impractical.
- You want to bootstrap a leaderboard before investing in a full Bradley-Terry fit, or you only have aggregated win fractions rather than raw match streams.

**When to avoid**

- Your comparison graph is disconnected or has a poor spectral gap (e.g., a barbell-shaped graph with two dense clusters and few bridges) — the stationary distribution becomes unreliable.
- You need calibrated win-probability predictions or standard errors for each rating — Rank Centrality outputs relative scores, not probabilistic estimates with uncertainty intervals.

**CLI usage**

```bash
propagon tournament rank-centrality chess-tournament.txt --iterations 200 --tolerance 1e-10
```

**Library usage**

```rust
use propagon::algos::RankCentrality;
use propagon::{PairwiseDataset, RankModel, Ranker};

let mut data = PairwiseDataset::new();
data.push("Alice", "Bob", 1.0);
data.push("Alice", "Charlie", 1.0);
data.push("Bob", "Charlie", 1.0);

let rc = RankCentrality::default();
let model = rc.fit(&data).unwrap();
for (name, score) in model.scores() {
    println!("{}: {:.4}", name, score);
}
```

**What the numbers mean**

Each entity receives a stationary-distribution probability between 0 and 1, and all scores sum to exactly 1. Higher scores mean stronger entities. The ratio of two entities' scores approximates their relative Bradley-Terry strength: if Alice scores 0.5 and Bob scores 0.25, Alice is roughly twice as strong as Bob (and the model predicts Alice winning about 67% of head-to-head games against Bob).

### SerialRank

**Real-world scenario**: You're running a round-robin chess tournament with 16 players and need a final leaderboard from the match results. SerialRank orders the players by strength without assuming any probabilistic model for win probabilities — no Elo curves, no Bradley-Terry likelihoods, just a mathematically grounded spectral ordering derived directly from the comparison structure.

**When to use**:

- You have a dense or near-complete set of pairwise comparisons (e.g., round-robin tournaments, all-vs-all A/B tests) and want a model-free ranking.
- You distrust parametric assumptions (Bradley-Terry, Elo) but still want a mathematically grounded spectral ordering.
- The data may contain a bounded fraction of corrupted or adversarial comparisons — SerialRank is provably robust to such noise.

**When to avoid**:

- You need interpretable strength scores or win-probability predictions — SerialRank outputs an ordering, not calibrated ratings.
- Your comparison graph is very sparse or has disconnected components — the similarity construction needs reasonably dense comparisons and relative placement between disconnected blocks is meaningless.

**CLI usage**:

```bash
propagon tournament serial-rank --iterations 500 --tolerance 1e-10 --seed 42 matches.txt
```

**Library usage**:

```rust
use propagon::algos::{SerialRank, SerialRankModel};
use propagon::{PairwiseDataset, Ranker};

let mut games = PairwiseDataset::new();
games.push("alice", "bob", 1.0);
games.push("bob", "carol", 1.0);
games.push("alice", "carol", 1.0);

let model: SerialRankModel = SerialRank::default().fit(&games)?;
println!("{:?}", model.sorted_scores());
```

**What the numbers mean**: Scores are Fiedler-vector coordinates — their ordering gives the ranking (higher score = stronger entity), but the magnitudes represent spacing along the seriation axis, not win probabilities or calibrated strengths. A large gap between consecutive scores suggests a clear separation in strength; near-equal scores indicate entities with similar sets of wins and losses. The sign is canonicalized so that more wins produce higher scores.

### Sliding-Window UCB

**Real-world scenario**

You're rotating banner ads, but what works changes with the season — last month's winner is this month's dud. A normal bandit that averages over *all* history is slow to notice the shift, because years of stale clicks drown out today's. Sliding-window UCB only trusts the most recent N observations: it computes the usual optimistic UCB index but over a moving window, so it forgets old data and adapts when the world moves under it.

**When to use**

- Reward rates drift over time (seasonality, trends, concept drift) and you want a bandit that tracks the *current* best arm, not the all-time best.
- You can pick a window length that matches how fast things change — short to react quickly, long for stability.
- You want UCB's no-Bayesian-machinery simplicity with built-in forgetting.

**When to avoid**

- Rewards are stationary — a standard UCB or Thompson over all history is more data-efficient; the window just throws away useful samples.
- You need exactly reproducible merges across shards — the window is order-dependent state and deliberately doesn't merge (unlike the stationary bandit policies).

**CLI usage**

```bash
propagon bandit sliding-window-ucb --window 1000 --exploration 2.0 ad-clicks.txt
propagon bandit sw-ucb --window 200 --select 1 ad-clicks.txt   # next arm to play
```

**Library usage**

```rust
use propagon::algos::SlidingWindowUcb;
use propagon::{OnlineRanker, RewardsDataset, RankModel};

let algo = SlidingWindowUcb { window: 40, exploration: 2.0 };
let mut model = algo.init();

let mut data = RewardsDataset::new();
data.push("a", 1.0);
data.push("b", 0.0);
data.push("a", 0.5);
algo.update(&mut model, &data)?;

let next = model.select()?;          // best arm to play now
for (arm, index) in model.sorted_scores() {
    println!("{arm}: {index:.3}");   // windowed UCB index
}
```

**What the numbers mean**

Each arm's score is its **windowed UCB index**: the arm's mean reward *over the last N events* plus an exploration bonus `sqrt(exploration · ln t / pulls)` computed within the window. An arm with no events left in the window reads as +∞ and is tried first. Because the statistics only see recent data, an arm that was great long ago but has gone cold falls in the ranking as its good results scroll out of the window. `--window` is the memory length (smaller forgets faster); `--exploration` is the usual UCB constant (2.0 is classic UCB1). Note there is no `--seed` — selection is a deterministic argmax.

### TD(0) Value Estimation

**Real-world scenario**

You want per-state values from session logs, like Monte Carlo gives you, but your rewards are sparse and arrive only at the very end of long episodes — so Monte Carlo's full-return averages are wildly noisy. TD(0) instead learns by bootstrapping: each step nudges a state's value toward the immediate reward plus the (discounted) value of the next state, letting reward signal seep backward through the chain a little at a time. It's the streaming, lower-variance cousin of Monte Carlo value estimation.

**When to use**

- You have episodic trajectories with sparse or delayed rewards where Monte Carlo returns are too noisy — TD's bootstrapping propagates signal with far less variance.
- You want online, incremental updates that fold new episodes into existing values without reprocessing history (it supports `--load-state`).
- You're fine trading a little bias (from bootstrapping off current estimates) for much lower variance.

**When to avoid**

- You want an unbiased estimate and have plenty of episodes — Monte Carlo is unbiased; TD's bootstrap introduces bias that only vanishes as values converge.
- You need uncertainty intervals or rigorous state-vs-state comparison — use Value Comparison (`compare`) instead.

**CLI usage**

```bash
# input rows are: state reward   (a blank line ends each episode)
propagon trajectories td --alpha 0.1 --gamma 0.95 --passes 5 sessions.txt
```

**Library usage**

```rust
use propagon::algos::TdValue;
use propagon::{OnlineRanker, RankModel, TrajectoriesDataset};

let mut d = TrajectoriesDataset::new();
d.push_step("landing", 0.0)?;
d.push_step("product", 0.0)?;
d.push_step("checkout", 1.0)?;
d.end_episode();

let td = TdValue { alpha: 0.5, gamma: 0.95, ..Default::default() };
let mut model = td.init();
td.update(&mut model, &d)?;
for (state, v) in model.sorted_scores() {
    println!("{state}: {v:.3}");
}
```

**What the numbers mean**

Each state's score is its current estimated value V(s), refined by the TD update `V(s) ← V(s) + α·(r + γ·V(s′) − V(s))` — the bracket is the *temporal-difference error*, the gap between the bootstrapped target and the current guess. Higher V(s) means the state tends to lead to more reward. `--alpha` is the learning rate (how far each step moves the estimate; too high is noisy, too low is slow); `--gamma` discounts future reward; `--passes` sweeps the batch multiple times per update for faster convergence on a fixed log. Because it's online, values depend on the order episodes arrive and keep adapting as you feed more.

### Team Bradley-Terry

**Real-world scenario**

In a 2v2 or 5v5 game you only observe which *team* won, but you want to rate the individual *players* — including new line-ups of players who've never been teamed before. Team Bradley-Terry models each team's strength as a combination of its members' individual strengths and fits those player strengths so the observed team results are most likely. Rate the players once and you can predict any roster, even one that never actually played together.

**When to use**

- Your results are team-vs-team but you want per-player ratings (esports, doubles, multiplayer matches with shifting rosters).
- Rosters vary, so you need a model that composes individual strengths rather than rating each fixed team as an opaque unit.
- You want the Bradley-Terry guarantees (maximum-likelihood, calibrated) lifted to the team setting, with a choice of additive or multiplicative team aggregation.

**When to avoid**

- Your teams are fixed and you don't care about individuals — just rate each team as an entity with plain Bradley-Terry.
- Player contributions are strongly synergistic (the whole differs from the sum/product of parts) — the additive/product aggregation can't capture chemistry effects; a per-roster model would.

**CLI usage**

```bash
# tie handling comes from the group-global --ties flag:
propagon tournament team-bradley-terry --aggregate additive --ties half-win games.tsv
```

**Library usage**

```rust
use propagon::algos::{TeamBradleyTerry, TeamAggregate};
use propagon::{Ranker, RankModel, GamesDataset, GameOutcome, TiePolicy};

let mut d = GamesDataset::new();
d.push_game(&["alice", "bob"], &["carol", "dave"], GameOutcome::Side1Win(1.0), 2.0)?;
d.push_game(&["alice", "carol"], &["bob", "dave"], GameOutcome::Tie, 1.0)?;

let algo = TeamBradleyTerry {
    aggregate: TeamAggregate::Additive,
    ties: TiePolicy::HalfWin,
    ..TeamBradleyTerry::default()
};
let model = algo.fit(&d)?;
for (player, strength) in model.sorted_scores() {
    println!("{player}: {strength:.4}");
}
```

**What the numbers mean**

Each score is an individual **player** strength (normalized to sum to 1, so larger is stronger). A team's predicted strength is built from its members' strengths — summed under `--aggregate additive` or multiplied under `product` — and two teams' strengths combine through the usual Bradley-Terry logistic to give a win probability. Because strengths are per-player, you can predict a brand-new line-up by composing the relevant members. Ties are handled by the group-global `--ties` flag (`half-win` splits a draw into two half-weight wins, `discard` drops it, `error` refuses tie rows).

### Thompson Sampling (Beta)

**Real-world scenario**
You're running an e-commerce A/B/n test with five ad creatives and want to maximize clicks over the next week. Instead of splitting traffic evenly across all variants, you show each visitor the creative most likely to win based on what you've learned so far — while still occasionally testing underperformers in case they improve with more data.

**When to use**
- You have sequential decisions with scalar rewards (clicks, conversions, ratings in [0,1]) and want to optimize cumulative payoff, not just produce a final ranking.
- You need a parameter-free exploration strategy with strong theoretical regret guarantees — Thompson Sampling has near-optimal O(log T) regret with no tuning knobs.
- You want to batch or merge results across servers: the Beta posterior parameters (α, β) are sufficient statistics that merge exactly, unlike many other methods.

**When to avoid**
- Your feedback is pairwise comparisons ("A beat B") rather than scalar rewards — use dueling bandits or a rating system instead.
- Arms share a structural relationship (e.g., beating a strong opponent should inform your estimate of a weaker one) — bandits treat arms independently, unlike Bradley-Terry or Elo which propagate strength through a comparison graph.

**CLI usage**
```bash
propagon bandit thompson-beta --seed 42 --prior-alpha 1.0 --prior-beta 1.0 --select 3 ad_rewards.tsv
```

**Library usage**
```rust
use propagon::algos::{Bandit, BanditModel, BanditPolicy};
use propagon::{OnlineRanker, RankModel, RewardsDataset};

let algo = Bandit {
    policy: BanditPolicy::ThompsonBeta { prior_alpha: 1.0, prior_beta: 1.0 },
    seed: 42,
};
let mut model: BanditModel = algo.init();
let mut data = RewardsDataset::new();
data.push_reward("ad_A", 1.0).unwrap();
data.push_reward("ad_B", 0.0).unwrap();
algo.update(&mut model, &data).unwrap();
let next_arm = model.select().unwrap();
```

**What the numbers mean**
Each arm gets a Beta posterior with parameters α = prior_alpha + total_rewards and β = prior_beta + total_trials − total_rewards. The score shown is the posterior mean (α / (α + β)), which is the estimated probability that the arm produces a reward of 1 on the next trial. An arm with score 0.75 means you expect it to succeed 75% of the time; arms with few trials have wide uncertainty and may be selected for exploration even if their current mean is lower.

### Thompson Sampling (Gaussian)

**Real-world scenario:** You are running an e-commerce site with 8 product-page layouts and want to maximize conversions over 30 days — instead of splitting traffic evenly, Thompson Sampling automatically shifts more visitors to the best-performing layouts while still occasionally testing the under-explored ones to make sure you haven't missed a winner.

**When to use:**
- You need to both rank arms by performance AND decide which arm to show next (online allocation).
- Rewards are continuous or approximately Gaussian (click-through rates, revenue per impression, latency, engagement time).
- You want minimal tuning — the Gaussian prior mean and weight are the only knobs, and the algorithm self-adjusts exploration as data accumulates.

**When to avoid:**
- Rewards are strictly binary (0/1) — use Thompson Sampling with a Beta posterior (thompson-beta) instead, which is the conjugate prior for Bernoulli data.
- Arms are related through a comparison graph (e.g., beating a strong opponent reveals more than beating a weak one) — bandit arms are independent, so a Bradley-Terry or Elo model would share strength across the graph.

**CLI usage:**
```bash
propagon bandit thompson-gaussian --prior-mean 0.0 --prior-weight 1.0 --seed 42 --select 3 ad-variants.rewards
```

**Library usage:**
```rust
use propagon::algos::{Bandit, BanditPolicy};
use propagon::{OnlineRanker, RewardsDataset};

let mut data = RewardsDataset::new();
data.push_reward("layout_A", 0.12).unwrap();
data.push_reward("layout_B", 0.08).unwrap();

let algo = Bandit {
    policy: BanditPolicy::ThompsonGaussian {
        prior_mean: 0.0,
        prior_weight: 1.0,
    },
    seed: 42,
};
let mut model = algo.init();
algo.update(&mut model, &data).unwrap();
let best = model.select_k(1).unwrap();
```

**What the numbers mean:** Each arm gets a posterior mean (the estimated expected reward) and a posterior variance (uncertainty). Arms with higher means are better on average; arms with high variance are uncertain and deserve more trials. When using --select N, the algorithm samples from each arm's posterior and returns the top N sampled arms — this naturally favors well-performing arms but occasionally picks uncertain ones, embodying the explore-exploit tradeoff. A score of 0.15 for a conversion-rate arm means roughly 15% average conversion; the gap between arms reflects how much better one is expected to perform.


### Thurstone-Mosteller

**Real-world scenario:** Imagine you have 20 image compression codecs and 100 testers each comparing random pairs side-by-side for visual quality. You want a single ranked list of codecs from best to worst based on those pairwise preferences. That's exactly what Thurstone-Mosteller gives you — it turns "A beat B" comparisons into a clean, probabilistic ranking.

**When to use:**
- You care about perceptual or psychological quality where the Gaussian latent assumption is natural (psychophysics, image/audio codec quality, user preference studies).
- You need a model that integrates cleanly with Gaussian message-passing systems (e.g., TrueSkill-style Bayesian online updates).
- You want a probit link instead of a logistic link for theoretical or compatibility reasons.

**When to avoid:**
- You only need a simple ranking and don't care about the link function: Bradley-Terry gives nearly identical results with simpler math.
- Your data has undefeated or winless entities (separation) and you don't want to manually tune pseudo-counts to avoid divergence.

**CLI usage:**
```bash
propagon tournament thurstone-mosteller codec-pairs.txt --pseudo-count 1.0 --iterations 500
```

**Library usage:**
```rust
use propagon::algos::ThurstoneMosteller;
use propagon::{PairwiseDataset, Ranker};

let mut data = PairwiseDataset::new();
data.push("codec_a", "codec_b", 1.0);
data.push("codec_a", "codec_c", 1.0);
data.push("codec_b", "codec_c", 1.0);

let algo = ThurstoneMosteller::default();
let model = algo.fit(&data)?;
for (id, score) in model.sorted_scores() {
    println!("{id}: {score:.4}");
}
```

**What the numbers mean:**

Each entity gets a score on a latent Gaussian scale, mean-centered to zero. Only score differences are meaningful: if codec A scores 0.50 and codec B scores -0.30, the difference of 0.80 means P(A ≻ B) = Φ(0.80) ≈ 79%. A difference of 0 means a coin flip; ±1 corresponds to roughly 84%/16% win probability; ±2 to roughly 98%/2%.


### Upper Confidence Bound

**Real-world scenario**

You are running an e-commerce site with 10 product banner designs and want to automatically show the highest-converting one to visitors while still gathering data on less-tested designs — each visitor click is a reward observation that refines your ranking in real time.

**When to use**

- You need to both rank alternatives and decide which to test next (active allocation), not just analyze historical data.
- Rewards are scalar and bounded (e.g., click/no-click, conversion rate, revenue per impression).
- You want a no-tuning-parameter baseline with provable sublinear regret — the classic UCB1 formula requires no hyperparameter search.

**When to avoid**

- Your feedback is only relative (e.g., "A preferred over B") rather than absolute rewards — use dueling bandits or Bradley-Terry instead.
- Arms are related and you want strength-sharing through a comparison graph — UCB treats each arm independently unless you upgrade to contextual variants like LinUCB.

**CLI usage**

```bash
propagon bandit upper-confidence-bound --exploration 2.0 --select 1 --seed 42 banner_rewards.txt
```

**Library usage**

```rust
use propagon::algos::{Bandit, BanditPolicy};
use propagon::{OnlineRanker, RankModel, RewardsDataset};

let policy = BanditPolicy::Ucb1 { exploration: 2.0 };
let mut model = Bandit { policy, seed: 42 }.init();
let mut rewards = RewardsDataset::new();
rewards.push("banner_a", 1.0);
rewards.push("banner_b", 0.0);
model.update(&mut rewards);
println!("{:?}", model.sorted_scores());
```

**What the numbers mean**

Each arm gets an index score equal to its average reward plus an exploration bonus (sqrt(exploration * ln(total_rounds) / arm_pulls)). A higher index means the arm is either performing well or hasn't been tested enough — the algorithm is optimistically assuming it might be good. Unpulled arms rank first (infinite index) to ensure every option gets tried. Arms sorted by descending index form the current ranking; the top arm is what the policy recommends playing next.

### Value Comparison

**Real-world scenario**

You ran two onboarding flows and logged thousands of user sessions for each. The average reward for flow B is a little higher — but is that a real improvement or just noise? Value Comparison answers it properly: it bootstraps over whole *episodes* (preserving the correlations inside each session) to put confidence intervals on every state's value, and can run pairwise tests reporting the probability that one state truly outperforms another, with a permutation p-value to back it.

**When to use**

- You need uncertainty, not just point estimates, on state values from trajectory logs — error bars for a report or a go/no-go decision.
- You want to compare states or variants rigorously: "P(B beats A)" plus a permutation p-value, not an eyeballed difference of means.
- Your episodes have internal correlation (a session's steps aren't independent) and you want a resampling scheme that respects that by resampling episodes, not steps.

**When to avoid**

- You only need a quick point value per state — plain Monte Carlo (`monte-carlo`) is cheaper; this runs thousands of refits.
- You have very few episodes — the bootstrap can't manufacture certainty that isn't in the data; intervals will be honestly wide.

**CLI usage**

```bash
# bootstrap CIs, plus pairwise exceedance tests with 999 permutations:
propagon trajectories compare --replicates 2000 --pairwise 999 --credible 0.95 sessions.txt
```

**Library usage**

```rust
use propagon::algos::{ValueCompare, PairwiseTests};
use propagon::{Ranker, TrajectoriesDataset};

let mut d = TrajectoriesDataset::new();
d.push_step("flow_a", 1.0)?; d.end_episode();
d.push_step("flow_b", 0.0)?; d.end_episode();
d.push_step("flow_b", 1.0)?; d.end_episode();

let vc = ValueCompare {
    replicates: 2000,
    pairwise: PairwiseTests::On { permutations: 999 },
    ..Default::default()
};
let model = vc.fit(&d)?;
for (name, point, lo, hi) in model.intervals() {
    println!("{name}: {point:.3} [{lo:.3}, {hi:.3}]");
}
for (a, b, exceed, p) in model.pairs() {
    println!("P({b} > {a}) = {exceed:.3}  perm-p = {p:.3}");
}
```

**What the numbers mean**

Each state gets a point value plus a credible interval (the central `--credible` mass of the bootstrap distribution, default 95%): a narrow interval means the value is well-pinned, a wide one means you don't have enough episodes to be sure. When `--pairwise` is on, each pair also gets an **exceedance** `P(V_b > V_a)` — the fraction of bootstrap replicates in which b's value beat a's, so 0.97 means b is very likely genuinely better — and a two-sided **permutation p-value** testing whether the two could have come from the same distribution. `--method bayesian` swaps Efron's bootstrap for the Bayesian (Gamma-weighted) one, which avoids empty resample cells on small data.

### Weng-Lin

**Real-world scenario**

Imagine you're running an online 5v5 team-based game with 2,000 players. After every match, each participant should receive an individual skill rating — a mu (skill estimate) and sigma (uncertainty) — without bogging down your server with a heavy Bayesian inference engine. Weng-Lin delivers TrueSkill-quality ratings using fast, closed-form math that runs in real time per match. It's the practical choice when you need multiplayer team ratings but don't want to implement Microsoft's TrueSkill from scratch.

**When to use**

- You need per-player ratings from team matches (not just head-to-head) and want TrueSkill-level accuracy without the implementation complexity.
- You want an open-source, patent-free alternative to Microsoft's TrueSkill for multiplayer rating.
- You need fast, closed-form updates that can be computed per-match in real time rather than batch-processed.

**When to avoid**

- You only have simple pairwise (1v1) results — use Elo, Glicko-2, or Bradley-Terry instead; Weng-Lin's team machinery adds unnecessary overhead.
- You need rigorous uncertainty quantification or production-grade refinements — TrueSkill 2 has more battle-tested calibration and edge-case handling.

**CLI usage**

```bash
propagon matchups weng-lin --variant thurstone-mosteller --tau 0.0833 games.matchups
```

**Library usage**

```rust
use propagon::algos::{WengLin, WengLinVariant};
use propagon::{MatchupsDataset, OnlineRanker, RankModel};

let mut dataset = MatchupsDataset::new();
dataset.push_matchup(&["Alice", "Bob"], &["Carol", "Dave"]).unwrap();

let mut wl = WengLin::default();
wl.variant = WengLinVariant::ThurstoneMostellerFull;
let mut model = wl.init();
wl.update(&mut model, &dataset).unwrap();

for (id, rating) in model.sorted_ratings() {
    println!("{}: mu={:.1} sigma={:.1}", id, rating.mu, rating.sigma);
}
```

**What the numbers mean**

Each player gets three numbers: mu (estimated skill mean, default 25), sigma (uncertainty, default ~8.3), and ordinal (conservative display rating = mu − 3×sigma). A higher mu means higher estimated skill; a higher sigma means the system is less certain about that player's true level. After many matches, sigma shrinks and mu stabilizes. The ordinal rating is what you'd show users — it only rises when the system is confident enough that even the lower bound of the player's skill has improved.

### Whole-History Rating

**Real-world scenario**

You want a Go server's ratings to be both current *and* historically honest: when a player improves, that should retro-actively sharpen what you believe about their *past* games too, not just their future ones. Elo and Glicko update forward in time and never look back. Whole-History Rating fits every player's entire rating *curve* at once — a smooth trajectory across time periods — so each game informs the player's strength before and after it. The result is the most statistically coherent "what was everyone's strength at every moment" you can get from the record.

**When to use**

- You care about ratings over time (a full rating *curve* per player), not just a single current number — historical analyses, "how strong was this player in 2019?"
- You want maximum statistical coherence: a batch fit where every game refines the whole timeline, beating the forward-only approximations of Elo/Glicko.
- Skill drifts smoothly and you can express the data as time periods (blank-line-separated batches).

**When to avoid**

- You need live, per-game updates on a busy server — WHR is a batch fit over all history and is heavier than an online update; Glicko-2 is the streaming choice.
- You have no meaningful time structure (all games are contemporaneous) — the timeline machinery buys nothing over a plain Bradley-Terry fit.

**CLI usage**

```bash
# blank lines separate time periods; --timeline prints the full curve per player:
propagon tournament whole-history-rating --w2 0.0006 --groups-are-separate --timeline history.tsv
```

**Library usage**

```rust
use propagon::algos::Whr;
use propagon::{Ranker, RankModel, PairwiseDataset};

let mut d = PairwiseDataset::new();
d.push("a", "b", 2.0);     // period 0
d.new_period();
d.push("b", "a", 1.0);     // period 1

let model = Whr::default().fit(&d)?;
for (name, last_rating) in model.sorted_scores() {
    println!("{name}: {last_rating:.1}");          // rating at its final period
}
if let Some((periods, ratings, sds)) = model.timeline("a") {
    for ((t, r), sd) in periods.iter().zip(ratings).zip(sds) {
        println!("  a @ {t}: {r:.1} ± {sd:.1}");
    }
}
```

**What the numbers mean**

Ratings are on the familiar Elo display scale, so a gap predicts win probability the usual way. The default leaderboard shows each player's rating at their *last* active period; `--timeline` (or `model.timeline(name)`) instead gives the whole curve — for each period a `(rating, standard deviation)` pair, where the SD is the model's uncertainty about that player at that moment (wider early or after long absences, tighter where they played a lot). `--w2` is the Wiener prior: how much a rating is allowed to drift between periods — small `w2` enforces nearly-flat careers, large `w2` lets ratings swing freely to fit each period. Periods come from blank-line batches with `--groups-are-separate`.

### Win Rate

**Real-world scenario:** Imagine you're running a chess tournament with 16 players. Some have only played 2 games while others have played 14. You want a leaderboard that's fair — one that doesn't overreward the lightly-played 2-0 player just because they haven't had a chance to lose yet. Win Rate handles this by applying a confidence-bound estimate so that a 2-0 newcomer doesn't outrank a 95-5 veteran.

**When to use:**

- You need a fast, explainable baseline ranking from pairwise outcomes and want small-sample entities handled sensibly.
- Your comparison schedule is roughly balanced (round-robin or randomized) so opponent strength doesn't systematically bias one entity over another.
- You are sorting user ratings, product reviews, or A/B test results where sample sizes vary widely across items.

**When to avoid:**

- Your data has non-uniform matchmaking (e.g., strong players face each other more, or some entities only play weak opponents) — the method is schedule-blind and will produce biased rankings, not just noisy ones.
- You need opponent-adjusted ratings or win-probability predictions; use Bradley-Terry, Elo, or Colley instead.

**CLI usage:**

```bash
propagon tournament win-rate matches.csv --confidence-interval 0.95
```

**Library usage:**

```rust
use propagon::algos::{Confidence, WinRate};
use propagon::{GamesDataset, OnlineRanker, RankModel};

let mut games = GamesDataset::new();
games.push_pair("Alice", "Bob", 1.0).unwrap();
games.push_pair("Alice", "Carol", 1.0).unwrap();
games.push_pair("Bob", "Carol", 1.0).unwrap();

let algo = WinRate { confidence: Confidence::P95 };
let model = algo.fit(&games).unwrap();
for (name, score) in model.sorted_scores() {
    println!("{}: {:.4}", name, score);
}
```

**What the numbers mean:**

Each entity gets a score between 0 and 1. At P95 (the default), the score is the upper bound of the 95% Wilson confidence interval for the entity's true win rate. A score of 0.95 means you can be 95% confident the entity's true win rate is at least that high. An undefeated player with only 2 games might score 0.82 (Wilson penalizes small samples), while a 90-10 veteran scores 0.97 — the veteran ranks higher despite a lower raw rate. At P50, the score is the plain win fraction with no smoothing.

## Input file formats

Propagon reads plain-text input files. Every algorithm accepts the format that matches its data type — here's what each format looks like.

### Tournament games

Used by: **every `propagon tournament` algorithm** — Win Rate, Bradley-Terry (MM/SGD, Bayesian, Generalized, Team, Covariate), Thurstone-Mosteller, Random Utility, Elo (and margin-of-victory Elo), Glicko-2, Luce Spectral Ranking, I-LSR, Rank Centrality, Keener, HodgeRank, Massey, Colley, Offense-Defense, Borda, Copeland, Kemeny, SerialRank, Random Walker, Whole-History Rating, mElo, Nash averaging, Blade-Chest.

**Format:** one game per line, **tab-separated** fields: `side1<TAB>side2<TAB>threshold[<TAB>count]`. Each side is a roster of **space-separated** player names. The signed `threshold` carries both winner and margin: `> 0` means side 1 won by that much, `< 0` means side 2 won, `0` is a tie. The optional 4th field is a repeat count (default 1). A blank line marks a rating-period boundary (used when a command takes `--groups-are-separate`).

```
BOS	NYY	1          # BOS beat NYY by 1 — side 1 is "home" when --home-advantage is on
LAD	SFG	4	2      # LAD beat SFG by 4; this row stands for 2 identical games
ARI	COL	0          # tie (threshold 0)
alice bob	carol dave	1   # a 2v2 team game (for team-bradley-terry), side 1 won
```

Win/loss-only data is just `±1` thresholds. Margin algorithms (Massey, Keener, Offense-Defense) read `|threshold|` as the margin; tie handling for win/loss algorithms is set by `--ties` (`error`/`discard`/`half-win`) and for margin algorithms by `--margin-ties`.

### N-way matchups

Used by: Weng-Lin

**Format:** one match per line, sides separated by `|`, players within a side space-separated; a trailing `=` marks a tie.

```
Alice Bob | Carol Dave    # Team AB vs Team CD
Alice Bob | Carol Dave =  # tie
```

### Rank aggregation (ballots)

Used by: Kemeny, Borda Count, Copeland, Plackett-Luce, I-LSR, Markov Chain (MC4), Mallows Model, Footrule

**Format:** one ballot per line, items listed best-first, space-separated. Partial ballots are allowed.

```
Alice Bob Carol   # Alice ranked 1st, Bob 2nd, Carol 3rd
Bob Alice Carol   # different voter's ranking
Carol Alice       # partial ballot (top-2 only)
```

### Crowd votes (annotated pairs)

Used by: Crowd Bradley-Terry

**Format:** one vote per line: `annotator_id winner_name loser_name [weight]`.

```
judge1 Alice Bob 1.0
judge2 Alice Carol 1.0
```

### Graph centrality (edges)

Used by: PageRank (and personalized PageRank via `--seeds`), LeaderRank, Harmonic Centrality, BiRank, HITS, Katz Centrality, Degree, k-Core Decomposition, Connected Components

**Format:** one edge per line: `source_id destination_id [weight]`.

```
pageA pageB 1.0   # pageA links to pageB
pageB pageC 1.0   # pageB links to pageC
pageC pageA 1.0   # pageC links to pageA (cycle)
```

### Bandit rewards

Used by: Greedy, Epsilon-Greedy, Upper Confidence Bound, KL-UCB, EXP3, Thompson Sampling (Beta), Thompson Sampling (Gaussian), Sliding-Window UCB

**Format:** one observation per line: `arm_name reward_value`.

```
banner_A 1.0   # click (reward=1)
banner_B 0.0   # no click (reward=0)
banner_A 1.0   # another click
```

### Contextual bandit rewards

Used by: LinUCB

**Format:** one observation per line: `arm_name reward_value x1 x2 ... xd` — the context features follow the reward, whitespace-separated. The dimension `d` is fixed by the first row.

```
sports  1.0 1.0 0.0   # arm "sports", reward 1, context (1, 0)
finance 0.0 0.0 1.0
sports  1.0 1.0 0.0
```

### Trajectories (episodes)

Used by: Monte Carlo Value, TD(0), Value Comparison, Behavior Cloning

**Format:** one step per line: `state reward`. A **blank line ends the current episode**.

```
landing  0.0
product  0.0
checkout 1.0
          # blank line: this session (episode) ends here
landing  0.0
exit     0.0
```

### Entity features (sidecar)

Used by: Covariate Bradley-Terry (`--features`)

**Format:** one entity per line: `entity_id x1 x2 ... xd`, whitespace-separated, every row the same width. Supplied alongside a tournament games file.

```
alice 1.0 -0.5
bob   0.0  0.5
carol -1.0 1.0
```

## State files and incremental updates

When you run an algorithm with `--save-state`, propagon writes the fitted model to a JSONL file. This file is human-readable and self-describing — the first line records which algorithm and parameters were used, and subsequent lines hold one entity per line.

```jsonl
{"propagon":1,"kind":"model","algorithm":"glicko2","params":{"tau":0.5,...},"entities":30}
{"id":"BOS","r":1670.8,"rd":40.5,"sigma":0.06}
```

To continue where you left off, pass `--load-state` with the same file along with new data. Online algorithms (Elo, Glicko-2, all bandits) fold the new results into the existing ratings without reprocessing history. Iterative algorithms (Bradley-Terry, Plackett-Luce, Rank Centrality) use the loaded state as a warm start, converging faster than from scratch.

## See also

- **[Algorithm survey](docs/algorithms.md)** — detailed write-ups with assumptions, citations, and complexity for every method.
- **[Developer docs](DEVELOPER.md)** — building from source, feature flags, and the test strategy.
- **[Examples](examples/)** — worked demos with real data (2018 MLB season, 2024 F1 season, Wikipedia link graph).
- **[Roadmap](docs/PRD.md)** — Python and WASM bindings are the next milestones.

Licensed under MIT OR Apache-2.0.
