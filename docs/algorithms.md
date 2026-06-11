# Ranking from Revealed Preferences: A Survey of Algorithms

This document surveys the algorithm landscape for **ranking entities from revealed preferences** — match outcomes, pairwise choices, multiway selections, clicks, link structure, and reward-bearing trajectories. It exists to ground propagon's evolution into a general-purpose ranking library: every method is described with its model, assumptions, estimation strategy, strengths, weaknesses, and use cases, and is tagged with its implementation status in propagon (or the sibling project mcrl-rs).

Citations appear inline as `[Author Year]` and resolve in [§17 References](#17-references).

## Table of Contents

- [§0 Scope & Philosophy](#0-scope--philosophy)
  - [§0.1 The Parametric vs Non-Parametric Axis](#01-the-parametric-vs-non-parametric-axis)
  - [§0.2 Data & Identifiability Primer](#02-data--identifiability-primer)
  - [§0.3 How to Read This Document](#03-how-to-read-this-document)
- [§1 Parametric Pairwise & Choice Models (Static)](#1-parametric-pairwise--choice-models-static)
- [§2 Online, Sequential & Dynamic Rating Systems](#2-online-sequential--dynamic-rating-systems)
- [§3 Spectral & Markov-Chain Methods on Comparison Data](#3-spectral--markov-chain-methods-on-comparison-data)
- [§4 Node Importance & Centrality (Graph Ranking)](#4-node-importance--centrality-graph-ranking)
- [§5 Least-Squares & Linear-Algebra Sports Ratings](#5-least-squares--linear-algebra-sports-ratings)
- [§6 Rank Aggregation & Social Choice](#6-rank-aggregation--social-choice)
- [§7 Non-Parametric & Robust Estimators](#7-non-parametric--robust-estimators)
- [§8 Active Ranking & Dueling Bandits](#8-active-ranking--dueling-bandits)
- [§9 Intransitivity & Multidimensional Skill](#9-intransitivity--multidimensional-skill)
- [§10 Feature-Based, Contextual & Learning-to-Rank](#10-feature-based-contextual--learning-to-rank)
- [§11 Bayesian & Uncertainty-Aware Inference](#11-bayesian--uncertainty-aware-inference)
- [§12 Applied Deep-Dive: Ranking LLMs & Model Evaluation](#12-applied-deep-dive-ranking-llms--model-evaluation)
- [§13 Value-Function & Trajectory-Based Ranking](#13-value-function--trajectory-based-ranking)
- [§14 Cross-Cutting Topics](#14-cross-cutting-topics)
- [§15 Method-Selection Decision Guide](#15-method-selection-decision-guide)
- [§16 Propagon Coverage Map](#16-propagon-coverage-map)
- [§17 References](#17-references)

---

## 0. Scope & Philosophy

A **revealed preference** is any observation in which an agent's behavior discloses a relative valuation, without anyone stating a score on an absolute scale:

| Data shape | Example | Primary consumers |
|---|---|---|
| Pairwise outcome `i ≻ j` | Team A beat team B; annotator preferred response A | §1, §2, §3, §7 |
| Pairwise with margin | A beat B by 14 points | §2 (MOV Elo), §5 (Massey), §3 (Keener) |
| Choice from a set | User picked item `i` out of slate `{i, j, k}` | §1 (Plackett-Luce, RUM) |
| Full or partial ranking | A judge ordered 10 essays | §1 (PL, Mallows), §6 |
| Multiple judges' rankings | Search engines' result lists; voters' ballots | §6 |
| Graph edges (links, follows, citations, purchases) | Page X links to page Y | §4 |
| Click / interaction logs | User clicked result 3, skipped 1–2 | §10 |
| Reward-bearing trajectories | Game states leading to a win; sessions ending in revenue | §13 |

Across nearly all of the comparison-based families, the unifying lens is:

$$P(i \succ j) = F(s_i - s_j)$$

where $s_i$ is a latent score for entity $i$ and $F$ is a link function. Methods differ along four axes:

1. **What is assumed about $F$** — logistic (Bradley-Terry), Gaussian CDF (Thurstone), *any* monotone function (stochastic transitivity models), or nothing at all (pure counting).
2. **How $s$ is estimated** — maximum likelihood (MM, Newton, SGD), spectral/stationary-distribution computation, least squares, Bayesian inference, or online updates.
3. **Whether $s$ is static or time-varying** — one fixed skill per entity vs. a trajectory of skill through time.
4. **What data shape is consumed** — pairs, sets, rankings, graphs, or trajectories.

Two families deliberately stretch the lens:

- **Node importance / centrality (§4)** treats the *structure* of a graph as the preference signal: a link, citation, follow, or purchase is an implicit endorsement of its target. There is no explicit "i beats j" event, yet the output is the same — a ranking. The bridge is concrete: the best-understood spectral comparison methods (Rank Centrality, Keener, random-walker rankings, §3) are *exactly* centrality computations applied to a graph whose edges are comparison outcomes.
- **Value-function ranking (§13)** treats *trajectories with rewards* as the preference signal: states (or entities identified with states — teams, UI variants, slot machines) are ranked by expected discounted return $V(s)$. Any comparison $V(a) > V(b)$ is itself a revealed-preference edge, so this family composes with every aggregator above it.

### 0.1 The Parametric vs Non-Parametric Axis

This is the most consequential modeling decision, so every method entry in this document carries a **Class** tag.

**Parametric** methods posit a finite-dimensional latent parameter per entity and a *fixed* link function $F$: Bradley-Terry assumes $F = \sigma$ (logistic), Thurstone assumes $F = \Phi$ (Gaussian CDF). What you buy:

- **Sample efficiency** — strength flows through the comparison graph; beating a strong opponent informs your rating against everyone, so sparse data ($O(n \log n)$ comparisons for $n$ items) can suffice.
- **Prediction of unseen matchups** — the model emits a calibrated $P(i \succ j)$ for pairs never observed.
- **Composability** — covariates, ties, home advantage, team structure, and time dynamics all attach naturally to the likelihood.

What you pay: **misspecification risk**. If the true comparison probabilities are intransitive (rock-paper-scissors, §9), heteroskedastic, or style-confounded (§12), the parametric fit is biased in ways that no amount of data fixes.

**Non-parametric** methods assume at most *stochastic transitivity* — if $P(i \succ j) \ge \tfrac12$ and $P(j \succ k) \ge \tfrac12$ then $P(i \succ k) \ge \max(\cdot,\cdot)$ (strong form, SST) — or nothing at all, and estimate ranks or the full probability matrix directly, typically by counting. What you buy: robustness over a class of models exponentially larger than the parametric ones. What you pay: weaker extrapolation and (sometimes) more data.

Two theoretical results anchor the tradeoff:

- **Counting is near-optimal.** Ranking items by a simple (Borda-style) win count against uniformly sampled opponents is minimax-optimal up to constant factors for rank recovery over SST models — and remains optimal even if the data really did come from Bradley-Terry [Shah & Wainwright 2018]. The parametric MLE's advantage is concentrated in *score* estimation and unseen-pair prediction, not in the ranking itself.
- **Parametric assumptions buy only a log factor.** In active top-$k$ ranking, sample complexity under parametric assumptions improves on the assumption-free setting by at most logarithmic factors [Heckel et al. 2019]. If you are paying a misspecification risk, you are buying surprisingly little statistical efficiency with it.

The practical synthesis used throughout this document: **parametric methods when you need calibrated predictions, uncertainty, covariates, or dynamics; non-parametric counting when you need a robust order and have reasonably dense data; spectral/algebraic methods as the scalable middle ground.**

A coarse map of the landscape (each method is detailed in its section):

| | **Parametric** | **Semi-parametric / Spectral / Algebraic** | **Non-parametric / Counting** |
|---|---|---|---|
| **Static** | Bradley-Terry, Thurstone, Plackett-Luce, RUM, Mallows (§1); Blade-Chest, mElo (§9); BT-with-covariates, RLHF reward models (§10, §12) | Rank Centrality, LSR, Keener, SerialRank, HodgeRank (§3); Massey, Colley (§5); PageRank, Katz, HITS & centrality family (§4); α-Rank (§9) | Borda/Copeland counting, SST models, noisy sorting, Wilson-score win rates (§6, §7); Kemeny & social choice (§6) |
| **Online / Dynamic** | Elo, Glicko-2, TrueSkill, Weng-Lin, WHR, dynamic BT (§2) | — | Dueling bandits, active ranking (§8) |
| **Trajectory-valued** | TD with function approximation (§13) | — | Monte Carlo V(s) + bootstrap/permutation inference (§13) |

### 0.2 Data & Identifiability Primer

Concepts referenced throughout the method entries:

**Connectivity (the Ford condition).** The Bradley-Terry MLE exists and is unique (up to normalization) iff in every possible partition of the items into two non-empty sets, some item in each set has beaten some item in the other [Ford 1957]. Intuitively: an undefeated item's rating diverges to $+\infty$, and disconnected components of the comparison graph cannot be placed on a common scale. Practical mitigations: regularization/priors (a pseudo-match against an average opponent), restricting to the largest strongly connected component, or propagon's options on `btm-mm` (`--remove-total-losers`, `--create-fake-games`, `--random-subgraph-links`).

**Scale and location invariance.** Latent scores are identified only up to an additive constant (multiplicative, on the $\pi_i = e^{s_i}$ scale). Any reported scores are *relative*; cross-dataset comparison of raw scores is meaningless without anchoring. Some models (e.g., the Gaussian RUM fit by propagon's `es-rum`) have additional non-identifiability (a common variance scale), so only the induced order and pairwise probabilities are meaningful.

**Ties.** Options: discard, count as half-wins (Elo's convention), or model explicitly with a tie parameter [Rao & Kupper 1967; Davidson 1970] — see §1.2. Tie modeling matters enormously in LLM-arena data, where ties are frequent (§12).

**Margins / scores.** Win-only models discard margin information; Massey (§5), Keener (§3), margin-of-victory Elo (§2), and TrueSkill 2 (§2) consume it. Margins add signal but import a new assumption (that margin is monotone in skill difference) and new failure modes (blowout-chasing).

**Order effects / home advantage.** A multiplicative advantage parameter for the first/home position attaches cleanly to BT-family likelihoods (§1.2). In arena settings the analogue is *position bias*, handled by randomizing presentation order.

**Teams and groups.** When outcomes are by team but ratings are wanted per player, the standard device is summing member skills (TrueSkill, §2.4) or multiplying abilities (group BT [Huang, Weng & Lin 2006]).

**Dependent comparisons.** Classical theory assumes independent comparisons; in reality the same player's matches share form, weather, lineups. Ignoring dependence biases uncertainty estimates (not necessarily point estimates). See [Cattelan 2012] for the survey of remedies (random effects, pairwise likelihoods).

### 0.3 How to Read This Document

Major methods get the full template below; minor or adjacent methods get a compact paragraph with the same **Class** tag. Centrality entries (§4) swap the `Handles` line for graph-relevant capabilities (directed / weighted / bipartite / dangling nodes / local-vs-global).

> **Template**
> - **TL;DR** — one sentence: what it solves.
> - **Inputs / Output** — data shape consumed; what comes out (scores, score+uncertainty, distribution, total order, stationary distribution).
> - **Class** — Parametric | Semi-parametric | Non-parametric | Algebraic/Spectral | Heuristic × Static | Online | Dynamic.
> - **Model & assumptions** — latent form, link $F$, identifiability conditions.
> - **Estimation & complexity** — algorithm, cost, convergence, scalability.
> - **Handles** — ties · margins · home advantage · teams · time dynamics · uncertainty · intransitivity.
> - **Pros / Cons.**
> - **Use cases.**
> - **Relationships** — generalizes / specializes / equivalent to.
> - **Propagon status** — implemented (`module.rs`) | candidate | sibling (mcrl-rs) | out-of-scope.
> - **References.**

In a hurry? Jump to the [decision guide (§15)](#15-method-selection-decision-guide) and the [propagon coverage map (§16)](#16-propagon-coverage-map).

---

## 1. Parametric Pairwise & Choice Models (Static)

The classical core. One latent strength per entity, a fixed link function, a likelihood, and seventy years of extensions. Recent umbrella treatments: [Hamilton, Tawn & Firth 2023] derive Bradley-Terry from at least four independent starting points (random utility, Luce's axiom, maximum entropy, network reversibility), and [Fang et al. 2026] survey modern estimation theory and ML applications.

### 1.1 Bradley-Terry (1952) — a.k.a. Zermelo's model, BTL

- **TL;DR** — The canonical model of pairwise comparison: each entity has one positive strength, and win probability is your share of the combined strength.
- **Inputs / Output** — pairwise outcomes (optionally weighted/aggregated) → real-valued scores (+ standard errors if you ask the likelihood nicely).
- **Class** — Parametric × Static.
- **Model & assumptions** — strengths $\pi_i > 0$ (equivalently $s_i = \log \pi_i$):

  $$P(i \succ j) = \frac{\pi_i}{\pi_i + \pi_j} = \sigma(s_i - s_j)$$

  Assumes independent comparisons and (implicitly) *strong stochastic transitivity* — BT cannot represent cyclic dominance. Identified up to a multiplicative constant given Ford connectivity (§0.2). First derived by [Zermelo 1929] for chess tournaments; rediscovered by [Bradley & Terry 1952]; existence/uniqueness by [Ford 1957].
- **Estimation & complexity** —
  - **MM (minorization-maximization)** [Hunter 2004]: iterate
    $\pi_i \leftarrow W_i \big/ \sum_{j \ne i} \tfrac{n_{ij}}{\pi_i + \pi_j}$
    where $W_i$ is $i$'s win count and $n_{ij}$ the number of $i$-$j$ comparisons. Monotone in likelihood, globally convergent under Ford's condition, embarrassingly cheap per pass ($O(\text{edges})$).
  - **Logistic regression**: BT is exactly logistic regression on indicator-difference features, so any logistic solver (Newton, L-BFGS, SGD) applies. The SGD form scales to streams and huge graphs and is the bridge to Elo (§2.1).
  - **Spectral**: Rank Centrality and LSR (§3) estimate the same model non-iteratively.
- **Handles** — ties ✗ (see §1.2) · margins ✗ · home-adv ✗ (see §1.2) · teams ✗ (see §1.2) · dynamics ✗ (see §2) · uncertainty △ (asymptotic SEs or bootstrap) · intransitivity ✗.
- **Pros** — interpretable; calibrated pairwise predictions; sparse-data efficient; the extension point for nearly everything else in this document; well-understood theory (consistency, asymptotic normality, finite-sample rates).
- **Cons** — no ties, margins, or dynamics out of the box; undefined on disconnected data; misspecified under cyclic or multidimensional skill; scores are relative only.
- **Use cases** — sports league tables from head-to-head results; LLM arena leaderboards (§12); paired-preference analysis of A/B alternatives; crowdsourced comparison aggregation; reward modeling (§10.6).
- **Relationships** — logistic special case of Thurstone's framework; pairwise marginal of Plackett-Luce; the model Elo tracks online; target model of Rank Centrality/LSR.
- **Propagon status** — **implemented**: `src/mm.rs` (`btm-mm`, Hunter's MM with connectivity mitigations) and `src/lr.rs` (`btm-lr`, logistic SGD with decay and a memory-thrifty mode).
- **References** — [Zermelo 1929; Bradley & Terry 1952; Ford 1957; Hunter 2004; Hamilton, Tawn & Firth 2023; Fang et al. 2026].

### 1.2 Bradley-Terry Extensions: Ties, Order Effects, Groups

- **TL;DR** — The BT likelihood is a chassis: ties, home advantage, and team structure bolt on as extra parameters without changing the estimation story.
- **Class** — Parametric × Static.
- **Ties.** [Rao & Kupper 1967] introduce a threshold $\theta \ge 1$: $P(i \succ j) = \tfrac{\pi_i}{\pi_i + \theta \pi_j}$, with the leftover mass as the tie probability (a draw happens when strengths are within a perceptual threshold). [Davidson 1970] instead sets $P(\text{tie}) \propto \nu \sqrt{\pi_i \pi_j}$, preserving the ratio structure. Both add one global parameter and remain MM-estimable [Hunter 2004]. Choose Davidson when tie rates are roughly symmetric in strength; Rao-Kupper when ties concentrate among near-equals.
- **Order effects / home advantage.** A multiplicative boost $\gamma > 0$ to whichever entity is at home (or shown first): $P(i \succ j \mid i\ \text{home}) = \tfrac{\gamma \pi_i}{\gamma \pi_i + \pi_j}$ [Agresti 2013]. One parameter, large realism gain in sports; the arena analogue is position bias.
- **Teams / groups.** Generalized BT models for team comparisons express team strength as a function of member strengths — e.g., $\pi_{\text{team}} = \sum_{k \in \text{team}} \pi_k$ or a product/geometric form — and recover individual ratings from team outcomes [Huang, Weng & Lin 2006]. The same paper covers multi-class probability estimation via pairwise coupling.
- **Covariates.** $s_i = \beta^\top x_i$ turns BT into a conditional logit on features — see §10.1 and the style-controlled variant in §12.2.
- **Pros / Cons** — each extension is one or two parameters, preserves convexity, and is testable by likelihood ratio; but each also adds an assumption (tie mechanism, constant home effect, additive team skill) that can be violated.
- **Propagon status** — **candidate**: none of the tie/home/team extensions are exposed by current modules; they are natural flags on a future BT implementation.
- **References** — [Rao & Kupper 1967; Davidson 1970; Agresti 2013; Huang, Weng & Lin 2006].

### 1.3 Thurstone-Mosteller (1927) — Case V, probit paired comparison

- **TL;DR** — The original latent-score model: each entity's perceived quality is Gaussian, and you prefer whichever sample came out higher.
- **Inputs / Output** — pairwise outcomes → scores on a latent Gaussian scale.
- **Class** — Parametric × Static.
- **Model & assumptions** — quality draws $X_i \sim \mathcal{N}(s_i, \sigma^2)$, preference is $X_i > X_j$, so

  $$P(i \succ j) = \Phi\!\left(\frac{s_i - s_j}{\sigma\sqrt{2}}\right)$$

  (Case V: equal variances, zero correlation) [Thurstone 1927], with estimation procedures consolidated by Mosteller. The practical guide of record for fitting and comparing Thurstone vs. BT is [Tsukida & Gupta 2011].
- **Estimation & complexity** — probit regression (MLE), or the classical closed-form inverse-Φ of empirical win rates when all pairs are densely observed.
- **Handles** — ties ✗ · margins ✗ · uncertainty △ · others ✗.
- **Pros** — the natural choice when the latent-Gaussian story is real (psychophysics, perceptual quality, image/audio MOS studies); conjugate with Gaussian message passing (TrueSkill, §2.4, is its online descendant).
- **Cons** — in practice nearly indistinguishable from BT (logistic ≈ scaled probit), so the extra integral rarely buys accuracy; same transitivity and connectivity constraints.
- **Use cases** — psychometrics; perceptual quality assessment (codec comparisons, image quality); anywhere Gaussian latents make priors natural.
- **Relationships** — BT swaps $\Phi$ for $\sigma$; TrueSkill is its Bayesian online version; Stern's gamma family (§1.6) interpolates between Thurstone-like and Luce-like models.
- **Propagon status** — candidate (probit link on the future BT chassis).
- **References** — [Thurstone 1927; Mosteller 1951; Tsukida & Gupta 2011].

### 1.4 Plackett-Luce (1959/1975) — multiway choices and rankings

- **TL;DR** — Bradley-Terry for sets: pick winners from slates (or produce whole rankings) with probability proportional to strength, repeatedly.
- **Inputs / Output** — choice-from-set events and/or full or partial rankings → scores.
- **Class** — Parametric × Static.
- **Model & assumptions** — choice from set $A$: $P(i \mid A) = \pi_i / \sum_{j \in A} \pi_j$ [Luce 1959]. A full ranking is a cascade of such choices (pick the winner, remove it, repeat) [Plackett 1975]:

  $$P(\sigma) = \prod_{r=1}^{m} \frac{\pi_{\sigma(r)}}{\sum_{k \ge r} \pi_{\sigma(k)}}$$

  Rests on **Luce's choice axiom** / IIA — the relative odds of $i$ over $j$ don't depend on what else is in the slate. IIA is the model's power and its Achilles' heel (red-bus/blue-bus substitution effects violate it).
- **Estimation & complexity** — MM [Hunter 2004]; spectral LSR / I-LSR (§3.2) with optimal-rate guarantees [Maystre & Grossglauser 2015]; SGD on the log-likelihood (this is exactly the listwise ListMLE loss, §10.4, and the softmax cross-entropy familiar from ML).
- **Handles** — ties ✗ · choice sets ✓ · partial rankings ✓ (top-$k$ truncation is likelihood-exact) · others as per BT.
- **Pros** — consumes far richer data than pairwise (a single 10-way ranking carries 45 pairwise constraints, coherently); top-$k$ data handled exactly; the de facto standard for discrete choice and listwise LTR.
- **Cons** — IIA; winner-focused (the cascade emphasizes top positions; bottom-of-ranking noise is modeled poorly); same connectivity caveats.
- **Use cases** — race/tournament results with full finishing orders; multiway A/B/n tests; recommender slates; ballot data; LLM evals where annotators rank $k$ responses.
- **Relationships** — restriction to pairs = BT; special case of RUM with Gumbel noise (§1.5); Mallows (§1.7) is the other major ranking distribution, with distance- rather than strength-based structure.
- **Propagon status** — **partially implemented**: `src/lsr.rs` fits PL from pairwise data; multiway/ranking input parsing and the exact I-LSR refinement are candidates.
- **References** — [Luce 1959; Plackett 1975; Hunter 2004; Maystre & Grossglauser 2015].

### 1.5 Random Utility Models & Conditional Logit (1974)

- **TL;DR** — The econometric umbrella: utility = systematic part + random noise; the noise distribution determines which comparison model you get.
- **Inputs / Output** — choices from sets (with or without covariates) → utility parameters / scores.
- **Class** — Parametric × Static.
- **Model & assumptions** — $U_i = s_i + \varepsilon_i$, choose $\arg\max_i U_i$. Gumbel $\varepsilon$ ⇒ multinomial/conditional logit = Plackett-Luce [McFadden 1974]; Gaussian $\varepsilon$ ⇒ Thurstone/probit (no closed form for sets > 2); nested and mixed logit relax IIA at the cost of more parameters.
- **Estimation & complexity** — MLE (convex for logit); simulation-based methods for probit/mixed logit; or gradient-free optimization when the likelihood is awkward (see ES-RUM below).
- **Pros / Cons** — connects ranking directly to seventy years of econometrics (welfare analysis, elasticities, demand prediction); covariates are first-class. Cost: distributional commitments, and the flexible variants (mixed logit) are expensive and weakly identified on small data.
- **Use cases** — purchase/choice logs; transport mode choice; assortment optimization; any "users chose X from slate S" dataset — the purest "revealed preference" setting in the economic sense (sidebar, §14).
- **ES-RUM (propagon).** `src/esrum.rs` fits a Gaussian RUM — each entity gets $(\mu_i, \sigma_i)$, win probability from the Gaussian difference — using **evolution strategies** (gradient-free, perturbation-based updates with regularization). Distinctive: per-entity *variance* is estimated, so an entity can be "good but erratic." Caveat documented in the module: the model is identified only up to location/scale, so outputs are meaningful **relatively**, not absolutely.
- **Propagon status** — **implemented** (`es-rum`) for the Gaussian-RUM special case; covariate-bearing conditional logit is a candidate.
- **References** — [McFadden 1974; Train 2009; Tsukida & Gupta 2011].

### 1.6 Stern's Gamma / Poisson-Race Family (1990) — compact

**Class: Parametric × Static.** [Stern 1990] models each entity as a gamma-distributed race time (equivalently, Poisson scoring processes racing to a threshold $r$); the entity that "finishes first" wins. $r = 1$ recovers Luce/BT-type structure and $r \to \infty$ approaches Thurstone — one continuous family containing both classic models. Practically, fitted rankings barely move across $r$, which is the cleanest evidence that the BT-vs-Thurstone choice is mostly aesthetic. Useful conceptually (model-sensitivity analysis) rather than as a distinct production method. *Propagon: out-of-scope (conceptual).* [Stern 1990].

### 1.7 Mallows Model (1957)

- **TL;DR** — A distribution over whole rankings: there is one true order, and observed rankings fall off exponentially in distance from it.
- **Inputs / Output** — full/partial rankings → central ranking $\sigma_0$ + dispersion $\phi$.
- **Class** — Parametric × Static (parametric over *permutations*, not scores).
- **Model & assumptions** — $P(\sigma) \propto \phi^{d(\sigma, \sigma_0)}$ with $d$ usually Kendall's tau distance [Mallows 1957]. No per-item strength: consensus and noise level are the only structure. The standard reference for this and other permutation models is [Marden 1995].
- **Estimation & complexity** — MLE of $\sigma_0$ under Kendall distance **is the Kemeny consensus problem (§6.3)** — NP-hard in general; $\phi$ then has a 1-D closed-form-ish MLE. Sampling and EM variants exist for mixtures.
- **Pros / Cons** — the right noise model when judges are noisy versions of one truth (and the basis for Kemeny's MLE interpretation [Young 1988]); mixtures capture sub-populations of taste. But: no item scores, no unseen-pair prediction, and hard combinatorics.
- **Use cases** — consensus from a small number of complete rankings (panels, juries, meta-search); modeling annotator noise over rankings.
- **Relationships** — MLE = Kemeny-Young (§6.3); contrast PL, which is strength-based and handles partial data more gracefully.
- **Propagon status** — covered implicitly via `kemeny.rs` (consensus); explicit $\phi$ estimation is a candidate nicety.
- **References** — [Mallows 1957; Marden 1995; Young 1988].

---

## 2. Online, Sequential & Dynamic Rating Systems

One family, one through-line: **skill changes over time, and/or data arrives as a stream**. Everything here is a state-space view of §1's models — a latent skill that drifts, observed through noisy comparisons — differing in how much of the posterior they keep (a point for Elo; mean+variance for Glicko; mean+variance+volatility for Glicko-2; full factor-graph messages for TrueSkill; the entire history for WHR). If your entities' true quality is *static* (e.g., frozen LLM checkpoints), be aware that online updating is the wrong tool — see §12.3.

### 2.1 Elo (1960s/1978)

- **TL;DR** — After each game, move both ratings toward what the result implied, by a step proportional to the surprise.
- **Inputs / Output** — a *stream* of pairwise outcomes (order matters) → one scalar rating per entity.
- **Class** — Parametric × Online.
- **Model & assumptions** — expected score $E_i = \sigma\big((R_i - R_j) \cdot \ln 10/400\big)$ (the 400/log-10 scaling is convention), update

  $$R_i \leftarrow R_i + K (S_i - E_i)$$

  with $S_i \in \{0, \tfrac12, 1\}$ [Elo 1978]. This is exactly **online SGD on the Bradley-Terry log-loss** with learning rate $K$ — Elo is not a different model from BT, it is a different *estimator* [Aldous 2017]. Tracks drift because SGD with constant step size never fully converges.
- **Estimation & complexity** — $O(1)$ per game; nothing to store but the ratings. The only knob, $K$, trades tracking speed against noise (chess federations use staged $K$ by experience/rating).
- **Handles** — ties △ (half-score convention) · margins △ (MOV variants below) · home-adv △ (additive offset) · teams ✗ · dynamics ✓ (implicitly) · uncertainty ✗ · intransitivity ✗.
- **Pros** — trivially simple, $O(1)$, transparent, battle-tested at planetary scale (chess, football, online games); self-correcting under drift.
- **Cons** — no uncertainty: a 3-game newcomer and a 3000-game veteran update identically; result-order dependence; sensitive to $K$; rating inflation/deflation across eras; provisional ratings are poor; for *static* skills, it is a noisy estimator of what BT MLE computes better offline [Boubdir et al. 2024].
- **Margin-of-victory variants** — scale $K$ or the score by a function of point differential; in association football, MOV-augmented Elo measurably improves match forecasts [Hvattum & Arntzen 2010]. Watch for blowout pathologies (teams running up scores).
- **Use cases** — live leaderboards with continuous play; matchmaking where simplicity and latency dominate; any setting with genuine skill drift.
- **Relationships** — SGD on BT (§1.1); Glicko adds variance; WHR replaces filtering with full-history smoothing.
- **Propagon status** — **candidate** (deliberately trivial to add; the offline equivalents `btm-mm`/`btm-lr` are implemented).
- **References** — [Elo 1978; Aldous 2017; Hvattum & Arntzen 2010].

### 2.2 Glicko (1999)

- **TL;DR** — Elo plus an honest variance: each player carries a rating *and* a rating deviation (RD) that shrinks with evidence and grows with inactivity.
- **Inputs / Output** — outcome stream grouped into rating periods → (rating, RD) per entity; RD yields confidence intervals and uncertainty-weighted updates.
- **Class** — Parametric (Bayesian approximation) × Online/Dynamic.
- **Model & assumptions** — Gaussian skill posterior per player; BT-style game likelihood; closed-form approximate posterior update per period; between periods, $RD^2 \leftarrow RD^2 + c^2 t$ injects drift uncertainty [Glickman 1999].
- **Estimation & complexity** — closed-form per period, $O(\text{games})$; a few global constants to set.
- **Handles** — everything Elo does, plus uncertainty ✓ and a principled inactivity story.
- **Pros / Cons** — fixes Elo's two worst flaws (no uncertainty, bad provisional ratings) at almost no cost; but skill volatility is assumed constant across players, and the period structure is a modeling choice that matters.
- **Relationships** — superseded in practice by Glicko-2; a one-Gaussian special case of what TrueSkill does with factor graphs.
- **Propagon status** — superseded by the implemented Glicko-2.
- **References** — [Glickman 1999].

### 2.3 Glicko-2 (2001)

- **TL;DR** — Glicko plus per-player *volatility*: the system learns who is steady and who is streaky, and lets surprising results move erratic players more.
- **Inputs / Output** — outcome stream in rating periods (propagon: blank-line-separated batches) → (μ, RD, σ) per entity; propagon emits μ ± 95% CI or μ alone (`--use-mu`).
- **Class** — Parametric (Bayesian approximation) × Online/Dynamic.
- **Model & assumptions** — adds a stochastic-variance state $\sigma_i$ (volatility) governing how fast $RD$ regrows; the volatility update is a 1-D iteration (Illinois algorithm) controlled by meta-parameter $\tau$ (≈0.3–1.2; lower = stable volatility) [Glickman 2001], with the implementable spec in [Glickman 2022].
- **Estimation & complexity** — closed-form except the scalar volatility solve; $O(\text{games})$ per period.
- **Handles** — ties ✓ (half-score) · uncertainty ✓✓ (two levels: rating noise and volatility) · dynamics ✓ · margins/teams ✗.
- **Pros** — best uncertainty story among the lightweight systems; robust default for player-vs-player products; widely deployed (chess servers, online games).
- **Cons** — rating-period bookkeeping; meta-parameters ($\tau$, defaults) need care; still pairwise-only, no teams or margins; per-period approximation degrades with very sparse periods.
- **Use cases** — ongoing competitions with intermittent participation; any leaderboard where "how sure are we?" must be displayed.
- **Relationships** — state-space elaboration of Glicko; the filtering counterpart of WHR's smoothing.
- **Propagon status** — **implemented**: `src/g2.rs` (`glicko2`), with batch periods and `--tau`.
- **References** — [Glickman 2001; Glickman 2022].

### 2.4 TrueSkill (2007), TrueSkill Through Time (2008), TrueSkill 2 (2018)

- **TL;DR** — Skill rating as full Bayesian inference on a factor graph: Gaussian skills, team sums, ordered-outcome likelihoods, solved by expectation propagation — built for multiplayer, multi-team games.
- **Inputs / Output** — match results among teams of players, with possible draws and >2 ranked teams per match → $(\mu_i, \sigma_i)$ per player; matchmaking uses the conservative estimate $\mu - k\sigma$.
- **Class** — Parametric (Bayesian) × Online/Dynamic.
- **Model & assumptions** — player skill $s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$; performance $p_i \sim \mathcal{N}(s_i, \beta^2)$; team performance = sum of member performances; observed match ranking constrains team performances with a draw margin $\varepsilon$. Inference by expectation propagation / Gaussian message passing on the factor graph [Herbrich, Minka & Graepel 2007] — the Thurstonian (probit) analogue of everything BT-shaped here.
- **Variants** — **TrueSkill Through Time** runs messages *backward* through history too (smoothing, not filtering), giving comparable ratings across eras — retro-rating 150 years of chess [Dangauthier et al. 2008]. **TrueSkill 2** adds what production games needed: experience effects, per-mode skills, individual statistics (kills/deaths) as extra evidence, squad correlation — tuned on Gears of War / Halo telemetry [Minka, Cleven & Zaykov 2018].
- **Estimation & complexity** — EP messages per match, near-$O(\text{players in match})$; ordered multi-team outcomes need an internal iteration.
- **Handles** — ties ✓ (draw margin) · teams ✓✓ (its raison d'être) · multi-way match outcomes ✓ · uncertainty ✓ · dynamics ✓ (skill dynamics noise) · margins △ (TS2 via stats) · intransitivity ✗.
- **Pros** — handles team/multiplayer structure nothing else here touches; principled uncertainty; fast convergence (tens of games); production-proven at Xbox scale.
- **Cons** — implementation complexity (EP, moment matching, truncated Gaussians); historical patent encumbrance pushed open implementations to Weng-Lin (§2.5); additive team skill is a strong assumption; probit machinery is opaque compared to Elo.
- **Use cases** — online game matchmaking; team esports; any multiplayer ladder.
- **Relationships** — Bayesian online Thurstone (§1.3); Weng-Lin approximates it cheaply; TTT is the smoothing variant, like WHR is for BT.
- **Propagon status** — candidate (high value, highest implementation cost in this family).
- **References** — [Herbrich, Minka & Graepel 2007; Dangauthier et al. 2008; Minka, Cleven & Zaykov 2018].

### 2.5 Weng-Lin Updates / OpenSkill (2011)

- **TL;DR** — TrueSkill-quality updates in closed form: Bayesian approximation yields simple per-match formulas for multi-team, multiplayer rating with your choice of BT or Thurstone likelihood.
- **Inputs / Output** — as TrueSkill → $(\mu_i, \sigma_i)$ per player.
- **Class** — Parametric (Bayesian approximation) × Online/Dynamic.
- **Model & assumptions** — Gaussian skills; match likelihoods from BT-style or Thurstone-style team comparison; Stein-lemma-based approximation produces closed-form mean/variance updates — no message-passing loop [Weng & Lin 2011].
- **Estimation & complexity** — strictly closed-form, faster than TrueSkill's EP, trivially parallel across matches.
- **Pros / Cons** — ~TrueSkill accuracy at a fraction of the implementation burden, with no patent baggage — the basis of the OpenSkill libraries [Joshy 2024]; slightly cruder approximation, fewer production refinements than TS2.
- **Use cases** — same as TrueSkill; the default open-source choice for multiplayer rating.
- **Propagon status** — **candidate — recommended**: best capability-to-effort ratio for adding team/multiplayer support.
- **References** — [Weng & Lin 2011; Joshy 2024].

### 2.6 Whole-History Rating (2008)

- **TL;DR** — Don't filter — smooth: model every player's entire skill *trajectory* and re-fit all of history at once, so past and present results inform each other.
- **Inputs / Output** — full timestamped game archive → a skill **curve** per player (+ uncertainty), comparable across time.
- **Class** — Parametric (Bayesian MAP) × Dynamic (batch).
- **Model & assumptions** — BT likelihood at each game; each player's $s_i(t)$ a Wiener process (Brownian drift) prior; MAP estimation by Newton's method over all parameters, exploiting the tridiagonal Hessian along each player's timeline [Coulom 2008].
- **Estimation & complexity** — iterated Newton sweeps; near-linear per pass in total games; incremental refresh ("update recently active players only") makes it serviceable online. Adopted in Go servers (KGS) and similar communities.
- **Handles** — dynamics ✓✓ (continuous-time drift) · uncertainty ✓ · ties/handicaps △ (likelihood tweaks) · teams ✗.
- **Pros** — strictly better use of information than filtering systems (Elo/Glicko discard the future's evidence about the past); era-comparable curves; accuracy benchmark for this family.
- **Cons** — batch refits; more code than Glicko; prior drift rate must be set; history storage required.
- **Use cases** — historical analyses ("strongest player of the 1990s"); server-side ratings where batch jobs are fine; ground truth to validate cheaper online systems against.
- **Relationships** — BT + Wiener prior; the BT counterpart of TrueSkill Through Time; the continuous-time limit of the state-space models below.
- **Propagon status** — candidate (good fit: batch CLI + Rayon).
- **References** — [Coulom 2008].

### 2.7 State-Space & Dynamic Bradley-Terry Models — compact family

**Class: Parametric × Dynamic.** The statistical literature's version of everything above: skill as a latent time series observed through paired-comparison likelihoods. [Fahrmeir & Tutz 1994] formulate dynamic ordered paired comparisons with Kalman-style estimation; [Glickman 1999] gives the large-scale approximate filtering treatment (the theory behind Glicko); [Cattelan, Varin & Firth 2013] take an exponentially-weighted-moving-average route fitted by pairwise likelihood, applied to basketball/soccer seasons; [Maystre, Kristof & Grossglauser 2019] generalize the dynamics to flexible Gaussian-process kernels (periodicity, long memory) with scalable inference. Choose this family over §2.2–2.6 when you need *custom dynamics* (seasonality, structural breaks) or rigorous inference rather than a deployable rating service. *Propagon: out-of-scope as a family; WHR/Glicko-2 cover the practical need.* [Fahrmeir & Tutz 1994; Glickman 1999; Cattelan, Varin & Firth 2013; Maystre, Kristof & Grossglauser 2019].

---

## 3. Spectral & Markov-Chain Methods on Comparison Data

Replace iterative likelihood optimization with **one linear-algebra primitive**: build a matrix from the comparison data, extract a leading eigenvector / stationary distribution / least-squares potential, read off the ranking. The payoffs are scalability (power iteration on sparse matrices), no convergence babysitting, and — for several of these — statistical guarantees matching the MLE. The deep connection to §4: every method in this section **is** a centrality measure computed on the comparison graph; the methods in §4 are the same machinery applied to graphs whose edges are endorsements rather than match results.

### 3.1 Rank Centrality (2012/2017)

- **TL;DR** — Random-walk where you drift toward whoever beats whom: the stationary distribution of "follow your losses" is a consistent, near-optimal BT estimate.
- **Inputs / Output** — aggregated pairwise win fractions → stationary probability per entity (≈ normalized BT strength).
- **Class** — Algebraic/Spectral (estimating a parametric model) × Static.
- **Model & assumptions** — Markov chain on entities: from $i$, transition to $j$ with probability proportional to the fraction of $i$–$j$ comparisons that $j$ won; the stationary distribution $\hat\pi$ ranks entities. Under BT data, $\hat\pi$ is consistent with finite-sample error optimal up to log factors — matching the MLE's minimax rate — provided the comparison graph has good spectral expansion [Negahban, Oh & Shah 2017].
- **Estimation & complexity** — power iteration: $O(\text{edges})$ per step, modest step counts; trivially parallel.
- **Handles** — ties △ (as half-wins in the fractions) · weights ✓ · uncertainty ✗ (bootstrap it) · others ✗.
- **Pros** — no iterative MLE, no learning rates; provably near-optimal; robust entry point at web scale; aggregation-friendly (only pairwise win fractions needed, not raw streams).
- **Cons** — needs connectivity *and* spectral-gap quality (a barbell-shaped comparison graph degrades it before it breaks the MLE); win-fraction aggregation discards sequence information; scores lack the likelihood's standard errors.
- **Use cases** — large-scale crowdsourced comparisons; bootstrapping a leaderboard before fitting full BT; ranking with millions of items where MM/Newton are awkward.
- **Relationships** — spectral estimator of BT (§1.1); a personalized-PageRank-flavored cousin of §4.4; generalized by LSR to choice sets.
- **Propagon status** — **candidate — recommended** (small delta on top of `lsr.rs`'s existing Markov-chain machinery).
- **References** — [Negahban, Oh & Shah 2017].

### 3.2 Luce Spectral Ranking — LSR / I-LSR (2015)

- **TL;DR** — Rank Centrality generalized to multiway choices: one Markov chain built from choice events gives an unbiased spectral estimate of Plackett-Luce, and iterating it converges to the exact MLE.
- **Inputs / Output** — pairwise outcomes, choice-from-set events, or (partial) rankings → PL strength estimates.
- **Class** — Algebraic/Spectral (→ exact ML with iteration) × Static.
- **Model & assumptions** — construct a Markov chain whose pairwise transition rates accumulate, for each choice event, mass from losers toward winners (inversely weighted by the choice-set strength sum); its stationary distribution is a consistent PL estimate (LSR). Re-weighting by the current estimate and repeating (I-LSR) yields the true MLE fixed point, with better conditioning than MM in practice [Maystre & Grossglauser 2015].
- **Estimation & complexity** — one (or a few) stationary-distribution computations; each is power iteration at $O(\text{events})$ per sweep. Propagon implements the stationary solve two ways: deterministic power method or Monte Carlo random-walk sampling (`--estimator`, `--steps`).
- **Handles** — choice sets ✓ · partial rankings ✓ · weights ✓ · ties/uncertainty ✗.
- **Pros** — fastest serious PL estimator; statistically efficient (I-LSR = MLE); one algorithm covers pairs, sets, and rankings.
- **Cons** — PL assumptions inherited (IIA, transitivity, connectivity); MC estimator introduces seed variance (fine for ranking, noisier for scores).
- **Use cases** — same as PL (§1.4) at scale: search-result preferences, race results, large k-way crowdsourcing.
- **Relationships** — restriction to pairs ≈ Rank Centrality; the spectral route to §1.4's model.
- **Propagon status** — **implemented**: `src/lsr.rs` (`lsr`, power-method and MC estimators); I-LSR refinement and native multiway input are candidates.
- **References** — [Maystre & Grossglauser 2015].

### 3.3 Keener's Method (1993)

- **TL;DR** — Your strength is proportional to the strength-weighted sum of what you took from each opponent: the Perron-Frobenius eigenvector of a nonnegative score matrix.
- **Inputs / Output** — pairwise results, ideally with scores/margins → positive rating vector (leading eigenvector).
- **Class** — Algebraic/Spectral × Static.
- **Model & assumptions** — build $A_{ij} = h\big(\tfrac{S_{ij} + 1}{S_{ij} + S_{ji} + 2}\big)$ from points $S_{ij}$ scored by $i$ against $j$ (Laplace-smoothed share, passed through a concave "skew" function $h$ to blunt blowouts); rating $r$ solves $A r = \lambda r$, which exists, is positive and unique by Perron-Frobenius given irreducibility [Keener 1993].
- **Estimation & complexity** — power iteration; $O(\text{edges})$ per step.
- **Handles** — margins ✓ (its specialty, with damping) · ties ✓ (smoothing) · uncertainty ✗.
- **Pros** — strength-of-schedule is automatic (points against strong teams are worth more); margin-aware; tiny implementation.
- **Cons** — the smoothing and skew choices are heuristic knobs, not likelihood-derived; sensitive to scheduling imbalance; no probabilistic semantics.
- **Use cases** — sports league ranking with scores (its NFL/college-football home turf); any margin-bearing tournament.
- **Relationships** — the bridge between least-squares ratings (§5) and PageRank-style eigenvector methods (§4.2–4.4); same fixed-point shape as eigenvector centrality on a judiciously built matrix.
- **Propagon status** — candidate (a natural `keener` subcommand; power iteration already exists in-house).
- **References** — [Keener 1993].

### 3.4 Random-Walker Rankings (2007) — compact

**Class: Algebraic/Spectral × Static.** A population of independent "fan" walkers each repeatedly picks one of its team's games and switches allegiance to the winner with probability $p > \tfrac12$ (losing with $1-p$): the stationary share of fans ranks the teams. With $p \to 1$ this approaches win-percentage extremism; $p$ near $\tfrac12$ smooths toward schedule structure. Essentially Rank Centrality with a bias knob, introduced for NCAA football where it gave sensible bowl rankings from very sparse schedules [Callaghan, Mucha & Porter 2007]. Pedagogically lovely; in production, prefer Rank Centrality (principled) or PageRank (general). *Propagon: out-of-scope (covered by Rank Centrality candidate).* [Callaghan, Mucha & Porter 2007].

### 3.5 SerialRank (2014)

- **TL;DR** — Ranking as *seriation*: entities that beat similar sets of opponents are similar; spectrally ordering the similarity matrix recovers the ranking.
- **Inputs / Output** — pairwise comparison matrix (possibly sparse/noisy) → a total **order** (not scores).
- **Class** — Algebraic/Spectral × Static (non-parametric in spirit: no link function).
- **Model & assumptions** — build similarity $S_{ij}$ = count of agreements between $i$'s and $j$'s comparison outcomes against common opponents; compute the Fiedler vector (second-smallest eigenvector) of the Laplacian of $S$; sorting by it recovers the exact order when comparisons derive from any total order, with robustness to a bounded fraction of corruptions [Fogel, d'Aspremont & Vojnović 2014].
- **Estimation & complexity** — one $n \times n$ similarity build (the $O(n^2)$ bottleneck, or $O(n \cdot \text{edges})$ sparse) + one sparse eigenvector.
- **Pros / Cons** — needs no probabilistic model at all (only "a true order exists"); robust to adversarial flips in ways likelihood methods aren't. But: outputs an order without strengths or win-probability predictions; the similarity construction wants reasonably dense comparisons; mostly displaced in practice by Rank Centrality and counting methods.
- **Use cases** — noisy near-complete tournaments; ranking when you distrust every parametric assumption but still want spectral speed; archaeology-style ordering problems (its seriation roots).
- **Relationships** — seriation/spectral-ordering literature; complements §7's counting estimators as the "assume only an order exists" toolkit.
- **Propagon status** — candidate (low priority).
- **References** — [Fogel, d'Aspremont & Vojnović 2014; Atkins, Boman & Hendrickson 1998].

### 3.6 HodgeRank & Least-Squares Ranking on Graphs (2011)

- **TL;DR** — Treat pairwise preferences as a *flow* on the comparison graph and decompose it: the gradient part is your global ranking, and the leftover curl is a certificate of how intransitive your data truly is.
- **Inputs / Output** — skew-symmetric edge data $Y_{ij}$ (mean margin, log-odds, net preference) → potential/score per node **plus** a decomposition diagnosing inconsistency (local cycles vs. global cycles).
- **Class** — Algebraic (combinatorial Hodge theory / least squares) × Static.
- **Model & assumptions** — find node potentials $s$ minimizing $\sum_{ij} w_{ij}\,(s_i - s_j - Y_{ij})^2$ — a graph least-squares/Poisson problem [Hirani, Kalyanaraman & Watts 2011]. The combinatorial Hodge decomposition splits $Y$ into **gradient** (consistent, explained by $s$) ⊕ **curl** (triangle-level cycles) ⊕ **harmonic** (long-range cycles), so you obtain both the optimal $L_2$ ranking *and* a quantitative inconsistency budget [Jiang, Lim, Yao & Ye 2011].
- **Estimation & complexity** — one sparse linear solve (graph Laplacian system): conjugate gradient / multigrid, near-linear in edges in practice.
- **Handles** — margins ✓ (native — any skew-symmetric statistic) · weights ✓ · ties ✓ (zero flows) · uncertainty △ (residual-based) · intransitivity ✓ **as a diagnostic** (it doesn't model cycles, it *measures* them).
- **Pros** — the only family that tells you *how rankable your data is* before you trust any ranking of it; margin-native; one linear solve; imbalance/incompleteness handled by design.
- **Cons** — $L_2$ on transformed preferences rather than a generative model: no calibrated win probabilities; choice of edge statistic $Y$ matters; harmonic component is hard to explain to stakeholders.
- **Use cases** — crowdsourced quality assessment (its original home: video QoE); detecting whether an LLM-arena dataset is even consistently rankable; sports with margins; any "rank + audit the rankability" need.
- **Relationships** — Massey's method (§5.1) is exactly this least-squares problem with point differentials as $Y$; SVD/low-rank cousins in §9 model the cycles instead of measuring them.
- **Propagon status** — **candidate — recommended** (a sparse Laplacian solve; high differentiating value: nobody's CLI ships an "inconsistency audit").
- **References** — [Jiang, Lim, Yao & Ye 2011; Hirani, Kalyanaraman & Watts 2011].

---

## 4. Node Importance & Centrality (Graph Ranking)

Here the preference signal is **structural**: a hyperlink, citation, follow, retweet, purchase, or trust assertion is an implicit endorsement of its target. No one ever said "X beats Y" — yet links are choices someone made, and ranking by importance is the natural aggregate of those choices. This family answers *"which nodes matter?"* rather than *"who would win?"*, so its outputs are importance scores, not calibrated matchup predictions. Entries below replace the `Handles` line with graph capabilities: **directed / weighted / bipartite / dangling-safe / locality** (local = decided by a node's neighborhood; global = by the whole graph). For rigorous guidance on choosing among these, see the axiomatic analysis in §4.10.

### 4.1 Degree & Strength — compact baseline

**Class: Non-parametric (counting) × Static. Locality: local.** In-degree (or weighted in-strength) is the centrality every other method is implicitly regularizing: citations counts, follower counts, raw win counts. $O(\text{edges})$ total, trivially interpretable, shockingly competitive on many tasks — and the exact graph analogue of win-rate ranking (§7.1), with the same flaw: it values *quantity* of endorsements, ignoring *who* endorses. Always compute it as the baseline before anything fancier. *Propagon: trivial candidate (one-liner over the edge list).* [Freeman 1979 for the classical framing].

### 4.2 Eigenvector Centrality (1972)

- **TL;DR** — You matter if those who endorse you matter: importance as the fixed point of mutual endorsement — the Perron eigenvector of the adjacency matrix.
- **Inputs / Output** — (weighted) graph → nonnegative importance per node.
- **Class** — Algebraic/Spectral × Static. **Graph**: directed △ (see cons) / weighted ✓ / dangling-safe ✗ / locality: global.
- **Model & assumptions** — $\lambda x = A^\top x$, take the leading eigenvector [Bonacich 1972]. Well-defined and positive on strongly connected (irreducible) graphs by Perron-Frobenius.
- **Estimation & complexity** — power iteration, $O(\text{edges})$ per step.
- **Pros / Cons** — the purest "recursive endorsement" formalism; parameter-free. But on directed graphs it pathologically zeroes out nodes outside the dominant strongly-connected component's upstream (a node with no in-links *contributes* nothing and scores nothing — acyclic graphs collapse entirely). Katz and PageRank exist precisely to fix this.
- **Use cases** — undirected/strongly-connected networks: co-authorship, correlation graphs, friendship networks.
- **Relationships** — Keener (§3.3) is this on a match-score matrix; Katz (§4.3) = damped fix; PageRank (§4.4) = stochastic-normalized fix.
- **Propagon status** — candidate (power iteration exists in `pr.rs`).
- **References** — [Bonacich 1972; Bonacich 1987].

### 4.3 Katz Centrality (1953)

- **TL;DR** — Count *all* walks that reach you, discounting longer ones geometrically — endorsement with exponentially decaying influence.
- **Inputs / Output** — (weighted, directed) graph → importance per node.
- **Class** — Algebraic/Spectral × Static. **Graph**: directed ✓ / weighted ✓ / dangling-safe ✓ / locality: global (tunably local via $\alpha$).
- **Model & assumptions** — $x = \sum_{k\ge1} \alpha^k (A^\top)^k \mathbf{1} = ((I - \alpha A^\top)^{-1} - I)\,\mathbf{1}$, convergent for $\alpha < 1/\lambda_{\max}$ [Katz 1953].
- **Estimation & complexity** — sparse linear solve or truncated walk sums; $O(\text{edges})$ per iteration.
- **Pros / Cons** — works where eigenvector centrality degenerates (DAGs, weakly connected digraphs); $\alpha$ interpolates between degree ($\alpha \to 0$) and eigenvector ($\alpha \to 1/\lambda_{\max}$). But $\alpha$ must be chosen, and high-degree hubs still dominate without out-degree normalization — which is precisely PageRank's edit.
- **Use cases** — citation and influence networks; status in sociometric data (its origin); feature for link prediction.
- **Relationships** — degree ⊂ Katz ⊂ eigenvector (limits); PageRank = Katz with column-stochastic normalization + teleportation.
- **Propagon status** — candidate.
- **References** — [Katz 1953].

### 4.4 PageRank & Personalized PageRank (1998)

- **TL;DR** — The random-surfer fixed point: follow a random out-link with probability $d$, teleport anywhere with $1-d$; stationary visit share = importance. Endorsements are *split* among everything you endorse.
- **Inputs / Output** — directed (weighted) graph → stationary probability per node.
- **Class** — Algebraic/Spectral × Static. **Graph**: directed ✓ / weighted ✓ / dangling △ (needs a policy — see below) / locality: global (personalized → local).
- **Model & assumptions** —

  $$p = d\,P^\top p + (1 - d)\,v$$

  with $P$ the out-degree-normalized transition matrix, damping $d \approx 0.85$, and teleport distribution $v$ (uniform = classic; concentrated = **personalized PageRank**, importance *relative to a seed set*) [Brin & Page 1998]. Teleportation guarantees irreducibility — this is eigenvector centrality made well-posed on arbitrary digraphs. Dangling nodes (no out-links) need a redistribution policy: propagon's `--sink-dispersion` offers `reverse` (back along in-edges), `all` (uniform), or `none`.
- **Estimation & complexity** — power iteration, $O(\text{edges})$ per step, ~50–100 steps at $d = 0.85$; massive-scale implementations are routine. Personalized variants admit fast local push algorithms.
- **Pros** — robust on real, messy directed graphs; spam-resistant relative to raw degree (endorsement mass is conserved and split); one tunable; the personalized variant turns it into a similarity/recommendation engine.
- **Cons** — topic drift toward dense old communities; $d$ matters; scores are graph-global (adding nodes shifts everyone); famously gameable at web scale (→ TrustRank, §4.8); no uncertainty.
- **Use cases** — web/citation/social importance; "rank items by interaction graph" in recsys (via PPR); also directly usable on *match graphs* (loser→winner edges), where it behaves as a margin-blind cousin of Rank Centrality (§3.1).
- **Relationships** — normalized-and-teleported Katz; the template for TrustRank, BiRank, LeaderRank; Rank Centrality is its statistically-grounded sibling for comparison data.
- **Propagon status** — **implemented**: `src/pr.rs` (`page-rank`, with damping and three sink policies).
- **References** — [Brin & Page 1998; Langville & Meyer 2006].

### 4.5 HITS — Hubs & Authorities (1999)

- **TL;DR** — Two mutually-defined scores: good **authorities** are pointed to by good **hubs**, and good hubs point to good authorities.
- **Inputs / Output** — directed graph (classically a query-focused subgraph) → two scores per node.
- **Class** — Algebraic/Spectral × Static. **Graph**: directed ✓ / weighted ✓ / dangling-safe ✓ / locality: global on the subgraph.
- **Model & assumptions** — iterate $a \leftarrow A^\top h$, $h \leftarrow A a$ with normalization; converges to the leading eigenvectors of $A^\top A$ and $A A^\top$ [Kleinberg 1999].
- **Estimation & complexity** — power iteration on the two coupled systems.
- **Pros / Cons** — the hub/authority split is genuinely informative on curation-shaped graphs (link directories, review aggregators, retweet networks: curators vs. sources). But: query-dependent in its original design; vulnerable to tightly-knit communities capturing the principal eigenvector (the TKC effect — SALSA's motivation); two scores complicate downstream use.
- **Use cases** — separating curators from content; citation networks (review papers vs. primary results); bipartite-ish endorsement structures.
- **Relationships** — SALSA = its random-walk normalization; CoHITS/BiRank carry the alternating idea to explicitly bipartite graphs.
- **Propagon status** — candidate.
- **References** — [Kleinberg 1999].

### 4.6 SALSA (2001) — compact

**Class: Algebraic/Spectral × Static.** HITS recomputed with stochastic matrices: alternate a backward and forward random-walk step, equivalent to independent walks on two bipartite projections. Authority scores reduce (on connected components) to in-degree weighted by component mass — which both explains SALSA's robustness to the tightly-knit-community capture that plagues HITS and reveals how degree-like it secretly is [Lempel & Moran 2001]. Historically important in web IR and link-recommendation systems (notably as the algorithmic basis of Twitter's who-to-follow GraphJet lineage). *Propagon: out-of-scope (BiRank covers the niche better for our data shapes).* [Lempel & Moran 2001].

### 4.7 Bipartite Ranking — BiRank (2017) & CoHITS (2009)

- **TL;DR** — Rank two node types that endorse each other (users↔items, reviewers↔products) by alternating smoothed propagation across the bipartite structure — interaction logs in, co-ranked sides out.
- **Inputs / Output** — weighted bipartite graph (e.g., user-item interactions, possibly with priors per side) → importance score per node on both sides.
- **Class** — Algebraic/Spectral × Static. **Graph**: bipartite ✓ (its purpose) / weighted ✓ / priors ✓ / locality: global.
- **Model & assumptions** — BiRank iterates symmetrically-normalized propagation $u \leftarrow \alpha\, S^\top v + (1-\alpha)\, u^0$, $v \leftarrow \beta\, S u + (1-\beta)\, v^0$ (with $S$ the degree-symmetric-normalized interaction matrix and $u^0, v^0$ query/prior vectors); converges to the optimum of a smoothness-plus-prior regularization objective [He, Gao, Kan & Wang 2017]. CoHITS is the earlier generalization unifying HITS-style co-ranking with personalization on bipartite graphs [Deng, Lyu & King 2009].
- **Estimation & complexity** — alternating sparse mat-vecs; $O(\text{interactions})$ per sweep; provable convergence (contraction).
- **Pros** — purpose-built for the most common revealed-preference data in industry (interaction matrices); symmetric normalization tames hub dominance; priors let you inject business signals; both sides ranked at once.
- **Cons** — scores are popularity-flavored importance, not preference strengths (no matchup semantics); $\alpha, \beta$ tuning; cold-start nodes need priors.
- **Use cases** — ranking items *and* users from purchase/click/rating logs; venue↔author co-ranking in bibliometrics; query↔document co-ranking.
- **Relationships** — bipartite descendant of HITS/SALSA with PageRank-style smoothing; complements §10.2's click models, which de-bias the same logs at the event level rather than the graph level.
- **Propagon status** — **implemented**: `src/birank.rs` (`birank`).
- **References** — [He, Gao, Kan & Wang 2017; Deng, Lyu & King 2009].

### 4.8 TrustRank (2004) — compact

**Class: Algebraic/Spectral × Static.** Personalized PageRank with the teleport vector restricted to a **manually vetted seed set** of trustworthy nodes: trust flows out from the seeds with damping, so spam farms unreachable from good neighborhoods score near zero [Gyöngyi, Garcia-Molina & Pedersen 2004]. The general pattern — *seeded propagation as adversarial robustness* — transfers to any endorsement graph with manipulation (fake-review rings, bot followings, citation cartels). Inverse/distrust variants propagate badness backward. *Propagon: covered by personalized teleport on a future PPR; flag as a documented recipe rather than a separate command.* [Gyöngyi, Garcia-Molina & Pedersen 2004].

### 4.9 LeaderRank (2011) — compact

**Class: Algebraic/Spectral × Static.** PageRank's teleportation replaced by a **ground node** bidirectionally linked to every node: the walk is automatically irreducible with *zero* parameters (no damping to tune), and the ground node adapts teleport mass to local degree. Reported to outrank PageRank in identifying influential spreaders and in robustness to noise/spam on social networks [Lü, Zhang, Yeung & Zhou 2011]. A small, cheap, parameter-free alternative worth benchmarking wherever PageRank is used. *Propagon: trivial variant of `pr.rs`; candidate.* [Lü, Zhang, Yeung & Zhou 2011].

### 4.10 Geodesic & Flow Centralities — compact group

**Class: Algebraic × Static. Locality: global.** Importance as *position* rather than endorsement — farther from the revealed-preference story, included for completeness and because they answer questions the eigenvector family cannot:

- **Closeness** — inverse mean shortest-path distance to all others; "who can reach everyone fastest." Breaks on disconnected graphs. [Bavelas 1950; Freeman 1979].
- **Harmonic centrality** — sum of inverse distances, $\sum_{j \ne i} 1/d(i,j)$: closeness fixed for disconnected graphs ($1/\infty = 0$), and the *only* measure satisfying all of Boldi-Vigna's axioms (see below). [Marchiori & Latora 2000; Boldi & Vigna 2014].
- **Betweenness** — share of all shortest paths passing through a node; "brokers and bottlenecks." Exact computation $O(VE)$ by Brandes' algorithm; sampling approximations for big graphs. Ranks *control* over flow, not endorsement. [Freeman 1977; Brandes 2001].
- **Current-flow / random-walk betweenness** — betweenness with electrical current (all paths, not just shortest) — better when flow doesn't know the shortest path; cubic-ish cost limits scale. [Newman 2005].
- **Subgraph/communicability centrality** — weighted count of closed walks through a node ($e^A$ diagonal); sensitive to local cliquishness. [Estrada & Rodríguez-Velázquez 2005].

**Choosing among centralities:** [Boldi & Vigna 2014] axiomatize (size, density, score monotonicity) and find harmonic centrality uniquely compliant — a rigorous default for "generic importance." But fitness-for-purpose beats axioms: endorsement questions → §4.2–4.9; reach → closeness/harmonic; brokerage → betweenness. *Propagon: out-of-scope except possibly harmonic (cheap, axiom-clean); the library's identity is endorsement-flavored ranking.*

### 4.11 k-Core Decomposition & Influential Spreaders (1983/2010) — compact

**Class: Non-parametric (combinatorial) × Static.** Iteratively strip nodes of degree < $k$; a node's **coreness** is the deepest shell it survives to [Seidman 1983]. Coreness — not degree or betweenness — best predicts spreading power in epidemic-style processes on real networks [Kitsak et al. 2010]: being embedded in a dense core beats having many fragile spokes. $O(\text{edges})$ total via bucket sort. Coarse (many ties) but ultra-cheap and robust; good as a *filter* (rank within the top cores by something finer). *Propagon: candidate utility (pairs naturally with the existing `extract-components`).* [Seidman 1983; Kitsak et al. 2010].

---

## 5. Least-Squares & Linear-Algebra Sports Ratings

Ratings as the solution of one linear system built from season results. Less statistical machinery than §1, more structure than counting — historically the backbone of (pre-playoff-era) American college sports rankings, and unreasonably useful anywhere margins exist. The umbrella reference for this whole section is [Langville & Meyer 2012].

### 5.1 Massey Ratings (1997)

- **TL;DR** — Ratings such that rating differences best predict observed margins, in the least-squares sense.
- **Inputs / Output** — pairwise results **with margins** → real ratings (mean-zero).
- **Class** — Algebraic (least squares) × Static.
- **Model & assumptions** — for each game, $r_{w} - r_{l} \approx \text{margin}$; stack into $X r = y$ and solve the normal equations $M r = p$, where $M$ is the graph Laplacian of the schedule ($M_{ii}$ = games played, $M_{ij} = -$games between $i,j$) and $p$ is net point differential. $M$ is singular (scores identified up to a constant); fix by replacing one row with $\sum r_i = 0$ [Massey 1997]. An offense/defense split falls out of a natural decomposition.
- **Estimation & complexity** — one sparse SPD solve; trivial at any sane league size, near-linear with CG at graph scale.
- **Handles** — margins ✓ (required) · ties ✓ (margin 0) · home-adv ✓ (add an intercept column) · uncertainty △ (regression SEs, caveat dependence) · dynamics ✗.
- **Pros** — dead simple; margin-efficient; strength-of-schedule automatic (it's a Laplacian system — your rating depends on opponents' ratings); the same machinery as HodgeRank's gradient component (§3.6), so it inherits that interpretation.
- **Cons** — Gaussian-margin assumption rewards blowouts unless margins are capped/transformed; win/loss only data degrades it to near-uselessness (use Colley); no probabilistic outputs without further assumptions.
- **Use cases** — any margin-bearing round-robin-ish competition; quick strength-of-schedule-adjusted league tables; baseline for fancier models.
- **Relationships** — special case of least-squares ranking on graphs (§3.6) with $Y$ = mean margin; Colley is its win-rate-only sibling; Elo-with-MOV is its online cousin.
- **Propagon status** — **candidate — recommended** (one sparse solve; pairs perfectly with the existing `.edges` weighted format).
- **References** — [Massey 1997; Langville & Meyer 2012].

### 5.2 Colley Matrix Method (2002)

- **TL;DR** — Laplace-smoothed win rates made schedule-aware: solve one linear system where your effective wins are adjusted by opponents' ratings — margins and history deliberately excluded.
- **Inputs / Output** — win/loss records only → ratings centered at $\tfrac12$.
- **Class** — Algebraic (least squares) × Static.
- **Model & assumptions** — start from the Laplace estimator $(1 + w_i)/(2 + n_i)$ and replace the prior half-games with opponents' actual ratings, yielding the SPD system $C r = b$ with $C = 2I + \text{Laplacian}$, $b_i = 1 + \tfrac{w_i - l_i}{2}$ [Colley 2002]. "Bias-free" by construction: no margins, no venue, no preseason prior.
- **Estimation & complexity** — one well-conditioned sparse SPD solve.
- **Pros / Cons** — fully deterministic and audit-friendly (it served in the BCS college-football formula); immune to blowout-chasing; the smoothing handles undefeated teams gracefully — but it throws away margin signal on purpose, has no probabilistic semantics, and underperforms BT/Massey on predictive tasks.
- **Use cases** — rankings that must be *defensible* and manipulation-resistant more than predictive (official standings, seeding).
- **Relationships** — Laplace-smoothed counting (§7.1) upgraded with schedule structure; the win-only counterpart of Massey.
- **Propagon status** — candidate (same solver as Massey; ship them together).
- **References** — [Colley 2002; Langville & Meyer 2012].

### 5.3 Offense-Defense Ratings (2009) — compact

**Class: Algebraic (matrix balancing) × Static.** Each team gets an offensive rating $o_i$ and defensive rating $d_i$: points scored by $i$ on $j$ should look like $o_i \cdot d_j$. Alternating updates ($o \leftarrow$ scores weighted by opponents' defense, $d \leftarrow$ allowed scores weighted by opponents' offense) are exactly **Sinkhorn-Knopp matrix balancing**, with the convergence theory that entails [Govan, Langville & Meyer 2009]. Gives *why*-decomposed ratings (great offense vs. great defense) that single-scalar methods can't, and previews the Sinkhorn machinery reappearing in differentiable ranking (§10.5). Aggregate rating = $o_i / d_i$ or similar. *Propagon: candidate (niche but cheap; fun output).* [Govan, Langville & Meyer 2009].

---

## 6. Rank Aggregation & Social Choice

Inputs here are **multiple rankings** (or a preference matrix distilled from them): voters' ballots, judges' orderings, search engines' result lists, per-task model leaderboards. The question shifts from "estimate latent strength" to "find the fairest consensus order" — and 250 years of voting theory (and its impossibility results) apply. These methods are predominantly **non-parametric**: they manipulate orders, not scores.

### 6.1 Borda Count (1781)

- **TL;DR** — Score each item by its average position across rankings (or its total wins across pairwise data); sort.
- **Inputs / Output** — full/partial rankings, or pairwise outcomes → positional scores → order.
- **Class** — Non-parametric (counting/positional) × Static.
- **Model & assumptions** — none. From rankings: item gets $m - \text{rank}$ points per ballot. From pairwise data: Borda score = win fraction against sampled opponents (this identification matters: it makes Borda the canonical *counting estimator*, §7.2).
- **Estimation & complexity** — $O(\text{data})$. Unbeatable.
- **Pros / Cons** — instant, transparent, and (from pairwise data, under uniform sampling) provably near-optimal for rank recovery over huge model classes [Shah & Wainwright 2018]. But: positional scores depend on slate composition (clone/irrelevant-alternative sensitivity); a Condorcet winner can lose Borda; with non-uniform comparison schedules it inherits schedule bias with no correction (BT's whole advantage).
- **Use cases** — first-cut consensus from heterogeneous rankers; metasearch; ensembling model output rankings; the always-on baseline.
- **Relationships** — = win-rate ranking on pairwise data (§7.1); the positional member of the social-choice canon vs. Copeland/Kemeny's pairwise-majority members.
- **Propagon status** — trivial candidate.
- **References** — [de Borda 1781; Shah & Wainwright 2018].

### 6.2 Copeland Method — compact

**Class: Non-parametric (pairwise majority) × Static.** Score = (pairwise majorities won) − (lost), over all opponents; sort. Condorcet-consistent (a Condorcet winner — beats everyone head-to-head — always tops the list), cheap ($O(n^2)$ majorities), and the natural "wins against the field" statistic. Coarse near the middle of the table (many tied Copeland scores) and needs most pairs observed. The dueling-bandit literature adopts it as a target when no Condorcet winner exists (§8.1). *Propagon: trivial candidate alongside Borda.* [Copeland 1951; Saari & Merlin 1996].

### 6.3 Kemeny-Young Optimal Consensus (1959)

- **TL;DR** — The consensus ranking minimizing total pairwise disagreement with the input rankings — the unique "right answer" under several axiom sets, and NP-hard to compute exactly.
- **Inputs / Output** — multiple rankings or a pairwise preference matrix → one total order (+ optimal-disagreement value).
- **Class** — Non-parametric (combinatorial optimization) × Static.
- **Model & assumptions** — find $\sigma^\* = \arg\min_\sigma \sum_v d_{K}(\sigma, \sigma_v)$ with $d_K$ = Kendall tau distance [Kemeny 1959]. Three load-bearing facts:
  1. **It's the MLE**: under the Condorcet/Mallows noise model (each voter is a noisy transposition-corrupting observation of one truth), Kemeny's rule is exactly maximum likelihood [Young 1988].
  2. **It's axiomatically singled out**: the unique rule that is neutral, consistent, and Condorcet [Young & Levenglick 1978].
  3. **It's NP-hard**, even with only four input rankings [Dwork, Kumar, Naor & Sivakumar 2001; Bartholdi, Tovey & Trick 1989].
- **Estimation & complexity** — exact: ILP/branch-and-bound to a few hundred items. Approximate: any positional start + **local Kemenization** (adjacent-swap descent) [Dwork et al. 2001]; pivot algorithms with constant-factor guarantees [Ailon, Charikar & Newman 2008]; a PTAS exists [Kenyon-Mathieu & Schudy 2007]. Propagon ships two heuristics: greedy insertion passes and differential-evolution search over orderings (`kemeny --algo insertion|de`).
- **Handles** — partial rankings △ (extensions exist) · ties in inputs △ · weights ✓ (vote multiplicity) · uncertainty ✗.
- **Pros** — the principled consensus: MLE + axiomatics in one object; immune to score-scale issues entirely (pure order manipulation).
- **Cons** — NP-hard; heuristics give no certificate of optimality (propagon reports the achieved objective only); no strengths/probabilities; needs reasonably complete preference matrices to be meaningful.
- **Use cases** — aggregating judge panels; combining heterogeneous benchmark leaderboards into one (§12.6); metasearch; biology (gene-list integration).
- **Relationships** — MLE of Mallows (§1.7); Borda and footrule (§6.5) are its polynomial-time approximation anchors; Slater's rule is the single-matrix analogue.
- **Propagon status** — **implemented** (heuristically): `src/kemeny.rs` + `src/de.rs`.
- **References** — [Kemeny 1959; Young & Levenglick 1978; Young 1988; Bartholdi, Tovey & Trick 1989; Dwork et al. 2001; Ailon, Charikar & Newman 2008; Kenyon-Mathieu & Schudy 2007].

### 6.4 Markov-Chain Rank Aggregation — MC1–MC4 (2001)

- **TL;DR** — Turn the input rankings into a random walk over items ("from item $i$, move to items commonly ranked above $i$"); rank by stationary distribution — rank aggregation borrowed PageRank's trick.
- **Inputs / Output** — multiple (partial) rankings → stationary scores → order.
- **Class** — Non-parametric (spectral over orders) × Static.
- **Model & assumptions** — four transition constructions of increasing majority-faithfulness, MC1–MC4; MC4 (move to $j$ if a majority of input lists rank $j$ above $i$) behaves best empirically [Dwork, Kumar, Naor & Sivakumar 2001].
- **Estimation & complexity** — power iteration; handles *partial* lists natively (a list only constrains items it contains) — the original motivation (metasearch over engines that each rank only some of the web).
- **Pros / Cons** — graceful with partial, overlapping, differently-sized input lists — exactly where Kemeny and Borda are awkward; spam-resistant in the metasearch setting; but heuristic (no MLE/axiomatic standing), and stationary scores have no preference-strength semantics.
- **Use cases** — metasearch; merging top-$k$ lists from many recommenders/benchmarks; any partial-list aggregation.
- **Relationships** — PageRank's machinery (§4.4) applied to ballots; Rank Centrality (§3.1) is the same idea with statistical guarantees for *comparison* data.
- **Propagon status** — candidate (shares all machinery with `pr.rs`/`lsr.rs`).
- **References** — [Dwork, Kumar, Naor & Sivakumar 2001].

### 6.5 Footrule / Spearman Aggregation — compact

**Class: Non-parametric × Static.** Minimize total **Spearman footrule** distance (sum of absolute rank displacements) instead of Kendall distance: unlike Kemeny, this is solvable **exactly in polynomial time** as a minimum-cost bipartite matching (items × positions, cost = total displacement). Because footrule and Kendall distances are within a factor of two of each other [Diaconis & Graham 1977], the footrule-optimal ranking is a 2-approximation to Kemeny — the standard polynomial-time anchor [Dwork et al. 2001]. *Propagon: candidate (gives `kemeny` a quality-guaranteed initialization).* [Diaconis & Graham 1977; Dwork et al. 2001].

### 6.6 Condorcet-Completion Methods: Schulze & Ranked Pairs — compact

**Class: Non-parametric (pairwise majority) × Static.** When a Condorcet winner doesn't exist, complete the majority relation as conservatively as possible. **Ranked pairs** [Tideman 1987]: lock in majorities from largest to smallest, skipping any that would create a cycle. **Schulze** [Schulze 2011]: rank by strongest beatpaths (widest-bottleneck paths in the majority graph). Both are Condorcet-consistent, clone-independent, and monotone — properties Borda and instant-runoff lack — and dominate modern organizational elections (Debian, Wikimedia). For data-science aggregation they're heavier than needed; their value to this library is as *references for fairness properties* (§12.6 applies them to LLM leaderboards). *Propagon: out-of-scope.* [Tideman 1987; Schulze 2011].

### 6.7 Social Choice for Agent Leaderboards (2023) — compact

**Class: Non-parametric × Static.** Treat each benchmark task as a *voter* casting a ranking over models/agents, then apply voting theory ("Voting-as-Evaluation"): Condorcet-consistent methods yield leaderboards far more robust to task duplication, task selection, and irrelevant alternatives than mean-score or Elo-style aggregation [Lanctot et al. 2023]. The cleanest available argument that **§6 belongs in a model-evaluation toolbox**, not just in election theory. See §12.6 for the arena context. *Propagon: enabled by the Borda/Copeland/Kemeny commands.* [Lanctot et al. 2023].

---

## 7. Non-Parametric & Robust Estimators

The "assume almost nothing" toolkit: count, smooth, bound. These methods trade away calibrated matchup prediction for robustness over model classes vastly larger than BT/Thurstone — and per §0.1, the surprise of the modern theory is how little they give up for it.

### 7.1 Win Rates with Wilson-Score Intervals (1927)

- **TL;DR** — Rank by a *confidence bound* on win rate instead of the raw rate, so a 2-0 newcomer stops outranking a 95-5 veteran.
- **Inputs / Output** — per-entity win/loss tallies → point estimate or interval bound per entity.
- **Class** — Non-parametric (counting + binomial inference) × Static.
- **Model & assumptions** — i.i.d. Bernoulli wins per entity (the big lie: it ignores *who* the opponents were). The Wilson score interval inverts the normal test of a binomial proportion:

  $$\frac{\hat p + \tfrac{z^2}{2n} \pm z \sqrt{\tfrac{\hat p (1 - \hat p)}{n} + \tfrac{z^2}{4 n^2}}}{1 + \tfrac{z^2}{n}}$$

  Ranking by the **lower bound** is the classic fix for small-sample inflation [Wilson 1927]; it behaves far better than the Wald interval at extreme $\hat p$ and small $n$ [Agresti & Coull 1998]. Bayesian Beta-Binomial smoothing is the same medicine in different packaging.
- **Estimation & complexity** — closed form, $O(\text{entities})$.
- **Handles** — uncertainty ✓ (the point of it) · everything else ✗ — *no opponent adjustment whatsoever*.
- **Pros** — instant, robust, explainable in one sentence; the correct way to sort user ratings, store reviews, and anything else with heterogeneous sample sizes.
- **Cons** — schedule-blind: in any setting with non-uniform matchmaking it is *biased*, not just noisy (farming weak opponents works). Use it as the baseline that schedule-aware methods must beat.
- **Use cases** — review/upvote sorting; balanced round-robins; smoke-testing data pipelines before real rankers.
- **Relationships** — degree centrality's tournament twin (§4.1); Borda counting (§6.1/§7.2) is the schedule-aware repair; Colley (§5.2) the structural one.
- **Propagon status** — **implemented**: `src/rate.rs` (`rate`, P50/P90/P95 bounds).
- **References** — [Wilson 1927; Agresti & Coull 1998].

### 7.2 Counting (Borda) Estimators — Simple, Robust, Optimal (2018)

- **TL;DR** — Under uniform random comparisons, ranking by raw win counts is minimax-optimal over the entire stochastically-transitive model class — the theoretical license to keep things simple.
- **Inputs / Output** — pairwise outcomes (≈uniform schedule) → win-fraction scores → order.
- **Class** — Non-parametric (counting) × Static.
- **Model & assumptions** — only SST (§0.1). The estimator is just each item's win fraction; the theory shows it attains, up to constants, the minimax rate for recovering the true order — *and remains optimal even when the data genuinely follows BT*, i.e., the parametric structure buys essentially nothing for rank recovery [Shah & Wainwright 2018].
- **Estimation & complexity** — $O(\text{comparisons})$. The fastest thing in this document.
- **Pros / Cons** — unbeatable simplicity-to-guarantee ratio; robust to every link-function misspecification. The catch is the *schedule assumption*: with adversarial or organic (non-uniform) matchmaking the guarantee evaporates, and BT-family models — which condition on the schedule — regain their advantage. No win-probability predictions, no uncertainty without bootstrap.
- **Use cases** — designed-experiment comparisons (you control the schedule: crowdsourcing with random pairing, A/B/n harnesses); sanity-checking parametric fits.
- **Relationships** — Borda count (§6.1) viewed as an estimator; the anchor result for §0.1's axis.
- **References** — [Shah & Wainwright 2018].

### 7.3 Stochastically Transitive Models (2016/2017) — compact

**Class: Non-parametric × Static.** Drop the link function entirely: assume only SST (the $n \times n$ probability matrix $M_{ij} = P(i \succ j)$ respects some total order monotonically) and estimate $M$ itself. The full matrix is estimable at rates dramatically better than parametric skeptics expected, but a **computational-statistical gap** appears: the minimax-optimal estimator is computationally hard, while efficient (SVD-thresholding) estimators lose a polynomial factor [Shah, Balakrishnan, Guntuboyina & Wainwright 2017]. Practically: this is the honest model class when you suspect BT is misspecified but transitivity still holds; estimate with counting + isotonic-regression smoothing. *Propagon: out-of-scope (research-grade); the counting estimator (§7.2) is its practical face.* [Shah et al. 2017].

### 7.4 Noisy Sorting (2008) — compact

**Class: Non-parametric × Static (active-adjacent).** Assume every comparison reports the true order with probability $\ge \tfrac12 + \gamma$ (constant noise margin). Then $O(n \log n)$ comparisons suffice to recover the exact order with high probability, via noisy binary-insertion machinery — matching noiseless sorting's query complexity up to constants [Braverman & Mossel 2008]. The information-theoretic floor for the whole field: any method demanding $\omega(n \log n)$ comparisons for a *total order under bounded noise* is leaving efficiency on the table. Modern refinements pin the exact constants. Bridges directly into §8 (it is, in effect, a fixed-policy active method). *Propagon: out-of-scope (theory anchor).* [Braverman & Mossel 2008].

---

## 8. Active Ranking & Dueling Bandits

Everything so far ranks a *given* dataset. This family chooses **which comparison to ask for next** — the right framing when comparisons cost money (crowdworkers), latency (live traffic), or user goodwill. Two cultures: bandit-style regret minimization (you pay for bad comparisons as you go) and pure-exploration/budget settings (you pay per query, then must answer).

### 8.1 Dueling Bandits (2009–2012)

- **TL;DR** — Multi-armed bandits where the only feedback is "which of these two was preferred" — find the best item (or ranking) from relative feedback while not embarrassing yourself en route.
- **Inputs / Output** — sequential choice of pairs + noisy preference feedback → best arm / ranking, with regret guarantees.
- **Class** — Non-parametric or semi-parametric × Online (active).
- **Model & assumptions** — unknown preference matrix $P(i \succ j)$; target is typically the **Condorcet winner** (beats every other arm in expectation), or — when none exists — Copeland/Borda/von Neumann winners. The interleaved-comparison formulation and the first regret analyses come from information retrieval (comparing search rankers via click preferences) [Yue, Broder, Kleinberg & Joachims 2012]. Algorithms span Interleaved Filter, RUCB, Copeland confidence bounds, Double Thompson Sampling; the survey of record is [Bengs, Busa-Fekete, El Mesaoudi-Paul & Hüllermeier 2021].
- **Estimation & complexity** — per-round index computations over the pairwise statistics; regret typically $O(K \log T)$ under a Condorcet winner.
- **Handles** — uncertainty ✓ (by construction) · intransitivity △ (Copeland/Borda targets) · the *schedule* is the algorithm's output, neatly sidestepping §7.2's caveat.
- **Pros** — query-efficient by design; principled stopping; the natural harness for online evaluation against live preferences.
- **Cons** — sequential infrastructure required (real-time assignment); analyses lean on stationarity; most algorithms target the *single best* item — full-ranking variants are costlier.
- **Use cases** — online evaluation of search rankers/recommenders via interleaving; live A/B/n with preference feedback; budgeted crowdsourcing for "find the best."
- **Relationships** — bandit counterpart of §1's estimation; degenerate full-feedback case is §7; preference-based RL generalizes it to trajectories (→ §13 and RLHF, §10.6).
- **Propagon status** — out-of-scope as an online service; a *simulator/policy library* over logged data is a plausible future direction.
- **References** — [Yue et al. 2012; Bengs et al. 2021].

### 8.2 Active Ranking from Pairwise Comparisons (2011) — compact

**Class: Semi-parametric (geometric) × Active.** If items embed in $\mathbb{R}^d$ and preferences follow distance to a reference point, adaptive query selection recovers the full ranking with $O(d \log n)$ comparisons instead of $\binom{n}{2}$ — exponential savings when the latent dimension is small, with robust variants under noise [Jamieson & Nowak 2011]. The geometric assumption is strong; the result matters as the template for "structure ⇒ logarithmic query complexity." *Propagon: out-of-scope.* [Jamieson & Nowak 2011].

### 8.3 Just Sort It — Quicksort as Active Ranking (2017) — compact

**Class: Semi-parametric × Active.** Run plain quicksort with noisy comparisons; under BT-type noise whose strength parameters are well-spread, a *single run's* output ranking is near-optimal, and aggregating a handful of runs estimates PL parameters at budget $O(n \log n)$ — matching specialized active-ranking machinery with an algorithm every engineer already knows [Maystre & Grossglauser 2017]. Excellent practical recipe for comparison-budgeted crowdsourcing: sort once, maybe thrice, then fit §1.4/§3.2 on the collected comparisons. *Propagon: documentation recipe (pairs with `lsr`).* [Maystre & Grossglauser 2017].

### 8.4 Active Top-k: When Parametric Assumptions Do Not Help (2019) — compact

**Class: theory anchor × Active.** For adaptively identifying the top-$k$ items (or full ranking), the sample complexity is governed by pairwise-probability gaps $|P(i \succ j) - \tfrac12|$ — and imposing parametric (BT-type) structure improves the worst-case budget by **at most logarithmic factors** over assuming nothing [Heckel, Shah, Ramchandran & Wainwright 2019]. Together with §7.2 this completes the argument of §0.1: parametric models earn their keep through *prediction, covariates, uncertainty, and dynamics* — not through fundamentally cheaper rank identification. *Propagon: out-of-scope (theory anchor).* [Heckel et al. 2019].

---

## 9. Intransitivity & Multidimensional Skill

Everything in §§1–8 (except HodgeRank's diagnostics) presumes a one-dimensional truth. Real competition is often cyclic: rock-paper-scissors dynamics in fighting games and metas, style matchups in sports ("team A always beats B, B beats C, C beats A"), strategy webs in multi-agent benchmarks. Forcing a scalar rating onto cyclic data doesn't just lose accuracy — it produces *confidently wrong* matchup predictions. This family models the cycles.

### 9.1 Blade-Chest Model (2016)

- **TL;DR** — Give each player two vectors — a "blade" (how it attacks) and a "chest" (where it's vulnerable): win probability depends on how your blade aligns with their chest, capturing rock-paper-scissors exactly.
- **Inputs / Output** — pairwise outcomes → per-entity embedding vectors (+ optional scalar) → matchup predictions.
- **Class** — Parametric (latent-embedding) × Static.
- **Model & assumptions** — matchup score $M_{ij} = b_i^\top c_j - b_j^\top c_i\ (+\ \gamma_i - \gamma_j)$, passed through a logistic link; the skew-symmetric bilinear term represents cyclic structure that no scalar model can [Chen & Joachims 2016]. Dimension $d$ controls cycle expressiveness; $d{=}0$ recovers BT.
- **Estimation & complexity** — SGD on logistic loss with regularization; scales like any embedding model; needs *repeated* matchups (or feature sharing) to pin down vectors.
- **Handles** — intransitivity ✓✓ (the point) · ties/margins/dynamics ✗ (extensions possible) · uncertainty ✗.
- **Pros** — strictly generalizes BT with a controllable budget for cyclic structure; measurably better matchup prediction on real game data (StarCraft, tennis shots, online games).
- **Cons** — data-hungry; embeddings are not a leaderboard (no canonical scalar order — that's the *honest* answer, but stakeholders want a list); regularization-sensitive.
- **Use cases** — matchup forecasting in esports/fighting-game metas; style-aware sports prediction; any domain where "who wins" depends on *pairing*, not just quality.
- **Relationships** — low-rank skew-symmetric completion of the log-odds matrix; mElo (§9.2) is the online/spectral cousin; HodgeRank (§3.6) *measures* what this *models*.
- **Propagon status** — candidate (medium effort: SGD machinery exists in `lr.rs`).
- **References** — [Chen & Joachims 2016].

### 9.2 Multidimensional Elo (mElo) & Nash Averaging (2018)

- **TL;DR** — Two repairs for evaluation in cyclic regimes: mElo augments Elo with low-rank cyclic terms; Nash averaging re-weights opponents/tasks by a maximum-entropy Nash equilibrium so redundant or irrelevant ones can't distort the ranking.
- **Inputs / Output** — pairwise win rates among agents (or agents × tasks scores) → mElo: rating + cyclic embedding; Nash: equilibrium weights + Nash-averaged skill.
- **Class** — Parametric (mElo) / game-theoretic (Nash averaging) × Static or Online.
- **Model & assumptions** — decompose the log-odds matrix as $s_i - s_j + \sum_k (c_{ik} c'_{jk} - c_{jk} c'_{ik})$: a transitive scalar plus rank-$2k$ skew-symmetric cycles, updated Elo-style online (mElo$_{2k}$). Nash averaging finds the max-entropy Nash equilibrium of the antisymmetric evaluation game and scores agents against that adversarially-chosen mixture — making evaluation **invariant to duplicating agents or tasks** [Balduzzi, Tuyls, Pérolat & Graepel 2018].
- **Estimation & complexity** — mElo: $O(k)$ per game; Nash: solve a zero-sum game (LP) on the win matrix — fine for hundreds of agents.
- **Handles** — intransitivity ✓ · redundancy-invariance ✓ (Nash) · uncertainty ✗ · dynamics △ (mElo is online).
- **Pros** — directly attacks the two classic leaderboard pathologies: cycles, and "spam the benchmark with near-duplicates"; mElo stays Elo-cheap.
- **Cons** — Nash equilibria can be non-unique and brittle to small win-rate noise; max-entropy selection helps but interpretability suffers; mElo's $k$ must be chosen.
- **Use cases** — multi-agent/games evaluation (its origin: AlphaGo-era agent comparisons); tournament metas; benchmark-suite aggregation where task redundancy is rampant.
- **Relationships** — mElo = online Blade-Chest-lite; Nash averaging anticipates §6.7's social-choice critique and α-Rank's game-theoretic turn.
- **Propagon status** — candidate (mElo: easy; Nash averaging: needs an LP solver dependency).
- **References** — [Balduzzi et al. 2018].

### 9.3 α-Rank — Evaluation by Evolution (2019)

- **TL;DR** — Rank strategies by evolutionary survival: simulate mutation-selection population dynamics over the empirical game and rank by time spent in each strategy profile — built for N-player, general-sum, intransitive worlds where Elo's assumptions are meaningless.
- **Inputs / Output** — payoff tensor over strategy profiles (from tournaments/simulations) → stationary mass per strategy (+ the chain's structure as a diagnostic).
- **Class** — Game-theoretic / Markov-chain × Static.
- **Model & assumptions** — construct a Markov chain over joint strategy profiles whose transitions follow imitation/mutation dynamics with selection intensity $\alpha$; as $\alpha \to \infty$ the chain's recurrent classes align with **Markov-Conley chains** — the game's genuinely stable cycles — and the stationary distribution provides the ranking [Omidshafiei et al. 2019].
- **Estimation & complexity** — the profile space is exponential in players (the chain is over *joint* profiles); tractable for the modest strategy sets of empirical game-theoretic evaluation; sparse solvers and sampling push it further.
- **Handles** — intransitivity ✓✓ · N-player & general-sum ✓ (unique here) · uncertainty △ (payoff noise propagates; resample).
- **Pros** — principled where nothing else here even applies (N-player, general-sum); dynamics-based stability is arguably the *right* notion of "good strategy" in multi-agent ecosystems; diagnostics expose the cycle structure itself.
- **Cons** — needs the full payoff tensor (expensive tournaments); $\alpha$ sweep required; rankings can be sensitive to payoff estimation noise; overkill for two-player transitive-ish data.
- **Use cases** — multi-agent RL evaluation; meta-game analysis (poker/MOBA/auction strategies); ranking *policies* rather than players.
- **Relationships** — generalizes Nash-averaging's game-theoretic stance from two-player zero-sum to N-player general-sum; machinery is §3/§4's stationary distributions over a profile graph.
- **Propagon status** — out-of-scope for the core library (input shape is a payoff tensor, not an edge list); worth a docs recipe.
- **References** — [Omidshafiei et al. 2019].

### 9.4 Games of Skill Look Like Spinning Tops (2020) — compact

**Class: empirical geometry (descriptive) × Static.** Across dozens of real games, the strategy landscape is a "spinning top": a long **transitive axis** (skill) plus a fat **cyclic disc** at intermediate skill that thins toward the extremes [Czarnecki et al. 2020]. Practical consequences for ranking: scalar ratings (§1–§2) work *better than they should* at the high end and worst mid-table — exactly where most entities live; and diverse opponent populations are needed to escape cyclic traps when *training* agents. Use as the empirical prior for deciding whether §9's machinery is worth its cost: measure your curl (§3.6) first. *Propagon: out-of-scope (conceptual).* [Czarnecki et al. 2020].

---

## 10. Feature-Based, Contextual & Learning-to-Rank

When entities (or comparison contexts) carry **features**, the latent score becomes a *function*, $s_i = f(x_i)$, and ranking meets supervised learning. This section is deliberately compact — full learning-to-rank is its own field with relevance-judgment data at its core — and focuses on the revealed-preference throughline: clicks, purchases, and human preference pairs as training signal.

### 10.1 Bradley-Terry with Covariates / Conditional Logit — compact

**Class: Parametric × Static.** Set $s_i = \beta^\top x_i$ (or $s_i = \beta^\top x_i + b_i$ for partial pooling): a conditional logit [McFadden 1974] on comparison data. This single move buys: cold-start scoring of *unseen* entities, deconfounding (style control, §12.2, is exactly this with style features), strength sharing across sparse entities, and hypothesis tests on *why* things win. The same chassis with $f$ a neural network is the modern reward model (§10.6). *Propagon: candidate — the `.features.id` file format already exists; a `--features` flag on a future BT command is the natural delivery.* [McFadden 1974; Agresti 2013].

### 10.2 Click Models & Counterfactual Learning-to-Rank

- **TL;DR** — Clicks are revealed preferences corrupted by *where things were shown*; click models estimate the corruption, and counterfactual LTR inverts it to train unbiased rankers from logs.
- **Inputs / Output** — logged result lists + clicks (+ optionally propensities) → de-biased relevance estimates / ranker training signal.
- **Class** — Semi-parametric (behavioral model + IPS) × Static (over logs).
- **Model & assumptions** — the **examination hypothesis**: $P(\text{click}) = P(\text{examined at position}) \times P(\text{relevant})$; the **cascade model**: users scan top-down until satisfied — both calibrated experimentally [Craswell, Zoeter, Taylor & Ramsey 2008]. Counterfactual LTR treats position bias as a known propensity and optimizes an **inverse-propensity-scored** ranking objective over logs, provably unbiased under correct propensities [Joachims, Swaminathan & Schnabel 2017]; propensities themselves come from randomization or joint estimation.
- **Pros / Cons** — converts the cheapest, largest preference signal you have (logs) into honest training data; but propensity misspecification silently re-biases everything, variance explodes for rarely-shown items, and click ≠ satisfaction.
- **Use cases** — search/recommendation training from interaction logs; correcting position bias in any presented-list setting — including arena-style UIs.
- **Relationships** — the event-level complement to BiRank's graph-level use of the same logs (§4.7); IPS machinery shared with §13.5's off-policy evaluation — the same statistical problem in different clothes.
- **Propagon status** — out-of-scope (needs presentation logs, not edge lists); the *concepts* inform §12's position-bias guidance.
- **References** — [Craswell et al. 2008; Joachims, Swaminathan & Schnabel 2017; Chuklin, Markov & de Rijke 2015].

### 10.3 RankNet → LambdaRank → LambdaMART (2005–2010) — compact

**Class: Parametric (neural/trees) × Static.** **RankNet**: train $f(x)$ with the BT/logistic loss on preference pairs — Bradley-Terry with a neural score function, full stop [Burges et al. 2005]. **LambdaRank**: skip the loss, define its *gradients* directly, scaled by the metric impact (|ΔNDCG|) of swapping a pair — optimizing non-smooth IR metrics by construction. **LambdaMART**: those lambda-gradients in gradient-boosted trees [Burges 2010] — still the tabular-LTR workhorse and a reminder that pairwise preference losses power most production rankers. *Propagon: out-of-scope (feature-based supervised learning); documented for the BT connection.* [Burges et al. 2005; Burges 2010].

### 10.4 Listwise Losses: ListNet & ListMLE (2007/2008) — compact

**Class: Parametric × Static.** Listwise LTR's main losses are Plackett-Luce in disguise: **ListMLE** maximizes exactly the PL likelihood of the observed ordering under scores $f(x)$ [Xia et al. 2008]; **ListNet** minimizes cross-entropy between PL-induced distributions [Cao et al. 2007]. Theoretical bridge worth stating plainly: *softmax classification, PL ranking, and listwise LTR are one model family*. Anything propagon learns about fitting PL well (§1.4, §3.2) transfers. *Propagon: out-of-scope; conceptual bridge.* [Cao et al. 2007; Xia et al. 2008].

### 10.5 Differentiable Sorting & Ranking (2011–2020) — compact

**Class: Algebraic/relaxation × Static.** Sorting/ranking as a *layer* inside gradient-trained models: relax permutation matrices to doubly-stochastic ones via Sinkhorn iterations [Adams & Zemel 2011], learn latent permutations with Gumbel-Sinkhorn [Mena et al. 2018], cast soft sorting/ranking as entropy-regularized optimal transport [Cuturi, Teboul & Vert 2019], or get exact $O(n \log n)$ differentiable sort/rank operators via isotonic-projection [Blondel, Teboul, Berthet & Djolonga 2020]. This is what makes "rank inside the loss function" possible (direct NDCG-ish optimization, permutation supervision, top-k selection layers). Note the rhyme: Sinkhorn balancing already appeared as offense-defense ratings (§5.3). *Propagon: out-of-scope (deep-learning tooling); documented because any future neural-feature ranker will reach for it.* [Adams & Zemel 2011; Mena et al. 2018; Cuturi et al. 2019; Blondel et al. 2020].

### 10.6 RLHF Reward Models & DPO — Bradley-Terry Eats the World

- **TL;DR** — The reward models behind aligned LLMs are Bradley-Terry with a transformer as the score function; DPO shows the policy itself can *be* the implicit BT reward — making §1's seventy-year-old model the load-bearing wall of modern AI training.
- **Inputs / Output** — human (or AI) preference pairs over model responses → a reward function $r(x, y)$ / directly a tuned policy.
- **Class** — Parametric (neural BT) × Static (per training round).
- **Model & assumptions** — collect preferences $y_w \succ y_l \mid x$; fit $P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$ — exactly BT with $s = r(x, \cdot)$ — then optimize the policy against $r$ with KL regularization [Christiano et al. 2017; Ouyang et al. 2022]. **DPO** eliminates the explicit reward: under the KL-regularized objective, the optimal policy satisfies $r(x,y) = \beta \log \tfrac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + C$, so substituting into the BT loss trains the policy directly on preferences [Rafailov et al. 2023].
- **Pros / Cons** — inherits *all* of BT's assumptions, at scale and with consequences: transitivity over responses, IIA, context-independence of annotator noise — each demonstrably violated in places. When BT is or isn't the right preference-model for reward learning, and which alternatives (classification-based, regression-based, general preference models) fix what, is now its own literature [Sun, Shen & Ton 2024].
- **Use cases** — LLM alignment; preference-tuning any generative system; the largest-scale deployment of revealed-preference ranking in existence.
- **Relationships** — BT with covariates (§10.1) at the far end of expressiveness; evaluation-side counterpart in §12; preference-based RL connects to §8's dueling bandits and §13's value functions.
- **Propagon status** — out-of-scope (training loop); in-scope insight: a fast, trustworthy BT/PL fitter is *also* preference-data tooling for the RLHF era.
- **References** — [Christiano et al. 2017; Ouyang et al. 2022; Rafailov et al. 2023; Sun, Shen & Ton 2024].

---

## 11. Bayesian & Uncertainty-Aware Inference

Less a family than a **lens applicable to every family**: replace point estimates with posteriors (or resampling distributions), and "the ranking" becomes a *distribution over rankings* — which is what you actually need for decisions ("is A better than B, p ≥ 0.95?"), for active sampling, and for honest leaderboards. Concrete cross-references: Glicko/TrueSkill are this lens applied to §2; Chatbot Arena's bootstrap CIs apply it to §1 (§12.1); mcrl-rs applies it to §13.

### 11.1 Bayesian Bradley-Terry via Latent Variables (2012)

- **TL;DR** — A data-augmentation trick (latent gamma/exponential race variables) makes BT and its generalizations conjugate, yielding simple Gibbs samplers and EM — full posteriors over strengths at modest cost.
- **Inputs / Output** — pairwise (and generalized: ties, home, multiway) outcomes + gamma priors → posterior samples over strengths → credible intervals, $P(i \succ j)$ posteriors, rank distributions.
- **Class** — Parametric (Bayesian) × Static.
- **Model & assumptions** — BT likelihood + independent Gamma priors on $\pi_i$; introducing latent exponential "arrival" variables per comparison restores conditional conjugacy, so Gibbs alternates closed-form draws; the same augmentation covers Rao-Kupper ties, home advantage, and PL choice sets [Caron & Doucet 2012].
- **Estimation & complexity** — Gibbs sweeps at $O(\text{comparisons})$ each; EM variant gives MAP at MM-like cost.
- **Handles** — uncertainty ✓✓ (full posterior) · ties/home/multiway ✓ (via the generalized family) · sparse data ✓ (priors regularize; Ford condition softened) · dynamics ✗.
- **Pros** — *the* answer to "BT, but with honest uncertainty and stability on sparse data"; priors fix divergent ratings (undefeated entities) gracefully; rank-probability outputs ("P(this item is top-3)") fall out free.
- **Cons** — MCMC bookkeeping (convergence, mixing); slower than MM/spectral point estimates; prior choice visible in sparse regimes.
- **Use cases** — small-to-medium leaderboards where uncertainty drives decisions; sparse tournaments; official rankings needing credible intervals.
- **Relationships** — Bayesian counterpart of §1.1/§1.4; Glicko-2 (§2.3) is its filtering approximation; bootstrap (§11.4) its frequentist sibling.
- **Propagon status** — **candidate — recommended** (pairs with `mm.rs`; Gibbs is simple and Rayon-friendly across chains).
- **References** — [Caron & Doucet 2012].

### 11.2 Crowd-BT — Annotator-Aware Ranking (2013)

- **TL;DR** — Crowdsourced comparisons come from annotators of wildly uneven quality; Crowd-BT learns each annotator's reliability *jointly* with the item ranking, down-weighting (or inverting) the noisy and the adversarial.
- **Inputs / Output** — (annotator, i ≻ j) triples → item scores **and** per-annotator reliability $\eta_k$; active-learning rule for routing the next query.
- **Class** — Parametric (Bayesian-flavored) × Static (online extension included).
- **Model & assumptions** — annotator $k$ reports the BT-preferred item with probability $\eta_k$, else the reverse: $P_k(i \succ j) = \eta_k\,\sigma(s_i - s_j) + (1 - \eta_k)\,\sigma(s_j - s_i)$; alternate updates of scores and reliabilities, with entropy-based active selection of (pair, annotator) [Chen, Bennett, Collins-Thompson & Horvitz 2013].
- **Pros / Cons** — substantial quality gains over vanilla BT whenever annotator quality varies (always); spammer/adversary detection for free. Identifiability needs annotator overlap on common pairs; mediocre-annotator regimes ($\eta \approx 0.5$) carry little signal; reliability assumed item-independent.
- **Use cases** — paid crowdsourcing pipelines; arena leaderboards with heterogeneous voters (§12); peer grading; any multi-judge comparison collection.
- **Relationships** — BT + the item-response-theory idea (rater models); complements aggregate-level style control (§12.2): one models *who voted*, the other *what confounded the vote*.
- **Propagon status** — **candidate — recommended** (input format extends `.edges` with an annotator column; high practical value).
- **References** — [Chen et al. 2013].

### 11.3 Gaussian-Process Preference Learning (2005) — compact

**Class: Semi-parametric (Bayesian, nonparametric-in-$f$) × Static.** Put a GP prior on the score function $f(x)$ over item *features* and observe preferences through a Thurstone-style probit likelihood: posterior inference (Laplace/EP) yields calibrated preference predictions for **unseen items** with kernel-controlled smoothness [Chu & Ghahramani 2005]. The classification subtlety: parametric in link, nonparametric in score function — the middle of §0.1's axis. Foundation of preference-based Bayesian optimization (preferential BO) and a serious tool when items have rich features but comparisons are scarce. Cubic-in-data cost without sparse approximations. *Propagon: out-of-scope (kernel stack); conceptually adjacent to §10.1.* [Chu & Ghahramani 2005].

### 11.4 Bootstrap & Bayesian-Bootstrap Rank Inference

- **TL;DR** — Model-free uncertainty for *any* ranker in this document: resample the data, re-rank, and read off the distribution of scores, ranks, and pairwise verdicts.
- **Inputs / Output** — any comparison/trajectory dataset + any ranking procedure → CIs on scores, rank distributions, $P(A \succ B)$ estimates, significance tests.
- **Class** — Non-parametric (resampling) × any.
- **Model & assumptions** — the classical bootstrap resamples observations with replacement [Efron 1979]; the **Bayesian bootstrap** draws Dirichlet$(\alpha, \ldots, \alpha)$ weights over observations — a smooth posterior over the empirical distribution [Rubin 1981]; **permutation tests** answer the sharper null "are A and B exchangeable?" exactly. All three are embarrassingly parallel.
- **Why it's load-bearing here** — it is *the* uncertainty mechanism for methods with no likelihood (counting §7, spectral §3, centrality §4, Kemeny §6) and the standard one even where likelihoods exist (Chatbot Arena's BT CIs, §12.1). Caveats: resampling unit must respect dependence (resample matches, not match-halves; cluster by player/session where §0.2's dependence bites); rank CIs are discrete and lumpy at the top.
- **Use cases** — leaderboard error bars; "is the new model actually better" gates; A/B test readouts on V(s) estimates (§13.2).
- **Relationships** — frequentist/cheap sibling of §11.1's posteriors; implemented for value-based ranking in mcrl-rs (bootstrap CIs, permutation tests, Bayesian-bootstrap $P(B > A)$).
- **Propagon status** — **sibling (mcrl-rs)** for trajectory data; **candidate — recommended** as a propagon-wide `--bootstrap N` flag (works for every scorer the library ships).
- **References** — [Efron 1979; Rubin 1981].

---

## 12. Applied Deep-Dive: Ranking LLMs & Model Evaluation

The most visible ranking problem of the 2020s, and a perfect stress test of this document: crowdsourced pairwise battles between *static* entities, with heavy ties, severe style confounding, heterogeneous voters, and enormous stakes attached to leaderboard positions. The methodology arc of 2023–2026 is essentially a guided tour of §§1, 6, 11 — the field tried Elo first, then rediscovered, one by one, why the rest of this document exists.

### 12.1 The Chatbot Arena Methodology: BT + Bootstrap (2024)

- **What it is** — crowdsourced, randomized, anonymized pairwise battles; users vote win/tie/loss; the leaderboard is fit by **Bradley-Terry MLE (logistic regression), not online Elo**, with **bootstrap confidence intervals** over battles, and model-pair sampling tuned to shrink the uncertainty fastest [Chiang et al. 2024].
- **Why BT-not-Elo matters** — model skill is *static* (frozen checkpoints): there is no drift for Elo's $K$-step to track, so its order-dependence and step-size noise are pure cost. Offline MLE uses every battle symmetrically and converges to the §1.1 estimate; bootstrap (§11.4) supplies honest error bars; rankings are presented as *overlapping-interval* bands rather than false precision.
- **Mechanics worth copying** — randomized presentation order (kills position bias, §0.2); anonymization until after the vote (kills brand bias); active pair sampling (a §8 idea in production); public battle data enabling reanalysis.
- **References** — [Chiang et al. 2024].

### 12.2 Style-Controlled Bradley-Terry (2024)

- **What it is** — the observation that *style predicts votes*: longer, better-formatted answers win disproportionately, confounding "writes prettily" with "is smarter." Fix: BT with covariates (§10.1) — add per-battle style features (token-length difference, markdown-density differences, etc., normalized) to the BT regression, so model coefficients estimate strength *at equalized style* [LMSYS 2024]. ANCOVA, rediscovered for leaderboards.
- **Effects** — substantial rank shifts among frontier models when length/format effects are absorbed by the style coefficients — direct evidence the unadjusted leaderboard partially measured verbosity.
- **Caveats** — controlling a *mediator* (if smarter models legitimately write better-structured answers, some true skill is regressed away); style features are proxies; causal interpretation requires care.
- **References** — [LMSYS 2024; Chiang et al. 2024].

### 12.3 Elo Uncovered: Why Online Elo Fails Static Skills (2023/2024)

- **What it is** — an axiomatic audit of Elo for LLM evaluation against two minimal properties: **reliability** (re-runs on reordered data give the same ranking) and **transitivity** (consistency of implied orderings). Online Elo violates both on realistic synthetic and real LLM-comparison data — outcomes depend on match *order* and on $K$ — unless updates are iterated to convergence over many permutations, at which point one has reinvented offline BT badly [Boubdir et al. 2024].
- **The takeaway for practitioners** — match the estimator to the data-generating process: drifting skill → §2; static skill → §1 fit offline, uncertainty by §11. "Elo" survives in LLM discourse as a *scale convention*, not as the update rule.
- **References** — [Boubdir et al. 2024].

### 12.4 Ties, Covariance & Factored Models for Arena Data (2025)

- **What it is** — a statistical upgrade of the arena pipeline addressing two neglected structures: **ties** (a large fraction of arena votes — naive BT discards or mishandles them) and **correlation between competitors** (battles share prompts and voters). The framework factors tie-generation separately from preference strength (generalizing Rao-Kupper/Davidson, §1.2), models covariance across competitors, and reports order-of-magnitude improvements in tie prediction with better-calibrated intervals [Ameli, Zhuang, Stoica & Mahoney 2025].
- **Why it matters here** — a live demonstration that §1.2's "minor extensions" and §0.2's "dependence" footnotes become first-order effects at scale.
- **References** — [Ameli et al. 2025].

### 12.5 Rethinking BT for Reward Modeling (2024) — compact

The training-side mirror of this section: when is BT the *right* objective for learning reward models from preferences, and what are the alternatives? [Sun, Shen & Ton 2024] ground BT's use in reward modeling theoretically (asymptotics, when order-consistency suffices), then catalog alternatives — classification-style objectives, margin-based losses, general preference models — and when each dominates. Read alongside §10.6; the practical message is that the choice of preference *likelihood* is a modeling decision with measurable downstream consequences, not boilerplate. [Sun, Shen & Ton 2024].

### 12.6 Social-Choice Leaderboards — compact

When aggregating *across benchmarks/tasks* (rather than within one arena), mean-score and Elo-style aggregation are manipulable by task selection and duplication; Condorcet-consistent voting over per-task rankings (§6.6–6.7) is provably more robust [Lanctot et al. 2023]. Combined with Nash averaging's redundancy-invariance (§9.2), the emerging best practice for *suite-level* model ranking is social-choice machinery, not score averaging. [Lanctot et al. 2023; Balduzzi et al. 2018].

**Section synthesis — an arena-grade pipeline assembled from this document:** randomize & anonymize (§12.1) → BT/PL MLE offline (§1) → tie-aware likelihood (§1.2/§12.4) → style & annotator covariates (§10.1/§11.2/§12.2) → bootstrap everything (§11.4) → audit rankability via Hodge curl (§3.6) → aggregate across suites by social choice (§6/§12.6). Every piece is an implemented-or-candidate propagon capability.

---

## 13. Value-Function & Trajectory-Based Ranking

The creative family, and mcrl-rs's home. The preference signal here is neither comparisons nor links but **trajectories with rewards**: sessions that end in revenue, games that end in wins, episodes that accumulate cost. Identify entities with *states* (a team, a UI variant, a slot machine, an opening position, a player) and rank by the **state-value function**:

$$V(s) = \mathbb{E}\left[\sum_{t \ge 0} \gamma^t r_t \,\middle|\, s_0 = s\right]$$

Two bridges make this a first-class member of the survey rather than a guest:

1. **Every value comparison is a revealed-preference edge.** $\hat V(a) > \hat V(b)$ — ideally with $P(V_a > V_b)$ attached (§13.2) — is exactly the `i ≻ j (weight)` atom every aggregator in §§1–6 consumes. Value estimation can *feed* comparison-based rankers when entities never meet head-to-head but do generate trajectories.
2. **It ranks what comparisons can't see.** Win/loss data reveals relative strength *given a matchup structure*; trajectories reveal *absolute expected outcomes* under real usage — including for entities that are never compared (states in one long process, variants shown to disjoint users).

### 13.1 Monte Carlo Value Estimation (first-visit / every-visit)

- **TL;DR** — Roll returns backward through each episode and average them per state: a model-free, assumption-light estimate of $V(s)$, rankable directly.
- **Inputs / Output** — episodes/trajectories (sequences of states + rewards) → $\hat V(s)$ per state (+ observation counts).
- **Class** — Non-parametric (empirical means) × Static (batch over logs).
- **Model & assumptions** — compute discounted returns $G_t = r_t + \gamma G_{t+1}$ backward per episode; average per state, counting either only each state's **first visit** per episode (cleaner i.i.d. structure) or **every visit** (more data, within-episode correlation) [Sutton & Barto 2018]. Model-free: no transition or reward model. The estimates are *on-policy* — they rank states under the behavior that generated the logs (see §13.5 for the counterfactual upgrade).
- **Estimation & complexity** — one backward pass per episode, $O(\text{total steps})$; embarrassingly parallel over episodes.
- **Handles** — uncertainty ✓ (via §13.2) · skew/outliers ✓ (median estimator, Winsorization) · time structure ✓ (γ trades near-term vs long-term) · intransitivity — not applicable (absolute scale!).
- **Pros** — minimal assumptions; consumes data no comparison method can; produces an *absolute* scale (a rare luxury here — "V(a) − V(b) = 3.2 expected dollars" means something); γ gives a principled near-term/long-term knob.
- **Cons** — needs episodic (or truncatable) data; high variance for long horizons; on-policy bias; state definition is a real modeling decision (too fine → no samples; too coarse → aggregation bias).
- **Use cases** — A/B variants ranked by downstream (not immediate) outcomes; game states/openings ranked from match logs; funnel/UX states ranked by expected conversion; bandit arms with delayed payoffs.
- **Relationships** — the RL textbook's policy-evaluation primitive turned ranking method; TD (§13.3) trades its variance for bias; feeds §§1–6 via bridge (1).
- **Propagon status** — **sibling (mcrl-rs)**: first/every-visit MC with configurable state fields, discount, mean/median estimators, Winsorization, and min-observation filters, over JSONL trajectories.
- **References** — [Sutton & Barto 2018].

### 13.2 Statistical Comparison of Value Estimates

- **TL;DR** — Turn noisy $\hat V$'s into defensible rankings: bootstrap CIs per state, permutation tests and exceedance probabilities per pair — the uncertainty layer that makes value-based ranking decision-grade.
- **Inputs / Output** — per-state return samples → CIs on $V(s)$; per pair: permutation $p$-values, $P(B > A)$ (a draw from B beats a draw from A), $P(\mathbb{E}[B] > \mathbb{E}[A])$, and Bayesian-bootstrap posteriors $P(V_B > V_A)$.
- **Class** — Non-parametric (resampling, §11.4 applied) × Static.
- **Model & assumptions** — bootstrap over episodes for CIs [Efron 1979]; permutation tests under exchangeability for sharp nulls; Dirichlet-weighted Bayesian bootstrap [Rubin 1981] for posterior-flavored exceedance probabilities; Winsorization for heavy-tailed returns (revenue!). Group-wise comparison (e.g., per-segment A-vs-B) refines rankings into conditional ones.
- **Pros / Cons** — exactly the right output type for decisions ("ship B: $P(V_B > V_A) = 0.97$") and immune to distributional fantasies; but exchangeability fails under interference/seasonality (resample blocks), and multiple comparisons across many states need correction.
- **Use cases** — experiment readouts; ranking many variants with honest "too close to call" verdicts; constructing **weighted preference edges** $a \succ b$ with weight $P(V_a > V_b)$ for downstream §1/§6 aggregation — the concrete mcrl-rs → propagon pipeline.
- **Propagon status** — **sibling (mcrl-rs)**: implemented (`--comparison-test`, `--bootstrap-samples`, Bayesian bootstrap with Dirichlet α, comparison groups).
- **References** — [Efron 1979; Rubin 1981; Sutton & Barto 2018].

### 13.3 Temporal-Difference & Function Approximation — compact

**Class: Parametric (in features) × Online.** TD(0) updates $\hat V(s) \leftarrow \hat V(s) + \alpha\,(r + \gamma \hat V(s') - \hat V(s))$ from *transitions* rather than complete episodes: lower variance, works on continuing (non-episodic) processes, online by nature — at the cost of bootstrapping bias and (with function approximation $V_\theta$) stability care [Sutton & Barto 2018]. Rank states by converged $\hat V$ exactly as in §13.1. Choose TD over MC when episodes are long/endless or states are too numerous to estimate independently (share strength through features — the §10.1 move, again). *Propagon/mcrl-rs: natural mcrl-rs extension.* [Sutton & Barto 2018].

### 13.4 Markov-Reward Player & Action Valuation (2015/2019)

- **TL;DR** — Rank *people* by the value they add: model the game as a Markov process, learn the value of every state, and credit each player's actions by how much they moved $V$ — ranking by contribution, not by team outcomes.
- **Inputs / Output** — fine-grained event/tracking data (play-by-play) → per-action values → aggregated per-player rankings.
- **Class** — Semi-parametric (Markov model or learned value model) × Static (per season/window).
- **Model & assumptions** — build a Markov (game) model over game contexts and compute action impact as $\Delta V$ — e.g., expected-goals-flavored state values in ice hockey via a large context-state Markov game [Routley & Schulte 2015]; or learn $P(\text{score soon} \mid \text{state})$ / $P(\text{concede soon} \mid \text{state})$ with supervised models and credit each on-ball action by the change (**VAEP**), valuing all soccer actions, not just shots [Decroos, Bransen, Van Haaren & Davis 2019].
- **Pros / Cons** — ranks contributors *within* a team sport where head-to-head player comparisons don't exist — a problem flatly outside §§1–8's input format; produces interpretable "value added" units. Costs: heavy data engineering; value-model misspecification leaks directly into player rankings; credit assignment among simultaneous contributors remains partly heuristic.
- **Use cases** — player scouting/valuation (sports analytics' core product); employee/process-step attribution analogies; ranking *components* of any pipeline by marginal value contribution.
- **Relationships** — §13.1's machinery + credit assignment; the team-decomposition counterpart of TrueSkill's additive skills (§2.4), with states instead of latent traits.
- **Propagon status** — out-of-scope for the core (needs domain event schemas); mcrl-rs covers the state-value layer it builds on.
- **References** — [Routley & Schulte 2015; Decroos et al. 2019].

### 13.5 Off-Policy Evaluation as Ranking — compact

**Class: Semi-parametric (importance weighting) × Static (over logs).** Rank *policies* (recommenders, treatment rules, agents) by estimated value **under data collected from a different policy**: importance-sampling estimators correct the distribution mismatch [Precup, Sutton & Singh 2000]; doubly-robust estimators combine a value model with IS for variance control and bias insurance [Dudík, Langford & Li 2011]. This is the counterfactual upgrade of §13.1's on-policy ranking — and the same IPS mathematics as counterfactual LTR (§10.2), confirming the deep link: *position bias and behavior-policy bias are the same disease*. Variance explodes with policy divergence; propensity logging is the price of admission. *Use cases: offline ranking of candidate recommenders/agents before any A/B test.* *Propagon: out-of-scope; mcrl-rs-adjacent future work.* [Precup, Sutton & Singh 2000; Dudík, Langford & Li 2011].

---

## 14. Cross-Cutting Topics

### 14.1 Identifiability & Connectivity

Every latent-score method needs the comparison graph connected (strongly, in the directed sense of [Ford 1957]) for a common scale to exist; spectral methods additionally want a healthy spectral gap (§3.1). Symptoms and treatments:

| Symptom | Cause | Treatments |
|---|---|---|
| Diverging rating | Undefeated/winless entity | Priors/regularization (§11.1); Laplace smoothing (§5.2); propagon `--remove-total-losers` |
| Incomparable score groups | Disconnected components | Rank within components (`extract-components`); bridge with priors or scheduled comparisons; propagon `--create-fake-games`, `--random-subgraph-links` |
| Unstable spectral scores | Weak spectral gap (barbell schedules) | More cross-group comparisons; fall back to MLE; regularized teleportation (§4.4's trick) |
| Scores drift across refits | Location/scale non-identifiability | Anchor (fix one entity, or zero-mean); never compare raw scores across datasets |

### 14.2 Sample Complexity at a Glance

Qualitative guide (n items; see sources for precise statements):

| Goal | Regime | Order of comparisons | Source |
|---|---|---|---|
| Full ranking, bounded-noise comparisons | active | $n \log n$ | [Braverman & Mossel 2008] |
| BT/PL scores to constant error | passive, random graph | $n \log n$ (spectral or MLE) | [Negahban, Oh & Shah 2017; Maystre & Grossglauser 2015] |
| Rank recovery, SST only | passive, uniform schedule | counting is minimax-optimal | [Shah & Wainwright 2018] |
| Top-$k$ identification | active | gap-dependent; parametric helps ≤ log factors | [Heckel et al. 2019] |

The recurring constant: $n \log n$ is the budget to think in. Dense all-pairs designs ($n^2$) buy estimation of the *full probability matrix* (§7.3), not better rankings.

### 14.3 Capability Matrix

The main implemented-or-recommended methods against the recurring requirements:

| Method (§) | Ties | Margins | Home adv | Teams | Dynamics | Uncertainty | Intransitivity | Features | Choice sets |
|---|---|---|---|---|---|---|---|---|---|
| BT-MM / BT-LR (§1.1) | ext. | ✗ | ext. | ext. | ✗ | △ | ✗ | ext. | ✗ |
| Plackett-Luce / LSR (§1.4, §3.2) | ✗ | ✗ | ✗ | ✗ | ✗ | △ | ✗ | ext. | ✓ |
| Elo (§2.1) | △ | ext. | △ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Glicko-2 (§2.3) | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| TrueSkill / Weng-Lin (§2.4–2.5) | ✓ | △ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| WHR (§2.6) | △ | ✗ | △ | ✗ | ✓✓ | ✓ | ✗ | ✗ | ✗ |
| Rank Centrality (§3.1) | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Keener (§3.3) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| HodgeRank (§3.6) | ✓ | ✓ | ✗ | ✗ | ✗ | △ | **measures** | ✗ | ✗ |
| Massey / Colley (§5) | ✓/✗ | ✓/✗ | ✓ | ✗ | ✗ | △ | ✗ | ✗ | ✗ |
| Borda / Copeland / Kemeny (§6) | △ | ✗ | ✗ | ✗ | ✗ | ✗ | △ | ✗ | rankings |
| Wilson rate (§7.1) | △ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Blade-Chest / mElo (§9) | ✗ | ✗ | ✗ | ✗ | △ | ✗ | ✓ | ext. | ✗ |
| Bayesian BT (§11.1) | ✓ | ✗ | ✓ | ✗ | ✗ | ✓✓ | ✗ | ext. | ✓ |
| Crowd-BT (§11.2) | ✗ | ✗ | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ |
| MC V(s) + bootstrap (§13) | n/a | n/a | n/a | n/a | γ | ✓✓ | n/a | state-def | n/a |

✓✓ core strength · ✓ supported · △ partial/convention · ext. = standard extension exists · ✗ unsupported.

### 14.4 Dependence Between Comparisons

Independence is the universal hidden assumption. Violations — same player's matches sharing form, multiple judgments per voter, battles sharing prompts — mostly leave *point estimates* serviceable but make *uncertainty* overconfident. Remedies: cluster bootstrap (resample players/voters/prompts, not comparisons); random-effects and pairwise-likelihood models [Cattelan 2012]; explicit covariance modeling for arena data [Ameli et al. 2025]. If your CIs decide anything, audit the resampling unit first.

### 14.5 Evaluating the Rankers Themselves

How to compare ranking methods on your data:

- **Order agreement**: Kendall's τ [Kendall 1938] (pairwise-disagreement-based; matches Kemeny's objective), Spearman's ρ and footrule (displacement-based [Diaconis & Graham 1977]); weighted/top-heavy variants when only the head matters.
- **Predictive quality** (parametric methods): held-out comparison log-loss and Brier score [Brier 1950], calibration curves, plain accuracy. *Temporal* splits for anything dynamic — random splits leak the future.
- **Top-set retrieval**: precision/overlap@k; NDCG [Järvelin & Kekäläinen 2002] when graded relevance exists.
- **Stability**: rank correlation across bootstrap resamples — a ranker that reshuffles under resampling isn't measuring anything.
- **Rankability**: before judging rankers, judge the data — Hodge curl share (§3.6), tournament-cycle counts, [Boldi & Vigna 2014]-style axiom checks for centrality choices.

Protocol note: comparing a parametric score's order against Kemeny-style consensus on the *same* data conflates model and metric (Kendall-τ-optimal is Kemeny's home turf). Use held-out prediction for parametric methods, order-agreement against ground truth (when one exists) for the rest.

### 14.6 Sidebar: Economic "Revealed Preference" (Afriat / GARP)

The phrase *revealed preference* originates in consumer theory: choices under budget constraints reveal preference relations, and [Afriat 1967] proved that finite expenditure data is consistent with utility maximization **iff** it satisfies cyclical consistency — operationalized as the Generalized Axiom of Revealed Preference (GARP), testable in polynomial time [Varian 1982]. This machinery *tests rationality and recovers utility bounds*; it does not produce an entity leaderboard, which is why it appears as lineage here rather than as a method entry. The conceptual through-line stands, though: this entire document is the statistical generalization of "what you chose tells us what you value" — with Afriat's cyclical consistency reappearing as stochastic transitivity (§7.3) and Hodge curl (§3.6).

---

## 15. Method-Selection Decision Guide

| Your situation | First choice | Also consider | Avoid |
|---|---|---|---|
| Pairwise outcomes, static skill, want a leaderboard | BT via MM (§1.1) | Rank Centrality (§3.1) at scale; Bayesian BT (§11.1) for CIs | Online Elo (§12.3) |
| Pairwise outcomes + many ties | Tie-aware BT (§1.2; §12.4) | Davidson vs Rao-Kupper by tie pattern | Discarding ties silently |
| Margins/scores available | Massey (§5.1), Keener (§3.3) | HodgeRank on margin flows (§3.6); MOV-Elo (§2.1) if online | Win-only methods (wasteful) |
| Skill drifts over time, streaming | Glicko-2 (§2.3) | Elo (simplicity); WHR (§2.6) for batch accuracy | Static BT refit naïvely |
| Teams of individuals, multiplayer | Weng-Lin/OpenSkill (§2.5) | TrueSkill (§2.4) | Per-player Elo on team results |
| Multiway choices / full rankings | Plackett-Luce via LSR/I-LSR (§1.4, §3.2) | Mallows/Kemeny for consensus-only | Decomposing to pairs ad hoc |
| Many judges' complete rankings | Kemeny (§6.3) | Borda (instant); footrule (poly-time guarantee, §6.5) | — |
| Partial, overlapping top-k lists | MC4 (§6.4) | Borda on shared items | Kemeny (ill-posed on partial lists) |
| Crowdsourced comparisons, uneven annotators | Crowd-BT (§11.2) | + style/position covariates (§12.2) | Trusting raw majorities |
| Suspected cycles / matchup effects | HodgeRank audit first (§3.6) | Blade-Chest, mElo (§9) if curl is high | Forcing a scalar and shipping it |
| Comparisons are expensive, you choose pairs | Just-Sort-It (§8.3) | Dueling bandits (§8.1) if live | Exhaustive all-pairs |
| Have a graph, no explicit comparisons | PageRank (§4.4) | BiRank (bipartite, §4.7); harmonic centrality (axiom-clean, §4.10); Katz (DAGs, §4.3) | Eigenvector centrality on digraphs |
| Interaction logs (users × items) | BiRank (§4.7) | PPR; counterfactual de-biasing (§10.2) if positions logged | Raw popularity |
| Trajectories with rewards, no head-to-head | MC V(s) + bootstrap (mcrl-rs, §13.1–13.2) | TD for long horizons (§13.3); then feed §1/§6 with $P(V_a > V_b)$ edges | Pretending sessions are matches |
| Ranking frozen models (LLM eval) | BT + bootstrap (§12.1) | + style control (§12.2), tie modeling (§12.4); social choice across suites (§12.6) | Online Elo; mean-score suite averaging |
| Need error bars on *any* of the above | Bootstrap it (§11.4) | Bayesian BT (§11.1) | Asymptotic SEs under dependence |
| Just need a sane baseline today | Wilson lower bound (§7.1) | Borda counting (§6.1) | Raw win % |

---

## 16. Propagon Coverage Map

Status of every method family against the current codebase (CLI subcommands in `src/main.rs`). "Recommended" candidates are this survey's suggested PRD priorities, chosen for value-to-effort on top of existing machinery.

### Implemented

| Method | Subcommand | Module | Notes |
|---|---|---|---|
| Bradley-Terry (MM) | `btm-mm` | `src/mm.rs` | Hunter 2004 MM; connectivity mitigations built in |
| Bradley-Terry (logistic SGD) | `btm-lr` | `src/lr.rs` | Streaming-friendly; thrifty mode |
| Glicko-2 | `glicko2` | `src/g2.rs` | Batch periods; τ; CI output |
| Luce Spectral Ranking | `lsr` | `src/lsr.rs` | Power-method & Monte Carlo estimators |
| Gaussian RUM (ES) | `es-rum` | `src/esrum.rs` | (μ, σ) per entity; relative-only caveat |
| Kemeny consensus (heuristic) | `kemeny` | `src/kemeny.rs` + `src/de.rs` | Insertion & differential evolution |
| Wilson-score win rate | `rate` | `src/rate.rs` | P50/P90/P95 bounds |
| PageRank | `page-rank` | `src/pr.rs` | Damping; 3 sink policies |
| BiRank | `birank` | `src/birank.rs` | Bipartite co-ranking |
| Components utility | `extract-components` | `src/cc.rs` | Connectivity triage (§14.1) |

### Sibling (mcrl-rs)

| Method | Notes |
|---|---|
| Monte Carlo V(s) (first/every-visit) | JSONL trajectories; γ; mean/median; Winsorization |
| Bootstrap / permutation / Bayesian-bootstrap comparison | Pairwise $P(V_B > V_A)$ tables; comparison groups |

### Candidates — recommended (PRD priorities)

| Method (§) | Effort | Leverage |
|---|---|---|
| BT extensions: ties, home, covariates (§1.2, §10.1) | Low | Arena-grade likelihoods; `.features.id` already exists |
| Rank Centrality (§3.1) | Low | Small delta on `lsr.rs`; web-scale BT |
| HodgeRank + LS-on-graphs (§3.6) | Medium | Unique "rankability audit" differentiator |
| Massey + Colley (§5.1–5.2) | Low | One sparse solver, two classic methods |
| Borda + Copeland (§6.1–6.2) | Trivial | Baselines + social-choice entry |
| Elo (§2.1) | Trivial | Ubiquitous ask; completes online story |
| Weng-Lin/OpenSkill (§2.5) | Medium | Teams/multiplayer without TrueSkill's EP |
| Bayesian BT (§11.1) | Medium | Posteriors; sparse-data robustness |
| Crowd-BT (§11.2) | Medium | Annotator column; crowdsourcing market |
| Library-wide `--bootstrap N` (§11.4) | Medium | Error bars on every scorer at once |
| I-LSR + native multiway input (§3.2) | Medium | Exact PL MLE; ranking-file ingestion |

### Candidates — worthwhile, lower priority

mElo (§9.2) · Blade-Chest (§9.1) · WHR (§2.6) · Keener (§3.3) · footrule 2-approx as Kemeny init (§6.5) · MC4 aggregation (§6.4) · Katz / eigenvector / HITS / LeaderRank / harmonic centrality (§4) · k-core (§4.11) · SerialRank (§3.5) · offense-defense (§5.3) · Mallows φ (§1.7) · TD extension in mcrl-rs (§13.3).

### Out-of-scope (documented for completeness)

TrueSkill proper (§2.4 — Weng-Lin covers the need) · dynamic-BT state-space family (§2.7) · dueling bandits as a live service (§8) · α-Rank (§9.3 — payoff-tensor input) · feature-based LTR & differentiable ranking (§10.3–10.5) · RLHF training loops (§10.6) · GP preference learning (§11.3) · click-log counterfactual LTR (§10.2) · VAEP-style domain valuation (§13.4) · OPE (§13.5) · geodesic centralities beyond harmonic (§4.10) · economic GARP testing (§14.6).

---

## 17. References

- Adams, R.P. & Zemel, R.S. (2011). *Ranking via Sinkhorn propagation.* [arXiv:1106.1925](https://arxiv.org/abs/1106.1925).
- Afriat, S.N. (1967). *The construction of utility functions from expenditure data.* International Economic Review 8(1), 67–77.
- Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.
- Agresti, A. & Coull, B.A. (1998). *Approximate is better than "exact" for interval estimation of binomial proportions.* The American Statistician 52(2), 119–126.
- Ailon, N., Charikar, M. & Newman, A. (2008). *Aggregating inconsistent information: Ranking and clustering.* Journal of the ACM 55(5), Article 23.
- Aldous, D. (2017). *Elo ratings and the sports model: A neglected topic in applied probability?* Statistical Science 32(4), 616–629.
- Ameli, S., Zhuang, S., Stoica, I. & Mahoney, M.W. (2025). *A statistical framework for ranking LLM-based chatbots.* ICLR 2025. [arXiv:2412.18407](https://arxiv.org/abs/2412.18407).
- Atkins, J.E., Boman, E.G. & Hendrickson, B. (1998). *A spectral algorithm for seriation and the consecutive ones problem.* SIAM Journal on Computing 28(1), 297–310.
- Balduzzi, D., Tuyls, K., Pérolat, J. & Graepel, T. (2018). *Re-evaluating evaluation.* NeurIPS 2018. [arXiv:1806.02643](https://arxiv.org/abs/1806.02643).
- Bartholdi, J., Tovey, C.A. & Trick, M.A. (1989). *Voting schemes for which it can be difficult to tell who won the election.* Social Choice and Welfare 6(2), 157–165.
- Bavelas, A. (1950). *Communication patterns in task-oriented groups.* Journal of the Acoustical Society of America 22(6), 725–730.
- Bengs, V., Busa-Fekete, R., El Mesaoudi-Paul, A. & Hüllermeier, E. (2021). *Preference-based online learning with dueling bandits: A survey.* JMLR 22(7), 1–108. [arXiv:1807.11398](https://arxiv.org/abs/1807.11398).
- Blondel, M., Teboul, O., Berthet, Q. & Djolonga, J. (2020). *Fast differentiable sorting and ranking.* ICML 2020. [arXiv:2002.08871](https://arxiv.org/abs/2002.08871).
- Boldi, P. & Vigna, S. (2014). *Axioms for centrality.* Internet Mathematics 10(3–4), 222–262. [arXiv:1308.2140](https://arxiv.org/abs/1308.2140).
- Bonacich, P. (1972). *Factoring and weighting approaches to status scores and clique identification.* Journal of Mathematical Sociology 2(1), 113–120.
- Bonacich, P. (1987). *Power and centrality: A family of measures.* American Journal of Sociology 92(5), 1170–1182.
- de Borda, J.-C. (1781). *Mémoire sur les élections au scrutin.* Histoire de l'Académie Royale des Sciences, Paris.
- Boubdir, M., Kim, E., Ermis, B., Hooker, S. & Fadaee, M. (2024). *Elo Uncovered: Robustness and best practices in language model evaluation.* [arXiv:2311.17295](https://arxiv.org/abs/2311.17295).
- Bradley, R.A. & Terry, M.E. (1952). *Rank analysis of incomplete block designs: I. The method of paired comparisons.* Biometrika 39(3/4), 324–345.
- Brandes, U. (2001). *A faster algorithm for betweenness centrality.* Journal of Mathematical Sociology 25(2), 163–177.
- Braverman, M. & Mossel, E. (2008). *Noisy sorting without resampling.* SODA 2008. [arXiv:0707.1051](https://arxiv.org/abs/0707.1051).
- Brier, G.W. (1950). *Verification of forecasts expressed in terms of probability.* Monthly Weather Review 78(1), 1–3.
- Brin, S. & Page, L. (1998). *The anatomy of a large-scale hypertextual Web search engine.* Computer Networks and ISDN Systems 30(1–7), 107–117.
- Burges, C.J.C. (2010). *From RankNet to LambdaRank to LambdaMART: An overview.* Microsoft Research Technical Report MSR-TR-2010-82.
- Burges, C.J.C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N. & Hullender, G. (2005). *Learning to rank using gradient descent.* ICML 2005.
- Callaghan, T., Mucha, P.J. & Porter, M.A. (2007). *Random walker ranking for NCAA Division I-A football.* American Mathematical Monthly 114(9), 761–777.
- Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F. & Li, H. (2007). *Learning to rank: From pairwise approach to listwise approach.* ICML 2007.
- Caron, F. & Doucet, A. (2012). *Efficient Bayesian inference for generalized Bradley-Terry models.* Journal of Computational and Graphical Statistics 21(1), 174–196. [arXiv:1011.1761](https://arxiv.org/abs/1011.1761).
- Cattelan, M. (2012). *Models for paired comparison data: A review with emphasis on dependent data.* Statistical Science 27(3), 412–433. [arXiv:1210.1016](https://arxiv.org/abs/1210.1016).
- Cattelan, M., Varin, C. & Firth, D. (2013). *Dynamic Bradley-Terry modelling of sports tournaments.* Journal of the Royal Statistical Society: Series C 62(1), 135–150.
- Chen, S. & Joachims, T. (2016). *Modeling intransitivity in matchup and comparison data.* WSDM 2016.
- Chen, X., Bennett, P.N., Collins-Thompson, K. & Horvitz, E. (2013). *Pairwise ranking aggregation in a crowdsourced setting.* WSDM 2013.
- Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A.N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J.E. & Stoica, I. (2024). *Chatbot Arena: An open platform for evaluating LLMs by human preference.* ICML 2024. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132).
- Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S. & Amodei, D. (2017). *Deep reinforcement learning from human preferences.* NeurIPS 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741).
- Chu, W. & Ghahramani, Z. (2005). *Preference learning with Gaussian processes.* ICML 2005.
- Chuklin, A., Markov, I. & de Rijke, M. (2015). *Click Models for Web Search.* Morgan & Claypool.
- Colley, W.N. (2002). *Colley's bias free college football ranking method.* Technical report, [colleyrankings.com](https://colleyrankings.com/matrate.pdf).
- Copeland, A.H. (1951). *A "reasonable" social welfare function.* Seminar on Mathematics in Social Sciences, University of Michigan.
- Coulom, R. (2008). *Whole-history rating: A Bayesian rating system for players of time-varying strength.* Computers and Games 2008, LNCS 5131. [Author PDF](https://www.remi-coulom.fr/WHR/WHR.pdf).
- Craswell, N., Zoeter, O., Taylor, M. & Ramsey, B. (2008). *An experimental comparison of click position-bias models.* WSDM 2008.
- Cuturi, M., Teboul, O. & Vert, J.-P. (2019). *Differentiable ranking and sorting using optimal transport.* NeurIPS 2019. [arXiv:1905.11885](https://arxiv.org/abs/1905.11885).
- Czarnecki, W.M., Gidel, G., Tracey, B., Tuyls, K., Omidshafiei, S., Balduzzi, D. & Jaderberg, M. (2020). *Real world games look like spinning tops.* NeurIPS 2020. [arXiv:2004.09468](https://arxiv.org/abs/2004.09468).
- Dangauthier, P., Herbrich, R., Minka, T. & Graepel, T. (2008). *TrueSkill through time: Revisiting the history of chess.* NIPS 2007.
- Davidson, R.R. (1970). *On extending the Bradley-Terry model to accommodate ties in paired comparison experiments.* JASA 65(329), 317–328.
- Decroos, T., Bransen, L., Van Haaren, J. & Davis, J. (2019). *Actions speak louder than goals: Valuing player actions in soccer.* KDD 2019. [arXiv:1802.07127](https://arxiv.org/abs/1802.07127).
- Deng, H., Lyu, M.R. & King, I. (2009). *A generalized Co-HITS algorithm and its application to bipartite graphs.* KDD 2009.
- Diaconis, P. & Graham, R.L. (1977). *Spearman's footrule as a measure of disarray.* Journal of the Royal Statistical Society: Series B 39(2), 262–268.
- Dudík, M., Langford, J. & Li, L. (2011). *Doubly robust policy evaluation and learning.* ICML 2011. [arXiv:1103.4601](https://arxiv.org/abs/1103.4601).
- Dwork, C., Kumar, R., Naor, M. & Sivakumar, D. (2001). *Rank aggregation methods for the Web.* WWW 2001.
- Efron, B. (1979). *Bootstrap methods: Another look at the jackknife.* Annals of Statistics 7(1), 1–26.
- Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco.
- Estrada, E. & Rodríguez-Velázquez, J.A. (2005). *Subgraph centrality in complex networks.* Physical Review E 71, 056103.
- Fahrmeir, L. & Tutz, G. (1994). *Dynamic stochastic models for time-dependent ordered paired comparison systems.* JASA 89(428), 1438–1449.
- Fang, S., Han, R., Luo, Y. & Xu, Y. (2026). *Recent advances in the Bradley-Terry model: Theory, algorithms, and applications.* [arXiv:2601.14727](https://arxiv.org/abs/2601.14727).
- Fogel, F., d'Aspremont, A. & Vojnović, M. (2014). *SerialRank: Spectral ranking using seriation.* NIPS 2014. [arXiv:1406.5370](https://arxiv.org/abs/1406.5370).
- Ford, L.R., Jr. (1957). *Solution of a ranking problem from binary comparisons.* American Mathematical Monthly 64(8, part 2), 28–33.
- Freeman, L.C. (1977). *A set of measures of centrality based on betweenness.* Sociometry 40(1), 35–41.
- Freeman, L.C. (1979). *Centrality in social networks: Conceptual clarification.* Social Networks 1(3), 215–239.
- Glickman, M.E. (1999). *Parameter estimation in large dynamic paired comparison experiments.* Journal of the Royal Statistical Society: Series C 48(3), 377–394.
- Glickman, M.E. (2001). *Dynamic paired comparison models with stochastic variances.* Journal of Applied Statistics 28(6), 673–689.
- Glickman, M.E. (2022). *Example of the Glicko-2 system.* Technical note, [glicko.net](http://www.glicko.net/glicko/glicko2.pdf).
- Govan, A.Y., Langville, A.N. & Meyer, C.D. (2009). *Offense-defense approach to ranking team sports.* Journal of Quantitative Analysis in Sports 5(1).
- Gyöngyi, Z., Garcia-Molina, H. & Pedersen, J. (2004). *Combating Web spam with TrustRank.* VLDB 2004.
- Hamilton, I., Tawn, N. & Firth, D. (2023). *The many routes to the ubiquitous Bradley-Terry model.* [arXiv:2312.13619](https://arxiv.org/abs/2312.13619).
- He, X., Gao, M., Kan, M.-Y. & Wang, D. (2017). *BiRank: Towards ranking on bipartite graphs.* IEEE TKDE 29(1), 57–71. [arXiv:1708.04396](https://arxiv.org/abs/1708.04396).
- Heckel, R., Shah, N.B., Ramchandran, K. & Wainwright, M.J. (2019). *Active ranking from pairwise comparisons and when parametric assumptions do not help.* Annals of Statistics 47(6), 3099–3126. [arXiv:1606.08842](https://arxiv.org/abs/1606.08842).
- Herbrich, R., Minka, T. & Graepel, T. (2007). *TrueSkill: A Bayesian skill rating system.* NIPS 2006.
- Hirani, A.N., Kalyanaraman, K. & Watts, S. (2011). *Least squares ranking on graphs.* [arXiv:1011.1716](https://arxiv.org/abs/1011.1716).
- Huang, T.-K., Weng, R.C. & Lin, C.-J. (2006). *Generalized Bradley-Terry models and multi-class probability estimates.* JMLR 7, 85–115.
- Hunter, D.R. (2004). *MM algorithms for generalized Bradley-Terry models.* Annals of Statistics 32(1), 384–406.
- Hvattum, L.M. & Arntzen, H. (2010). *Using ELO ratings for match result prediction in association football.* International Journal of Forecasting 26(3), 460–470.
- Jamieson, K.G. & Nowak, R. (2011). *Active ranking using pairwise comparisons.* NIPS 2011. [arXiv:1109.3701](https://arxiv.org/abs/1109.3701).
- Järvelin, K. & Kekäläinen, J. (2002). *Cumulated gain-based evaluation of IR techniques.* ACM TOIS 20(4), 422–446.
- Jiang, X., Lim, L.-H., Yao, Y. & Ye, Y. (2011). *Statistical ranking and combinatorial Hodge theory.* Mathematical Programming 127(1), 203–244. [arXiv:0811.1067](https://arxiv.org/abs/0811.1067).
- Joachims, T., Swaminathan, A. & Schnabel, T. (2017). *Unbiased learning-to-rank with biased feedback.* WSDM 2017. [arXiv:1608.04468](https://arxiv.org/abs/1608.04468).
- Joshy, V. (2024). *OpenSkill: A faster asymmetric multi-team, multiplayer rating system.* [arXiv:2401.05451](https://arxiv.org/abs/2401.05451).
- Katz, L. (1953). *A new status index derived from sociometric analysis.* Psychometrika 18(1), 39–43.
- Keener, J.P. (1993). *The Perron-Frobenius theorem and the ranking of football teams.* SIAM Review 35(1), 80–93.
- Kemeny, J.G. (1959). *Mathematics without numbers.* Daedalus 88(4), 577–591.
- Kendall, M.G. (1938). *A new measure of rank correlation.* Biometrika 30(1/2), 81–93.
- Kenyon-Mathieu, C. & Schudy, W. (2007). *How to rank with few errors.* STOC 2007.
- Kitsak, M., Gallos, L.K., Havlin, S., Liljeros, F., Muchnik, L., Stanley, H.E. & Makse, H.A. (2010). *Identification of influential spreaders in complex networks.* Nature Physics 6, 888–893. [arXiv:1001.5285](https://arxiv.org/abs/1001.5285).
- Kleinberg, J.M. (1999). *Authoritative sources in a hyperlinked environment.* Journal of the ACM 46(5), 604–632.
- Lanctot, M., Larson, K., Bachrach, Y., Marris, L., Li, Z., Bhoopchand, A., Anthony, T., Tanner, B. & Koop, A. (2023). *Evaluating agents using social choice theory.* [arXiv:2312.03121](https://arxiv.org/abs/2312.03121).
- Langville, A.N. & Meyer, C.D. (2006). *Google's PageRank and Beyond: The Science of Search Engine Rankings.* Princeton University Press.
- Langville, A.N. & Meyer, C.D. (2012). *Who's #1? The Science of Rating and Ranking.* Princeton University Press.
- Lempel, R. & Moran, S. (2001). *SALSA: The stochastic approach for link-structure analysis.* ACM TOIS 19(2), 131–160.
- LMSYS (2024). *Does style matter? Disentangling style and substance in Chatbot Arena.* [lmsys.org blog, 2024-08-28](https://lmsys.org/blog/2024-08-28-style-control/).
- Lü, L., Zhang, Y.-C., Yeung, C.H. & Zhou, T. (2011). *Leaders in social networks, the Delicious case.* PLoS ONE 6(6), e21202.
- Luce, R.D. (1959). *Individual Choice Behavior: A Theoretical Analysis.* Wiley.
- Mallows, C.L. (1957). *Non-null ranking models. I.* Biometrika 44(1/2), 114–130.
- Marchiori, M. & Latora, V. (2000). *Harmony in the small-world.* Physica A 285(3–4), 539–546. [arXiv:cond-mat/0008357](https://arxiv.org/abs/cond-mat/0008357).
- Marden, J.I. (1995). *Analyzing and Modeling Rank Data.* Chapman & Hall.
- Massey, K. (1997). *Statistical models applied to the rating of sports teams.* Bachelor's thesis, Bluefield College. [masseyratings.com](https://masseyratings.com/theory/massey97.pdf).
- Maystre, L. & Grossglauser, M. (2015). *Fast and accurate inference of Plackett-Luce models.* NIPS 2015. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2015/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html).
- Maystre, L. & Grossglauser, M. (2017). *Just sort it! A simple and effective approach to active preference learning.* ICML 2017. [arXiv:1502.05556](https://arxiv.org/abs/1502.05556).
- Maystre, L., Kristof, V. & Grossglauser, M. (2019). *Pairwise comparisons with flexible time-dynamics.* KDD 2019. [arXiv:1903.07746](https://arxiv.org/abs/1903.07746).
- McFadden, D. (1974). *Conditional logit analysis of qualitative choice behavior.* In P. Zarembka (ed.), *Frontiers in Econometrics*, Academic Press.
- Mena, G., Belanger, D., Linderman, S. & Snoek, J. (2018). *Learning latent permutations with Gumbel-Sinkhorn networks.* ICLR 2018. [arXiv:1802.08665](https://arxiv.org/abs/1802.08665).
- Minka, T., Cleven, R. & Zaykov, Y. (2018). *TrueSkill 2: An improved Bayesian skill rating system.* Microsoft Research Technical Report MSR-TR-2018-8.
- Mosteller, F. (1951). *Remarks on the method of paired comparisons: I.* Psychometrika 16(1), 3–9.
- Negahban, S., Oh, S. & Shah, D. (2017). *Rank Centrality: Ranking from pairwise comparisons.* Operations Research 65(1), 266–287. [arXiv:1209.1688](https://arxiv.org/abs/1209.1688).
- Newman, M.E.J. (2005). *A measure of betweenness centrality based on random walks.* Social Networks 27(1), 39–54.
- Omidshafiei, S., Papadimitriou, C., Piliouras, G., Tuyls, K., Rowland, M., Lespiau, J.-B., Czarnecki, W.M., Lanctot, M., Pérolat, J. & Munos, R. (2019). *α-Rank: Multi-agent evaluation by evolution.* Scientific Reports 9, 9937. [arXiv:1903.01373](https://arxiv.org/abs/1903.01373).
- Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155).
- Plackett, R.L. (1975). *The analysis of permutations.* Journal of the Royal Statistical Society: Series C 24(2), 193–202.
- Precup, D., Sutton, R.S. & Singh, S. (2000). *Eligibility traces for off-policy policy evaluation.* ICML 2000.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D. & Finn, C. (2023). *Direct preference optimization: Your language model is secretly a reward model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
- Rao, P.V. & Kupper, L.L. (1967). *Ties in paired-comparison experiments: A generalization of the Bradley-Terry model.* JASA 62(317), 194–204.
- Routley, K. & Schulte, O. (2015). *A Markov game model for valuing player actions in ice hockey.* UAI 2015.
- Rubin, D.B. (1981). *The Bayesian bootstrap.* Annals of Statistics 9(1), 130–134.
- Saari, D.G. & Merlin, V.R. (1996). *The Copeland method I: Relationships and the dictionary.* Economic Theory 8, 51–76.
- Schulze, M. (2011). *A new monotonic, clone-independent, reversal symmetric, and Condorcet-consistent single-winner election method.* Social Choice and Welfare 36(2), 267–303.
- Seidman, S.B. (1983). *Network structure and minimum degree.* Social Networks 5(3), 269–287.
- Shah, N.B. & Wainwright, M.J. (2018). *Simple, robust and optimal ranking from pairwise comparisons.* JMLR 18(199), 1–38. [arXiv:1512.08949](https://arxiv.org/abs/1512.08949).
- Shah, N.B., Balakrishnan, S., Guntuboyina, A. & Wainwright, M.J. (2017). *Stochastically transitive models for pairwise comparisons: Statistical and computational issues.* IEEE Transactions on Information Theory 63(2), 934–959. [arXiv:1510.05610](https://arxiv.org/abs/1510.05610).
- Stern, H. (1990). *Models for distributions on permutations.* JASA 85(410), 558–564.
- Sun, H., Shen, Y. & Ton, J.-F. (2024). *Rethinking Bradley-Terry models in preference-based reward modeling: Foundations, theory, and alternatives.* [arXiv:2411.04991](https://arxiv.org/abs/2411.04991).
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Thurstone, L.L. (1927). *A law of comparative judgment.* Psychological Review 34(4), 273–286.
- Tideman, T.N. (1987). *Independence of clones as a criterion for voting rules.* Social Choice and Welfare 4(3), 185–206.
- Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.
- Tsukida, K. & Gupta, M.R. (2011). *How to analyze paired comparison data.* University of Washington Technical Report UWEETR-2011-0004.
- Varian, H.R. (1982). *The nonparametric approach to demand analysis.* Econometrica 50(4), 945–973.
- Weng, R.C. & Lin, C.-J. (2011). *A Bayesian approximation method for online ranking.* JMLR 12, 267–300. [jmlr.org](https://www.jmlr.org/papers/v12/weng11a.html).
- Wilson, E.B. (1927). *Probable inference, the law of succession, and statistical inference.* JASA 22(158), 209–212.
- Xia, F., Liu, T.-Y., Wang, J., Zhang, W. & Li, H. (2008). *Listwise approach to learning to rank: Theory and algorithm.* ICML 2008.
- Young, H.P. (1988). *Condorcet's theory of voting.* American Political Science Review 82(4), 1231–1244.
- Young, H.P. & Levenglick, A. (1978). *A consistent extension of Condorcet's election principle.* SIAM Journal on Applied Mathematics 35(2), 285–300.
- Yue, Y., Broder, J., Kleinberg, R. & Joachims, T. (2012). *The K-armed dueling bandits problem.* Journal of Computer and System Sciences 78(5), 1538–1556.
- Zermelo, E. (1929). *Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung.* Mathematische Zeitschrift 29, 436–460.









