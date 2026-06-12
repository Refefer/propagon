//! `propagon` — ranking from revealed preferences.
//!
//! Subcommands are grouped by the shape of the input data:
//!
//! - `propagon tournament <algo> <path>` — pairwise comparison rankers
//!   (win-rate, elo, glicko2, bradley-terry-model, thurstone-mosteller,
//!   luce-spectral-ranking, i-luce-spectral-ranking, rank-centrality,
//!   serial-rank, random-walker, random-utility-model, kemeny, borda-count,
//!   copeland, massey, colley, keener, offense-defense,
//!   bayesian-bradley-terry, hodge-rank) over `winner loser [weight]` rows.
//! - `propagon rankings <algo> <path>` — ballot aggregation (plackett-luce,
//!   i-luce-spectral-ranking, markov-chain, borda-count, kemeny, mallows,
//!   footrule) over one-ballot-per-line files.
//! - `propagon graph <algo> <path>` — node-importance rankers and utilities
//!   (page-rank with optional teleport seeds, birank, hits, katz-centrality,
//!   leader-rank, harmonic, degree, k-core, components) over
//!   `src dst [weight]` edges.
//! - `propagon bandit <policy> <path>` — multi-armed bandits (greedy,
//!   epsilon-greedy, upper-confidence-bound, kl-ucb, exp3, thompson-beta,
//!   thompson-gaussian) over `arm reward` rows.
//!
//! Cross-cutting: string ids natively, `--save-state`/`--load-state` for
//! incremental and warm-started runs, `--format tsv|jsonl`, `--threads`.

mod emit;
mod io;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use propagon::algos::{
    Aggregate, Bandit, BanditModel, BanditPolicy, BayesianBradleyTerry, BehaviorCloning, BiRank,
    BladeChest, BladeChestVariant, Borda, BradleyTerryLR, BradleyTerryMM, Colley, Confidence,
    Copeland, CovariateBt, CrowdBt, Degree, Direction, EdgeCost, Elo, EsRum, Estimator, Footrule,
    GammaPolicy, GeneralizedBt, Glicko2, Granularity, Harmonic, Hits, HodgeFlow, HodgeRank,
    HomeAdvantage, ILsr, KCore, Katz, Keener, Kemeny, KemenyAlgo, KemenyPasses, LeaderRank, LinUcb,
    LinUcbModel, Lsr, MElo, Mallows, Massey, Mc4, McValue, MovElo, NashAveraging, OffenseDefense,
    PageRank, PairwiseTests, PlackettLuce, RandomWalker, RankCentrality, ResampleScheme,
    RumDistribution, SerialRank, Sink, SlidingWindowUcb, SourceBudget, SwUcbModel, TdValue,
    TeamAggregate, TeamBradleyTerry, Teleport, ThurstoneMosteller, TieModel, ValueCompare, Visit,
    WengLin, WengLinVariant, Whr, WinRate, Winsorize, extract_components,
};
use propagon::{
    Error, FitOptions, MarginTies, OnlineRanker, Progress, RankModel, Ranker, Resample, Result,
    TiePolicy,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------- progress

/// Renders core progress on stderr when it is a terminal.
#[derive(Default)]
struct CliProgress {
    bar: Mutex<Option<indicatif::ProgressBar>>,
}

impl Progress for CliProgress {
    fn start(&self, phase: &str, total: Option<u64>) {
        let pb = match total {
            Some(t) => indicatif::ProgressBar::new(t),
            None => indicatif::ProgressBar::new_spinner(),
        };
        pb.set_style(
            indicatif::ProgressStyle::with_template("[{msg}] {wide_bar} {pos}/{len} {eta_precise}")
                .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()),
        );
        pb.set_message(phase.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(200));
        *self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(pb);
    }

    fn update(&self, done: u64) {
        if let Some(pb) = self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            pb.set_position(done);
        }
    }

    fn message(&self, msg: &str) {
        if let Some(pb) = self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            pb.set_message(msg.to_string());
        }
    }

    fn finish(&self) {
        if let Some(pb) = self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            pb.finish_and_clear();
        }
    }
}

// ---------------------------------------------------------------- cli

fn flag(name: &'static str, help: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .help(help)
        .action(ArgAction::SetTrue)
}

fn opt<T: Clone + Send + Sync + std::str::FromStr + 'static>(
    name: &'static str,
    help: &'static str,
) -> Arg
where
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    Arg::new(name)
        .long(name)
        .help(help)
        .value_parser(value_parser!(T))
}

/// Adds the input-path positional every leaf command takes.
fn with_path(cmd: Command) -> Command {
    cmd.arg(Arg::new("path").required(true).help("Input data file"))
}

fn cli() -> Command {
    Command::new("propagon")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Ranking from revealed preferences (see docs/algorithms.md)")
        .subcommand_required(true)
        .arg(opt::<usize>("threads", "Worker threads (default: all cores)").global(true))
        .arg(
            Arg::new("format")
                .long("format")
                .value_parser(["tsv", "jsonl"])
                .default_value("tsv")
                .help("Output format: 'id: score' tsv or model-state jsonl")
                .global(true),
        )
        .arg(opt::<PathBuf>("save-state", "Write the fitted model state to this file").global(true))
        .arg(
            opt::<PathBuf>(
                "load-state",
                "Resume from a saved model state (update or warm start)",
            )
            .global(true),
        )
        .subcommand(tournament_cmd())
        .subcommand(rankings_cmd())
        .subcommand(crowd_cmd())
        .subcommand(matchups_cmd())
        .subcommand(graph_cmd())
        .subcommand(bandit_cmd())
        .subcommand(trajectories_cmd())
}

fn tournament_cmd() -> Command {
    Command::new("tournament")
        .about(
            "Rank entities from game results: 'side1<TAB>side2<TAB>threshold[<TAB>count]' \
             rows (threshold >0: side 1 wins by that margin, <0: side 2 wins, =0: tie)",
        )
        .subcommand_required(true)
        .arg(
            opt::<usize>(
                "bootstrap",
                "Resample-and-refit N times; emit score and rank intervals",
            )
            .global(true),
        )
        .arg(
            opt::<f64>(
                "bootstrap-credible",
                "Central interval mass for --bootstrap (default 0.95)",
            )
            .global(true),
        )
        .arg(opt::<u64>("bootstrap-seed", "Resampling seed for --bootstrap").global(true))
        .arg(
            opt::<usize>(
                "min-count",
                "Iteratively drop games whose players appear fewer times",
            )
            .global(true),
        )
        .arg(
            flag(
                "groups-are-separate",
                "Treat blank-line-separated batches as rating periods",
            )
            .global(true),
        )
        .arg(
            Arg::new("ties")
                .long("ties")
                .value_parser(["error", "discard", "half-win"])
                .default_value("error")
                .help("How win/loss algorithms lower tie games")
                .global(true),
        )
        .arg(
            Arg::new("margin-ties")
                .long("margin-ties")
                .value_parser(["error", "discard", "zero"])
                .default_value("error")
                .help("How margin algorithms (massey, keener, offense-defense) lower ties")
                .global(true),
        )
        .subcommand(with_path(
            Command::new("win-rate")
                .visible_alias("rate")
                .about("Win rates with Wilson confidence intervals")
                .arg(
                    Arg::new("confidence-interval")
                        .long("confidence-interval")
                        .value_parser(["0.95", "0.9", "0.90", "0.5"])
                        .default_value("0.95"),
                ),
        ))
        .subcommand(with_path(
            Command::new("elo")
                .about("Elo ratings (order-dependent online updates; ties score half)")
                .arg(opt::<f64>("k", "Update step size"))
                .arg(opt::<f64>("initial-rating", "Rating for unseen entities"))
                .arg(opt::<f64>("scale", "Logistic scale (default 400)"))
                .arg(flag(
                    "margin-of-victory",
                    "Scale K by ln(1+margin)/ln 2 (margin-1 wins match plain elo)",
                ))
                .arg(opt::<f64>(
                    "mov-exponent",
                    "Exponent on the margin multiplier (with --margin-of-victory)",
                )),
        ))
        .subcommand(with_path(
            Command::new("glicko2")
                .about("Glicko-2 ratings (periods via --groups-are-separate)")
                .arg(opt::<f64>("tau", "Volatility constraint (0.3-1.2)"))
                .arg(flag("use-mu", "Emit only the internal-scale rating")),
        ))
        .subcommand(with_path(
            Command::new("bradley-terry-model")
                .visible_alias("btm")
                .about("Bradley-Terry strengths; pick the estimator with --estimator")
                .arg(
                    Arg::new("estimator")
                        .long("estimator")
                        .value_parser(["mm", "sgd"])
                        .default_value("mm")
                        .help("mm: minorization-maximization; sgd: logistic gradient descent"),
                )
                .arg(opt::<usize>("iterations", "(mm) Maximum MM sweeps"))
                .arg(opt::<f64>("tolerance", "(mm) Convergence tolerance").alias("tol"))
                .arg(opt::<usize>(
                    "min-graph-size",
                    "(mm) Skip components smaller than this",
                ))
                .arg(flag(
                    "remove-total-losers",
                    "(mm) Also remove never-won entities",
                ))
                .arg(opt::<f64>(
                    "create-fake-games",
                    "(mm) Patch one-sided entities with fake games of this weight",
                ))
                .arg(opt::<usize>(
                    "random-subgraph-links",
                    "(mm) Random links between components",
                ))
                .arg(opt::<f64>(
                    "random-subgraph-weight",
                    "(mm) Weight of those links",
                ))
                .arg(opt::<u64>("seed", "(mm) Seed for random component links"))
                .arg(opt::<usize>("passes", "(sgd) Passes per period"))
                .arg(opt::<f64>("alpha", "(sgd) Learning rate"))
                .arg(opt::<f64>("decay", "(sgd) L2 shrinkage per pass"))
                .arg(flag("thrifty", "(sgd) Sequential in-place updates")),
        ))
        .subcommand(with_path(
            Command::new("luce-spectral-ranking")
                .visible_alias("lsr")
                .about("Luce spectral ranking (fast Plackett-Luce estimate)")
                .arg(opt::<usize>(
                    "steps",
                    "Power passes / walk steps (0 = auto)",
                ))
                .arg(
                    Arg::new("estimator")
                        .long("estimator")
                        .value_parser(["power", "monte-carlo"])
                        .default_value("power"),
                )
                .arg(opt::<u64>("seed", "Seed for Monte Carlo walks")),
        ))
        .subcommand(with_path(
            Command::new("rank-centrality")
                .about("Rank Centrality spectral Bradley-Terry estimate")
                .arg(opt::<usize>("iterations", "Maximum sweeps"))
                .arg(opt::<f64>("tolerance", "Early-exit tolerance")),
        ))
        .subcommand(with_path(
            Command::new("random-utility-model")
                .visible_alias("rum")
                .about("Gaussian random-utility model via evolution strategies")
                .arg(opt::<usize>("passes", "ES iterations"))
                .arg(opt::<f64>("alpha", "Initial perturbation scale"))
                .arg(opt::<f64>("gamma", "L2 regularization"))
                .arg(opt::<usize>("min-obs", "Minimum comparisons per entity"))
                .arg(opt::<usize>("prior", "Pseudo-count smoothing"))
                .arg(flag("fixed", "Pin all variances to 1 (Thurstone-style)"))
                .arg(opt::<u64>("seed", "Random seed")),
        ))
        .subcommand(with_path(
            Command::new("kemeny")
                .about("Kemeny-optimal consensus ranking")
                .arg(opt::<usize>(
                    "passes",
                    "Insertion passes / DE evaluations (0 = auto)",
                ))
                .arg(opt::<usize>("min-obs", "Minimum comparisons per entity"))
                .arg(
                    Arg::new("algo")
                        .long("algo")
                        .value_parser(["insertion", "de"])
                        .default_value("insertion"),
                )
                .arg(opt::<u64>("seed", "Seed for the DE search")),
        ))
        .subcommand(with_path(
            Command::new("borda-count")
                .visible_alias("borda")
                .about("Borda count (weighted win totals)"),
        ))
        .subcommand(with_path(
            Command::new("copeland").about("Copeland pairwise-majority scores"),
        ))
        .subcommand(with_path(
            Command::new("massey")
                .about("Massey least-squares ratings (weight column = margin of victory)")
                .arg(opt::<usize>("iterations", "Maximum solver iterations"))
                .arg(opt::<f64>("tolerance", "Relative residual target")),
        ))
        .subcommand(with_path(
            Command::new("colley")
                .about("Colley bias-free ratings from wins and losses only")
                .arg(opt::<usize>("iterations", "Maximum solver iterations"))
                .arg(opt::<f64>("tolerance", "Relative residual target")),
        ))
        .subcommand(with_path(
            Command::new("keener")
                .about("Keener eigenvector ratings (rows are 'scorer opponent amount')")
                .arg(flag("no-skew", "Skip Keener's blowout-damping skew"))
                .arg(flag(
                    "no-normalize-games",
                    "Skip the per-game row normalization",
                ))
                .arg(opt::<usize>("iterations", "Power-iteration budget"))
                .arg(opt::<f64>("tolerance", "Early-exit threshold")),
        ))
        .subcommand(with_path(
            Command::new("bayesian-bradley-terry")
                .visible_alias("bayes-bt")
                .about("Bradley-Terry posterior with credible intervals (Gibbs)")
                .arg(opt::<f64>("shape", "Gamma prior shape (1.0: MAP = MLE)"))
                .arg(opt::<f64>("rate", "Gamma prior rate (scale anchor)"))
                .arg(opt::<usize>("samples", "Posterior draws after burn-in"))
                .arg(opt::<usize>("burn-in", "Warm-up sweeps to discard"))
                .arg(opt::<f64>(
                    "credible",
                    "Central credible mass (default 0.9)",
                ))
                .arg(opt::<u64>("seed", "Sampler seed")),
        ))
        .subcommand(with_path(
            Command::new("hodge-rank")
                .visible_alias("hodge")
                .about("HodgeRank potentials plus a how-rankable-is-this diagnostic")
                .arg(
                    Arg::new("flow")
                        .long("flow")
                        .value_parser(["log-odds", "win-rate-delta", "mean-margin"])
                        .default_value("log-odds")
                        .help("Pairwise statistic to decompose"),
                )
                .arg(opt::<usize>("iterations", "Maximum solver iterations"))
                .arg(opt::<f64>("tolerance", "Relative residual target")),
        ))
        .subcommand(with_path(
            Command::new("thurstone-mosteller")
                .visible_alias("tm")
                .about("Thurstone-Mosteller Case V scale values (probit link)")
                .arg(opt::<usize>("iterations", "Maximum Newton sweeps"))
                .arg(opt::<f64>("tolerance", "Mean-change stopping rule"))
                .arg(opt::<f64>(
                    "pseudo-count",
                    "Virtual games both ways per observed pair (separation fix)",
                )),
        ))
        .subcommand(with_path(
            Command::new("i-luce-spectral-ranking")
                .visible_alias("ilsr")
                .about("I-LSR: iterated spectral solves to the exact Plackett-Luce MLE")
                .arg(opt::<usize>("outer", "Chain-rebuild iterations"))
                .arg(opt::<usize>(
                    "inner-steps",
                    "Power passes per stationary solve",
                ))
                .arg(opt::<f64>("tolerance", "Outer L1 stopping rule")),
        ))
        .subcommand(with_path(
            Command::new("serial-rank")
                .about("SerialRank seriation ordering (scores are Fiedler coordinates)")
                .arg(opt::<usize>("iterations", "Power-iteration budget"))
                .arg(opt::<f64>("tolerance", "Alignment stopping rule"))
                .arg(opt::<u64>("seed", "Start-vector seed")),
        ))
        .subcommand(with_path(
            Command::new("random-walker")
                .about("Random-walker rankings: fans follow winners with bias p")
                .arg(opt::<f64>(
                    "bias",
                    "Winner-following bias p, strictly in (0.5, 1)",
                ))
                .arg(opt::<usize>("iterations", "Maximum sweeps"))
                .arg(opt::<f64>("tolerance", "Early-exit tolerance")),
        ))
        .subcommand(with_path(
            Command::new("offense-defense")
                .visible_alias("od")
                .about("Offense-defense Sinkhorn ratings (margins as scores)")
                .arg(opt::<usize>("iterations", "Maximum Sinkhorn sweeps"))
                .arg(opt::<f64>("tolerance", "Relative-change stopping rule")),
        ))
        .subcommand(with_path(
            Command::new("generalized-bradley-terry")
                .visible_alias("gbt")
                .about("Bradley-Terry with native ties and home advantage (side 1 = home)")
                .arg(
                    Arg::new("tie-model")
                        .long("tie-model")
                        .value_parser(["none", "davidson", "rao-kupper"])
                        .default_value("davidson")
                        .help("Tie likelihood (davidson: P(tie) ~ nu sqrt(pi_i pi_j))"),
                )
                .arg(flag(
                    "home-advantage",
                    "Estimate a multiplicative advantage for side 1 (the home side)",
                ))
                .arg(opt::<usize>("iterations", "Maximum MM sweeps"))
                .arg(opt::<f64>("tolerance", "Convergence tolerance")),
        ))
        .subcommand(with_path(
            Command::new("team-bradley-terry")
                .visible_alias("team-bt")
                .about("Player strengths from team games (multi-player sides)")
                .arg(
                    Arg::new("aggregate")
                        .long("aggregate")
                        .value_parser(["additive", "product"])
                        .default_value("additive")
                        .help("Team strength: sum or product of member strengths"),
                )
                .arg(opt::<usize>("iterations", "Maximum sweeps"))
                .arg(opt::<f64>("tolerance", "Convergence tolerance")),
        ))
        .subcommand(with_path(
            Command::new("whole-history-rating")
                .visible_alias("whr")
                .about("WHR skill curves over rating periods (Wiener prior + Newton)")
                .arg(opt::<f64>(
                    "w2",
                    "Wiener variance per period (natural log-odds units)",
                ))
                .arg(opt::<f64>(
                    "prior-games",
                    "Virtual anchor draws on each player's first period",
                ))
                .arg(opt::<usize>("iterations", "Maximum Newton sweeps"))
                .arg(opt::<f64>("tolerance", "Max rating change to stop at"))
                .arg(flag(
                    "timeline",
                    "Emit every (period, rating, sd) point per player",
                )),
        ))
        .subcommand(with_path(
            Command::new("melo")
                .about("Multidimensional Elo: rating + cyclic vectors (mElo_2k)")
                .arg(opt::<usize>("k", "Cyclic dimension pairs"))
                .arg(opt::<f64>("lr-rating", "Rating learning rate"))
                .arg(opt::<f64>("lr-vector", "Cyclic-vector learning rate"))
                .arg(opt::<f64>("initial-rating", "Rating for unseen entities"))
                .arg(opt::<f64>("init-scale", "Stddev of fresh cyclic vectors"))
                .arg(opt::<u64>("seed", "Vector-initialization seed")),
        ))
        .subcommand(with_path(
            Command::new("nash-averaging")
                .visible_alias("nash")
                .about("Maxent-Nash weighted skill (redundancy-invariant evaluation)")
                .arg(opt::<usize>("iterations", "Multiplicative-weights budget"))
                .arg(opt::<f64>("tolerance", "Duality-gap target"))
                .arg(opt::<f64>("learning-rate", "MW step in (0, 1]"))
                .arg(opt::<usize>(
                    "anneal-every",
                    "Iterations per temperature halving",
                )),
        ))
        .subcommand(with_path(
            Command::new("blade-chest")
                .about("Blade-chest embeddings for intransitive matchups")
                .arg(
                    Arg::new("variant")
                        .long("variant")
                        .value_parser(["inner", "dist"])
                        .default_value("inner")
                        .help("Matchup score: inner products or squared distances"),
                )
                .arg(opt::<usize>("dims", "Embedding dimension"))
                .arg(opt::<f64>("lr", "SGD learning rate"))
                .arg(opt::<usize>("epochs", "Shuffled passes over the data"))
                .arg(opt::<f64>("l2", "Vector shrinkage per touch"))
                .arg(opt::<f64>("init-scale", "Stddev of initial vectors"))
                .arg(opt::<u64>("seed", "Init/shuffle seed")),
        ))
        .subcommand(with_path(
            Command::new("covariate-bradley-terry")
                .visible_alias("cbt")
                .about("Conditional logit: strengths from entity features (s = beta . x)")
                .arg(
                    opt::<PathBuf>("features", "File of 'entity x1 x2 ... xd' rows").required(true),
                )
                .arg(opt::<f64>("l2", "Ridge penalty on coefficients"))
                .arg(flag(
                    "intercepts",
                    "Add per-entity intercepts (partial pooling; needs --l2 > 0)",
                ))
                .arg(opt::<usize>("iterations", "Maximum Newton steps"))
                .arg(opt::<f64>("tolerance", "Gradient-norm stopping rule")),
        ))
}

fn rankings_cmd() -> Command {
    Command::new("rankings")
        .visible_alias("ballots")
        .about("Aggregate full or partial rankings: one ballot per line, best first")
        .subcommand_required(true)
        .arg(
            opt::<usize>(
                "bootstrap",
                "Resample-and-refit N times; emit score and rank intervals",
            )
            .global(true),
        )
        .arg(
            opt::<f64>(
                "bootstrap-credible",
                "Central interval mass for --bootstrap (default 0.95)",
            )
            .global(true),
        )
        .arg(opt::<u64>("bootstrap-seed", "Resampling seed for --bootstrap").global(true))
        .subcommand(with_path(
            Command::new("plackett-luce")
                .visible_alias("pl")
                .about("Plackett-Luce maximum likelihood (Hunter's MM)")
                .arg(opt::<usize>("iterations", "Maximum MM sweeps"))
                .arg(opt::<f64>("tolerance", "Mean-change stopping rule")),
        ))
        .subcommand(with_path(
            Command::new("markov-chain")
                .visible_alias("mc4")
                .about("MC4 Markov-chain aggregation (majority-move random walk)")
                .arg(opt::<f64>("damping", "Teleport mix in [0,1]"))
                .arg(opt::<usize>("iterations", "Power-iteration budget"))
                .arg(opt::<f64>("tolerance", "Early-exit threshold")),
        ))
        .subcommand(with_path(
            Command::new("borda-count")
                .visible_alias("borda")
                .about("Positional Borda points (m-rank per ballot)"),
        ))
        .subcommand(with_path(
            Command::new("kemeny")
                .about("Kemeny consensus over the ballots' implied pairwise preferences")
                .arg(opt::<usize>(
                    "passes",
                    "Insertion passes / DE evaluation budget",
                ))
                .arg(opt::<usize>(
                    "min-obs",
                    "Drop entities with fewer comparisons",
                ))
                .arg(
                    Arg::new("algo")
                        .long("algo")
                        .value_parser(["insertion", "de"])
                        .default_value("insertion")
                        .help("Search heuristic"),
                )
                .arg(opt::<u64>("seed", "Seed for the DE search")),
        ))
        .subcommand(with_path(
            Command::new("i-luce-spectral-ranking")
                .visible_alias("ilsr")
                .about("I-LSR Plackett-Luce MLE fitted natively on the ballots")
                .arg(opt::<usize>("outer", "Chain-rebuild iterations"))
                .arg(opt::<usize>(
                    "inner-steps",
                    "Power passes per stationary solve",
                ))
                .arg(opt::<f64>("tolerance", "Outer L1 stopping rule")),
        ))
        .subcommand(with_path(
            Command::new("mallows")
                .about("Mallows model: Kemeny consensus plus dispersion phi")
                .arg(opt::<usize>(
                    "passes",
                    "Kemeny insertion budget (omit for auto)",
                ))
                .arg(opt::<u64>("seed", "Consensus-search seed")),
        ))
        .subcommand(with_path(Command::new("footrule").about(
            "Footrule-optimal aggregation (exact matching; 2-approximates Kemeny)",
        )))
}

fn crowd_cmd() -> Command {
    Command::new("crowd")
        .about("Rank from annotator-tagged votes: 'annotator winner loser [weight]' rows")
        .subcommand_required(true)
        .arg(
            opt::<usize>(
                "bootstrap",
                "Resample-and-refit N times; emit score and rank intervals",
            )
            .global(true),
        )
        .arg(
            opt::<f64>(
                "bootstrap-credible",
                "Central interval mass for --bootstrap (default 0.95)",
            )
            .global(true),
        )
        .arg(opt::<u64>("bootstrap-seed", "Resampling seed for --bootstrap").global(true))
        .subcommand(with_path(
            Command::new("bradley-terry")
                .visible_alias("crowd-bt")
                .about("Joint item ranking and annotator-reliability estimation")
                .arg(opt::<f64>("lambda", "Virtual-node regularization weight"))
                .arg(opt::<f64>(
                    "eta-prior-alpha",
                    "Beta prior alpha on reliability",
                ))
                .arg(opt::<f64>(
                    "eta-prior-beta",
                    "Beta prior beta on reliability",
                ))
                .arg(opt::<usize>("iterations", "Outer EM iteration cap"))
                .arg(opt::<f64>(
                    "tolerance",
                    "Relative log-likelihood stopping rule",
                ))
                .arg(opt::<usize>("inner-sweeps", "MM sweeps per M-step")),
        ))
}

fn matchups_cmd() -> Command {
    Command::new("matchups")
        .about("Rate players from team matches: teams '|'-separated best-first, '=' ties players whitespace-separated")
        .subcommand_required(true)
        .subcommand(with_path(
            Command::new("weng-lin")
                .visible_alias("openskill")
                .about("Bayesian (mu, sigma) ratings for multi-team multiplayer matches")
                .arg(
                    Arg::new("variant")
                        .long("variant")
                        .value_parser(["bradley-terry", "thurstone-mosteller"])
                        .default_value("bradley-terry")
                        .help("Pairwise likelihood"),
                )
                .arg(opt::<f64>("mu", "Initial rating mean"))
                .arg(opt::<f64>("sigma", "Initial rating deviation"))
                .arg(opt::<f64>("beta", "Performance noise"))
                .arg(opt::<f64>("kappa", "Variance floor multiplier"))
                .arg(opt::<f64>("epsilon", "Draw margin (thurstone-mosteller)"))
                .arg(opt::<f64>("tau", "Pre-match sigma inflation (0 = off)"))
                .arg(
                    Arg::new("gamma")
                        .long("gamma")
                        .value_parser(["sigma-over-c", "one-over-k"])
                        .default_value("sigma-over-c")
                        .help("Variance-update damping policy"),
                ),
        ))
}

fn graph_cmd() -> Command {
    Command::new("graph")
        .about("Rank nodes by graph structure: 'src dst [weight]' edges")
        .subcommand_required(true)
        .arg(
            opt::<usize>(
                "bootstrap",
                "Resample-and-refit N times; emit score and rank intervals",
            )
            .global(true),
        )
        .arg(
            opt::<f64>(
                "bootstrap-credible",
                "Central interval mass for --bootstrap (default 0.95)",
            )
            .global(true),
        )
        .arg(opt::<u64>("bootstrap-seed", "Resampling seed for --bootstrap").global(true))
        .subcommand(with_path(
            Command::new("page-rank")
                .about("PageRank over an endorsement graph (src endorses dst)")
                .arg(opt::<usize>("iterations", "Power iterations"))
                .arg(opt::<f64>("damping-factor", "Damping factor"))
                .arg(
                    Arg::new("sink-dispersion")
                        .long("sink-dispersion")
                        .value_parser(["reverse", "all", "uniform", "none"])
                        .default_value("reverse"),
                )
                .arg(
                    opt::<String>(
                        "seeds",
                        "Comma-separated teleport seeds (personalized PageRank / RWR)",
                    )
                    .conflicts_with("seeds-file"),
                )
                .arg(opt::<PathBuf>(
                    "seeds-file",
                    "File of 'name [weight]' teleport seeds",
                ))
                .arg(flag(
                    "matches",
                    "Treat rows as match results: 'winner loser' becomes loser -> winner",
                )),
        ))
        .subcommand(with_path(
            Command::new("leader-rank")
                .about("LeaderRank: parameter-free ground-node random walk")
                .arg(opt::<usize>("iterations", "Power-iteration budget"))
                .arg(opt::<f64>("tolerance", "Early-exit threshold")),
        ))
        .subcommand(with_path(
            Command::new("harmonic")
                .about("Harmonic centrality: sum of inverse distances")
                .arg(
                    Arg::new("direction")
                        .long("direction")
                        .value_parser(["in", "out", "total"])
                        .default_value("in")
                        .help("Distance orientation (in: how easily others reach you)"),
                )
                .arg(flag(
                    "weighted",
                    "Use edge weights as lengths (Dijkstra; weights must be > 0)",
                ))
                .arg(opt::<usize>(
                    "sample-sources",
                    "Estimate from N sampled sources instead of all",
                ))
                .arg(opt::<u64>("seed", "Source-sampling seed")),
        ))
        .subcommand(with_path(
            Command::new("birank")
                .about("BiRank over a bipartite interaction graph")
                .arg(opt::<usize>("iterations", "Sweeps"))
                .arg(opt::<f64>("alpha", "dst-side propagation weight"))
                .arg(opt::<f64>("beta", "src-side propagation weight"))
                .arg(opt::<u64>("seed", "Initialization seed")),
        ))
        .subcommand(with_path(
            Command::new("hits")
                .about("HITS hubs and authorities (two scores per node)")
                .arg(opt::<usize>("iterations", "Power-iteration budget"))
                .arg(opt::<f64>("tolerance", "Early-exit threshold")),
        ))
        .subcommand(with_path(
            Command::new("katz-centrality")
                .visible_alias("katz")
                .about("Katz centrality: all walks, geometrically discounted")
                .arg(opt::<f64>("alpha", "Walk discount (< 1/lambda_max)"))
                .arg(opt::<usize>("iterations", "Iteration budget"))
                .arg(opt::<f64>("tolerance", "Early-exit threshold")),
        ))
        .subcommand(with_path(
            Command::new("degree")
                .about("Weighted degree/strength baseline")
                .arg(
                    Arg::new("direction")
                        .long("direction")
                        .value_parser(["in", "out", "total"])
                        .default_value("in")
                        .help("Which incident edges count"),
                ),
        ))
        .subcommand(with_path(
            Command::new("k-core")
                .visible_alias("kcore")
                .about("k-core decomposition: coreness per node (undirected)"),
        ))
        .subcommand(with_path(
            Command::new("components")
                .about("Write each connected component to <path>.<i>")
                .arg(opt::<usize>("min-graph-size", "Minimum component size")),
        ))
}

fn bandit_cmd() -> Command {
    let common = |cmd: Command| -> Command {
        with_path(cmd)
            .arg(opt::<u64>("seed", "Policy randomness seed"))
            .arg(opt::<usize>(
                "select",
                "Print the next N arms to play instead of scores",
            ))
    };
    Command::new("bandit")
        .about("Multi-armed bandits over 'arm reward' rows")
        .subcommand_required(true)
        .subcommand(common(
            Command::new("greedy").about("Always exploit the best empirical mean"),
        ))
        .subcommand(common(
            Command::new("epsilon-greedy")
                .about("Exploit with probability 1-epsilon, explore uniformly otherwise")
                .arg(opt::<f64>("epsilon", "Exploration rate")),
        ))
        .subcommand(common(
            Command::new("upper-confidence-bound")
                .visible_alias("ucb1")
                .about("Optimism under uncertainty: try arms you know least about")
                .arg(opt::<f64>(
                    "exploration",
                    "Exploration constant (classic UCB1: 2.0)",
                )),
        ))
        .subcommand(common(
            Command::new("kl-ucb")
                .about("KL-UCB: tighter-than-UCB1 indices for [0,1] rewards")
                .arg(opt::<f64>("c", "ln ln t scale (theory: >= 3; practice: 0)")),
        ))
        .subcommand(common(
            Command::new("exp3")
                .about("EXP3 adversarial exponential weights (offline replay)")
                .arg(opt::<f64>("gamma", "Exploration mix in (0,1]")),
        ))
        .subcommand(common(
            Command::new("thompson-beta")
                .visible_alias("ts-beta")
                .about("Thompson Sampling with a Beta posterior (rewards in [0,1])")
                .arg(opt::<f64>("prior-alpha", "Beta prior alpha"))
                .arg(opt::<f64>("prior-beta", "Beta prior beta")),
        ))
        .subcommand(common(
            Command::new("thompson-gaussian")
                .visible_alias("ts-gaussian")
                .about("Thompson Sampling with a Gaussian posterior")
                .arg(opt::<f64>("prior-mean", "Prior mean"))
                .arg(opt::<f64>("prior-weight", "Prior pseudo-observations")),
        ))
        .subcommand(
            with_path(
                Command::new("sliding-window-ucb")
                    .visible_alias("sw-ucb")
                    .about("UCB over the last N events only (tracks drifting arms)")
                    .arg(opt::<usize>("window", "Events kept in the window"))
                    .arg(opt::<f64>("exploration", "Exploration constant")),
            )
            .arg(opt::<usize>(
                "select",
                "Print the next N arms to play instead of scores",
            )),
        )
        .subcommand(with_path(
            Command::new("linucb")
                .about("Contextual LinUCB over 'arm reward x1 x2 ... xd' rows")
                .arg(opt::<f64>("alpha", "Exploration width"))
                .arg(opt::<f64>("ridge", "Ridge regularization"))
                .arg(opt::<String>(
                    "select-for",
                    "Print the arm to play for this comma-separated context",
                )),
        ))
}

fn trajectories_cmd() -> Command {
    Command::new("trajectories")
        .about("Rank states from reward-bearing episodes: 'state reward' rows, blank line ends an episode")
        .subcommand_required(true)
        .arg(
            opt::<usize>(
                "bootstrap",
                "Resample-and-refit N times; emit score and rank intervals",
            )
            .global(true),
        )
        .arg(
            opt::<f64>(
                "bootstrap-credible",
                "Central interval mass for --bootstrap (default 0.95)",
            )
            .global(true),
        )
        .arg(opt::<u64>("bootstrap-seed", "Resampling seed for --bootstrap").global(true))
        .subcommand(with_path(
            Command::new("monte-carlo")
                .visible_alias("mc")
                .about("Monte Carlo state values: discounted returns averaged per state")
                .arg(opt::<f64>("gamma", "Discount factor in (0, 1]"))
                .arg(
                    Arg::new("visit")
                        .long("visit")
                        .value_parser(["first", "every"])
                        .default_value("first")
                        .help("Count each state once per episode, or every occurrence"),
                )
                .arg(
                    Arg::new("aggregate")
                        .long("aggregate")
                        .value_parser(["mean", "median"])
                        .default_value("mean"),
                )
                .arg(opt::<f64>(
                    "winsorize",
                    "Clamp returns into the [q, 1-q] quantiles before aggregating",
                ))
                .arg(opt::<usize>(
                    "min-observations",
                    "Drop states with fewer return samples",
                )),
        ))
        .subcommand(with_path(
            Command::new("td")
                .about("TD(0) state values (order-dependent online updates)")
                .arg(opt::<f64>("alpha", "Learning rate in (0, 1]"))
                .arg(opt::<f64>("gamma", "Discount factor in (0, 1]"))
                .arg(opt::<usize>("passes", "Sweeps over the batch per update"))
                .arg(opt::<f64>("initial-value", "Value for unseen states")),
        ))
        .subcommand(with_path(
            Command::new("compare")
                .about("Bootstrap CIs on state values, plus pairwise exceedance tests")
                .arg(opt::<f64>("gamma", "Discount factor in (0, 1]"))
                .arg(
                    Arg::new("visit")
                        .long("visit")
                        .value_parser(["first", "every"])
                        .default_value("first"),
                )
                .arg(opt::<usize>("replicates", "Bootstrap replicates"))
                .arg(
                    Arg::new("method")
                        .long("method")
                        .value_parser(["bootstrap", "bayesian"])
                        .default_value("bootstrap")
                        .help("Episode resampling scheme"),
                )
                .arg(opt::<f64>("credible", "Central interval mass"))
                .arg(opt::<usize>(
                    "pairwise",
                    "Also run pairwise tests with this many permutations",
                ))
                .arg(opt::<usize>(
                    "min-observations",
                    "Drop states seen in fewer episodes",
                ))
                .arg(opt::<u64>("seed", "Resampling seed")),
        ))
        .subcommand(with_path(
            Command::new("behavior-cloning")
                .visible_alias("bc")
                .about("Rank actions by expert frequency (rewards ignored)")
                .arg(opt::<char>(
                    "per-state",
                    "Split tokens as state<SEP>action on this separator",
                ))
                .arg(opt::<f64>("smoothing", "Laplace smoothing pseudo-count"))
                .arg(flag(
                    "emit-pairs",
                    "Print the implied preference edges as tournament rows instead of scores",
                )),
        ))
}

// ---------------------------------------------------------------- helpers

/// Overrides `field` only when the user actually passed the flag — the
/// param-struct `Default` stays the single source of numeric defaults
/// (AGENTS.md rule 1).
fn set<T: Clone + Send + Sync + 'static>(m: &ArgMatches, name: &str, field: &mut T) {
    if let Some(v) = m.get_one::<T>(name) {
        *field = v.clone();
    }
}

/// Reads a clap-defaulted choice flag; the fallback literal mirrors the
/// `default_value` and is unreachable.
fn choice<'a>(m: &'a ArgMatches, name: &str, fallback: &'a str) -> &'a str {
    m.get_one::<String>(name)
        .map(String::as_str)
        .unwrap_or(fallback)
}

struct Ctx<'a> {
    path: &'a Path,
    format: String,
    save_state: Option<PathBuf>,
    load_state: Option<PathBuf>,
    min_count: usize,
    periods: bool,
    ties: TiePolicy,
    margin_ties: MarginTies,
    progress: Option<CliProgress>,
    pool_holder: Option<rayon::ThreadPool>,
}

impl<'a> Ctx<'a> {
    fn from_matches(sm: &'a ArgMatches) -> Result<Self> {
        let pool_holder = match sm.get_one::<usize>("threads") {
            Some(&n) => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| Error::InvalidInput(format!("--threads: {e}")))?,
            ),
            None => None,
        };
        let Some(path) = sm.get_one::<String>("path") else {
            return Err(Error::InvalidInput("missing input path".into()));
        };

        Ok(Ctx {
            path: Path::new(path.as_str()),
            format: sm
                .get_one::<String>("format")
                .cloned()
                .unwrap_or_else(|| "tsv".into()),
            save_state: sm.get_one::<PathBuf>("save-state").cloned(),
            load_state: sm.get_one::<PathBuf>("load-state").cloned(),
            min_count: sm
                .try_get_one::<usize>("min-count")
                .ok()
                .flatten()
                .copied()
                .unwrap_or(1),
            periods: sm
                .try_get_one::<bool>("groups-are-separate")
                .ok()
                .flatten()
                .copied()
                == Some(true),
            ties: match sm
                .try_get_one::<String>("ties")
                .ok()
                .flatten()
                .map(String::as_str)
            {
                Some("discard") => TiePolicy::Discard,
                Some("half-win") => TiePolicy::HalfWin,
                _ => TiePolicy::Error,
            },
            margin_ties: match sm
                .try_get_one::<String>("margin-ties")
                .ok()
                .flatten()
                .map(String::as_str)
            {
                Some("discard") => MarginTies::Discard,
                Some("zero") => MarginTies::Zero,
                _ => MarginTies::Error,
            },
            progress: std::io::stderr().is_terminal().then(CliProgress::default),
            pool_holder,
        })
    }

    fn opts(&self) -> FitOptions<'_> {
        FitOptions {
            progress: match &self.progress {
                Some(p) => p,
                None => &propagon::SILENT,
            },
            threading: match &self.pool_holder {
                Some(pool) => propagon::Threading::Dedicated(pool),
                None => propagon::Threading::Shared,
            },
        }
    }

    fn games(&self) -> Result<propagon::GamesDataset> {
        let g = io::read_games(self.path, self.periods)?;
        Ok(if self.min_count > 1 {
            g.filter_min_count(self.min_count)
        } else {
            g
        })
    }

    /// Win/loss pairs for the BT family (ties per `--ties`).
    fn pairwise(&self) -> Result<propagon::PairwiseDataset> {
        self.games()?.to_pairwise(self.ties)
    }

    /// Margin rows for Massey/Keener/Hodge mean-margin (ties per
    /// `--margin-ties`).
    fn margin_pairs(&self) -> Result<propagon::PairwiseDataset> {
        self.games()?.margin_pairs(self.margin_ties)
    }

    fn reject_load_state(&self, sub: &str) -> Result<()> {
        if self.load_state.is_some() {
            return Err(Error::InvalidInput(format!(
                "{sub} does not support --load-state (no incremental or warm-start form)"
            )));
        }
        Ok(())
    }

    fn emit<M: RankModel>(&self, model: &M) -> Result<()> {
        self.save(model)?;
        let mut out = std::io::stdout().lock();
        match self.format.as_str() {
            "jsonl" => emit::jsonl(&mut out, model),
            _ => emit::scores(&mut out, model),
        }
    }

    fn save<M: RankModel>(&self, model: &M) -> Result<()> {
        if let Some(p) = &self.save_state {
            model.save_to_path(p)?;
        }
        Ok(())
    }
}

/// Fit cold or warm-start from `--load-state`.
fn fit_maybe_warm<A: Ranker>(algo: &A, data: &A::Data, ctx: &Ctx<'_>) -> Result<A::Model> {
    match &ctx.load_state {
        Some(p) => {
            let init = A::Model::load_from_path(p)?;
            algo.fit_warm_opts(data, &init, &ctx.opts())
        }
        None => algo.fit_opts(data, &ctx.opts()),
    }
}

/// Runs the bootstrap wrapper instead of a plain fit when `--bootstrap N`
/// was passed; returns true when it handled the command (model emitted).
fn maybe_bootstrap<A>(algo: A, data: &A::Data, ctx: &Ctx<'_>, sm: &ArgMatches) -> Result<bool>
where
    A: Ranker + Sync,
    A::Data: Resample + Sync,
    A::Model: Send,
{
    let n = match sm.try_get_one::<usize>("bootstrap").ok().flatten() {
        Some(&n) => n,
        None => return Ok(false),
    };
    ctx.reject_load_state("--bootstrap")?;
    let mut bs = propagon::algos::Bootstrap::new(algo);
    bs.replicates = n;
    set(sm, "bootstrap-credible", &mut bs.credible);
    set(sm, "bootstrap-seed", &mut bs.seed);

    let model = bs.fit_opts(data, &ctx.opts())?;
    if model.replicates_ok() < n {
        eprintln!("replicates ok: {}/{n}", model.replicates_ok());
    }
    ctx.save(&model)?;
    let mut out = std::io::stdout().lock();
    match ctx.format.as_str() {
        "jsonl" => emit::jsonl(&mut out, &model)?,
        _ => emit::bootstrap(&mut out, &model)?,
    }
    Ok(true)
}

/// Rejects `--bootstrap` on algorithms it cannot wrap (online updates, or
/// batch fits whose semantics resampling would silently change).
fn reject_bootstrap(sm: &ArgMatches, algo: &str, why: &str) -> Result<()> {
    if sm
        .try_get_one::<usize>("bootstrap")
        .ok()
        .flatten()
        .is_some()
    {
        return Err(Error::InvalidInput(format!(
            "--bootstrap does not support {algo}: {why}"
        )));
    }
    Ok(())
}

/// Init or load from `--load-state`, then fold the new data in.
fn update_maybe_loaded<A: OnlineRanker>(
    algo: &A,
    data: &A::Data,
    ctx: &Ctx<'_>,
) -> Result<A::Model> {
    let mut model = match &ctx.load_state {
        Some(p) => A::Model::load_from_path(p)?,
        None => algo.init(),
    };
    algo.update_opts(&mut model, data, &ctx.opts())?;
    Ok(model)
}

// ---------------------------------------------------------------- dispatch

fn run() -> Result<()> {
    let matches = cli().get_matches();

    // `subcommand_required(true)` makes the misses below unreachable.
    let Some((group, gm)) = matches.subcommand() else {
        return Err(Error::InvalidInput("a subcommand is required".into()));
    };
    let Some((leaf, sm)) = gm.subcommand() else {
        return Err(Error::InvalidInput(format!(
            "{group} requires an algorithm subcommand"
        )));
    };

    match group {
        "tournament" => run_tournament(leaf, sm),
        "rankings" => run_rankings(leaf, sm),
        "crowd" => run_crowd(leaf, sm),
        "matchups" => run_matchups(leaf, sm),
        "graph" => run_graph(leaf, sm),
        "bandit" => run_bandit(leaf, sm),
        "trajectories" => run_trajectories(leaf, sm),
        other => Err(Error::InvalidInput(format!("unknown subcommand {other:?}"))),
    }
}

fn run_tournament(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    match algo {
        "win-rate" => {
            reject_bootstrap(sm, "win-rate", "it folds tallies online")?;
            let confidence = match choice(sm, "confidence-interval", "0.95") {
                "0.95" => Confidence::P95,
                "0.9" | "0.90" => Confidence::P90,
                _ => Confidence::P50,
            };
            let algo = WinRate { confidence };
            let model = update_maybe_loaded(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "elo" => {
            reject_bootstrap(sm, "elo", "online updates are order-dependent")?;
            if sm.get_flag("margin-of-victory") {
                let mut algo = MovElo::default();
                set(sm, "k", &mut algo.k);
                set(sm, "initial-rating", &mut algo.initial_rating);
                set(sm, "scale", &mut algo.scale);
                set(sm, "mov-exponent", &mut algo.mov_exponent);

                let model = update_maybe_loaded(&algo, &ctx.games()?, &ctx)?;
                return ctx.emit(&model);
            }
            let mut algo = Elo::default();
            set(sm, "k", &mut algo.k);
            set(sm, "initial-rating", &mut algo.initial_rating);
            set(sm, "scale", &mut algo.scale);

            let model = update_maybe_loaded(&algo, &ctx.games()?, &ctx)?;
            ctx.emit(&model)
        }
        "glicko2" => {
            reject_bootstrap(sm, "glicko2", "online updates are order-dependent")?;
            let mut algo = Glicko2::default();
            set(sm, "tau", &mut algo.tau);

            let model = update_maybe_loaded(&algo, &ctx.games()?, &ctx)?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::glicko2(&mut out, &model, sm.get_flag("use-mu")),
            }
        }
        "bradley-terry-model" => match choice(sm, "estimator", "mm") {
            "sgd" => {
                let mut algo = BradleyTerryLR::default();
                set(sm, "passes", &mut algo.passes);
                set(sm, "alpha", &mut algo.alpha);
                set(sm, "decay", &mut algo.decay);
                algo.thrifty = sm.get_flag("thrifty");

                let data = ctx.pairwise()?;
                if maybe_bootstrap(algo, &data, &ctx, sm)? {
                    return Ok(());
                }
                let model = fit_maybe_warm(&algo, &data, &ctx)?;
                ctx.emit(&model)
            }
            _ => {
                let mut algo = BradleyTerryMM::default();
                set(sm, "iterations", &mut algo.iterations);
                set(sm, "tolerance", &mut algo.tolerance);
                set(sm, "min-graph-size", &mut algo.min_graph_size);
                set(sm, "create-fake-games", &mut algo.create_fake_games);
                set(sm, "random-subgraph-links", &mut algo.random_subgraph_links);
                set(
                    sm,
                    "random-subgraph-weight",
                    &mut algo.random_subgraph_weight,
                );
                set(sm, "seed", &mut algo.seed);
                algo.remove_total_losers = sm.get_flag("remove-total-losers");

                let data = ctx.pairwise()?;
                if maybe_bootstrap(algo, &data, &ctx, sm)? {
                    return Ok(());
                }
                let model = fit_maybe_warm(&algo, &data, &ctx)?;
                ctx.save(&model)?;
                let mut out = std::io::stdout().lock();

                match ctx.format.as_str() {
                    "jsonl" => emit::jsonl(&mut out, &model),
                    _ => emit::btm_mm(&mut out, &model),
                }
            }
        },
        "luce-spectral-ranking" => {
            let mut algo = Lsr::default();

            if choice(sm, "estimator", "power") == "monte-carlo" {
                algo.estimator = Estimator::MonteCarlo;
                algo.steps = algo.estimator.default_steps();
            }
            set(sm, "steps", &mut algo.steps);
            set(sm, "seed", &mut algo.seed);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(algo, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = fit_maybe_warm(&algo, &data, &ctx)?;
            ctx.emit(&model)
        }
        "rank-centrality" => {
            ctx.reject_load_state(algo)?;
            let mut rc = RankCentrality::default();
            set(sm, "iterations", &mut rc.iterations);
            set(sm, "tolerance", &mut rc.tolerance);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(rc, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = rc.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "random-utility-model" => {
            ctx.reject_load_state(algo)?;
            let mut es = EsRum::default();
            set(sm, "passes", &mut es.passes);
            set(sm, "alpha", &mut es.alpha);
            set(sm, "gamma", &mut es.gamma);
            set(sm, "min-obs", &mut es.min_obs);
            set(sm, "prior", &mut es.prior);
            set(sm, "seed", &mut es.seed);

            if sm.get_flag("fixed") {
                es.distribution = RumDistribution::FixedNormal;
            }

            let data = ctx.pairwise()?;
            if maybe_bootstrap(es, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = es.fit_opts(&data, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::esrum(&mut out, &model),
            }
        }
        "kemeny" => {
            ctx.reject_load_state(algo)?;
            let mut km = Kemeny::default();
            set(sm, "min-obs", &mut km.min_obs);
            set(sm, "seed", &mut km.seed);

            if choice(sm, "algo", "insertion") == "de" {
                km.algo = KemenyAlgo::DiffEvo;
            }
            if let Some(&n) = sm.get_one::<usize>("passes") {
                km.passes = KemenyPasses::Fixed(n);
            }

            let data = ctx.pairwise()?;
            if maybe_bootstrap(km, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = km.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "borda-count" => {
            ctx.reject_load_state(algo)?;
            let data = ctx.pairwise()?;
            if maybe_bootstrap(Borda::default(), &data, &ctx, sm)? {
                return Ok(());
            }
            let model = Borda::default().fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "copeland" => {
            ctx.reject_load_state(algo)?;
            let data = ctx.pairwise()?;
            if maybe_bootstrap(Copeland::default(), &data, &ctx, sm)? {
                return Ok(());
            }
            let model = Copeland::default().fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "massey" => {
            ctx.reject_load_state(algo)?;
            let mut ms = Massey::default();
            set(sm, "iterations", &mut ms.iterations);
            set(sm, "tolerance", &mut ms.tolerance);

            let data = ctx.margin_pairs()?;
            if maybe_bootstrap(ms, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = ms.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "colley" => {
            ctx.reject_load_state(algo)?;
            let mut cl = Colley::default();
            set(sm, "iterations", &mut cl.iterations);
            set(sm, "tolerance", &mut cl.tolerance);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(cl, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = cl.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "keener" => {
            ctx.reject_load_state(algo)?;
            let mut kn = Keener::default();
            set(sm, "iterations", &mut kn.iterations);
            set(sm, "tolerance", &mut kn.tolerance);
            kn.skew = !sm.get_flag("no-skew");
            kn.normalize_games = !sm.get_flag("no-normalize-games");

            let data = ctx.margin_pairs()?;
            if maybe_bootstrap(kn, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = kn.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "bayesian-bradley-terry" => {
            ctx.reject_load_state(algo)?;
            let mut bb = BayesianBradleyTerry::default();
            set(sm, "shape", &mut bb.shape);
            set(sm, "rate", &mut bb.rate);
            set(sm, "samples", &mut bb.samples);
            set(sm, "burn-in", &mut bb.burn_in);
            set(sm, "credible", &mut bb.credible);
            set(sm, "seed", &mut bb.seed);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(bb, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = bb.fit_opts(&data, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::bayes_bt(&mut out, &model),
            }
        }
        "hodge-rank" => {
            ctx.reject_load_state(algo)?;
            let mut hr = HodgeRank::default();
            set(sm, "iterations", &mut hr.iterations);
            set(sm, "tolerance", &mut hr.tolerance);

            hr.flow = match choice(sm, "flow", "log-odds") {
                "win-rate-delta" => HodgeFlow::WinRateDelta,
                "mean-margin" => HodgeFlow::MeanMargin,
                _ => HodgeFlow::LogOdds,
            };

            let data = match hr.flow {
                HodgeFlow::MeanMargin => ctx.margin_pairs()?,
                _ => ctx.pairwise()?,
            };
            if maybe_bootstrap(hr, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = hr.fit_opts(&data, &ctx.opts())?;
            eprintln!(
                "inconsistency (cyclic flow share): {:.4}",
                model.inconsistency()
            );
            ctx.emit(&model)
        }
        "thurstone-mosteller" => {
            ctx.reject_load_state(algo)?;
            let mut tm = ThurstoneMosteller::default();
            set(sm, "iterations", &mut tm.iterations);
            set(sm, "tolerance", &mut tm.tolerance);
            set(sm, "pseudo-count", &mut tm.pseudo_count);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(tm, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = tm.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "i-luce-spectral-ranking" => {
            let mut il = ILsr::default();
            set(sm, "outer", &mut il.outer);
            set(sm, "inner-steps", &mut il.inner_steps);
            set(sm, "tolerance", &mut il.tolerance);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(il, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = fit_maybe_warm(&il, &data, &ctx)?;
            ctx.emit(&model)
        }
        "serial-rank" => {
            ctx.reject_load_state(algo)?;
            let mut sr = SerialRank::default();
            set(sm, "iterations", &mut sr.iterations);
            set(sm, "tolerance", &mut sr.tolerance);
            set(sm, "seed", &mut sr.seed);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(sr, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = sr.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "random-walker" => {
            ctx.reject_load_state(algo)?;
            let mut rw = RandomWalker::default();
            set(sm, "bias", &mut rw.p);
            set(sm, "iterations", &mut rw.iterations);
            set(sm, "tolerance", &mut rw.tolerance);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(rw, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = rw.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "offense-defense" => {
            ctx.reject_load_state(algo)?;
            let mut od = OffenseDefense::default();
            set(sm, "iterations", &mut od.iterations);
            set(sm, "tolerance", &mut od.tolerance);

            let data = ctx.margin_pairs()?;
            if maybe_bootstrap(od, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = od.fit_opts(&data, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::offense_defense(&mut out, &model),
            }
        }
        "generalized-bradley-terry" => {
            ctx.reject_load_state(algo)?;
            let ties = match choice(sm, "tie-model", "davidson") {
                "none" => TieModel::None,
                "rao-kupper" => TieModel::RaoKupper,
                _ => TieModel::Davidson,
            };
            let mut gb = GeneralizedBt {
                ties,
                ..GeneralizedBt::default()
            };
            if sm.get_flag("home-advantage") {
                gb.home = HomeAdvantage::Estimate;
            }
            set(sm, "iterations", &mut gb.iterations);
            set(sm, "tolerance", &mut gb.tolerance);

            let data = ctx.games()?;
            if maybe_bootstrap(gb, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = gb.fit_opts(&data, &ctx.opts())?;
            match (gb.ties, gb.home) {
                (TieModel::None, HomeAdvantage::None) => {}
                (TieModel::None, _) => {
                    eprintln!("home advantage gamma: {:.4}", model.home_advantage());
                }
                (_, HomeAdvantage::None) => {
                    eprintln!("tie parameter: {:.4}", model.tie_parameter());
                }
                _ => eprintln!(
                    "tie parameter: {:.4}  home advantage gamma: {:.4}",
                    model.tie_parameter(),
                    model.home_advantage()
                ),
            }
            ctx.emit(&model)
        }
        "team-bradley-terry" => {
            ctx.reject_load_state(algo)?;
            let mut tb = TeamBradleyTerry::default();
            if choice(sm, "aggregate", "additive") == "product" {
                tb.aggregate = TeamAggregate::Product;
            }
            tb.ties = ctx.ties;
            set(sm, "iterations", &mut tb.iterations);
            set(sm, "tolerance", &mut tb.tolerance);

            let data = ctx.games()?;
            if maybe_bootstrap(tb, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = tb.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "whole-history-rating" => {
            reject_bootstrap(
                sm,
                "whole-history-rating",
                "row resampling drops the rating periods WHR is built on",
            )?;
            let mut wh = Whr::default();
            set(sm, "w2", &mut wh.w2);
            set(sm, "prior-games", &mut wh.prior_games);
            set(sm, "iterations", &mut wh.iterations);
            set(sm, "tolerance", &mut wh.tolerance);

            let model = fit_maybe_warm(&wh, &ctx.pairwise()?, &ctx)?;
            if sm.get_flag("timeline") {
                ctx.save(&model)?;
                let mut out = std::io::stdout().lock();
                match ctx.format.as_str() {
                    "jsonl" => emit::jsonl(&mut out, &model)?,
                    _ => emit::whr_timeline(&mut out, &model)?,
                }
                Ok(())
            } else {
                ctx.emit(&model)
            }
        }
        "melo" => {
            reject_bootstrap(sm, "melo", "online updates are order-dependent")?;
            let mut me = MElo::default();
            set(sm, "k", &mut me.k);
            set(sm, "lr-rating", &mut me.lr_rating);
            set(sm, "lr-vector", &mut me.lr_vector);
            set(sm, "initial-rating", &mut me.initial_rating);
            set(sm, "init-scale", &mut me.init_scale);
            set(sm, "seed", &mut me.seed);

            let model = update_maybe_loaded(&me, &ctx.games()?, &ctx)?;
            ctx.emit(&model)
        }
        "nash-averaging" => {
            ctx.reject_load_state(algo)?;
            let mut na = NashAveraging::default();
            set(sm, "iterations", &mut na.iterations);
            set(sm, "tolerance", &mut na.tolerance);
            set(sm, "learning-rate", &mut na.learning_rate);
            set(sm, "anneal-every", &mut na.anneal_every);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(na, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = na.fit_opts(&data, &ctx.opts())?;
            eprintln!("duality gap: {:.3e}", model.gap());
            ctx.emit(&model)
        }
        "blade-chest" => {
            ctx.reject_load_state(algo)?;
            let mut bc = BladeChest::default();
            if choice(sm, "variant", "inner") == "dist" {
                bc.variant = BladeChestVariant::Dist;
            }
            set(sm, "dims", &mut bc.dims);
            set(sm, "lr", &mut bc.lr);
            set(sm, "epochs", &mut bc.epochs);
            set(sm, "l2", &mut bc.l2);
            set(sm, "init-scale", &mut bc.init_scale);
            set(sm, "seed", &mut bc.seed);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(bc, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = bc.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "covariate-bradley-terry" => {
            ctx.reject_load_state(algo)?;
            let features_path = sm
                .get_one::<PathBuf>("features")
                .ok_or_else(|| Error::InvalidInput("--features is required".into()))?;
            let mut cb = CovariateBt::new(io::read_features(features_path)?);
            set(sm, "l2", &mut cb.l2);
            cb.intercepts = sm.get_flag("intercepts");
            set(sm, "iterations", &mut cb.iterations);
            set(sm, "tolerance", &mut cb.tolerance);

            let data = ctx.pairwise()?;
            if maybe_bootstrap(cb.clone(), &data, &ctx, sm)? {
                return Ok(());
            }
            let model = cb.fit_opts(&data, &ctx.opts())?;
            eprintln!(
                "coefficients: {}",
                model
                    .coefficients()
                    .iter()
                    .map(|c| format!("{c:.6}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            ctx.emit(&model)
        }
        other => Err(Error::InvalidInput(format!(
            "unknown tournament algorithm {other:?}"
        ))),
    }
}

fn run_rankings(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    let data = io::read_rankings(ctx.path)?;

    match algo {
        "plackett-luce" => {
            let mut pl = PlackettLuce::default();
            set(sm, "iterations", &mut pl.iterations);
            set(sm, "tolerance", &mut pl.tolerance);

            if maybe_bootstrap(pl, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = fit_maybe_warm(&pl, &data, &ctx)?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::plackett_luce(&mut out, &model),
            }
        }
        "markov-chain" => {
            ctx.reject_load_state(algo)?;
            let mut mc = Mc4::default();
            set(sm, "damping", &mut mc.damping);
            set(sm, "iterations", &mut mc.iterations);
            set(sm, "tolerance", &mut mc.tolerance);

            if maybe_bootstrap(mc, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = mc.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "borda-count" => {
            ctx.reject_load_state(algo)?;
            reject_bootstrap(
                sm,
                "rankings borda-count",
                "the positional path has no resampling wrapper yet",
            )?;
            let model = Borda::default().fit_rankings(&data)?;
            ctx.emit(&model)
        }
        "kemeny" => {
            ctx.reject_load_state(algo)?;
            let mut km = Kemeny::default();
            set(sm, "min-obs", &mut km.min_obs);
            set(sm, "seed", &mut km.seed);

            if choice(sm, "algo", "insertion") == "de" {
                km.algo = KemenyAlgo::DiffEvo;
            }
            if let Some(&n) = sm.get_one::<usize>("passes") {
                km.passes = KemenyPasses::Fixed(n);
            }

            let pairs = data.to_pairwise();
            if maybe_bootstrap(km, &pairs, &ctx, sm)? {
                return Ok(());
            }
            let model = km.fit_opts(&pairs, &ctx.opts())?;
            ctx.emit(&model)
        }
        "i-luce-spectral-ranking" => {
            ctx.reject_load_state(algo)?;
            let mut il = ILsr::default();
            set(sm, "outer", &mut il.outer);
            set(sm, "inner-steps", &mut il.inner_steps);
            set(sm, "tolerance", &mut il.tolerance);

            reject_bootstrap(
                sm,
                "rankings i-luce-spectral-ranking",
                "the native ballot path has no resampling wrapper yet",
            )?;
            let model = il.fit_rankings_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "mallows" => {
            ctx.reject_load_state(algo)?;
            let mut ml = Mallows::default();
            set(sm, "seed", &mut ml.seed);
            if let Some(&n) = sm.get_one::<usize>("passes") {
                ml.passes = KemenyPasses::Fixed(n);
            }

            if maybe_bootstrap(ml, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = ml.fit_opts(&data, &ctx.opts())?;
            eprintln!(
                "phi (dispersion): {:.4}  mean Kendall distance: {:.4}",
                model.phi(),
                model.mean_distance()
            );
            ctx.emit(&model)
        }
        "footrule" => {
            ctx.reject_load_state(algo)?;
            if maybe_bootstrap(Footrule::default(), &data, &ctx, sm)? {
                return Ok(());
            }
            let model = Footrule::default().fit_opts(&data, &ctx.opts())?;
            eprintln!("total footrule distance: {}", model.cost());
            ctx.emit(&model)
        }
        other => Err(Error::InvalidInput(format!(
            "unknown rankings algorithm {other:?}"
        ))),
    }
}

fn run_crowd(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    match algo {
        "bradley-terry" => {
            ctx.reject_load_state(algo)?;
            let mut cb = CrowdBt::default();
            set(sm, "lambda", &mut cb.lambda);
            set(sm, "eta-prior-alpha", &mut cb.eta_prior_alpha);
            set(sm, "eta-prior-beta", &mut cb.eta_prior_beta);
            set(sm, "iterations", &mut cb.iterations);
            set(sm, "tolerance", &mut cb.tolerance);
            set(sm, "inner-sweeps", &mut cb.inner_sweeps);

            let data = io::read_annotated(ctx.path)?;
            if maybe_bootstrap(cb, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = cb.fit_opts(&data, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::crowd_bt(&mut out, &model),
            }
        }
        other => Err(Error::InvalidInput(format!(
            "unknown crowd algorithm {other:?}"
        ))),
    }
}

fn run_matchups(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    match algo {
        "weng-lin" => {
            let mut wl = WengLin::default();
            set(sm, "mu", &mut wl.mu);
            set(sm, "sigma", &mut wl.sigma);
            set(sm, "beta", &mut wl.beta);
            set(sm, "kappa", &mut wl.kappa);
            set(sm, "epsilon", &mut wl.epsilon);
            set(sm, "tau", &mut wl.tau);

            if choice(sm, "variant", "bradley-terry") == "thurstone-mosteller" {
                wl.variant = WengLinVariant::ThurstoneMostellerFull;
            }
            if choice(sm, "gamma", "sigma-over-c") == "one-over-k" {
                wl.gamma = GammaPolicy::OneOverK;
            }

            let data = io::read_matchups(ctx.path)?;
            let model = update_maybe_loaded(&wl, &data, &ctx)?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::weng_lin(&mut out, &model),
            }
        }
        other => Err(Error::InvalidInput(format!(
            "unknown matchups algorithm {other:?}"
        ))),
    }
}

fn run_graph(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    match algo {
        "page-rank" => {
            ctx.reject_load_state(algo)?;
            let mut pr = PageRank::default();
            set(sm, "damping-factor", &mut pr.damping);
            set(sm, "iterations", &mut pr.iterations);

            pr.sink = match choice(sm, "sink-dispersion", "reverse") {
                "all" => Sink::All,
                "uniform" => Sink::Uniform,
                "none" => Sink::None,
                _ => Sink::Reverse,
            };
            if let Some(list) = sm.get_one::<String>("seeds") {
                pr.teleport = Teleport::Seeds(
                    list.split(',')
                        .map(|n| (n.trim().to_string(), 1.0))
                        .collect(),
                );
            } else if let Some(p) = sm.get_one::<PathBuf>("seeds-file") {
                pr.teleport = Teleport::Seeds(io::read_seeds(p)?);
            }

            // --matches: rows are 'winner loser', i.e. the endorsement
            // loser -> winner (the tournament-file orientation).
            let graph = io::read_graph(ctx.path, sm.get_flag("matches"))?;
            if maybe_bootstrap(pr.clone(), &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = pr.fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "leader-rank" => {
            ctx.reject_load_state(algo)?;
            let mut lr = LeaderRank::default();
            set(sm, "iterations", &mut lr.iterations);
            set(sm, "tolerance", &mut lr.tolerance);

            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(lr, &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = lr.fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "harmonic" => {
            ctx.reject_load_state(algo)?;
            let direction = match choice(sm, "direction", "in") {
                "out" => Direction::Out,
                "total" => Direction::Total,
                _ => Direction::In,
            };
            let mut hc = Harmonic {
                direction,
                ..Harmonic::default()
            };
            if sm.get_flag("weighted") {
                hc.cost = EdgeCost::Weight;
            }
            if let Some(&count) = sm.get_one::<usize>("sample-sources") {
                let mut seed = Harmonic::DEFAULT_SAMPLE_SEED;
                set(sm, "seed", &mut seed);
                hc.sources = SourceBudget::Sample { count, seed };
            }

            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(hc, &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = hc.fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "birank" => {
            ctx.reject_load_state(algo)?;
            let mut br = BiRank::default();
            set(sm, "iterations", &mut br.iterations);
            set(sm, "alpha", &mut br.alpha);
            set(sm, "beta", &mut br.beta);
            set(sm, "seed", &mut br.seed);

            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(br, &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = br.fit_opts(&graph, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::birank(&mut out, &model),
            }
        }
        "hits" => {
            ctx.reject_load_state(algo)?;
            let mut ht = Hits::default();
            set(sm, "iterations", &mut ht.iterations);
            set(sm, "tolerance", &mut ht.tolerance);

            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(ht, &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = ht.fit_opts(&graph, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::hits(&mut out, &model),
            }
        }
        "katz-centrality" => {
            ctx.reject_load_state(algo)?;
            let mut kz = Katz::default();
            set(sm, "alpha", &mut kz.alpha);
            set(sm, "iterations", &mut kz.iterations);
            set(sm, "tolerance", &mut kz.tolerance);

            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(kz, &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = kz.fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "degree" => {
            ctx.reject_load_state(algo)?;
            let direction = match choice(sm, "direction", "in") {
                "out" => Direction::Out,
                "total" => Direction::Total,
                _ => Direction::In,
            };

            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(Degree { direction }, &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = Degree { direction }.fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "k-core" => {
            ctx.reject_load_state(algo)?;
            let graph = io::read_graph(ctx.path, false)?;
            if maybe_bootstrap(KCore::default(), &graph, &ctx, sm)? {
                return Ok(());
            }
            let model = KCore::default().fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "components" => {
            ctx.reject_load_state(algo)?;
            let graph = io::read_graph(ctx.path, false)?;

            let mut min_size = 1usize;
            set(sm, "min-graph-size", &mut min_size);

            let comps = extract_components(graph.view(), min_size);
            for (i, comp) in comps.iter().enumerate() {
                let out_path = format!("{}.{i}", ctx.path.display());
                let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path)?);

                for (s, d, w) in comp.view().edges() {
                    use std::io::Write;
                    let sn = comp.interner().name(s).unwrap_or("<unresolved>");
                    let dn = comp.interner().name(d).unwrap_or("<unresolved>");
                    writeln!(f, "{sn} {dn} {w}")?;
                }
            }

            eprintln!("wrote {} components", comps.len());
            Ok(())
        }
        other => Err(Error::InvalidInput(format!(
            "unknown graph algorithm {other:?}"
        ))),
    }
}

fn run_trajectories(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    let data = io::read_trajectories(ctx.path)?;

    match algo {
        "monte-carlo" => {
            ctx.reject_load_state(algo)?;
            let mut mc = McValue::default();
            set(sm, "gamma", &mut mc.gamma);
            set(sm, "min-observations", &mut mc.min_observations);
            if choice(sm, "visit", "first") == "every" {
                mc.visit = Visit::Every;
            }
            if choice(sm, "aggregate", "mean") == "median" {
                mc.aggregate = Aggregate::Median;
            }
            if let Some(&q) = sm.get_one::<f64>("winsorize") {
                mc.winsorize = Winsorize::Percentile(q);
            }

            if maybe_bootstrap(mc, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = mc.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "td" => {
            reject_bootstrap(sm, "td", "online updates are order-dependent")?;
            let mut td = TdValue::default();
            set(sm, "alpha", &mut td.alpha);
            set(sm, "gamma", &mut td.gamma);
            set(sm, "passes", &mut td.passes);
            set(sm, "initial-value", &mut td.initial_value);

            let model = update_maybe_loaded(&td, &data, &ctx)?;
            ctx.emit(&model)
        }
        "compare" => {
            ctx.reject_load_state(algo)?;
            let mut vc = ValueCompare::default();
            set(sm, "gamma", &mut vc.gamma);
            set(sm, "replicates", &mut vc.replicates);
            set(sm, "credible", &mut vc.credible);
            set(sm, "min-observations", &mut vc.min_observations);
            set(sm, "seed", &mut vc.seed);
            if choice(sm, "visit", "first") == "every" {
                vc.visit = Visit::Every;
            }
            if choice(sm, "method", "bootstrap") == "bayesian" {
                vc.method = ResampleScheme::BayesianBootstrap;
            }
            if let Some(&permutations) = sm.get_one::<usize>("pairwise") {
                vc.pairwise = PairwiseTests::On { permutations };
            }

            reject_bootstrap(sm, "compare", "it is already a bootstrap procedure")?;
            let model = vc.fit_opts(&data, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();
            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::value_compare(&mut out, &model),
            }
        }
        "behavior-cloning" => {
            ctx.reject_load_state(algo)?;
            let mut bc = BehaviorCloning::default();
            set(sm, "smoothing", &mut bc.smoothing);
            if let Some(&separator) = sm.get_one::<char>("per-state") {
                bc.granularity = Granularity::PerState { separator };
            }

            if maybe_bootstrap(bc, &data, &ctx, sm)? {
                return Ok(());
            }
            let model = bc.fit_opts(&data, &ctx.opts())?;
            if sm.get_flag("emit-pairs") {
                ctx.save(&model)?;
                let pairs = model.implied_pairs();
                use std::io::Write;
                let mut out = std::io::stdout().lock();
                for (w, l, x) in pairs.rows() {
                    let wn = pairs.interner().name(w).unwrap_or("<unresolved>");
                    let ln = pairs.interner().name(l).unwrap_or("<unresolved>");
                    writeln!(out, "{wn}\t{ln}\t1\t{x}")?;
                }
                Ok(())
            } else {
                ctx.emit(&model)
            }
        }
        other => Err(Error::InvalidInput(format!(
            "unknown trajectories algorithm {other:?}"
        ))),
    }
}

fn run_bandit(policy_name: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;

    if policy_name == "sliding-window-ucb" {
        let mut algo = SlidingWindowUcb::default();
        set(sm, "window", &mut algo.window);
        set(sm, "exploration", &mut algo.exploration);

        let data = io::read_rewards(ctx.path)?;
        let mut model: SwUcbModel = match &ctx.load_state {
            Some(p) => SwUcbModel::load_from_path(p)?,
            None => algo.init(),
        };
        algo.update_opts(&mut model, &data, &ctx.opts())?;

        if let Some(&k) = sm.get_one::<usize>("select") {
            ctx.save(&model)?;
            for name in model.select_k(k)? {
                println!("{name}");
            }
            return Ok(());
        }
        return ctx.emit(&model);
    }
    if policy_name == "linucb" {
        let mut algo = LinUcb::default();
        set(sm, "alpha", &mut algo.alpha);
        set(sm, "ridge", &mut algo.ridge);

        let data = io::read_contextual(ctx.path)?;
        let mut model: LinUcbModel = match &ctx.load_state {
            Some(p) => LinUcbModel::load_from_path(p)?,
            None => algo.init(),
        };
        algo.update_opts(&mut model, &data, &ctx.opts())?;
        ctx.save(&model)?;

        if let Some(spec) = sm.get_one::<String>("select-for") {
            let x: Vec<f64> = spec
                .split(',')
                .map(|v| {
                    v.trim()
                        .parse::<f64>()
                        .map_err(|e| Error::InvalidInput(format!("bad context value {v:?}: {e}")))
                })
                .collect::<Result<_>>()?;
            println!("{}", model.select_for(&x)?);
            return Ok(());
        }
        let mut out = std::io::stdout().lock();
        return match ctx.format.as_str() {
            "jsonl" => emit::jsonl(&mut out, &model),
            _ => emit::scores(&mut out, &model),
        };
    }

    let policy = match policy_name {
        "greedy" => BanditPolicy::Greedy,
        "epsilon-greedy" => {
            let mut epsilon = BanditPolicy::DEFAULT_EPSILON;
            set(sm, "epsilon", &mut epsilon);
            BanditPolicy::EpsilonGreedy { epsilon }
        }
        "upper-confidence-bound" => {
            let mut exploration = BanditPolicy::DEFAULT_EXPLORATION;
            set(sm, "exploration", &mut exploration);
            BanditPolicy::Ucb1 { exploration }
        }
        "kl-ucb" => {
            let mut c = BanditPolicy::DEFAULT_KL_C;
            set(sm, "c", &mut c);
            BanditPolicy::KlUcb { c }
        }
        "exp3" => {
            let mut gamma = BanditPolicy::DEFAULT_EXP3_GAMMA;
            set(sm, "gamma", &mut gamma);
            BanditPolicy::Exp3 { gamma }
        }
        "thompson-beta" => {
            let mut prior_alpha = BanditPolicy::DEFAULT_PRIOR_ALPHA;
            let mut prior_beta = BanditPolicy::DEFAULT_PRIOR_BETA;
            set(sm, "prior-alpha", &mut prior_alpha);
            set(sm, "prior-beta", &mut prior_beta);
            BanditPolicy::ThompsonBeta {
                prior_alpha,
                prior_beta,
            }
        }
        "thompson-gaussian" => {
            let mut prior_mean = BanditPolicy::DEFAULT_PRIOR_MEAN;
            let mut prior_weight = BanditPolicy::DEFAULT_PRIOR_WEIGHT;
            set(sm, "prior-mean", &mut prior_mean);
            set(sm, "prior-weight", &mut prior_weight);
            BanditPolicy::ThompsonGaussian {
                prior_mean,
                prior_weight,
            }
        }
        other => {
            return Err(Error::InvalidInput(format!(
                "unknown bandit policy {other:?}"
            )));
        }
    };

    let mut algo = Bandit {
        policy,
        ..Default::default()
    };
    set(sm, "seed", &mut algo.seed);

    let data = io::read_rewards(ctx.path)?;
    let mut model: BanditModel = match &ctx.load_state {
        Some(p) => BanditModel::load_from_path(p)?,
        None => algo.init(),
    };
    algo.update_opts(&mut model, &data, &ctx.opts())?;

    if let Some(&k) = sm.get_one::<usize>("select") {
        let arms = model.select_k(k)?;
        ctx.save(&model)?; // after select: the draw counter advanced

        for id in arms {
            println!("{}", model.arm_name(id).unwrap_or("<unresolved>"));
        }
        Ok(())
    } else {
        ctx.emit(&model)
    }
}
