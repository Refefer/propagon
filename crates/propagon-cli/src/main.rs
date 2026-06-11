//! `propagon` — ranking from revealed preferences.
//!
//! Subcommands are grouped by the shape of the input data:
//!
//! - `propagon tournament <algo> <path>` — pairwise comparison rankers
//!   (win-rate, elo, glicko2, bradley-terry-model, luce-spectral-ranking,
//!   rank-centrality, random-utility-model, kemeny, borda-count, copeland)
//!   over `winner loser [weight]` rows.
//! - `propagon graph <algo> <path>` — node-importance rankers and utilities
//!   (page-rank, birank, components) over `src dst [weight]` edges.
//! - `propagon bandit <policy> <path>` — multi-armed bandits (greedy,
//!   epsilon-greedy, upper-confidence-bound, thompson-beta,
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
    Bandit, BanditModel, BanditPolicy, BiRank, Borda, BradleyTerryLR, BradleyTerryMM, Colley,
    Confidence, Copeland, Elo, EsRum, Estimator, Glicko2, HodgeFlow, HodgeRank, Keener, Kemeny,
    KemenyAlgo, KemenyPasses, Lsr, Massey, Mc4, PageRank, PlackettLuce, RankCentrality,
    RumDistribution, Sink, WinRate, extract_components,
};
use propagon::{Error, FitOptions, OnlineRanker, Progress, RankModel, Ranker, Result};

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
        .subcommand(graph_cmd())
        .subcommand(bandit_cmd())
}

fn tournament_cmd() -> Command {
    Command::new("tournament")
        .about("Rank entities from pairwise outcomes: 'winner loser [weight]' rows")
        .subcommand_required(true)
        .arg(
            opt::<usize>(
                "min-count",
                "Iteratively drop rows whose endpoints appear fewer times",
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
                .about("Elo ratings (order-dependent online updates)")
                .arg(opt::<f64>("k", "Update step size"))
                .arg(opt::<f64>("initial-rating", "Rating for unseen entities"))
                .arg(opt::<f64>("scale", "Logistic scale (default 400)")),
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
}

fn rankings_cmd() -> Command {
    Command::new("rankings")
        .visible_alias("ballots")
        .about("Aggregate full or partial rankings: one ballot per line, best first")
        .subcommand_required(true)
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
}

fn graph_cmd() -> Command {
    Command::new("graph")
        .about("Rank nodes by graph structure: 'src dst [weight]' edges")
        .subcommand_required(true)
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
                .arg(flag(
                    "matches",
                    "Treat rows as match results: 'winner loser' becomes loser -> winner",
                )),
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

    fn pairwise(&self) -> Result<propagon::PairwiseDataset> {
        io::read_pairwise(self.path, self.periods, self.min_count)
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
        "graph" => run_graph(leaf, sm),
        "bandit" => run_bandit(leaf, sm),
        other => Err(Error::InvalidInput(format!("unknown subcommand {other:?}"))),
    }
}

fn run_tournament(algo: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;
    match algo {
        "win-rate" => {
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
            let mut algo = Elo::default();
            set(sm, "k", &mut algo.k);
            set(sm, "initial-rating", &mut algo.initial_rating);
            set(sm, "scale", &mut algo.scale);

            let model = update_maybe_loaded(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "glicko2" => {
            let mut algo = Glicko2::default();
            set(sm, "tau", &mut algo.tau);

            let model = update_maybe_loaded(&algo, &ctx.pairwise()?, &ctx)?;
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

                let model = fit_maybe_warm(&algo, &ctx.pairwise()?, &ctx)?;
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

                let model = fit_maybe_warm(&algo, &ctx.pairwise()?, &ctx)?;
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

            let model = fit_maybe_warm(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "rank-centrality" => {
            ctx.reject_load_state(algo)?;
            let mut rc = RankCentrality::default();
            set(sm, "iterations", &mut rc.iterations);
            set(sm, "tolerance", &mut rc.tolerance);

            let model = rc.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
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

            let model = es.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
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

            let model = km.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "borda-count" => {
            ctx.reject_load_state(algo)?;
            let model = Borda::default().fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "copeland" => {
            ctx.reject_load_state(algo)?;
            let model = Copeland::default().fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "massey" => {
            ctx.reject_load_state(algo)?;
            let mut ms = Massey::default();
            set(sm, "iterations", &mut ms.iterations);
            set(sm, "tolerance", &mut ms.tolerance);

            let model = ms.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "colley" => {
            ctx.reject_load_state(algo)?;
            let mut cl = Colley::default();
            set(sm, "iterations", &mut cl.iterations);
            set(sm, "tolerance", &mut cl.tolerance);

            let model = cl.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "keener" => {
            ctx.reject_load_state(algo)?;
            let mut kn = Keener::default();
            set(sm, "iterations", &mut kn.iterations);
            set(sm, "tolerance", &mut kn.tolerance);
            kn.skew = !sm.get_flag("no-skew");
            kn.normalize_games = !sm.get_flag("no-normalize-games");

            let model = kn.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
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

            let model = hr.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            eprintln!(
                "inconsistency (cyclic flow share): {:.4}",
                model.inconsistency()
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

            let model = mc.fit_opts(&data, &ctx.opts())?;
            ctx.emit(&model)
        }
        "borda-count" => {
            ctx.reject_load_state(algo)?;
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

            let model = km.fit_opts(&data.to_pairwise(), &ctx.opts())?;
            ctx.emit(&model)
        }
        other => Err(Error::InvalidInput(format!(
            "unknown rankings algorithm {other:?}"
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

            // --matches: rows are 'winner loser', i.e. the endorsement
            // loser -> winner (the tournament-file orientation).
            let graph = io::read_graph(ctx.path, sm.get_flag("matches"))?;
            let model = pr.fit_opts(&graph, &ctx.opts())?;
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
            let model = br.fit_opts(&graph, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();

            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::birank(&mut out, &model),
            }
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

fn run_bandit(policy_name: &str, sm: &ArgMatches) -> Result<()> {
    let ctx = Ctx::from_matches(sm)?;

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
