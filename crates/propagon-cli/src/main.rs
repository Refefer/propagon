//! `propagon` — ranking from revealed preferences.
//!
//! v1-compatible surface: `propagon <path> <subcommand> [flags]`, same
//! subcommand names and flags, `"{id}: {value}"` output. New in v2: string
//! ids natively (no dehydrate step), `--save-state`/`--load-state` for
//! incremental and warm-started runs, `--format jsonl`, `--threads`, and the
//! `elo`, `borda`, `copeland`, `rank-centrality`, and `bandit` subcommands.

mod emit;
mod io;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use propagon::algos::{
    Bandit, BanditModel, BanditPolicy, BiRank, Borda, BradleyTerryLR, BradleyTerryMM, Confidence,
    Copeland, Elo, EsRum, Estimator, Glicko2, Kemeny, KemenyAlgo, Lsr, PageRank, RankCentrality,
    RumDistribution, Sink, WinRate, extract_components,
};
use propagon::{
    Error, FitOptions, OnlineRanker, Progress, RankModel, Ranker, Result,
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
            indicatif::ProgressStyle::with_template(
                "[{msg}] {wide_bar} {pos}/{len} {eta_precise}",
            )
            .expect("static template"),
        );
        pb.set_message(phase.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(200));
        *self.bar.lock().unwrap() = Some(pb);
    }

    fn update(&self, done: u64) {
        if let Some(pb) = self.bar.lock().unwrap().as_ref() {
            pb.set_position(done);
        }
    }

    fn message(&self, msg: &str) {
        if let Some(pb) = self.bar.lock().unwrap().as_ref() {
            pb.set_message(msg.to_string());
        }
    }

    fn finish(&self) {
        if let Some(pb) = self.bar.lock().unwrap().take() {
            pb.finish_and_clear();
        }
    }
}

// ---------------------------------------------------------------- cli

fn flag(name: &'static str, help: &'static str) -> Arg {
    Arg::new(name).long(name).help(help).action(ArgAction::SetTrue)
}

fn opt<T: Clone + Send + Sync + std::str::FromStr + 'static>(
    name: &'static str,
    help: &'static str,
) -> Arg
where
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    Arg::new(name).long(name).help(help).value_parser(value_parser!(T))
}

fn cli() -> Command {
    Command::new("propagon")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Ranking from revealed preferences (see docs/algorithms.md)")
        .subcommand_required(true)
        .arg(Arg::new("path").required(true).help("Input data file"))
        .arg(
            opt::<usize>("min-count", "Iteratively drop edges whose endpoints appear in fewer rows")
                .global(true),
        )
        .arg(
            flag("groups-are-separate", "Treat blank-line-separated batches as separate periods")
                .global(true),
        )
        .arg(opt::<usize>("threads", "Worker threads (default: all cores)").global(true))
        .arg(
            Arg::new("format")
                .long("format")
                .value_parser(["tsv", "jsonl"])
                .default_value("tsv")
                .help("Output format: v1-style tsv or model-state jsonl")
                .global(true),
        )
        .arg(
            opt::<PathBuf>("save-state", "Write the fitted model state to this file")
                .global(true),
        )
        .arg(
            opt::<PathBuf>("load-state", "Resume from a saved model state (update or warm start)")
                .global(true),
        )
        .subcommand(
            Command::new("rate").about("Win rates with Wilson intervals").arg(
                Arg::new("confidence-interval")
                    .long("confidence-interval")
                    .value_parser(["0.95", "0.9", "0.90", "0.5"])
                    .default_value("0.95"),
            ),
        )
        .subcommand(
            Command::new("glicko2")
                .about("Glicko-2 ratings (periods via --groups-are-separate)")
                .arg(opt::<f64>("tau", "Volatility constraint (0.3-1.2)"))
                .arg(flag("use-mu", "Emit only the internal-scale rating")),
        )
        .subcommand(
            Command::new("btm-lr")
                .about("Bradley-Terry via logistic SGD")
                .arg(opt::<f64>("alpha", "Learning rate"))
                .arg(opt::<f64>("decay", "L2 shrinkage per pass"))
                .arg(opt::<usize>("passes", "Passes per period"))
                .arg(flag("thrifty", "Sequential in-place updates")),
        )
        .subcommand(
            Command::new("btm-mm")
                .about("Bradley-Terry via minorization-maximization")
                .arg(opt::<usize>("min-graph-size", "Skip components smaller than this"))
                .arg(opt::<usize>("iterations", "Maximum MM sweeps"))
                .arg(opt::<f64>("tol", "Convergence tolerance").alias("tolerance"))
                .arg(flag("remove-total-losers", "Also remove never-won entities"))
                .arg(opt::<f64>("create-fake-games", "Patch one-sided entities with fake games of this weight"))
                .arg(opt::<usize>("random-subgraph-links", "Random links between components"))
                .arg(opt::<f64>("random-subgraph-weight", "Weight of those links"))
                .arg(opt::<u64>("seed", "Seed for random component links")),
        )
        .subcommand(
            Command::new("es-rum")
                .about("Gaussian random-utility model via evolution strategies")
                .arg(opt::<usize>("passes", "ES iterations"))
                .arg(opt::<f64>("alpha", "Initial perturbation scale"))
                .arg(opt::<f64>("gamma", "L2 regularization"))
                .arg(opt::<usize>("min-obs", "Minimum comparisons per entity").alias("min-observations"))
                .arg(opt::<usize>("prior", "Pseudo-count smoothing"))
                .arg(flag("fixed", "Pin all variances to 1 (Thurstone-style)"))
                .arg(opt::<u64>("seed", "Random seed")),
        )
        .subcommand(
            Command::new("kemeny")
                .about("Kemeny-optimal consensus ranking")
                .arg(opt::<usize>("passes", "Insertion passes / DE evaluations (0 = auto)"))
                .arg(opt::<usize>("min-obs", "Minimum comparisons per entity"))
                .arg(
                    Arg::new("algo")
                        .long("algo")
                        .value_parser(["insertion", "de"])
                        .default_value("insertion"),
                )
                .arg(opt::<u64>("seed", "Seed for the DE search")),
        )
        .subcommand(
            Command::new("lsr")
                .about("Luce spectral ranking (Plackett-Luce)")
                .arg(opt::<usize>("steps", "Power passes / walk steps (0 = auto)"))
                .arg(
                    Arg::new("estimator")
                        .long("estimator")
                        .alias("algo")
                        .value_parser(["power", "monte-carlo"])
                        .default_value("power"),
                )
                .arg(opt::<u64>("seed", "Seed for Monte Carlo walks")),
        )
        .subcommand(
            Command::new("page-rank")
                .about("PageRank over the match/endorsement graph")
                .arg(opt::<usize>("iterations", "Power iterations"))
                .arg(opt::<f64>("damping-factor", "Damping factor"))
                .arg(
                    Arg::new("sink-dispersion")
                        .long("sink-dispersion")
                        .value_parser(["reverse", "all", "none"])
                        .default_value("reverse"),
                ),
        )
        .subcommand(
            Command::new("birank")
                .about("BiRank over a bipartite interaction graph")
                .arg(opt::<usize>("iterations", "Sweeps"))
                .arg(opt::<f64>("alpha", "dst-side propagation weight"))
                .arg(opt::<f64>("beta", "src-side propagation weight"))
                .arg(opt::<u64>("seed", "Initialization seed")),
        )
        .subcommand(
            Command::new("extract-components")
                .about("Write each connected component to <path>.<i>")
                .arg(opt::<usize>("min-graph-size", "Minimum component size").alias("min-size")),
        )
        .subcommand(
            Command::new("elo")
                .about("Elo ratings (order-dependent online updates)")
                .arg(opt::<f64>("k", "Update step size"))
                .arg(opt::<f64>("initial-rating", "Rating for unseen entities"))
                .arg(opt::<f64>("scale", "Logistic scale (default 400)")),
        )
        .subcommand(Command::new("borda").about("Borda count (weighted win totals)"))
        .subcommand(Command::new("copeland").about("Copeland pairwise-majority scores"))
        .subcommand(
            Command::new("rank-centrality")
                .about("Rank Centrality spectral Bradley-Terry estimate")
                .arg(opt::<usize>("iterations", "Maximum sweeps"))
                .arg(opt::<f64>("tolerance", "Early-exit tolerance")),
        )
        .subcommand(
            Command::new("bandit")
                .about("Multi-armed bandit over 'arm reward' rows")
                .arg(
                    Arg::new("policy")
                        .long("policy")
                        .value_parser(["greedy", "epsilon-greedy", "ucb1", "ts-beta", "ts-gaussian"])
                        .default_value("ucb1"),
                )
                .arg(opt::<f64>("epsilon", "Exploration rate (epsilon-greedy)"))
                .arg(opt::<f64>("exploration", "UCB exploration constant"))
                .arg(opt::<f64>("prior-alpha", "Beta prior alpha (ts-beta)"))
                .arg(opt::<f64>("prior-beta", "Beta prior beta (ts-beta)"))
                .arg(opt::<f64>("prior-mean", "Gaussian prior mean (ts-gaussian)"))
                .arg(opt::<f64>("prior-weight", "Gaussian prior pseudo-observations"))
                .arg(opt::<u64>("seed", "Policy randomness seed"))
                .arg(opt::<usize>("select", "Print the next N arms to play instead of scores")),
        )
        .subcommand(
            Command::new("dehydrate")
                .about("[deprecated] Map string ids to dense integers (v2 reads strings natively)")
                .arg(opt::<String>("delim", "Field delimiter (default: tab)"))
                .arg(opt::<PathBuf>("features", "Optional features file to remap")),
        )
        .subcommand(
            Command::new("hydrate")
                .about("[deprecated] Map score-file ids back to names")
                .arg(opt::<PathBuf>("vocab", "Vocab file from dehydrate").required(true))
                .arg(opt::<PathBuf>("output", "Output path (default: stdout)")),
        )
}

// ---------------------------------------------------------------- helpers

fn get_or<T: Clone + Send + Sync + 'static>(m: &ArgMatches, name: &str, default: T) -> T {
    m.get_one::<T>(name).cloned().unwrap_or(default)
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
    fn opts(&self) -> FitOptions<'_> {
        let mut opts = FitOptions::default();
        if let Some(p) = &self.progress {
            opts.progress = Some(p);
        }
        opts.pool = self.pool_holder.as_ref();
        opts
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

// ---------------------------------------------------------------- main

fn run() -> Result<()> {
    let matches = cli().get_matches();
    let (sub, sm) = matches.subcommand().expect("subcommand required");

    let pool_holder = match matches.get_one::<usize>("threads") {
        Some(&n) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| Error::InvalidInput(format!("--threads: {e}")))?,
        ),
        None => None,
    };

    let ctx = Ctx {
        path: Path::new(matches.get_one::<String>("path").expect("required").as_str()),
        format: sm.get_one::<String>("format").cloned().unwrap_or_else(|| "tsv".into()),
        save_state: sm.get_one::<PathBuf>("save-state").cloned(),
        load_state: sm.get_one::<PathBuf>("load-state").cloned(),
        min_count: get_or(sm, "min-count", 1usize),
        periods: sm.get_flag("groups-are-separate"),
        progress: std::io::stderr().is_terminal().then(CliProgress::default),
        pool_holder,
    };

    match sub {
        "rate" => {
            let confidence = match sm.get_one::<String>("confidence-interval").unwrap().as_str() {
                "0.95" => Confidence::P95,
                "0.9" | "0.90" => Confidence::P90,
                _ => Confidence::P50,
            };
            let algo = WinRate { confidence };
            let model = update_maybe_loaded(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "glicko2" => {
            let algo = Glicko2 { tau: get_or(sm, "tau", 0.5), ..Default::default() };
            let model = update_maybe_loaded(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();
            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::glicko2(&mut out, &model, sm.get_flag("use-mu")),
            }
        }
        "btm-lr" => {
            let algo = BradleyTerryLR {
                passes: get_or(sm, "passes", 10),
                alpha: get_or(sm, "alpha", 1.0),
                decay: get_or(sm, "decay", 1e-5),
                thrifty: sm.get_flag("thrifty"),
            };
            let model = fit_maybe_warm(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "btm-mm" => {
            let algo = BradleyTerryMM {
                iterations: get_or(sm, "iterations", 10_000),
                tolerance: get_or(sm, "tol", 1e-6),
                min_graph_size: get_or(sm, "min-graph-size", 1),
                remove_total_losers: sm.get_flag("remove-total-losers"),
                create_fake_games: get_or(sm, "create-fake-games", 0.0),
                random_subgraph_links: get_or(sm, "random-subgraph-links", 0),
                random_subgraph_weight: get_or(sm, "random-subgraph-weight", 1e-3),
                seed: get_or(sm, "seed", 1221),
            };
            let model = fit_maybe_warm(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();
            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::btm_mm(&mut out, &model),
            }
        }
        "es-rum" => {
            ctx.reject_load_state(sub)?;
            let algo = EsRum {
                distribution: if sm.get_flag("fixed") {
                    RumDistribution::FixedNormal
                } else {
                    RumDistribution::Gaussian
                },
                passes: get_or(sm, "passes", 100),
                alpha: get_or(sm, "alpha", 1.0),
                gamma: get_or(sm, "gamma", 1e-3),
                min_obs: get_or(sm, "min-obs", 1),
                prior: get_or(sm, "prior", 0),
                seed: get_or(sm, "seed", 2019),
            };
            let model = algo.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();
            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::esrum(&mut out, &model),
            }
        }
        "kemeny" => {
            ctx.reject_load_state(sub)?;
            let algo = Kemeny {
                passes: get_or(sm, "passes", 0),
                min_obs: get_or(sm, "min-obs", 1),
                algo: match sm.get_one::<String>("algo").unwrap().as_str() {
                    "de" => KemenyAlgo::DiffEvo,
                    _ => KemenyAlgo::Insertion,
                },
                seed: get_or(sm, "seed", 2020),
            };
            let model = algo.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "lsr" => {
            let estimator = match sm.get_one::<String>("estimator").unwrap().as_str() {
                "monte-carlo" => Estimator::MonteCarlo,
                _ => Estimator::PowerMethod,
            };
            let steps = match get_or(sm, "steps", 0usize) {
                0 => {
                    if estimator == Estimator::MonteCarlo {
                        1000
                    } else {
                        10
                    }
                }
                n => n,
            };
            let algo = Lsr { steps, estimator, seed: get_or(sm, "seed", 2020) };
            let model = fit_maybe_warm(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "page-rank" => {
            ctx.reject_load_state(sub)?;
            let algo = PageRank {
                damping: get_or(sm, "damping-factor", 0.85),
                iterations: get_or(sm, "iterations", 10),
                sink: match sm.get_one::<String>("sink-dispersion").unwrap().as_str() {
                    "all" => Sink::All,
                    "none" => Sink::None,
                    _ => Sink::Reverse,
                },
            };
            // v1 match-file orientation: a row `winner loser` is the
            // endorsement `loser -> winner`.
            let graph = io::read_graph(ctx.path, true)?;
            let model = algo.fit_opts(&graph, &ctx.opts())?;
            ctx.emit(&model)
        }
        "birank" => {
            ctx.reject_load_state(sub)?;
            let algo = BiRank {
                iterations: get_or(sm, "iterations", 10),
                alpha: get_or(sm, "alpha", 1.0),
                beta: get_or(sm, "beta", 1.0),
                seed: get_or(sm, "seed", 2019),
            };
            let graph = io::read_graph(ctx.path, false)?;
            let model = algo.fit_opts(&graph, &ctx.opts())?;
            ctx.save(&model)?;
            let mut out = std::io::stdout().lock();
            match ctx.format.as_str() {
                "jsonl" => emit::jsonl(&mut out, &model),
                _ => emit::birank(&mut out, &model),
            }
        }
        "extract-components" => {
            ctx.reject_load_state(sub)?;
            let graph = io::read_graph(ctx.path, false)?;
            let comps = extract_components(graph.view(), get_or(sm, "min-graph-size", 1));
            for (i, comp) in comps.iter().enumerate() {
                let out_path = format!("{}.{i}", ctx.path.display());
                let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
                for (s, d, w) in comp.view().edges() {
                    use std::io::Write;
                    let sn = comp.interner().name(s).expect("id resolves");
                    let dn = comp.interner().name(d).expect("id resolves");
                    writeln!(f, "{sn} {dn} {w}")?;
                }
            }
            eprintln!("wrote {} components", comps.len());
            Ok(())
        }
        "elo" => {
            let algo = Elo {
                k: get_or(sm, "k", 32.0),
                initial_rating: get_or(sm, "initial-rating", 1500.0),
                scale: get_or(sm, "scale", 400.0),
            };
            let model = update_maybe_loaded(&algo, &ctx.pairwise()?, &ctx)?;
            ctx.emit(&model)
        }
        "borda" => {
            ctx.reject_load_state(sub)?;
            let model = Borda::default().fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "copeland" => {
            ctx.reject_load_state(sub)?;
            let model = Copeland::default().fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "rank-centrality" => {
            ctx.reject_load_state(sub)?;
            let algo = RankCentrality {
                iterations: get_or(sm, "iterations", 200),
                tolerance: get_or(sm, "tolerance", 1e-10),
            };
            let model = algo.fit_opts(&ctx.pairwise()?, &ctx.opts())?;
            ctx.emit(&model)
        }
        "bandit" => {
            let policy = match sm.get_one::<String>("policy").unwrap().as_str() {
                "greedy" => BanditPolicy::Greedy,
                "epsilon-greedy" => {
                    BanditPolicy::EpsilonGreedy { epsilon: get_or(sm, "epsilon", 0.1) }
                }
                "ts-beta" => BanditPolicy::ThompsonBeta {
                    prior_alpha: get_or(sm, "prior-alpha", 1.0),
                    prior_beta: get_or(sm, "prior-beta", 1.0),
                },
                "ts-gaussian" => BanditPolicy::ThompsonGaussian {
                    prior_mean: get_or(sm, "prior-mean", 0.0),
                    prior_weight: get_or(sm, "prior-weight", 1.0),
                },
                _ => BanditPolicy::Ucb1 { exploration: get_or(sm, "exploration", 2.0) },
            };
            let algo = Bandit { policy, seed: get_or(sm, "seed", 42) };
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
                    println!("{}", model.arm_name(id).expect("id resolves"));
                }
                Ok(())
            } else {
                ctx.emit(&model)
            }
        }
        "dehydrate" => {
            eprintln!(
                "warning: dehydrate is deprecated — v2 reads string ids natively; \
                 this subcommand will be removed in a future release"
            );
            dehydrate(
                ctx.path,
                &get_or(sm, "delim", "\t".to_string()),
                sm.get_one::<PathBuf>("features"),
            )
        }
        "hydrate" => {
            eprintln!(
                "warning: hydrate is deprecated — v2 emits names directly; \
                 this subcommand will be removed in a future release"
            );
            hydrate(
                ctx.path,
                sm.get_one::<PathBuf>("vocab").expect("required"),
                sm.get_one::<PathBuf>("output"),
            )
        }
        other => Err(Error::InvalidInput(format!("unknown subcommand {other:?}"))),
    }
}

// ------------------------------------------------- deprecated v1 utilities

fn dehydrate(path: &Path, delim: &str, features: Option<&PathBuf>) -> Result<()> {
    use std::io::Write;
    let text = std::fs::read_to_string(path)?;
    let mut interner = propagon::Interner::new();
    let mut edges: Vec<(u32, u32, String)> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split(delim);
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            return Err(Error::parse(lineno + 1, format!("expected two fields: {line:?}")));
        };
        let w = it.next().unwrap_or("1").to_string();
        let a = interner.intern(a.trim());
        let b = interner.intern(b.trim());
        edges.push((a, b, w));
    }

    let base = path.display();
    let mut vocab = std::io::BufWriter::new(std::fs::File::create(format!("{base}.vocab"))?);
    let mut feat_id =
        std::io::BufWriter::new(std::fs::File::create(format!("{base}.features.id"))?);
    for (idx, name) in interner.names().enumerate() {
        writeln!(vocab, "{idx} {name}")?;
        writeln!(feat_id, "{idx} {idx}")?;
    }
    let mut edges_f = std::io::BufWriter::new(std::fs::File::create(format!("{base}.edges"))?);
    for (a, b, w) in edges {
        writeln!(edges_f, "{a} {b} {w}")?;
    }
    if let Some(fpath) = features {
        let mut out =
            std::io::BufWriter::new(std::fs::File::create(format!("{base}.features"))?);
        for line in std::fs::read_to_string(fpath)?.lines() {
            if let Some((name, rest)) = line.split_once(delim)
                && let Some(id) = interner.get(name.trim())
            {
                writeln!(out, "{id} {rest}")?;
            }
        }
    }
    Ok(())
}

fn hydrate(path: &Path, vocab: &Path, output: Option<&PathBuf>) -> Result<()> {
    use std::io::Write;
    let mut names = std::collections::HashMap::new();
    for line in std::fs::read_to_string(vocab)?.lines() {
        if let Some((idx, name)) = line.split_once(' ') {
            names.insert(idx.to_string(), name.to_string());
        }
    }
    let mut out: Box<dyn Write> = match output {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::stdout().lock()),
    };
    for line in std::fs::read_to_string(path)?.lines() {
        if let Some((id, value)) = line.split_once(": ") {
            let name = names.get(id).map(String::as_str).unwrap_or(id);
            writeln!(out, "{name}\t{value}")?;
        }
    }
    Ok(())
}
