mod mm;
mod lr;
mod g2;
mod reader;
mod pr;
mod birank;
mod vp;
mod vw;
mod rw;
mod lpa;
mod labelrankplus;
mod chashmap;
mod walker;
mod gcs;
mod de;
mod metric;
mod converter;
mod mccluster;
mod pb;
mod he;
mod cc;
mod cluster_strat;
mod esrum;
mod kemeny;
mod rate;

mod utils;

#[macro_use]
extern crate clap;
extern crate hashbrown;

use clap::{Arg, App, ArgMatches, SubCommand};

use std::fmt::{Write,Display};
use hashbrown::{HashMap,HashSet};

type Match = (u32, u32, f32);
type Games = Vec<Match>;

fn filter_edges(game_sets: Vec<Games>, min_count: usize) -> Vec<Games> {
    let mut new_games = Vec::new();
    if min_count > 1 {
        for mut games in game_sets.into_iter() {
            let total_games = games.len();
            let mut hm: HashMap<u32,usize> = HashMap::new();
            loop {
                let len = games.len();
                hm.clear();
                for (w, l, _) in games.iter() {
                    for n in [w,l].iter() {
                        let e = hm.entry(**n).or_insert(0usize);
                        *e += 1;
                    }
                }
                games = games.into_iter().filter(|(w,l,_)| {
                    hm.get(w).unwrap() >= &min_count && hm.get(l).unwrap() >= &min_count
                }).collect();
                if games.len() == len { break }
                eprintln!("Filtered {} edges ({} remaining)", len - games.len(), len);
            }
            eprintln!("Filtered out {} total edges", total_games - games.len());
            new_games.push(games);
        }
    } else {
        new_games = game_sets;
    }
    new_games
}

fn emit_scores<K: Display, V: Display>(it: impl Iterator<Item=(K,V)>) {
    use std::io::{BufWriter,stdout,Write};
    let stdout = stdout();
    let mut handle = BufWriter::new(stdout.lock());
    for (idx, (id, s)) in it.enumerate() {
        if idx > 0 {
            write!(handle, "\n")
                .expect("Error when writing out results!");
        }
        write!(handle, "{}: {}", id, s)
            .expect("Error when writing out results!");
    }
}

// computes the Action rates
fn rate(args: &&clap::ArgMatches<'_>, games: Games) {
    let ci = value_t!(args, "confidence-interval", f32).unwrap_or(0.95);

    // Compute rate stats
    let (mut winners, losers) = utils::tally_winners_losers(&games);
    
    // Just return the rate
    if ci == 0.5 {
        let it = losers.into_iter().map(|(team, (_, l_score))| {
            let e = winners.entry(team).or_insert((0, 0.));
            (team, e.1 / (e.1 + l_score))
        });
        emit_scores(it)
    } else {
        // Get z
        let z: f32 = if ci == 0.95 {
            1.96
        } else {
            1.645
        };

        let z_sqr = z * z;
        
        // Get all keys
        let mut all_teams: HashSet<_> = winners.keys().collect();
        all_teams.extend(losers.keys());
        let mut scores = Vec::new();
        for team in all_teams.into_iter() {
            let x = winners.get(team).unwrap_or(&(0, 0.)).1;
            let n = x + losers.get(team).unwrap_or(&(0, 0.)).1;
            let n_t = n as f32 + z_sqr;
            let p_t = (x as f32 + z_sqr / 2.) / n_t;
            let range = z * (p_t / n_t * (1. - p_t)).powf(0.5);
            scores.push((team, p_t - range));
        }
        scores.sort_by(|a, b| (b.1).partial_cmp(&a.1).expect("Shouldn't blow up!"));
        emit_scores(scores.into_iter());
    }
           
}

// glicko score
fn glicko(args: &&clap::ArgMatches<'_>, games: Vec<Games>) {
    let tau        = value_t!(args, "tau", f64).unwrap_or(0.5);
    let use_mu     = args.is_present("use-mu");
    let env        = g2::Env::default();
    let mut series = env.new_match_set(tau);

    for (i, game_set) in games.into_iter().enumerate() {
        eprintln!("Game set {}, Known teams: {}", i, series.teams().len());
        let gs = game_set.into_iter().map(|(w,l,_)| (w, l)).collect();
        series.update(gs);
    }

    let mut scores: Vec<_> = series.teams().iter()
        .map(|(t, p)| {
            if use_mu {
                (t, format!("{}",p.mu(&env)))
            } else {
                let (lb, ub) = p.bounds();
                let mu = p.mu(&env);
                (t, format!("{:.4}\t{:.4}\t{:.4}\t{:.4}", mu, p.rd, lb, ub))
            }
        }).collect();

    scores.sort_by(|a, b| (b.1).partial_cmp(&a.1).expect("blew up in sort"));
    emit_scores(scores.into_iter());

}

fn btm_lr(args: &&clap::ArgMatches<'_>, games: Vec<Games>) {
    let alpha = value_t!(args, "alpha", f32).unwrap_or(1.);
    let decay = value_t!(args, "decay", f32).unwrap_or(1e-5);
    let passes = value_t!(args, "passes", usize).unwrap_or(10);
    let thrifty = args.is_present("thrifty");
    let mut btm = lr::BtmLr::new(passes, alpha, decay, thrifty);
    for (i, games_set) in games.into_iter().enumerate() {
        eprintln!("Processing GameSet {}", i);
        btm.update(&games_set);
    }

    let mut scores: Vec<_> = btm.scores.into_iter().collect();
    
    scores.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());
    emit_scores(scores.into_iter());
}

fn es_rum(args: &&clap::ArgMatches<'_>, games: Games) {
    let passes  = value_t!(args, "passes", usize).unwrap_or(100);
    let alpha   = value_t!(args, "alpha", f32).unwrap_or(1f32);
    let gamma   = value_t!(args, "gamma", f32).unwrap_or(1e-3f32);
    let min_obs = value_t!(args, "min-observations", usize).unwrap_or(1);
    let seed    = value_t!(args, "seed", u64).unwrap_or(2019);

    let distribution = if args.is_present("fixed") {
        esrum::Distribution::FixedNormal
    } else {
        esrum::Distribution::Gaussian
    };

    let esrum = esrum::EsRum {
        distribution,
        passes,
        alpha,
        gamma,
        min_obs,
        seed
    };

    // Load priors
    let rums = esrum.fit(games.into_iter());
    emit_scores(rums.into_iter().map(|(k, v)| (k, format!("{:.6} {:.6}", v[0], v[1]))));
}

fn kemeny(args: &&clap::ArgMatches<'_>, games: Games) {
    let passes  = value_t!(args, "passes", usize);
    let min_obs = value_t!(args, "min-obs", usize).unwrap_or(1);
    let algo    = value_t!(args, "algo", String).unwrap_or("insertion".into());

    let (algo, passes) = match algo.as_ref() {
        "insertion" => (kemeny::Algorithm::Insertion, passes.unwrap_or(1)),
        _           => (kemeny::Algorithm::DiffEvo, passes.unwrap_or(50000)),
    };

    let kemeny = kemeny::Kemeny {
        passes: passes,
        algo: algo,
        min_obs: min_obs
    };

    // Load priors
    let ranked = kemeny.fit(games.into_iter());
    let n_items = ranked.len();
    emit_scores(ranked.into_iter().enumerate().map(|(idx, item)| (item, n_items - idx)));
}

fn page_rank(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations = value_t!(args, "iterations", usize).unwrap_or(10);
    let df = value_t!(args, "damping-factor", f32).unwrap_or(0.85);
    let sd = value_t!(args, "sink-dispersion", String).unwrap_or("reverse".into());

    let sink = match sd.as_ref() {
        "reverse" => pr::Sink::Reverse,
        "all"     => pr::Sink::All,
        _         => pr::Sink::None
    };

    let page_rank = pr::PageRank::new(df, iterations, sink);
    let scores = page_rank.compute(games);
    emit_scores(scores.into_iter());
}

fn birank(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations = value_t!(args, "iterations", usize).unwrap_or(10);
    let alpha      = value_t!(args, "alpha", f32).unwrap_or(1.0);
    let beta       = value_t!(args, "beta", f32).unwrap_or(1.0);
    let seed       = value_t!(args, "seed", u64).unwrap_or(2019);

    let settings = birank::Settings {
        n_iters: iterations,
        alpha,
        beta,
        seed: seed
    };

    let mut birank = birank::BiRank::build(games.into_iter());
    birank.randomize(&settings);
    birank.compute(&settings, HashMap::new(), HashMap::new());
    birank.emit();
}

fn fast_json<K,N: std::hash::Hash + Eq + std::fmt::Display>(
    it: impl Iterator<Item=(K, Vec<(N,f32)>)>, 
    idx_to_vocab: HashMap<N,String>
) -> impl Iterator<Item=(K,String)> {
    it.map(move |(id, mut emb)| {
        let mut string = String::new();
        string.push_str("{");
        if emb.len() > 0 {
            emb.sort_by(|a,b| (b.1).partial_cmp(&a.1).unwrap());
            emb.into_iter().for_each(|(f, v)| {
                if idx_to_vocab.contains_key(&f) {
                    string.push_str(format!("\"{}\":{},", idx_to_vocab[&f], v).as_str());
                } else {
                    string.push_str(format!("\"{}\":{},", f, v).as_str());
                }
            });
            string.pop();
        }
        string.push_str("}");
        (id, string)
    })
}

fn vec_prop(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations = value_t!(args, "iterations", usize).unwrap_or(10);
    let alpha      = value_t!(args, "alpha", f32).unwrap_or(0.9);
    let prior      = value_t!(args, "prior", String).expect("Required");
    let max_terms  = value_t!(args, "max-terms", usize).unwrap_or(100);
    let error      = value_t!(args, "error", f32).unwrap_or(1e-5);
    let l2_output  = args.is_present("l2-output");
    let chunks     = value_t!(args, "chunks", usize).unwrap_or(91);
    let seed       = value_t!(args, "seed", u64).unwrap_or(2019);

    let reg = match args.value_of("regularizer").unwrap_or("symmetric") {
        "l1" => vp::Regularizer::L1,
        "l2" => vp::Regularizer::L2,
        _    => vp::Regularizer::Symmetric
    };

    let vp = vp::VecProp {
        n_iters: iterations,
        regularizer: reg,
        alpha,
        max_terms,
        error,
        chunks,
        normalize: l2_output,
        seed: seed 
    };

    // Load priors
    let (priors, idx_to_vocab) = vp::load_priors(prior.as_str());
    let embeddings = vp.fit(games.into_iter(), &priors);
    let it = embeddings.into_iter().map(|(k, v)| (k, (v.0).0));
    emit_scores(fast_json(it, idx_to_vocab));
}

fn vec_walk(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations     = value_t!(args, "iterations", usize).unwrap_or(10);
    let alpha          = value_t!(args, "alpha", f32).unwrap_or(0.9);
    let prior          = value_t!(args, "prior", String).expect("Required");
    let walk_len       = value_t!(args, "walk-len", usize).unwrap_or(20);
    let biased_walk    = !value_t!(args, "uniform-walk", bool).unwrap_or(false);
    let context_window = value_t!(args, "context-window", usize).unwrap_or(2);
    let max_terms      = value_t!(args, "max-terms", usize).unwrap_or(100);
    let error          = value_t!(args, "error", f32).unwrap_or(1e-5);
    let chunks         = value_t!(args, "chunks", usize).unwrap_or(91);
    let neg_sample     = value_t!(args, "negative-sample", usize).unwrap_or(5);
    let seed           = value_t!(args, "seed", u64).unwrap_or(2019);

    let vw = vw::VecWalk {
        n_iters: iterations,
        alpha,
        max_terms,
        walk_len,
        biased_walk,
        context_window,
        error,
        chunks,
        negative_sample: neg_sample,
        seed: seed 
    };

    // Load priors
    let (priors, idx_to_vocab) = vp::load_priors(prior.as_str());
    let embeddings = vw.fit(games.into_iter(), &priors);

    let it = embeddings.into_iter().map(|(k, v)| (k, v.0));
    emit_scores(fast_json(it, idx_to_vocab));
}


fn lpa(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations = if let Some(n_iter) = value_t!(args, "iterations", usize).ok() {
        if n_iter == 0 {
            None
        } else {
            Some(n_iter)
        }
    } else {
        Some(10)
    };
    let chunks = value_t!(args, "chunks", usize).unwrap_or(10);
    let seed   = value_t!(args, "seed", u64).unwrap_or(2019);

    let lpa = lpa::LPA {
        n_iters: iterations,
        chunks: chunks,
        seed: seed 
    };

    let clusters = lpa.fit(games.into_iter());
    // Count cluster sizes, sort by them, and emit in most to least popular
    let mut counts = HashMap::new();
    let mut t_vec = Vec::with_capacity(clusters.len());
    for (k, c) in clusters.into_iter() {
        *counts.entry(c).or_insert(0) += 1;
        t_vec.push((k, c));
    }

    eprintln!("Total Clusters: {}", counts.len());

    t_vec.sort_by_key(|(k, c)| (counts[c], *c, *k));
    let it = t_vec.into_iter().rev();
    emit_scores(it);
}

fn label_rank(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations = value_t!(args, "iterations", usize).unwrap_or(10);
    let inflation  = value_t!(args, "inflation", f32).unwrap_or(2.);
    let prior      = value_t!(args, "prior", f32).unwrap_or(0.1);
    let q          = value_t!(args, "q", f32).unwrap_or(0.1);
    let max_terms  = value_t!(args, "max-terms", usize).unwrap_or(10);

    let label_rank = labelrankplus::LabelRankPlus {
        n_iters: iterations,
        inflation,
        prior,
        q,
        max_terms
    };

    let clusters = label_rank.fit(games.into_iter());
    // Count cluster sizes, sort by them, and emit in most to least popular
    let mut counts = HashMap::new();
    let mut t_vec = Vec::with_capacity(clusters.len());
    for (k, c) in clusters.into_iter() {
        *counts.entry(c).or_insert(0) += 1;
        t_vec.push((k, c));
    }

    eprintln!("Total Clusters: {}", counts.len());

    t_vec.sort_by_key(|(k, c)| (counts[c], *c, *k));
    let it = t_vec.into_iter().rev();
    emit_scores(it);
}

fn random_walk(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations  = value_t!(args, "iterations", usize).unwrap_or(10);
    let walk_len    = value_t!(args, "walk-len", usize).unwrap_or(20);
    let biased_walk = !value_t!(args, "uniform-walk", bool).unwrap_or(false);
    let buffer_size = value_t!(args, "buffer-size", usize).unwrap_or(10000);
    let seed        = value_t!(args, "seed", u64).unwrap_or(2019);

    let random_walk = rw::RandomWalk {
        iterations,
        buffer_size,
        walk_len,
        biased_walk,
        seed: seed
    };
    let mut s = String::new();
    for walk in random_walk.generate(games.into_iter()) {
        for (i, k) in walk.into_iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            write!(s, "{}", k).expect("Should never fail!");
        }
        println!("{}", s);
        s.clear();
    }
}

fn euc_emb(args: &&clap::ArgMatches<'_>, games: Games) {
    let dims         = value_t!(args, "dims", usize).unwrap();
    let landmarks    = value_t!(args, "landmarks", usize).unwrap();
    let global_fns   = value_t!(args, "global-embed-fns", usize).unwrap_or(1_000_000);
    let local_fns    = value_t!(args, "local-embed-fns", usize).unwrap_or(1_000);
    let seed         = value_t!(args, "seed", u64).unwrap_or(2019);
    let chunks       = value_t!(args, "chunks", usize).unwrap_or(91);
    let global_bias  = value_t!(args, "global-bias", f32).unwrap_or(0.7);
    let passes       = value_t!(args, "passes", usize).unwrap_or(3);
    let l2norm       = args.is_present("l2");
    let only_walks   = args.is_present("only-walks");

    let distance = match args.value_of("weighting").unwrap_or("uniform") {
        "original" => gcs::Distance::Original,
        "uniform"  => gcs::Distance::Uniform,
        "edge"     => gcs::Distance::EdgeWeighted,
        _          => gcs::Distance::DegreeWeighted
    };

    let selection = match args.value_of("selection").unwrap_or("degree") {
        "random" => gcs::LandmarkSelection::Random,
        _        => gcs::LandmarkSelection::Degree
    };

    let metric = match args.value_of("space").unwrap_or("euclidean") {
        "euclidean"   => metric::Space::Euclidean,
        "hyperboloid" => metric::Space::Hyperboloid,
        "manhattan"   => metric::Space::Manhattan,
        _             => metric::Space::Poincare
    };

    let emb = gcs::GCS {
        metric,
        landmarks,
        only_walks,
        dims,
        global_fns,
        local_fns,
        distance,
        selection,
        chunks,
        l2norm,
        global_bias,
        passes,
        seed
    };

    let embeddings = emb.fit(games.into_iter());
    emit_scores(embeddings.into_iter().map(|(k, v)| {
        let mut s = String::new();
        s.push('[');
        for (i, vi) in v.into_iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write!(&mut s, "{}", vi).expect("Should never fail");
        }
        s.push(']');
        (k, s)
    }));
}

fn mc_cluster(args: &&clap::ArgMatches<'_>, games: Games) {
    let max_steps        = value_t!(args, "steps", usize).unwrap_or(10000);
    let restarts         = value_t!(args, "restarts", f32).unwrap_or(0.5);
    let ppr              = args.is_present("ppr");
    let max_terms        = value_t!(args, "max-terms", usize).unwrap_or(50);
    let seed             = value_t!(args, "seed", u64).unwrap_or(2020);
    let min_cluster_size = value_t!(args, "min-cluster-size", usize).unwrap_or(0);
    let emb_path         = value_t!(args, "save-embeddings", String).ok();

    let sampler = match args.value_of("sampler").unwrap_or("random-walk") {
        "metropolis-hastings" => mccluster::Sampler::MetropolisHastings,
        _                     => mccluster::Sampler::RandomWalk
    };

    let clusterer = match args.value_of("clusterer").unwrap_or("attractor") {
        "similarity" => {
            let best_only        = args.is_present("best-only");
            let rem_weak_links   = args.is_present("rem-weak-links");
            let metric = match args.value_of("metric").unwrap_or("cosine") {
                "cosine"  => cluster_strat::Metric::Cosine,
                "jaccard" => cluster_strat::Metric::Jaccard,
                _         => cluster_strat::Metric::Overlap
            };

            let strategy = cluster_strat::SimStrategy {
                best_only,
                seed,
                min_cluster_size,
                rem_weak_links,
                metric
            };
            cluster_strat::ClusterStrategy::Similarity(strategy)
        },
        _            => {
            let strategy = cluster_strat::AttractorStrategy {
                num: value_t!(args, "num-attractors", usize).unwrap_or(1),
                min_cluster_size
            };
            cluster_strat::ClusterStrategy::Attractors(strategy)
        }
    };
    
    let mc = mccluster::MCCluster {
        max_steps,
        restarts,
        ppr,
        max_terms,
        sampler,
        emb_path,
        seed,
        clusterer
    };

    // Load priors
    let embeddings = mc.fit(games.into_iter());
    emit_scores(embeddings.into_iter());
}

fn hash_embedding(args: &&clap::ArgMatches<'_>, games: Games) {
    let dims      = value_t!(args, "dims", u16).expect("Requires dimensions!");
    let hashes    = value_t!(args, "hashes", usize).unwrap_or(3);
    let max_steps = value_t!(args, "steps", usize).unwrap_or(10000);
    let restarts  = value_t!(args, "restarts", f32).unwrap_or(0.1);
    let weighted  = args.is_present("weighted");
    let ppr       = args.is_present("ppr");
    let directed  = args.is_present("directed");
    let seed      = value_t!(args, "seed", u64).unwrap_or(2020);

    let sampler = match args.value_of("sampler").unwrap_or("random-walk") {
        "metropolis-hastings" => he::Sampler::MetropolisHastings,
        _                     => he::Sampler::RandomWalk
    };

    let norm = match args.value_of("normalize").unwrap_or("none") {
        "none" => he::Norm::None,
        "l1"   => he::Norm::L1,
        _      => he::Norm::L2
    };

    let hash_emb = he::HashEmbeddings {
        dims,
        hashes,
        max_steps,
        restarts,
        sampler,
        norm,
        weighted,
        directed,
        ppr,
        seed
    };

    // Load priors
    let (names, embeddings) = hash_emb.fit(games.into_iter());
    emit_scores(embeddings.chunks(dims as usize).enumerate().map(|(i, v)| {
        let mut s = String::new();
        s.push('[');
        for (i, vi) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write!(&mut s, "{}", vi).expect("Should never fail");
        }
        s.push(']');
        (names[i], s)
    }));
}

fn components(args: &&clap::ArgMatches<'_>, path: &str, games: Games) {
    let min_size = value_t!(args, "min-graph-size", usize).unwrap_or(1);
    cc::extract_components(path, min_size, games.into_iter());
}

fn dehydrate(path: &str, args: &&clap::ArgMatches<'_>) {
    let delim = value_t!(args, "delim", String).unwrap_or('\t'.to_string());
    let features = value_t!(args, "features", String).ok();
    converter::Converter::dehydrate(path, &delim, features.as_deref());
}

fn hydrate(embedding: &str, args: &&clap::ArgMatches<'_>) {
    let vocab = value_t!(args, "vocab", String).unwrap();
    let output = value_t!(args, "output", String).ok();
    converter::Converter::hydrate(embedding, vocab.as_str(), output.as_deref());
}


fn parse<'a>() -> ArgMatches<'a> {
    App::new("propagon")
        .version("0.1.0")
        .author("Andrew S. <refefer@gmail.com>")
        .about("Suite of high performance graph tools")
        .arg(Arg::with_name("path")
             .min_values(1)
             .required(true)
             .help("Path to edges"))
        .arg(Arg::with_name("min-count")
             .long("min-count")
             .takes_value(true)
             .help("Items with fewer will be omitted"))
        .arg(Arg::with_name("groups-are-separate")
             .long("groups-are-separate")
             .help("Treats separate files and groups as separate datasets to process"))

        .subcommand(SubCommand::with_name("btm-mm")
            .about("Computes bradley-terry model rankings using the minor-maxim method")
            .arg(Arg::with_name("min-graph-size")
                 .long("min-graph-size")
                 .takes_value(true)
                 .help("Minimum size of graph to compute"))
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of passes to run the MM algorithm"))
            .arg(Arg::with_name("tolerance")
                 .long("tol")
                 .takes_value(true)
                 .help("When error is less than tolerance, exits optimizer"))
            .arg(Arg::with_name("remove-total-losers")
                 .long("remove-total-losers")
                 .help("Removes teams which have no wins"))
            .arg(Arg::with_name("create-fake-games")
                 .long("create-fake-games")
                 .takes_value(true)
                 .help("Creates fake games instead of removing unanimous winners/losers.  Value is the weight of the games"))
            .arg(Arg::with_name("random-subgraph-links")
                 .long("random-subgraph-links")
                 .takes_value(true)
                 .help("Creates K random games between subgraphs to create fully connected graph"))
            .arg(Arg::with_name("random-subgraph-weight")
                 .long("random-subgraph-weight")
                 .takes_value(true)
                 .help("Weight of the games.  Defaults to 1e-3")))

        .subcommand(SubCommand::with_name("rate")
            .about("Computes rankings based on win loss ratios")
            .arg(Arg::with_name("confidence-interval")
                 .long("confidence-interval")
                 .takes_value(true)
                 .possible_values(&["0.95", "0.9", "0.5"])
                 .help("Confidence interval to compute the rank")))

        .subcommand(SubCommand::with_name("glicko2")
            .about("Computes glicko2 rankings for pairs of games.")
            .arg(Arg::with_name("tau")
                 .long("tau")
                 .takes_value(true)
                 .help("Uncertainty metric, tupically between 0.3 and 1.5.  Lower means more luck based"))
            .arg(Arg::with_name("use-mu")
                 .long("use-mu")
                 .help("If provided, uses mu instead of 95% lower-bound")))

        .subcommand(SubCommand::with_name("btm-lr")
            .about("Computes bradley-terry model rankings logistic regression")
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Learning rate for each pass.  Defaults to 1e-2."))
            .arg(Arg::with_name("decay")
                 .long("decay")
                 .takes_value(true)
                 .help("Decays weights each pass."))
            .arg(Arg::with_name("passes")
                 .long("passes")
                 .takes_value(true)
                 .help("Number of passes to perform SGD.  Default is 10"))
            .arg(Arg::with_name("thrifty")
                 .long("thrifty")
                 .help("Reduces allocations by processing each record 
                        independently")))

        .subcommand(SubCommand::with_name("page-rank")
            .about("Computes the page rank of each vertex in a directed graph")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("damping-factor")
                 .long("damping-factor")
                 .takes_value(true)
                 .help("Damping Factor to use.  Default is 0.85"))
            .arg(Arg::with_name("sink-dispersion")
                 .long("sink-dispersion")
                 .takes_value(true)
                 .possible_values(&["none", "reverse", "all"])
                 .help("How sink nodes are dispersed.  Default is reverse")))

        .subcommand(SubCommand::with_name("birank")
            .about("Computes importance using the BiRank method.")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Blend coefficiant for p_0 vector"))
            .arg(Arg::with_name("beta")
                 .long("beta")
                 .takes_value(true)
                 .help("Blend coefficiant for u_0 vector"))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use.")))

        .subcommand(SubCommand::with_name("vec-prop")
            .about("Propagates sparse vectors within a graph using the Vec-Prop algorithm.")
            .arg(Arg::with_name("prior")
                 .long("prior")
                 .takes_value(true)
                 .required(true)
                 .help("Prior dataset file"))
            .arg(Arg::with_name("regularizer")
                 .long("regularizer")
                 .takes_value(true)
                 .possible_values(&["l1", "l2", "symmetric"])
                 .help("Embedding regularizer.  Default is 'symmetric'."))
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Blend coefficiant for prior vector.  Higher values assumes a weaker prior."))
            .arg(Arg::with_name("max-terms")
                 .long("max-terms")
                 .takes_value(true)
                 .help("Max terms to keep between propagations"))
            .arg(Arg::with_name("error")
                 .long("error")
                 .takes_value(true)
                 .help("Max error rate before suppressing the data"))
            .arg(Arg::with_name("l2-output")
                 .long("l2-output")
                 .help("If provided, L2 normalizes the final embeddings"))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use."))
            .arg(Arg::with_name("chunks")
                 .long("chunks")
                 .takes_value(true)
                 .help("Tunes concurrent hashmap.  Default is 91.")))

        .subcommand(SubCommand::with_name("vec-walk")
            .about("Propagates sparse vectors within a graph using weighted random walks.")
            .arg(Arg::with_name("prior")
                 .long("prior")
                 .takes_value(true)
                 .required(true)
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph. Default is 10"))
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Update parameter, bound between (0, 1).  Higher learns faster, lower has lower volatility."))
            .arg(Arg::with_name("max-terms")
                 .long("max-terms")
                 .takes_value(true)
                 .help("Max terms to keep between propagations.  Default is 100"))
            .arg(Arg::with_name("error")
                 .long("error")
                 .takes_value(true)
                 .help("Max error rate before suppressing the data. Default is 1e-5"))
            .arg(Arg::with_name("chunks")
                 .long("chunks")
                 .takes_value(true)
                 .help("Tunes concurrent hashmap.  Default is 91."))
            .arg(Arg::with_name("walk-len")
                 .long("walk-len")
                 .takes_value(true)
                 .help("Number of steps in the random walk"))
            .arg(Arg::with_name("uniform-walk")
                 .long("uniform-walk")
                 .help("Whether to do an unweighted random walk."))
            .arg(Arg::with_name("context-window")
                 .long("context-window")
                 .takes_value(true)
                 .help("Number of adjacent embeddings to average together. Default is 2"))
            .arg(Arg::with_name("negative-sample")
                 .long("negative-sample")
                 .takes_value(true)
                 .help("Number of vertices to randomly sample to subtract.  Default is 5"))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use.")))

        .subcommand(SubCommand::with_name("lpa")
            .about("Computes clusters using label propagation")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph.  Default is 10.  When set to 0, will run until there are no more node membership changes."))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use."))
            .arg(Arg::with_name("chunks")
                 .long("chunks")
                 .takes_value(true)
                 .help("Number of vertices to perform in parallel.  Default is 10")))

        .subcommand(SubCommand::with_name("random-walk")
            .about("Generates random walks from a graph")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of random walks starting at each node to run.  Default is 10"))
            .arg(Arg::with_name("walk-len")
                 .long("walk-len")
                 .takes_value(true)
                 .help("Number of steps in the random walk."))
            .arg(Arg::with_name("uniform-walk")
                 .long("uniform-walk")
                 .help("If provided, performs a uniform walk instead of a biased walk."))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use."))
            .arg(Arg::with_name("buffer-size")
                 .long("buffer-size")
                 .takes_value(true)
                 .help("Amount of temp space to use for parallel computation.  Default is 10000.")))

        .subcommand(SubCommand::with_name("dehydrate")
            .about("Converts a human readable graph format to a propagon compatible version.")
            .arg(Arg::with_name("delim")
                 .long("delim")
                 .takes_value(true)
                 .help("Uses the provided delimiter for splitting lines.  Default is tab"))
            .arg(Arg::with_name("features")
                 .long("features")
                 .takes_value(true)
                 .help("Converts an optional features file to the index format.")))

        .subcommand(SubCommand::with_name("hydrate")
            .about("Converts an embedding back to a readable format")
            .arg(Arg::with_name("vocab")
                 .long("vocab")
                 .takes_value(true)
                 .required(true)
                 .help("Vocab file to convert the embedding back from."))
            .arg(Arg::with_name("output")
                 .long("output")
                 .takes_value(true)
                 .help("If provided, writes to the given file.  Otherwise, to stdout.")))

        .subcommand(SubCommand::with_name("label-rank")
            .about("Computes clusters using LabelRank")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph.  Default is 10."))
            .arg(Arg::with_name("inflation")
                 .long("inflation")
                 .takes_value(true)
                 .help("Label inflation exponent.  Default is 2."))
            .arg(Arg::with_name("prior")
                 .long("prior")
                 .takes_value(true)
                 .help("Label prior.  Default is 0.1"))
            .arg(Arg::with_name("q")
                 .long("q")
                 .takes_value(true)
                 .help("q value for stopping criteria.  Default is 0.1"))
            .arg(Arg::with_name("max-terms")
                 .long("max-terms")
                 .takes_value(true)
                 .help("Maximum number of terms to store.  Default is 10.")))

        .subcommand(SubCommand::with_name("gcs")
            .about("Generates dense embeddings based on the graph coordinate system of shortest distance")
            .arg(Arg::with_name("dims")
                 .long("dims")
                 .required(true)
                 .takes_value(true)
                 .help("Embedding dimensions to use"))
            .arg(Arg::with_name("landmarks")
                 .long("landmarks")
                 .required(true)
                 .takes_value(true)
                 .help("Number of landmarks to use"))
            .arg(Arg::with_name("only-walks")
                 .long("only-walks")
                 .help("If enabled, only emits the walk distances"))
            .arg(Arg::with_name("global-embed-fns")
                 .long("global-embed-fns")
                 .takes_value(true)
                 .help("Number of function calls for the global optimization step.  Default is 1,000,000"))
            .arg(Arg::with_name("local-embed-fns")
                 .long("local-embed-fns")
                 .takes_value(true)
                 .help("Number of Function calls for the local optimization step.  Default is 1,000"))
            .arg(Arg::with_name("global-bias")
                 .long("global-bias")
                 .takes_value(true)
                 .help("Amount to bias toward global distance versus local.  Defaults to 0.7"))
            .arg(Arg::with_name("passes")
                 .long("passes")
                 .takes_value(true)
                 .help("Number of passes to fine-tune on.  Default is 3."))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use."))
            .arg(Arg::with_name("weighting")
                 .long("weighting")
                 .takes_value(true)
                 .possible_values(&["uniform", "degree", "edge", "original"])
                 .help("Selects how edge weights are treated during distance calculations.  Default is uniform."))
            .arg(Arg::with_name("chunks")
                 .long("chunks")
                 .takes_value(true)
                 .help("Tunes concurrent hashmap.  Default is 91."))
            .arg(Arg::with_name("selection")
                 .long("selection")
                 .takes_value(true)
                 .possible_values(&["random", "degree"])
                 .help("Pick which mechanism to use for landmark selection.  Default is degree"))
            .arg(Arg::with_name("space")
                 .long("space")
                 .takes_value(true)
                 .possible_values(&["euclidean", "poincare", "hyperboloid", "manhattan"])
                 .help("Space to learn embeddings.  Default is Euclidean"))
            .arg(Arg::with_name("l2")
                 .long("l2")
                 .help("If provided, L2-norms the embeddings.")))

        .subcommand(SubCommand::with_name("mc-cluster")
            .about("Community detection via random walks.")
            .arg(Arg::with_name("steps")
                 .long("steps")
                 .takes_value(true)
                 .help("Total number of steps to take for sampling. Default is 10000."))
            .arg(Arg::with_name("restarts")
                 .long("restarts")
                 .takes_value(true)
                 .help("Probability that a random walk restarts.  Default is 0.5"))
            .arg(Arg::with_name("ppr")
                 .long("ppr")
                 .help("Walk uses Personalized Page Rank instead of all-node accounting"))
            .arg(Arg::with_name("max-terms")
                 .long("max-terms")
                 .takes_value(true)
                 .help("Keeps only the top K terms.  Default is 50"))
            .arg(Arg::with_name("sampler")
                 .long("sampler")
                 .takes_value(true)
                 .possible_values(&["random-walk", "metropolis-hastings"])
                 .help("How to sample the distribution around the node.  Default is 'metropolist-hastings'"))
            .arg(Arg::with_name("metric")
                 .long("metric")
                 .takes_value(true)
                 .possible_values(&["cosine", "jaccard", "ratio"])
                 .help("Similarity metric to use.  Default is Cosine.  Used by 'similarity'"))
            .arg(Arg::with_name("best-only")
                 .long("best-only")
                 .help("Chooses only the max score for each node.  Creates sparser, smaller graphs.  Used by 'similarity'"))
            .arg(Arg::with_name("min-cluster-size")
                 .long("min-cluster-size")
                 .takes_value(true)
                 .help("Minimum cluster size to emit.  Default is 1"))
            .arg(Arg::with_name("rem-weak-links")
                 .long("rem-weak-links")
                 .help("If provided, removes weak links within clusters.  Used by 'similarity'"))
            .arg(Arg::with_name("clusterer")
                 .long("clusterer")
                 .takes_value(true)
                 .possible_values(&["attractor", "similarity"])
                 .help("Which embedding clusterer to use.  Default is 'attractor'"))
            .arg(Arg::with_name("num-attractors")
                 .long("num-attractors")
                 .takes_value(true)
                 .help("Number of attractors to use.  Used by 'attractor'.  Default is '1'"))
            .arg(Arg::with_name("save-embeddings")
                 .long("save-embeddings")
                 .takes_value(true)
                 .help("If provided, writes out the markov embeddings to the provided file"))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use.")))

       .subcommand(SubCommand::with_name("hash-embedding")
            .about("Generates node embeddings based on hash kernels.")
            .arg(Arg::with_name("dims")
                 .long("dims")
                 .required(true)
                 .takes_value(true)
                 .help("Embedding dimensions to use"))
            .arg(Arg::with_name("hashes")
                 .long("hashes")
                 .takes_value(true)
                 .help("Number of random hashes to use per value.  Default is 3"))
            .arg(Arg::with_name("steps")
                 .long("steps")
                 .takes_value(true)
                 .help("Total number of steps to take for sampling. Default is 10000."))
            .arg(Arg::with_name("restarts")
                 .long("restarts")
                 .takes_value(true)
                 .help("Probability that a random walk restarts.  Default is 0.1"))
            .arg(Arg::with_name("ppr")
                 .long("ppr")
                 .help("If provided, estimates using PPR estimation."))
            .arg(Arg::with_name("weighted")
                 .long("weighted")
                 .help("If provided, uses edge weights to guide hashing"))
            .arg(Arg::with_name("sampler")
                 .long("sampler")
                 .takes_value(true)
                 .possible_values(&["random-walk", "metropolis-hastings"])
                 .help("How to sample the distribution around the node.  Default is 'metropolist-hastings'"))
            .arg(Arg::with_name("normalize")
                 .long("normalize")
                 .takes_value(true)
                 .possible_values(&["none", "l1", "l2"])
                 .help("Normalizes the embeddings.  Default is 'none'"))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use.")))

        .subcommand(SubCommand::with_name("es-rum")
            .about("Fits a Random Utility Model to partial orderings")
            .arg(Arg::with_name("passes")
                 .long("passes")
                 .takes_value(true)
                 .help("Number of optimization steps to run"))
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Learning rate for the gradient-free step.  Default is 1"))
            .arg(Arg::with_name("gamma")
                 .long("gamma")
                 .takes_value(true)
                 .help("Regularization on the distribution scores.  Default is 1e-5"))
            .arg(Arg::with_name("min-observations")
                 .long("min-obs")
                 .takes_value(true)
                 .help("Only emits scores observed more than K times.  Default is 1"))
            .arg(Arg::with_name("fixed")
                 .long("fixed")
                 .help("If set, fixes the variance for each distribution to 1, only learning mu."))
            .arg(Arg::with_name("seed")
                 .long("seed")
                 .takes_value(true)
                 .help("Random seed to use.")))

        .subcommand(SubCommand::with_name("kemeny")
            .about("Optimizes a Kemeny ranking for the provided pairs")
            .arg(Arg::with_name("passes")
                 .long("passes")
                 .takes_value(true)
                 .help("Number of passes, dependent on algorithm.  If 'insertion' is used, 
                        it is the number of refinements we run (roughly O(K*N^2)) - default is 1.  
                        If 'de' is used, passes is the number of function calls to run.  Default is 50,000."))
            .arg(Arg::with_name("min-obs")
                 .long("min-obs")
                 .takes_value(true)
                 .help("Only omits alternatives with at least K appearances.  Default is 1"))
            .arg(Arg::with_name("algo")
                 .long("algo")
                 .takes_value(true)
                 .possible_values(&["insertion", "de"])
                 .help("Algorithm to optimize with.  Default is 'insertion'.")))

        .subcommand(SubCommand::with_name("extract-components")
            .about("Extracts fully connected components from a graph and writes them to separate files")
            .arg(Arg::with_name("min-graph-size")
                 .long("min-size")
                 .takes_value(true)
                 .help("Minimum graph size to emit.  Default is 1")))
 
        .get_matches()
}


fn main() {
    let args = parse();
    let path: Vec<String> = args.values_of("path")
        .expect("Need a path to edges")
        .map(|x| x.into())
        .collect();

    if let Some(ref sub_args) = args.subcommand_matches("dehydrate") {
        dehydrate(&path[0], sub_args);
        return
    } else if let Some(ref sub_args) = args.subcommand_matches("hydrate") {
        hydrate(&path[0], sub_args);
        return
    }

    let min_count  = value_t!(args, "min-count", usize).unwrap_or(1);

    let mut reader: Box<dyn reader::GameReader> = if args.is_present("groups-are-separate") {
        Box::new(reader::EachSetSeparate::new(path.clone()))
    } else {
        Box::new(reader::AllGames::new(path.clone()))
    };

    loop {
        if let Ok(Some(games)) = reader.next_set() {
            for (i, g) in games.iter().enumerate() {
                eprintln!("Set {}: Read in {} edges", i, g.len());
            }

            // Weed out matches with competitors less than min-count
            let games = filter_edges(games, min_count);
            for (i, g) in games.iter().enumerate() {
                eprintln!("Post Filter - Set {}: Read in {} edges", i, g.len());
            }

            if let Some(ref sub_args) = args.subcommand_matches("btm-mm") {
                // Flatten games
                let all_games = games.into_iter().flatten().collect();
                mm::minor_maxim(sub_args, all_games, min_count);
            } else if let Some(ref sub_args) = args.subcommand_matches("rate") {
                let all_games = games.into_iter().flatten().collect();
                rate(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("glicko2") {
                glicko(sub_args, games);
            } else if let Some(ref sub_args) = args.subcommand_matches("btm-lr") {
                btm_lr(sub_args, games);
            } else if let Some(ref sub_args) = args.subcommand_matches("es-rum") {
                let all_games = games.into_iter().flatten().collect();
                es_rum(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("kemeny") {
                let all_games = games.into_iter().flatten().collect();
                kemeny(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("page-rank") {
                let all_games = games.into_iter().flatten().collect();
                page_rank(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("birank") {
                let all_games = games.into_iter().flatten().collect();
                birank(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("vec-prop") {
                let all_games = games.into_iter().flatten().collect();
                vec_prop(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("vec-walk") {
                let all_games = games.into_iter().flatten().collect();
                vec_walk(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("lpa") {
                let all_games = games.into_iter().flatten().collect();
                lpa(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("label-rank") {
                let all_games = games.into_iter().flatten().collect();
                label_rank(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("random-walk") {
                let all_games = games.into_iter().flatten().collect();
                random_walk(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("gcs") {
                let all_games = games.into_iter().flatten().collect();
                euc_emb(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("mc-cluster") {
                let all_games = games.into_iter().flatten().collect();
                mc_cluster(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("hash-embedding") {
                let all_games = games.into_iter().flatten().collect();
                hash_embedding(sub_args, all_games);
            } else if let Some(ref sub_args) = args.subcommand_matches("extract-components") {
                let all_games = games.into_iter().flatten().collect();
                components(sub_args, &path[0], all_games);
            }

            // print a separator
            println!("");
        } else {
            break
        }
    }
}
