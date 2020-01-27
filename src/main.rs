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
                eprintln!("Filtered {} matches ({} remaining)", len - games.len(), len);
            }
            eprintln!("Filtered out {} total matches", total_games - games.len());
            new_games.push(games);
        }
    } else {
        new_games = game_sets;
    }
    new_games
}

fn tally_winners_losers(games: &Games) -> (HashMap<u32,(usize,f32)>, HashMap<u32,(usize,f32)>) {
    let mut w = HashMap::new();
    let mut l = HashMap::new();
    for (winner, loser, s) in games.iter() {
        let e = w.entry(*winner).or_insert((0, 0.));
        e.0 += 1;
        e.1 += s;
        let e = l.entry(*loser).or_insert((0, 0.));
        e.0 += 1;
        e.1 += s;
    }
    (w, l)
}

fn emit_scores<K: Display, V: Display>(it: impl Iterator<Item=(K,V)>) {
    for (id, s) in it {
        println!("{}: {}", id, s);
    }
}

// computes the Action rates
fn rate(args: &&clap::ArgMatches<'_>, games: Games) {
    let ci = value_t!(args, "confidence-interval", f32).unwrap_or(0.95);

    // Compute rate stats
    let (mut winners, losers) = tally_winners_losers(&games);
    
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
    let tau = value_t!(args, "tau", f64).unwrap_or(0.5);
    let use_mu = args.is_present("use-mu");
    let env = g2::Env::default();
    let mut series = env.new_match_set(tau);
    for (i, game_set) in games.into_iter().enumerate() {
        eprintln!("Game set {}, Known teams: {}", i, series.teams().len());
        let gs = game_set.into_iter().map(|(w,l,_)| (w, l)).collect();
        series.update(gs);
    }
    let mut scores: Vec<_> = series.teams().iter()
        .map(|(t, p)| {
            if use_mu {
                (t, p.mu(&env))
            } else {
                (t, p.bounds().0)
            }
        }).collect();

    scores.sort_by(|a, b| (b.1).partial_cmp(&a.1).expect("blew up in sort"));
    emit_scores(scores.into_iter());

}

fn btm_lr(args: &&clap::ArgMatches<'_>, games: Vec<Games>) {
    let alpha = value_t!(args, "alpha", f32).unwrap_or(1.);
    let decay = value_t!(args, "decay", f32).unwrap_or(1.);
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

    let settings = birank::Settings {
        n_iters: iterations,
        alpha,
        beta,
        seed: 2019
    };

    let mut birank = birank::BiRank::build(games.into_iter());
    birank.randomize(&settings);
    birank.compute(&settings, HashMap::new(), HashMap::new());
    birank.emit();
}

fn fast_json<K>(
    it: impl Iterator<Item=(K, Vec<(usize,f32)>)>, 
    idx_to_vocab: HashMap<usize,String>
) -> impl Iterator<Item=(K,String)> {
    it.map(move |(id, mut emb)| {
        let mut string = String::new();
        string.push_str("{");
        if emb.len() > 0 {
            emb.sort_by(|a,b| (b.1).partial_cmp(&a.1).unwrap());
            emb.into_iter().for_each(|(f, v)| {
                string.push_str(format!("\"{}\":{},", idx_to_vocab[&f], v).as_str());
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
    let chunks     = value_t!(args, "chunks", usize).unwrap_or(10);

    let reg = match args.value_of("regularizer").unwrap() {
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
        seed: 2019
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
        seed: 2019
    };

    // Load priors
    let (priors, idx_to_vocab) = vp::load_priors(prior.as_str());
    let embeddings = vw.fit(games.into_iter(), &priors);

    let it = embeddings.into_iter().map(|(k, v)| (k, v.0));
    emit_scores(fast_json(it, idx_to_vocab));
}


fn lpa(args: &&clap::ArgMatches<'_>, games: Games) {
    let iterations = value_t!(args, "iterations", usize).ok();
    let chunks     = value_t!(args, "chunks", usize).unwrap_or(10);

    let lpa = lpa::LPA {
        n_iters: iterations,
        chunks: chunks,
        seed: 2019
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

    let random_walk = rw::RandomWalk {
        iterations,
        buffer_size,
        walk_len,
        biased_walk,
        seed: 2019
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


fn parse<'a>() -> ArgMatches<'a> {
    App::new("btm")
        .version("0.0.1")
        .author("Andrew S. <refefer@gmail.com>")
        .about("Computes the bradley-terry model of a given set")
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
                 .help("Blend coefficiant for u_0 vector")))

        .subcommand(SubCommand::with_name("vec-prop")
            .about("Propagates sparse vectors within a graph using the Vec-Prop algorithm.")
            .arg(Arg::with_name("prior")
                 .long("prior")
                 .takes_value(true)
                 .required(true)
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("regularizer")
                 .long("regularizer")
                 .takes_value(true)
                 .possible_values(&["l1", "l2", "symmetric"])
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph"))
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Blend coefficiant for prior vector"))
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
            .arg(Arg::with_name("chunks")
                 .long("chunks")
                 .takes_value(true)
                 .help("Number of vertices to perform in parallel.  Default is 10")))

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
                 .long("uniform-len")
                 .takes_value(true)
                 .help("Whether to do an unweighted random walk."))
            .arg(Arg::with_name("context-window")
                 .long("context-window")
                 .takes_value(true)
                 .help("Number of adjacent embeddings to average together. Default is 2"))
            .arg(Arg::with_name("negative-sample")
                 .long("negative-sample")
                 .takes_value(true)
                 .help("Number of vertices to randomly sample to subtract.  Default is 5")))

        .subcommand(SubCommand::with_name("lpa")
            .about("Computes clusters using label propagation")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph.  If omitted, runs until there are no more node membership shanges"))
            .arg(Arg::with_name("chunks")
                 .long("chunks")
                 .takes_value(true)
                 .help("Number of vertices to perform in parallel.  Default is 10")))

        .subcommand(SubCommand::with_name("random-walk")
            .about("Generates random walks from a graph")
            .arg(Arg::with_name("iterations")
                 .long("iterations")
                 .takes_value(true)
                 .help("Number of iterations to compute on the graph.  If omitted, runs until there are no more node membership shanges"))
            .arg(Arg::with_name("walk-len")
                 .long("walk-len")
                 .takes_value(true)
                 .help("Number of steps in the random walk."))
            .arg(Arg::with_name("unbiased-walk")
                 .long("uniform-walk")
                 .help("If provided, performs a uniform walk instead of a biased walk."))
            .arg(Arg::with_name("buffer-size")
                 .long("buffer-size")
                 .takes_value(true)
                 .help("Amount of temp space to use for parallel computation.  Default is 10000.")))

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


        .get_matches()
}

fn main() {
    let args = parse();
    let path: Vec<_> = args.values_of("path")
        .expect("Need a path to edges")
        .map(|x| x.into())
        .collect();

    let min_count  = value_t!(args, "min-count", usize).unwrap_or(1);

    let mut reader: Box<dyn reader::GameReader> = if args.is_present("groups-are-separate") {
        Box::new(reader::EachSetSeparate::new(path))
    } else {
        Box::new(reader::AllGames::new(path))
    };

    loop {
        if let Ok(Some(games)) = reader.next_set() {
            for (i, g) in games.iter().enumerate() {
                eprintln!("Set {}: Read in {} games", i, g.len());
            }

            // Weed out matches with competitors less than min-count
            let games = filter_edges(games, min_count);
            for (i, g) in games.iter().enumerate() {
                eprintln!("Post Filter - Set {}: Read in {} games", i, g.len());
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
            }

            // print a separator
            println!("");
        } else {
            break
        }
    }
}
