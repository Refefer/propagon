mod mm;
mod lr;
mod g2;
mod reader;

#[macro_use]
extern crate clap;

use clap::{Arg, App, ArgMatches, SubCommand};

use std::fmt::Display;
use std::collections::{HashMap,HashSet};

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

fn btm_lr(args: &&clap::ArgMatches<'_>, games: Games) {
    let alpha = value_t!(args, "alpha", f32).unwrap_or(10.);
    let passes = value_t!(args, "passes", usize).unwrap_or(10);
    let mut scores = lr::lr(&games, passes, alpha);
    
    scores.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());
    emit_scores(scores.into_iter());
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
            .arg(Arg::with_name("confidence-interval")
                 .long("confidence-interval")
                 .takes_value(true)
                 .possible_values(&["0.95", "0.9", "0.5"])
                 .help("Confidence interval to compute the rank")))

        .subcommand(SubCommand::with_name("glicko2")
            .arg(Arg::with_name("tau")
                 .long("tau")
                 .takes_value(true)
                 .help("Uncertainty metric, tupically between 0.3 and 1.5.  Lower means more luck based"))
            .arg(Arg::with_name("use-mu")
                 .long("use-mu")
                 .help("If provided, uses mu instead of 95% lower-bound")))

        .subcommand(SubCommand::with_name("btm-lr")
            .arg(Arg::with_name("alpha")
                 .long("alpha")
                 .takes_value(true)
                 .help("Learning rate for each pass.  Defaults to 1e-2."))
            .arg(Arg::with_name("passes")
                 .long("passes")
                 .takes_value(true)
                 .help("Number of passes to perform SGD.  Default is 10")))

        .get_matches()
}

fn main() {
    let args = parse();
    let path: Vec<_> = args.values_of("path")
        .expect("Need a path to edges")
        .map(|x| x.into())
        .collect();

    let min_count  = value_t!(args, "min-count", usize).unwrap_or(1);

    let mut reader: Box<reader::GameReader> = if args.is_present("groups-are-separate") {
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
                let all_games = games.into_iter().flatten().collect();
                btm_lr(sub_args, all_games);
            }
            // print a separator
            println!("");
        } else {
            break
        }
    }
}
