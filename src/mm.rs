extern crate random;
use super::{Games,Match,emit_scores,filter_edges,tally_winners_losers};

use random::Source;

use hashbrown::{HashMap,HashSet};

fn compute_err(p: &HashMap<u32,f32>, np: &HashMap<u32, f32>) -> f32 {
    let mut err = 0.;
    let mut cnt = 0;
    for (k, v) in p.iter() {
        err += (*v - np[k]).abs();
        cnt += 1;
    }
    err / cnt as f32
}

fn get_subgraphs(games: &[Match]) -> Vec<Vec<u32>> {
    let mut graph = HashMap::new();
    // Build adjacencies
    for (w, l, _) in games.iter() {
        let e = graph.entry(*w).or_insert_with(|| Vec::new());
        e.push(*l);
        let e = graph.entry(*l).or_insert_with(|| Vec::new());
        e.push(*w);
    }

    let mut seen = HashSet::new();

    let mut subgraphs = Vec::new();
    let mut stack = Vec::new();
    for start in graph.keys() {
        if seen.contains(start) { continue }
        let mut subgraph = Vec::new();
        stack.clear();
        stack.push(*start);
        while !stack.is_empty() {
            let node = stack.pop().unwrap();
            subgraph.push(node);
            seen.insert(node);

            // Get neighbors
            for out in graph[&node].iter() {
                if !seen.contains(out) {
                    stack.push(*out)
                }
            }
        }
        subgraphs.push(subgraph);
    }
    subgraphs
}

// Creating fake games on winners or losers allows us to optimize everything
// within the same framework rather than pruning undefeated/unwinning teams
fn create_fake_games(games: &mut Games, weight: f32) {
    let (w, l) = tally_winners_losers(&games);

    let mut new_games = Vec::new();
    for (winner, loser, _s) in games.iter() {
        if w.contains_key(winner) ^ l.contains_key(winner) {
            new_games.push((*winner, *loser, weight));
            new_games.push((*loser, *winner, weight));
        }
    }
    eprintln!("Created {} fake games", new_games.len());
    games.extend_from_slice(new_games.as_slice());
}

fn remove_undefeated(mut games: Games, losers_too: bool) -> (Games, HashMap<u32,usize>, HashMap<u32,usize>) {
    let mut undef = HashMap::new();
    let mut last_place = HashMap::new();
    loop {
        let (w, l) = tally_winners_losers(&games);
        let b_len = games.len();
        if losers_too {
            let contains = |id: &u32| {w.contains_key(id) ^ l.contains_key(id)};
            games = games.into_iter()
                .filter(|(wi, li, _s)| {
                    if contains(wi) || contains(li) {
                        if w.contains_key(wi) {
                            undef.insert(*wi, w[wi].0);
                        }
                        if l.contains_key(li) {
                            last_place.insert(*li, l[li].0);
                        }
                        false
                    } else {
                        true
                    }
                }).collect();

        } else {
            let all_winners = |id: &u32| {w.contains_key(id) && !l.contains_key(id)};
            games = games.into_iter()
                .filter(|(wi, _li, _s)| {
                    if all_winners(wi) {
                        undef.insert(*wi, w[wi].0);
                        false
                    } else {
                        true
                    }
                }).collect();
        }

        if games.len() == b_len {
            break
        }
    }
    (games, undef, last_place)
}

fn create_random_subgraphs(games: &mut Games, sgs: &Vec<Vec<u32>>, weight: f32) {
    let mut source = random::default().seed([1221, 2019]);

    for (i, gs) in sgs.iter().enumerate() {
        // Select random id
        let src = source.read::<usize>() % gs.len();
        let mut dest_graph;
        loop {
            dest_graph = source.read::<usize>() % sgs.len();
            if dest_graph != i { break };
        }

        let dest = source.read::<usize>() % sgs[dest_graph].len();

        // Bidirectional games
        games.push((gs[src], sgs[dest_graph][dest], weight));
        games.push((sgs[dest_graph][dest], gs[src], weight))
    }
}

// Minorization-Maximization approach to bradley-terry models
#[allow(non_snake_case)] 
fn mm_opti(games: &Games, sgi: usize, sg: Vec<u32>, iterations: usize, tol: f32) -> Vec<(u32,f32)> {
    let sg: HashSet<_> = sg.into_iter().collect();

    // Set up the book keeping
    let mut wij = HashMap::new();
    let mut Wi  = HashMap::new();
    let mut nij = HashMap::new();

    for (w, l, s) in games.iter() {
        if !sg.contains(w) { continue }
        
        // Add winners
        let e = wij.entry(*w).or_insert_with(|| HashMap::new());
        let count = e.entry(*l).or_insert(0.0);
        *count += s;

        let e = Wi.entry(*w).or_insert(0.);
        *e += s;
        Wi.entry(*l).or_insert(0.);

        let e = nij.entry((*w, *l)).or_insert(0.);
        *e += s;
        let e = nij.entry((*l, *w)).or_insert(0.);
        *e += s;
    }

    let mut policy = HashMap::new();
    let mut new_policy: HashMap<_,_> = Wi.keys()
        .map(|x| (*x, 1. / Wi.len() as f32))
        .collect();

    // Update the policy each iteration
    for iter in 0..iterations {
        std::mem::swap(&mut policy, &mut new_policy);
        new_policy.clear();

        for (team, pi) in policy.iter() {
            // Sum[(nij) / Sum[p(i) + p(j)]]
            let npi = if let Some(hm) = wij.get(team) {
                let mut denom = 0f32;
                for (oteam, _wins) in hm.iter() {
                    denom += nij[&(*team, *oteam)] as f32 / (pi + policy[oteam]);
                }
                Wi[team] as f32 / denom
            } else {
                0f32
            };
            new_policy.insert(*team, npi);
        }

        // Normalize it to 1
        let total: f32 = new_policy.values().sum();
        for v in new_policy.values_mut() {
            *v /= total;
        }

        // compute err
        let err = compute_err(&policy, &new_policy);
        if iter % 10 == 0 {
            eprintln!("Iteration: {}, Error: {}", iter, err);
        }
        if err < tol {
            eprintln!("Error is less than tolerance, exiting early...");
            break;
        }
    }

    // Output it
    if sgi > 0 {
        println!("");
    }

    let mut np: Vec<_> = new_policy.into_iter().collect();
    np.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());
    np
}

// Runs an MM variation of BTM, utilizing a variety of workarounds to either fully
// connect the strongly connected graphs or ensure invariants (such as no undefeated teams)
pub fn minor_maxim(args: &&clap::ArgMatches<'_>, mut games: Games, min_count: usize)  {
    let min_graph_size = value_t!(args, "min-graph-size", usize).unwrap_or(1);
    let iterations     = value_t!(args, "iterations", usize).unwrap_or(10000);
    let tol            = value_t!(args, "tolerance", f32).unwrap_or(1e-6);
    let rsl            = value_t!(args, "random-subgraph-links", usize).unwrap_or(0);
    let rsw            = value_t!(args, "random-subgraph-weight", f32).unwrap_or(1e-3);
    let rtl            = args.is_present("remove-total-losers");
    let cfg            = value_t!(args, "create-fake-games", f32).unwrap_or(0.);

    // Remove undefeated teams
    let (games, undef, last_place) = if cfg > 0. {
        create_fake_games(&mut games, cfg);
        (games, HashMap::new(), HashMap::new())
    } else {
        eprintln!("Removing undefeated/never won teams");
        let out = remove_undefeated(games, rtl);
        eprintln!("Removed {} undefeated, {} last place teams", out.1.len(), out.2.len());
        out
    };

    // Weed out matches with competitors less than min-count
    let mut games = filter_edges(vec![games], min_count).pop()
        .expect("Should never be empty");
    
    eprintln!("Total Remaining Matches: {}", games.len());
    let mut sgs = get_subgraphs(&games);
    eprintln!("Total Graphs : {}", sgs.len());

    let sgs = if rsl > 0 && sgs.len() > 1 {
        eprintln!("Creating random links between graphs");
        for _ in 0..rsl {
            create_random_subgraphs(&mut games, &sgs, rsw);
        }
        vec![sgs.into_iter().flatten().collect()]
    } else {
        sgs.sort_by_key(|sg| -(sg.len() as i32));
        sgs
    };

    for (sgi, sg) in sgs.into_iter().enumerate() {
        if sg.len() < min_graph_size {
            continue
        }
        let policy = mm_opti(&games, sgi, sg, iterations, tol);
        emit_scores(policy.into_iter());
    }

    // Emit undefeated first
    let mut vec: Vec<_> = undef.into_iter().filter(|(_,c)| c >= &min_count).collect();
    vec.sort_by_key(|(_, c)| -(*c as i64));
    println!("");
    emit_scores(vec.into_iter().map(|(k, cnt)| (k, cnt as f32)));

    let mut vec: Vec<_> = last_place.into_iter().filter(|(_,c)| c >= &min_count).collect();
    vec.sort_by_key(|(_, c)| *c as i64);
    println!("");
    emit_scores(vec.into_iter().map(|(k, cnt)| (k, - (cnt as f32))));
}
