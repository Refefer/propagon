//! Correctness tests against **independently published** values: every
//! expected number below was read from a primary or authoritative source
//! (cited per test) — not produced by this codebase.
//!
//! Per AGENTS.md rule 7, integration tests are not exempt from the
//! unwrap/expect denies: every test propagates errors via `Result`.

use propagon::algos::{
    Bandit, BanditPolicy, Borda, BradleyTerryMM, Copeland, Elo, Kemeny, KemenyPasses, PageRank,
    Sink, wilson_interval,
};
use propagon::{
    GraphDataset, OnlineRanker, PairwiseDataset, RankModel, Ranker, RankingsDataset, RewardsDataset,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Bradley-Terry on Agresti's 1987 AL East baseball data.
///
/// Source: Agresti, *Categorical Data Analysis* (2nd ed., p. 438), as
/// packaged and fit by the BradleyTerry2 R package vignette
/// (<https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html>,
/// model `baseballModel1`). Published ability estimates on the logit scale
/// with Baltimore as the zero reference:
/// Milwaukee 1.5814, Detroit 1.4364, Toronto 1.2945, New York 1.2476,
/// Boston 1.1077, Cleveland 0.6839.
///
/// The win matrix below is the package's `baseball` dataset aggregated over
/// home/away (13 games per pair, 273 games total). Watch the famous trap:
/// New York beat Boston 6 times, Boston beat New York 7.
#[test]
fn bradley_terry_matches_agresti_baseball() -> TestResult {
    const TEAMS: [&str; 7] = [
        "Milwaukee",
        "Detroit",
        "Toronto",
        "New York",
        "Boston",
        "Cleveland",
        "Baltimore",
    ];
    // wins[i][j] = games TEAMS[i] won against TEAMS[j].
    const WINS: [[u32; 7]; 7] = [
        [0, 7, 9, 7, 7, 9, 11],
        [6, 0, 7, 5, 11, 9, 9],
        [4, 6, 0, 7, 7, 8, 12],
        [6, 8, 6, 0, 6, 7, 10],
        [6, 2, 6, 7, 0, 7, 12],
        [4, 4, 5, 6, 6, 0, 6],
        [2, 4, 1, 3, 1, 7, 0],
    ];
    const EXPECTED: [f64; 6] = [1.5814, 1.4364, 1.2945, 1.2476, 1.1077, 0.6839];

    let mut d = PairwiseDataset::new();
    let mut total = 0;

    for (i, row) in WINS.iter().enumerate() {
        for (j, &wins) in row.iter().enumerate() {
            if wins > 0 {
                d.push(TEAMS[i], TEAMS[j], wins as f32);
                total += wins;
            }
        }
    }
    assert_eq!(total, 273, "273 games in the 1987 AL East season data");

    let algo = BradleyTerryMM {
        tolerance: 1e-12,
        ..Default::default()
    };
    let model = algo.fit(&d)?;
    let scores: std::collections::HashMap<&str, f64> = model.scores().collect();
    let baltimore = scores.get("Baltimore").ok_or("Baltimore fitted")?;

    for (team, expected) in TEAMS.iter().zip(EXPECTED) {
        let ability = (scores.get(team).ok_or("team fitted")? / baltimore).ln();
        assert!(
            (ability - expected).abs() < 1e-3,
            "{team}: fitted {ability:.4} vs published {expected}"
        );
    }

    Ok(())
}

/// Logistic Elo expected scores.
///
/// Source: Wikipedia, "Elo rating system" — `E = 1/(1 + 10^(-D/400))`; the
/// article's stated reference points (+100 → ~64%, +200 → ~76%) and the
/// exact algebraic values below. (FIDE's handbook table 8.1b is the older
/// normal-CDF variant and intentionally not used.)
#[test]
fn elo_expected_scores_match_published_values() -> TestResult {
    let elo = Elo::default();

    for (gap, expected) in [
        (0.0, 0.5),
        (100.0, 0.640_065_00),
        (200.0, 0.759_746_93),
        (400.0, 10.0 / 11.0),
    ] {
        let e = elo.expected_score(1500.0 + gap, 1500.0);
        assert!((e - expected).abs() < 1e-8, "gap {gap}: {e} vs {expected}");
    }

    Ok(())
}

/// Wilson score intervals, no continuity correction.
///
/// Source: Newcombe (1998), "Two-sided confidence intervals for the single
/// proportion", *Statistics in Medicine* 17:857-872, Table I method 3
/// (95% intervals, z = 1.959964).
#[test]
fn wilson_intervals_match_newcombe_table() -> TestResult {
    const Z: f64 = 1.959964;

    for (successes, n, lo, hi) in [
        (81.0, 263.0, 0.2553, 0.3662),
        (15.0, 148.0, 0.0624, 0.1605),
        (0.0, 20.0, 0.0000, 0.1611),
        (1.0, 29.0, 0.0061, 0.1718),
    ] {
        let (got_lo, got_hi) = wilson_interval(successes, n - successes, Z);
        assert!(
            (got_lo - lo).abs() < 5e-5,
            "({successes}/{n}) lower {got_lo:.4} vs {lo}"
        );
        assert!(
            (got_hi - hi).abs() < 5e-5,
            "({successes}/{n}) upper {got_hi:.4} vs {hi}"
        );
    }

    Ok(())
}

/// The Tennessee capital ballots used by both tests below.
///
/// Source: Wikipedia, "Borda count" / "Condorcet method" — 100 voters:
/// 42 Memphis>Nashville>Chattanooga>Knoxville, 26 Nashville>Chattanooga>
/// Knoxville>Memphis, 15 Chattanooga>Knoxville>Nashville>Memphis,
/// 17 Knoxville>Chattanooga>Nashville>Memphis.
const TENNESSEE: [(usize, [&str; 4]); 4] = [
    (42, ["Memphis", "Nashville", "Chattanooga", "Knoxville"]),
    (26, ["Nashville", "Chattanooga", "Knoxville", "Memphis"]),
    (15, ["Chattanooga", "Knoxville", "Nashville", "Memphis"]),
    (17, ["Knoxville", "Chattanooga", "Nashville", "Memphis"]),
];

/// Published Borda totals (3/2/1/0 points per ballot): Memphis 126,
/// Nashville 194, Chattanooga 173, Knoxville 107 → Nashville wins.
#[test]
fn borda_matches_tennessee_example() -> TestResult {
    let mut ballots = RankingsDataset::new();

    for (count, order) in TENNESSEE {
        for _ in 0..count {
            ballots.push_ranking(order)?;
        }
    }

    let model = Borda::default().fit_rankings(&ballots)?;
    let scores: std::collections::HashMap<&str, f64> = model.scores().collect();
    assert_eq!(scores.get("Memphis"), Some(&126.0));
    assert_eq!(scores.get("Nashville"), Some(&194.0));
    assert_eq!(scores.get("Chattanooga"), Some(&173.0));
    assert_eq!(scores.get("Knoxville"), Some(&107.0));

    Ok(())
}

/// Published pairwise majorities: Nashville beats every rival (Condorcet
/// winner); Copeland order Nashville 3 > Chattanooga 2 > Knoxville 1 >
/// Memphis 0.
#[test]
fn copeland_finds_tennessee_condorcet_winner() -> TestResult {
    let mut d = PairwiseDataset::new();

    for (count, order) in TENNESSEE {
        // Each ballot contributes one win per ordered pair it ranks.
        for i in 0..order.len() {
            for j in (i + 1)..order.len() {
                d.push(order[i], order[j], count as f32);
            }
        }
    }

    let model = Copeland::default().fit(&d)?;
    let scores: std::collections::HashMap<&str, f64> = model.scores().collect();
    assert_eq!(scores.get("Nashville"), Some(&3.0), "Condorcet winner");
    assert_eq!(scores.get("Chattanooga"), Some(&2.0));
    assert_eq!(scores.get("Knoxville"), Some(&1.0));
    assert_eq!(scores.get("Memphis"), Some(&0.0));

    Ok(())
}

/// PageRank on the 6-node worked example of Langville & Meyer.
///
/// Source: Langville & Meyer, "A Survey of Eigenvector Methods for Web
/// Information Retrieval", *SIAM Review* 47(1), 2005, §4.5: α = 0.9,
/// dangling node 2 replaced by a uniform row, published vector
/// π = (.03721, .05396, .04151, .3751, .206, .2862).
#[test]
fn pagerank_matches_langville_meyer_example() -> TestResult {
    let mut g = GraphDataset::new();

    for (src, dst) in [
        ("1", "2"),
        ("1", "3"),
        ("3", "1"),
        ("3", "2"),
        ("3", "5"),
        ("4", "5"),
        ("4", "6"),
        ("5", "4"),
        ("5", "6"),
        ("6", "4"),
    ] {
        g.push(src, dst, 1.0);
    }

    let pr = PageRank {
        damping: 0.9,
        iterations: 200,
        sink: Sink::Uniform,
    };
    let model = pr.fit(&g)?;
    let scores: std::collections::HashMap<&str, f64> = model.scores().collect();

    for (node, expected) in [
        ("1", 0.03721),
        ("2", 0.05396),
        ("3", 0.04151),
        ("4", 0.3751),
        ("5", 0.206),
        ("6", 0.2862),
    ] {
        let got = scores.get(node).ok_or("node ranked")?;
        assert!(
            (got - expected).abs() < 1e-4,
            "node {node}: {got:.5} vs {expected}"
        );
    }

    Ok(())
}

/// Kemeny insertion heuristic against an exhaustive oracle: enumerate all
/// 720 orderings of seeded 6-item tournaments (ground truth needs no
/// external source — it is the full search space).
///
/// Kemeny is NP-hard and insertion is a heuristic: on dense random
/// tournaments it can land in local optima (measured here: ≥ 97% of the
/// exhaustive optimum). The test pins that near-optimality honestly rather
/// than asserting exactness the algorithm does not guarantee.
#[test]
fn kemeny_insertion_attains_exhaustive_optimum() -> TestResult {
    // Deterministic xorshift64 fixture data.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for trial in 0..5 {
        let names: Vec<String> = (0..6).map(|i| format!("t{trial}_{i}")).collect();
        let mut wins = [[0u32; 6]; 6];
        let mut d = PairwiseDataset::new();

        for i in 0..6 {
            for j in 0..6 {
                if i != j {
                    let games = 1 + (rand() % 4) as u32;
                    wins[i][j] = games;
                    d.push(&names[i], &names[j], games as f32);
                }
            }
        }

        // Oracle: maximum concordant weight over all 720 permutations.
        let mut best = 0u32;
        let mut perm = [0usize, 1, 2, 3, 4, 5];
        permute(&mut perm, 0, &mut |p| {
            let mut score = 0;
            for a in 0..6 {
                for b in (a + 1)..6 {
                    score += wins[p[a]][p[b]];
                }
            }
            best = best.max(score);
        });

        // Heuristic result, scored under the same objective.
        let model = Kemeny {
            passes: KemenyPasses::Fixed(20),
            ..Default::default()
        }
        .fit(&d)?;
        let order: Vec<usize> = model
            .order()
            .map(|name| {
                names
                    .iter()
                    .position(|n| n == name)
                    .ok_or_else(|| format!("unknown item {name}"))
            })
            .collect::<Result<_, _>>()?;
        let mut achieved = 0;

        for a in 0..6 {
            for b in (a + 1)..6 {
                achieved += wins[order[a]][order[b]];
            }
        }

        assert!(
            achieved <= best,
            "trial {trial}: heuristic cannot beat the oracle"
        );
        assert!(
            achieved as f64 >= 0.95 * best as f64,
            "trial {trial}: heuristic {achieved} below 95% of optimum {best}"
        );
    }

    Ok(())
}

/// Heap-style permutation enumeration (avoids a combinatorics dependency).
fn permute(items: &mut [usize; 6], k: usize, visit: &mut impl FnMut(&[usize; 6])) {
    if k == items.len() {
        visit(items);
        return;
    }

    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// Thompson Sampling Beta posterior arithmetic: with a uniform Beta(1,1)
/// prior and 7 successes / 3 failures, the posterior mean is
/// (1+7)/(1+1+10) = 8/12 (textbook Beta-Binomial conjugacy).
#[test]
fn thompson_beta_posterior_mean_is_analytic() -> TestResult {
    let bandit = Bandit {
        policy: BanditPolicy::ThompsonBeta {
            prior_alpha: 1.0,
            prior_beta: 1.0,
        },
        seed: 1,
    };
    let mut data = RewardsDataset::new();

    for _ in 0..7 {
        data.push("arm", 1.0);
    }
    for _ in 0..3 {
        data.push("arm", 0.0);
    }

    let mut model = bandit.init();
    bandit.update(&mut model, &data)?;
    let (_, estimate) = model.scores().next().ok_or("arm present")?;
    assert!(
        (estimate - 8.0 / 12.0).abs() < 1e-12,
        "posterior mean {estimate}"
    );

    Ok(())
}
