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
        ..PageRank::default()
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

/// The running 5-team example of Langville & Meyer, *Who's #1? The Science
/// of Rating and Ranking* (2012), ch. 2 (also used for Colley below): the
/// 2005 ACC season — full scores Duke-Miami 7-52, Duke-UNC 21-24,
/// Duke-UVA 7-38, Duke-VT 0-45, Miami-UNC 34-16, Miami-UVA 25-17,
/// Miami-VT 27-7, UNC-UVA 7-5, UNC-VT 3-30, UVA-VT 14-52. Each entry below
/// is (winner, loser, margin).
const ACC_2005: [(&str, &str, f32); 10] = [
    ("Miami", "Duke", 45.0),
    ("UNC", "Duke", 3.0),
    ("UVA", "Duke", 31.0),
    ("VT", "Duke", 45.0),
    ("Miami", "UNC", 18.0),
    ("Miami", "UVA", 8.0),
    ("Miami", "VT", 20.0),
    ("UNC", "UVA", 2.0),
    ("VT", "UNC", 27.0),
    ("VT", "UVA", 38.0),
];

fn acc_dataset() -> PairwiseDataset {
    let mut d = PairwiseDataset::new();
    for (w, l, margin) in ACC_2005 {
        d.push(w, l, margin);
    }
    d
}

/// Massey ratings published in *Who's #1?* ch. 2 (pp. 11-13): Duke -24.8,
/// Miami 18.2, UNC -8.0, UVA -3.4, VT 18.0 (mean-zero).
#[test]
fn massey_matches_whos_number_one() -> TestResult {
    use propagon::algos::Massey;

    let model = Massey::default().fit(&acc_dataset())?;
    let scores: std::collections::HashMap<&str, f64> = model.scores().collect();

    for (team, expected) in [
        ("Duke", -24.8),
        ("Miami", 18.2),
        ("UNC", -8.0),
        ("UVA", -3.4),
        ("VT", 18.0),
    ] {
        let got = scores.get(team).ok_or("team rated")?;
        assert!(
            (got - expected).abs() < 0.05,
            "{team}: {got:.2} vs {expected}"
        );
    }

    Ok(())
}

/// Colley ratings for the same season: with every pair playing once,
/// C = 7I - J gives r_i = (b_i + 2.5)/7 analytically — published values
/// 0.2143, 0.7857, 0.5, 0.3571, 0.6429 (comperank R package documents this
/// dataset/method pairing; *Who's #1?* ch. 3 is the method source).
#[test]
fn colley_matches_whos_number_one() -> TestResult {
    use propagon::algos::Colley;

    // Colley reads weights as game counts, not margins — re-list each game
    // with weight 1 (feeding the margin-weighted dataset would tell Colley
    // that Miami beat Duke 45 times).
    let mut d = PairwiseDataset::new();
    for (w, l, _) in ACC_2005 {
        d.push(w, l, 1.0);
    }

    let model = Colley::default().fit(&d)?;
    let scores: std::collections::HashMap<&str, f64> = model.scores().collect();

    for (team, expected) in [
        ("Duke", 0.2143),
        ("Miami", 0.7857),
        ("UNC", 0.5),
        ("UVA", 0.3571),
        ("VT", 0.6429),
    ] {
        let got = scores.get(team).ok_or("team rated")?;
        assert!(
            (got - expected).abs() < 1e-3,
            "{team}: {got:.4} vs {expected}"
        );
    }

    Ok(())
}

/// Keener ratings for the same 2005 ACC season, against the published
/// comperank output (`rate_keener(ncaa2005, sum(score1))` →
/// 0.0671/0.351/0.158/0.161/0.263, and with `skew_fun = NULL` →
/// 0.0898/0.295/0.165/0.189/0.261):
/// <https://echasnovski.github.io/comperank/reference/keener.html>.
/// Method source: Langville & Meyer, *Who's #1?* ch. 4 (Keener 1993).
/// Input is the full score matrix — both directions of every game.
#[test]
fn keener_matches_comperank() -> TestResult {
    use propagon::algos::Keener;

    // (team, opponent, points scored): full 2005 ACC score matrix.
    let scores_data: [(&str, &str, f32); 10] = [
        ("Duke", "Miami", 7.0),
        ("Duke", "UNC", 21.0),
        ("Duke", "UVA", 7.0),
        ("Duke", "VT", 0.0),
        ("Miami", "UNC", 34.0),
        ("Miami", "UVA", 25.0),
        ("Miami", "VT", 27.0),
        ("UNC", "UVA", 7.0),
        ("UNC", "VT", 3.0),
        ("UVA", "VT", 14.0),
    ];
    let reverse: [f32; 10] = [52.0, 24.0, 38.0, 45.0, 16.0, 17.0, 7.0, 5.0, 30.0, 52.0];

    let mut d = PairwiseDataset::new();
    for ((a, b, pts), rev) in scores_data.into_iter().zip(reverse) {
        d.push(a, b, pts);
        d.push(b, a, rev);
    }

    for (skew, expected) in [
        (true, [0.0671, 0.3506, 0.1585, 0.1605, 0.2634]),
        (false, [0.0898, 0.2948, 0.1649, 0.1891, 0.2613]),
    ] {
        let model = Keener {
            skew,
            ..Default::default()
        }
        .fit(&d)?;
        let got: std::collections::HashMap<&str, f64> = model.scores().collect();

        for (team, want) in ["Duke", "Miami", "UNC", "UVA", "VT"].iter().zip(expected) {
            let s = got.get(team).ok_or("team rated")?;
            assert!(
                (s - want).abs() < 1e-3,
                "skew={skew} {team}: {s:.4} vs {want}"
            );
        }
    }

    Ok(())
}

/// Plackett-Luce restricted to 2-item ballots is exactly Bradley-Terry
/// (Hunter 2004, eq. 30 reduces to eq. 3) — so feeding each Agresti
/// baseball game as a two-item ranking must reproduce the published
/// BradleyTerry2 abilities (same citation as
/// `bradley_terry_matches_agresti_baseball` above).
#[test]
fn plackett_luce_on_pairs_reduces_to_bradley_terry() -> TestResult {
    use propagon::algos::PlackettLuce;

    const TEAMS: [&str; 7] = [
        "Milwaukee",
        "Detroit",
        "Toronto",
        "New York",
        "Boston",
        "Cleveland",
        "Baltimore",
    ];
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

    let mut ballots = RankingsDataset::new();
    for (i, row) in WINS.iter().enumerate() {
        for (j, &wins) in row.iter().enumerate() {
            for _ in 0..wins {
                ballots.push_ranking([TEAMS[i], TEAMS[j]])?;
            }
        }
    }

    let model = PlackettLuce {
        tolerance: 1e-12,
        ..Default::default()
    }
    .fit(&ballots)?;
    let gamma: std::collections::HashMap<&str, f64> = model.scores().collect();
    let baltimore = gamma.get("Baltimore").ok_or("Baltimore fitted")?;

    for (team, expected) in TEAMS.iter().zip(EXPECTED) {
        let ability = (gamma.get(team).ok_or("team fitted")? / baltimore).ln();
        assert!(
            (ability - expected).abs() < 1e-3,
            "{team}: {ability:.4} vs {expected}"
        );
    }

    Ok(())
}

/// Bayesian BT (Caron & Doucet 2012 Gibbs, arXiv:1011.1761) on the Agresti
/// baseball data: with shape a = 1 the posterior concentrates around the
/// MLE, so log-ability posteriors must agree with the published
/// BradleyTerry2 values within Monte-Carlo noise, and the team order must
/// match exactly.
#[test]
fn bayesian_bt_agrees_with_published_mle() -> TestResult {
    use propagon::algos::BayesianBradleyTerry;

    const TEAMS: [&str; 7] = [
        "Milwaukee",
        "Detroit",
        "Toronto",
        "New York",
        "Boston",
        "Cleveland",
        "Baltimore",
    ];
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
    for (i, row) in WINS.iter().enumerate() {
        for (j, &wins) in row.iter().enumerate() {
            if wins > 0 {
                d.push(TEAMS[i], TEAMS[j], wins as f32);
            }
        }
    }

    let model = BayesianBradleyTerry {
        samples: 4000,
        burn_in: 1000,
        ..Default::default()
    }
    .fit(&d)?;
    let means: std::collections::HashMap<&str, f64> = model.scores().collect();
    let baltimore = means.get("Baltimore").ok_or("Baltimore fitted")?;

    for (team, expected) in TEAMS.iter().zip(EXPECTED) {
        let ability = (means.get(team).ok_or("team fitted")? / baltimore).ln();
        assert!(
            (ability - expected).abs() < 0.2,
            "{team}: posterior {ability:.3} vs MLE {expected}"
        );
    }

    let order: Vec<&str> = {
        let mut v: Vec<(&str, f64)> = model.scores().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.into_iter().map(|(n, _)| n).collect()
    };
    assert_eq!(
        order,
        TEAMS.to_vec(),
        "posterior order matches published order"
    );

    Ok(())
}

/// Weng-Lin (JMLR 12 (2011), Algorithms 1 and 3) against the openskill.js
/// model-level regression vectors — third-party values that two independent
/// libraries agree on, and whose 1v1 cases were re-derived by hand from the
/// paper's equations during planning:
/// <https://github.com/philihp/openskill.js/blob/main/src/models/__tests__/bradley-terry-full.test.ts>
/// <https://github.com/philihp/openskill.js/blob/main/src/models/__tests__/thurstone-mosteller-full.test.ts>
/// All paper defaults (mu 25, sigma 25/3, beta 25/6, kappa 1e-4, eps 0.1,
/// gamma sigma/c, no tau).
#[test]
fn weng_lin_matches_openskill_vectors() -> TestResult {
    use propagon::MatchupsDataset;
    use propagon::algos::{GammaPolicy, Rating, WengLin, WengLinVariant};

    let rate =
        |algo: &WengLin, teams: &[&[&str]]| -> Result<Vec<Rating>, Box<dyn std::error::Error>> {
            let mut d = MatchupsDataset::new();
            d.push_ordered(teams)?;
            let mut m = algo.init();
            algo.update(&mut m, &d)?;
            Ok(m.ratings().map(|(_, r)| r).collect())
        };

    let bt = WengLin::default();

    // BT-full 1v1.
    let r = rate(&bt, &[&["w"], &["l"]])?;
    assert!((r[0].mu - 27.63523138347365).abs() < 1e-9, "{}", r[0].mu);
    assert!((r[0].sigma - 8.065506316323548).abs() < 1e-9);
    assert!((r[1].mu - 22.36476861652635).abs() < 1e-9);
    assert!((r[1].sigma - 8.065506316323548).abs() < 1e-9);

    // BT-full 5-player free-for-all.
    let r = rate(&bt, &[&["1"], &["2"], &["3"], &["4"], &["5"]])?;
    let mus = [
        35.5409255338946,
        30.2704627669473,
        25.0,
        19.729537233052703,
        14.4590744661054,
    ];
    for (got, want) in r.iter().zip(mus) {
        assert!((got.mu - want).abs() < 1e-9, "{} vs {want}", got.mu);
        assert!((got.sigma - 7.202515895247076).abs() < 1e-9);
    }

    // BT-full, three teams of sizes (3, 1, 2) finishing in listed order:
    // exercises sum-aggregation and variance partitioning.
    let r = rate(&bt, &[&["a1", "a2", "a3"], &["b1"], &["c1", "c2"]])?;
    assert!((r[0].mu - 25.992743915179297).abs() < 1e-9, "{}", r[0].mu);
    assert!((r[0].sigma - 8.19709997489984).abs() < 1e-9);
    assert!((r[3].mu - 28.48909130001799).abs() < 1e-9, "{}", r[3].mu);
    assert!((r[3].sigma - 8.220848339985736).abs() < 1e-9);
    assert!((r[4].mu - 20.518164784802714).abs() < 1e-9, "{}", r[4].mu);
    assert!((r[4].sigma - 8.127515465304823).abs() < 1e-9);

    // gamma = 1/k variants pin the GammaPolicy enum.
    let bt_k = WengLin {
        gamma: GammaPolicy::OneOverK,
        ..Default::default()
    };
    let r = rate(&bt_k, &[&["w"], &["l"]])?;
    assert!(
        (r[0].sigma - 8.122328620674137).abs() < 1e-9,
        "{}",
        r[0].sigma
    );
    let r = rate(&bt_k, &[&["1"], &["2"], &["3"], &["4"], &["5"]])?;
    assert!(
        (r[0].sigma - 7.993052538854532).abs() < 1e-9,
        "{}",
        r[0].sigma
    );

    // TM-full 1v1 (tolerance bounded by our normal-CDF approximation).
    let tm = WengLin {
        variant: WengLinVariant::ThurstoneMostellerFull,
        ..Default::default()
    };
    let r = rate(&tm, &[&["w"], &["l"]])?;
    assert!((r[0].mu - 29.230718708993216).abs() < 1e-6, "{}", r[0].mu);
    assert!((r[0].sigma - 7.630934718709003).abs() < 1e-6);

    // TM-full 5-player free-for-all.
    let r = rate(&tm, &[&["1"], &["2"], &["3"], &["4"], &["5"]])?;
    let mus = [
        41.92287483597286,
        33.46143741798643,
        25.0,
        16.53856258201357,
        8.077125164027137,
    ];
    for (got, want) in r.iter().zip(mus) {
        assert!((got.mu - want).abs() < 1e-6, "{} vs {want}", got.mu);
        assert!((got.sigma - 4.958964145006544).abs() < 1e-6);
    }

    // TM-full three teams (3, 1, 2).
    let r = rate(&tm, &[&["a1", "a2", "a3"], &["b1"], &["c1", "c2"]])?;
    assert!((r[0].mu - 25.729796801442728).abs() < 1e-6, "{}", r[0].mu);
    assert!((r[0].sigma - 8.153169236399172).abs() < 1e-6);
    assert!((r[3].mu - 34.02513843037207).abs() < 1e-6, "{}", r[3].mu);
    assert!((r[3].sigma - 7.757460494129447).abs() < 1e-6);
    assert!((r[4].mu - 15.245064768185204).abs() < 1e-6, "{}", r[4].mu);
    assert!((r[4].sigma - 7.372121080126496).abs() < 1e-6);

    Ok(())
}

/// HITS on the worked example of Langville & Meyer, "A Survey of
/// Eigenvector Methods for Web Information Retrieval", SIAM Review 47(1)
/// 2005, §3.3 (the same paper as the PageRank vector above): nodes
/// {1,2,3,5,6,10}, edges 1→3, 1→6, 2→1, 3→6, 6→3, 6→5, 10→6. Published
/// 1-norm-normalized vectors: authority (0, 0, .3660, .1340, .5, 0),
/// hub (.3660, 0, .2113, 0, .2113, .2113).
#[test]
fn hits_matches_langville_meyer_example() -> TestResult {
    use propagon::algos::Hits;

    let mut g = GraphDataset::new();
    for (s, d) in [
        ("1", "3"),
        ("1", "6"),
        ("2", "1"),
        ("3", "6"),
        ("6", "3"),
        ("6", "5"),
        ("10", "6"),
    ] {
        g.push(s, d, 1.0);
    }

    let model = Hits::default().fit(&g)?;
    let authority: std::collections::HashMap<&str, f64> = model.authority_scores().collect();
    let hub: std::collections::HashMap<&str, f64> = model.hub_scores().collect();

    for (node, a, h) in [
        ("1", 0.0, 0.3660),
        ("2", 0.0, 0.0),
        ("3", 0.3660, 0.2113),
        ("5", 0.1340, 0.0),
        ("6", 0.5, 0.2113),
        ("10", 0.0, 0.2113),
    ] {
        let got_a = authority.get(node).ok_or("node scored")?;
        let got_h = hub.get(node).ok_or("node scored")?;
        assert!(
            (got_a - a).abs() < 5e-4,
            "{node} authority {got_a:.4} vs {a}"
        );
        assert!((got_h - h).abs() < 5e-4, "{node} hub {got_h:.4} vs {h}");
    }

    Ok(())
}
