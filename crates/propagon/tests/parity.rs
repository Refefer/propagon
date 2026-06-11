//! Early numeric-parity checks against the captured v1 golden outputs
//! (`crates/propagon-cli/tests/golden/`), at the library level. The CLI-level
//! harness (S6) additionally checks output *formatting*; here we check that
//! the ported math reproduces v1's numbers on the example tournament data.

use std::collections::HashMap;
use std::path::PathBuf;

use propagon::algos::{BradleyTerryMM, Confidence, Glicko2, SectionKind, WinRate};
use propagon::{OnlineRanker, PairwiseDataset, RankModel, Ranker};

fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

/// Loads the example edge file the way the CLI will: whitespace-separated
/// `winner loser [weight]`, blank lines as period boundaries.
fn baseball() -> PairwiseDataset {
    let text = std::fs::read_to_string(repo_path("example/tournament/baseball.2018.edges"))
        .expect("example data present");
    let mut d = PairwiseDataset::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            d.new_period();
            continue;
        }
        let mut it = line.split_whitespace();
        let w = it.next().unwrap();
        let l = it.next().unwrap();
        let x: f32 = it.next().map(|t| t.parse().unwrap()).unwrap_or(1.0);
        d.push(w, l, x);
    }
    d
}

/// Parses golden `id: v1[\t v2 ...]` lines into id -> columns.
fn golden(name: &str) -> HashMap<String, Vec<f64>> {
    let text = std::fs::read_to_string(repo_path(&format!(
        "crates/propagon-cli/tests/golden/{name}"
    )))
    .expect("golden file present");
    let mut out = HashMap::new();
    for line in text.lines() {
        let Some((id, rest)) = line.split_once(": ") else { continue };
        let cols: Vec<f64> = rest
            .split_whitespace()
            .map(|t| t.parse().expect("numeric golden column"))
            .collect();
        if !cols.is_empty() {
            out.insert(id.to_string(), cols);
        }
    }
    out
}

#[test]
fn btm_mm_matches_v1_golden() {
    let model = BradleyTerryMM::default().fit(&baseball()).unwrap();
    let want = golden("btm-mm.out");

    let ranked = &model.sections()[0];
    assert_eq!(ranked.kind, SectionKind::Ranked);
    assert_eq!(ranked.entries.len(), want.len(), "entity count");
    for &(id, score) in &ranked.entries {
        let name = model.name(id).unwrap();
        let expected = want[name][0];
        assert!(
            (score - expected).abs() < 1e-4,
            "{name}: v2 {score} vs v1 {expected}"
        );
    }
}

#[test]
fn glicko2_matches_v1_golden() {
    let algo = Glicko2::default();
    let mut model = algo.init();
    algo.update(&mut model, &baseball()).unwrap();
    let want = golden("glicko2.out");

    let mut checked = 0;
    for (name, p) in model.players() {
        let cols = &want[name]; // mu, rd, lower, upper (4dp in golden)
        let mu = model.mu(p);
        assert!((mu - cols[0]).abs() < 2e-3, "{name} mu: {mu} vs {}", cols[0]);
        assert!((p.rd - cols[1]).abs() < 2e-3, "{name} rd: {} vs {}", p.rd, cols[1]);
        let (lo, hi) = p.bounds();
        assert!((lo - cols[2]).abs() < 5e-3, "{name} lower: {lo} vs {}", cols[2]);
        assert!((hi - cols[3]).abs() < 5e-3, "{name} upper: {hi} vs {}", cols[3]);
        checked += 1;
    }
    assert_eq!(checked, want.len());
}

/// v1 quirk: its CLI advertised `--confidence-interval 0.9` but the handler
/// matched the string `"0.90"`, so `0.9` silently fell through to the point
/// estimate. `rate-090.out` was captured with `0.9` and therefore holds P50
/// values; `rate-095.out` holds genuine Wilson P95 values.
#[test]
fn rate_matches_v1_golden() {
    for (confidence, file, tol) in
        [(Confidence::P50, "rate-090.out", 1e-6), (Confidence::P95, "rate-095.out", 1e-5)]
    {
        let algo = WinRate { confidence };
        let mut model = algo.init();
        algo.update(&mut model, &baseball()).unwrap();
        let want = golden(file);

        let mut checked = 0;
        for (name, score) in model.scores() {
            let expected = want[name][0];
            assert!(
                (score - expected).abs() < tol,
                "{file} {name}: v2 {score} vs v1 {expected}"
            );
            checked += 1;
        }
        assert_eq!(checked, want.len(), "{file}");
    }
}

#[test]
fn btm_lr_matches_v1_golden() {
    let model = propagon::algos::BradleyTerryLR::default().fit(&baseball()).unwrap();
    let want = golden("btm-lr.out");

    // v1 accumulated in f32 with nondeterministic-order parallel reductions;
    // compare values loosely and the induced ranking exactly.
    let mut v2: Vec<(&str, f64)> = model.sorted_scores();
    let mut v1: Vec<(String, f64)> = want.iter().map(|(k, v)| (k.clone(), v[0])).collect();
    v1.sort_by(|a, b| b.1.total_cmp(&a.1));
    v2.sort_by(|a, b| b.1.total_cmp(&a.1));
    assert_eq!(v1.len(), v2.len());
    for ((n1, s1), (n2, s2)) in v1.iter().zip(&v2) {
        assert_eq!(n1, n2, "ranking order diverged");
        assert!((s1 - s2).abs() < 1e-3, "{n1}: v2 {s2} vs v1 {s1}");
    }
}
