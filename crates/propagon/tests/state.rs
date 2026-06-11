//! Cross-cutting state guarantees (PRD FR-4/FR-5):
//! - every model type save → load → save is byte-identical;
//! - warm starts beat cold starts on appended data (BT-MM acceptance).

use propagon::algos::{Borda, BradleyTerryMM, Copeland, Lsr, PageRank, RankCentrality};
use propagon::{GraphDataset, PairwiseDataset, Progress, RankModel, Ranker};

fn pairwise() -> PairwiseDataset {
    let mut d = PairwiseDataset::new();
    for _ in 0..3 {
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        d.push("a", "c", 1.0);
    }
    d.push("b", "a", 1.0);
    d.push("c", "b", 1.0);
    d.push("c", "a", 1.0);
    d
}

fn graph() -> GraphDataset {
    let mut g = GraphDataset::new();
    g.push("a", "b", 1.0);
    g.push("b", "c", 1.0);
    g.push("c", "a", 1.0);
    g.push("a", "c", 1.0);
    g
}

fn assert_byte_identical<M: RankModel>(m: &M) {
    let mut first = Vec::new();
    m.save_jsonl(&mut first).unwrap();
    let loaded = M::load_jsonl(first.as_slice()).unwrap();
    let mut second = Vec::new();
    loaded.save_jsonl(&mut second).unwrap();
    assert_eq!(
        String::from_utf8(first).unwrap(),
        String::from_utf8(second).unwrap(),
        "{} round trip not byte-identical",
        m.algorithm()
    );
}

#[test]
fn remaining_models_round_trip_byte_identical() {
    // Models without their own in-module round-trip tests.
    assert_byte_identical(&Borda::default().fit(&pairwise()).unwrap());
    assert_byte_identical(&Copeland::default().fit(&pairwise()).unwrap());
    assert_byte_identical(&Lsr::default().fit(&pairwise()).unwrap());
    assert_byte_identical(&RankCentrality::default().fit(&pairwise()).unwrap());
    assert_byte_identical(&PageRank::default().fit(&graph()).unwrap());
}

/// Records the highest sweep index reported through `Progress`.
#[derive(Default)]
struct SweepCounter(std::sync::atomic::AtomicU64);

impl Progress for SweepCounter {
    fn update(&self, done: u64) {
        self.0.fetch_max(done, std::sync::atomic::Ordering::Relaxed);
    }
}

/// PRD FR-5 acceptance: warm BT-MM on a small data increment converges in
/// strictly fewer sweeps than a cold start. MM converges linearly, so the
/// saving is the head of the error curve (log-scale): meaningful, not 10×.
#[test]
fn warm_start_converges_much_faster() {
    // A larger random-ish tournament so convergence takes real work.
    let mut base = PairwiseDataset::new();
    let names: Vec<String> = (0..40).map(|i| format!("t{i}")).collect();
    let mut state = 88172645463325252u64;
    let mut rand = || {
        // xorshift64 — deterministic fixture data
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for round in 0..6 {
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                if (i + j + round) % 3 == 0 {
                    // stronger (lower index) wins more often
                    if rand() % 100 < 70 {
                        base.push(&names[i], &names[j], 1.0);
                    } else {
                        base.push(&names[j], &names[i], 1.0);
                    }
                }
            }
        }
    }

    let algo = BradleyTerryMM {
        tolerance: 1e-6,
        ..Default::default()
    };
    let cold_model = algo.fit(&base).unwrap();

    // Append a small increment (the weekly-update scenario): a handful of
    // new results on top of ~1500 existing rows.
    let mut extended = base.clone();
    for i in 0..4 {
        let j = i + 7;
        extended.push(&names[i], &names[j], 1.0);
    }

    let count_sweeps = |run: &dyn Fn(&SweepCounter)| -> u64 {
        let counter = SweepCounter::default();
        run(&counter);
        counter.0.load(std::sync::atomic::Ordering::Relaxed)
    };

    let cold_sweeps = count_sweeps(&|c| {
        let opts = propagon::FitOptions {
            progress: Some(c),
            ..Default::default()
        };
        algo.fit_opts(&extended, &opts).unwrap();
    });
    let warm_sweeps = count_sweeps(&|c| {
        let opts = propagon::FitOptions {
            progress: Some(c),
            ..Default::default()
        };
        algo.fit_warm_opts(&extended, &cold_model, &opts).unwrap();
    });

    assert!(
        warm_sweeps < cold_sweeps && warm_sweeps * 4 <= cold_sweeps * 3,
        "warm start took {warm_sweeps} sweeps vs cold {cold_sweeps} (target ≤ 75% and strictly fewer)"
    );
}
