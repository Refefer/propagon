//! Counting behavior cloning (`docs/algorithms.md` §13.6; Pomerleau 1989).
//!
//! The actions a demonstrator takes are revealed preferences over the
//! actions available: count what experts actually do and rank by visitation
//! frequency, with optional Laplace smoothing — the tabular MLE of the
//! behavior policy `π(a | s)`.
//!
//! **Rewards are ignored.** This ranks by *imitation*, not outcome:
//! popular ≠ good. [`McValue`](crate::algos::McValue) on the same dataset
//! ranks by what pays off; disagreement between the two is itself
//! informative.
//!
//! Two granularities: [`Granularity::Global`] treats every step token as an
//! opaque action and ranks them by overall frequency;
//! [`Granularity::PerState`] splits each token as `state<sep>action` on the
//! *first* separator and normalizes within each state. Emitted entities
//! keep the composite token name either way. Only observed tokens are
//! emitted; smoothing shifts mass between them but invents no entities.
//! [`BcModel::implied_pairs`] exports the counts as net preference edges
//! for downstream pairwise rankers (Bradley-Terry, Kemeny, LSR).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::{self, ScoreCountLine};
use crate::dataset::{PairwiseDataset, TrajectoriesDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// How step tokens map to (state, action) pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Granularity {
    /// Every token is an opaque action in one implicit state.
    #[default]
    Global,
    /// Tokens are `state<separator>action`, split on the first separator;
    /// frequencies are normalized within each state.
    PerState { separator: char },
}

/// Behavior-cloning parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorCloning {
    pub granularity: Granularity,
    /// Laplace smoothing α ≥ 0: score = (count + α) / (N + α·K) with K
    /// distinct actions in the normalization group.
    pub smoothing: f64,
}

impl Default for BehaviorCloning {
    fn default() -> Self {
        Self {
            granularity: Granularity::default(),
            smoothing: 0.0,
        }
    }
}

impl BehaviorCloning {
    /// Rejects a negative or non-finite smoothing constant (NaN included —
    /// the comparison is written to fail on it).
    fn validate(&self) -> Result<()> {
        if !(self.smoothing >= 0.0 && self.smoothing.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "smoothing must be finite and non-negative, got {}",
                self.smoothing
            )));
        }
        Ok(())
    }
}

/// Smoothed visitation frequencies per (composite) token.
#[derive(Debug, Clone)]
pub struct BcModel {
    params: BehaviorCloning,
    names: Interner,
    scores: Vec<f64>,
    counts: Vec<u64>,
}

impl BcModel {
    /// The counts as net preference edges: within each state (the single
    /// implicit one under [`Granularity::Global`]), every ordered action
    /// pair with `count(a|s) > count(b|s)` becomes one aggregated
    /// `a ≻ b` row of weight `count(a|s) − count(b|s)` — ready for
    /// Bradley-Terry, Kemeny, or LSR downstream. Rows are emitted in
    /// deterministic order: states by first occurrence, then both action
    /// ids ascending.
    pub fn implied_pairs(&self) -> PairwiseDataset {
        let mut groups: Vec<Vec<u32>> = Vec::new();

        match self.params.granularity {
            Granularity::Global => groups.push((0..self.names.len() as u32).collect()),
            Granularity::PerState { separator } => {
                let mut index: HashMap<&str, usize> = HashMap::new();

                for (id, name) in self.names.names().enumerate() {
                    // A separator-less name is unreachable from fit/load
                    // (both validate); degrade to whole-token grouping
                    // rather than panicking.
                    let state = name.split_once(separator).map_or(name, |(s, _)| s);

                    match index.get(state) {
                        Some(&g) => groups[g].push(id as u32),
                        None => {
                            index.insert(state, groups.len());
                            groups.push(vec![id as u32]);
                        }
                    }
                }
            }
        }

        let mut out = PairwiseDataset::new();
        for group in &groups {
            for &a in group {
                for &b in group {
                    let (ca, cb) = (self.counts[a as usize], self.counts[b as usize]);
                    if ca > cb {
                        out.push(
                            self.names.resolve(a),
                            self.names.resolve(b),
                            (ca - cb) as f32,
                        );
                    }
                }
            }
        }
        out
    }
}

impl RankModel for BcModel {
    fn algorithm(&self) -> &'static str {
        "behavior-cloning"
    }

    /// Smoothed visitation frequency per token (normalized within its
    /// state under [`Granularity::PerState`]).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.scores.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines = common::score_count_lines(&self.names, &self.scores, &self.counts);
        state::save_model(w, "behavior-cloning", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (BehaviorCloning, Vec<ScoreCountLine>) =
            state::load_model(r, "behavior-cloning")?;

        // Re-validate the per-state invariant so implied_pairs can group
        // loaded models without a fallible path.
        if let Granularity::PerState { separator } = params.granularity {
            for line in &lines {
                if !line.id.contains(separator) {
                    return Err(Error::State(format!(
                        "token {:?} has no separator {separator:?} required by per-state mode",
                        line.id
                    )));
                }
            }
        }

        let (names, scores, counts) = common::from_score_count_lines(lines)?;
        Ok(Self {
            params,
            names,
            scores,
            counts,
        })
    }
}

impl Ranker for BehaviorCloning {
    type Data = TrajectoriesDataset;
    type Model = BcModel;

    /// One counting pass over all steps (episode boundaries and rewards are
    /// irrelevant to frequencies), then a smoothing-normalized emission in
    /// dataset-id order.
    fn fit_opts(&self, data: &TrajectoriesDataset, _opts: &FitOptions<'_>) -> Result<BcModel> {
        self.validate()?;
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let mut counts = vec![0u64; data.n_entities()];
        for (s, _) in data.steps() {
            counts[s as usize] += 1;
        }

        // Per-group normalizers (N = steps, K = distinct actions). Global
        // mode uses the single implicit state keyed by "".
        let alpha = self.smoothing;
        let mut norms: HashMap<&str, (u64, u64)> = HashMap::new();

        for (id, &c) in counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let key = match self.granularity {
                Granularity::Global => "",
                Granularity::PerState { separator } => {
                    let name = data.interner().resolve(id as u32);
                    match name.split_once(separator) {
                        Some((state, _action)) => state,
                        None => {
                            return Err(Error::InvalidInput(format!(
                                "token {name:?} has no separator {separator:?}; per-state \
                                 mode expects \"state{separator}action\""
                            )));
                        }
                    }
                }
            };
            let e = norms.entry(key).or_default();
            e.0 += c;
            e.1 += 1;
        }

        let mut names = Interner::new();
        let mut scores = Vec::new();
        let mut emitted = Vec::new();

        for (id, &c) in counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let name = data.interner().resolve(id as u32);
            let key = match self.granularity {
                Granularity::Global => "",
                // The first pass already rejected separator-less tokens.
                Granularity::PerState { separator } => {
                    name.split_once(separator).map_or(name, |(s, _)| s)
                }
            };
            let (n_s, k_s) = norms.get(key).copied().unwrap_or((c, 1));

            names.intern(name);
            scores.push((c as f64 + alpha) / (n_s as f64 + alpha * k_s as f64));
            emitted.push(c);
        }

        Ok(BcModel {
            params: *self,
            names,
            scores,
            counts: emitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::BradleyTerryMM;

    fn steps(tokens: &[&str]) -> TrajectoriesDataset {
        let mut d = TrajectoriesDataset::new();
        for t in tokens {
            d.push_step(t, 0.0).unwrap();
        }
        d.end_episode();
        d
    }

    fn scores(m: &BcModel) -> std::collections::HashMap<String, f64> {
        m.scores().map(|(n, s)| (n.to_string(), s)).collect()
    }

    /// Global counts a=3, b=1 (N = 4, K = 2): unsmoothed 3/4 and 1/4;
    /// α = 2 gives (3+2)/(4+4) = 5/8 and (1+2)/8 = 3/8.
    #[test]
    fn global_frequencies_match_hand_counts() {
        let d = steps(&["a", "a", "b", "a"]);

        let s = scores(&BehaviorCloning::default().fit(&d).unwrap());
        assert!((s["a"] - 0.75).abs() < 1e-15);
        assert!((s["b"] - 0.25).abs() < 1e-15);

        let smoothed = BehaviorCloning {
            smoothing: 2.0,
            ..Default::default()
        };
        let s = scores(&smoothed.fit(&d).unwrap());
        assert!((s["a"] - 5.0 / 8.0).abs() < 1e-15);
        assert!((s["b"] - 3.0 / 8.0).abs() < 1e-15);
    }

    /// Two states, three actions:
    /// s1 sees a×2, b×1 (N = 3, K = 2); s2 sees y×3, x×1 (N = 4, K = 2).
    /// Unsmoothed: s1:a = 2/3, s1:b = 1/3, s2:y = 3/4, s2:x = 1/4.
    /// α = 1: s1:a = 3/5, s1:b = 2/5, s2:y = 4/6, s2:x = 2/6.
    #[test]
    fn per_state_frequencies_match_hand_counts() {
        let d = steps(&["s1:a", "s1:a", "s1:b", "s2:y", "s2:y", "s2:y", "s2:x"]);
        let per_state = |smoothing| BehaviorCloning {
            granularity: Granularity::PerState { separator: ':' },
            smoothing,
        };

        let s = scores(&per_state(0.0).fit(&d).unwrap());
        assert!((s["s1:a"] - 2.0 / 3.0).abs() < 1e-15);
        assert!((s["s1:b"] - 1.0 / 3.0).abs() < 1e-15);
        assert!((s["s2:y"] - 0.75).abs() < 1e-15);
        assert!((s["s2:x"] - 0.25).abs() < 1e-15);

        let s = scores(&per_state(1.0).fit(&d).unwrap());
        assert!((s["s1:a"] - 3.0 / 5.0).abs() < 1e-15);
        assert!((s["s1:b"] - 2.0 / 5.0).abs() < 1e-15);
        assert!((s["s2:y"] - 4.0 / 6.0).abs() < 1e-15);
        assert!((s["s2:x"] - 2.0 / 6.0).abs() < 1e-15);
    }

    /// Counts s1:{a 2, b 1}, s2:{y 3, x 1} imply exactly two net edges:
    /// s1:a ≻ s1:b (weight 1) and s2:y ≻ s2:x (weight 2), in state order.
    #[test]
    fn implied_pairs_exact_rows() {
        let d = steps(&["s1:a", "s1:a", "s1:b", "s2:y", "s2:y", "s2:y", "s2:x"]);
        let bc = BehaviorCloning {
            granularity: Granularity::PerState { separator: ':' },
            smoothing: 0.0,
        };
        let pairs = bc.fit(&d).unwrap().implied_pairs();

        let rows: Vec<(String, String, f32)> = pairs
            .rows()
            .map(|(w, l, x)| {
                (
                    pairs.interner().name(w).unwrap().to_string(),
                    pairs.interner().name(l).unwrap().to_string(),
                    x,
                )
            })
            .collect();
        assert_eq!(
            rows,
            vec![
                ("s1:a".to_string(), "s1:b".to_string(), 1.0),
                ("s2:y".to_string(), "s2:x".to_string(), 2.0),
            ]
        );

        // Global mode: a=3 ≻ b=1 over the single implicit state.
        let g = BehaviorCloning::default()
            .fit(&steps(&["a", "a", "b", "a"]))
            .unwrap()
            .implied_pairs();
        let rows: Vec<_> = g.rows().collect();
        assert_eq!(rows, vec![(0, 1, 2.0)]);
    }

    /// implied_pairs → Bradley-Terry recovers the per-state frequency
    /// order. The implied edges are one-directional, so the BT fit needs
    /// `create_fake_games` to keep Ford's condition (a pure frequency DAG
    /// has undefeated/winless entities).
    #[test]
    fn implied_pairs_feed_bradley_terry() {
        let d = steps(&[
            "s1:a", "s1:a", "s1:a", "s1:a", "s1:a", // a×5
            "s1:b", "s1:b", "s1:b", // b×3
            "s1:c", // c×1
            "s2:y", "s2:y", "s2:y", // y×3
            "s2:x", // x×1
        ]);
        let bc = BehaviorCloning {
            granularity: Granularity::PerState { separator: ':' },
            smoothing: 0.0,
        };
        let pairs = bc.fit(&d).unwrap().implied_pairs();

        let bt = BradleyTerryMM {
            create_fake_games: 1.0,
            ..Default::default()
        };
        let s: std::collections::HashMap<String, f64> = bt
            .fit(&pairs)
            .unwrap()
            .scores()
            .map(|(n, v)| (n.to_string(), v))
            .collect();
        assert!(s["s1:a"] > s["s1:b"] && s["s1:b"] > s["s1:c"]);
        assert!(s["s2:y"] > s["s2:x"]);
    }

    #[test]
    fn missing_separator_names_the_token() {
        let d = steps(&["s1:a", "plain"]);
        let bc = BehaviorCloning {
            granularity: Granularity::PerState { separator: ':' },
            smoothing: 0.0,
        };

        match bc.fit(&d) {
            Err(Error::InvalidInput(msg)) => assert!(msg.contains("plain"), "{msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let d = steps(&["s1:a", "s1:a", "s1:b", "s2:y"]);
        let bc = BehaviorCloning {
            granularity: Granularity::PerState { separator: ':' },
            smoothing: 0.5,
        };
        let m = bc.fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = BcModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);

        // implied_pairs works identically on the loaded model
        assert_eq!(
            m.implied_pairs().rows().collect::<Vec<_>>(),
            loaded.implied_pairs().rows().collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(matches!(
            BehaviorCloning::default().fit(&TrajectoriesDataset::new()),
            Err(Error::EmptyDataset)
        ));

        let d = steps(&["a"]);
        for smoothing in [-1.0, f64::NAN, f64::INFINITY] {
            let bc = BehaviorCloning {
                smoothing,
                ..Default::default()
            };
            assert!(matches!(bc.fit(&d), Err(Error::InvalidInput(_))));
        }
    }
}
