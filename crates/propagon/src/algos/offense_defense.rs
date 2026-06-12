//! Offense-defense ratings (`docs/algorithms.md` §5.3; Govan, Langville &
//! Meyer 2009).
//!
//! Sinkhorn-Knopp matrix balancing on the aggregated score matrix: every
//! entity gets an offensive rating `o` and a defensive rating `d` such that
//! the points `j` scores on `i` look like `o_j · d_i`. Alternating updates
//! `o_j = Σ_i a_ij / d_i` (production weighted by opponents' defense) and
//! `d_i = Σ_j a_ij / o_j` (points allowed weighted by opponents' offense),
//! with `a_ij` the points `j` scored against `i`, starting from `d = 1`.
//! The aggregate rating is `s = o / d`. Higher `o` means stronger offense;
//! **lower `d` means stronger defense** (it scales what opponents manage to
//! score against you).
//!
//! Assumes rows mean **`(scorer, opponent, amount)`** — for score data push
//! both directions of every game (winner's points and loser's points), the
//! same convention Keener uses.
//!
//! Gotcha: balancing requires full support. An entity that never scores (or
//! never concedes) drives its rating to zero and the iteration has no fixed
//! point — surfaced as a numeric error naming the entity; the paper's
//! ε-perturbation fix is the caller's job, applied to the data.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Offense-defense parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OffenseDefense {
    /// Maximum Sinkhorn sweeps.
    pub iterations: usize,
    /// Early exit when the max relative change of `o` per sweep drops
    /// below this.
    pub tolerance: f64,
}

impl Default for OffenseDefense {
    fn default() -> Self {
        Self {
            iterations: 500,
            tolerance: 1e-9,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OdLine {
    id: String,
    o: f64,
    d: f64,
    s: f64,
}

/// Fitted offense/defense split (`s = o/d`; higher `s` is better).
#[derive(Debug, Clone)]
pub struct OffenseDefenseModel {
    params: OffenseDefense,
    names: Interner,
    offense: Vec<f64>,
    defense: Vec<f64>,
    scores: Vec<f64>,
}

impl OffenseDefenseModel {
    /// `(name, offensive rating)` per entity; higher = stronger offense.
    pub fn offense(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.offense.iter().copied())
    }

    /// `(name, defensive rating)` per entity; **lower** = stronger defense.
    pub fn defense(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.defense.iter().copied())
    }
}

impl RankModel for OffenseDefenseModel {
    fn algorithm(&self) -> &'static str {
        "offense-defense"
    }

    /// Aggregate ratings `s = o/d`.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.scores.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<OdLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| OdLine {
                id: id.to_string(),
                o: self.offense[i],
                d: self.defense[i],
                s: self.scores[i],
            })
            .collect();
        state::save_model(w, "offense-defense", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (OffenseDefense, Vec<OdLine>) =
            state::load_model(r, "offense-defense")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            offense: lines.iter().map(|l| l.o).collect(),
            defense: lines.iter().map(|l| l.d).collect(),
            scores: lines.iter().map(|l| l.s).collect(),
        })
    }
}

/// First entity whose rating lost support (non-finite or effectively zero),
/// if any — Sinkhorn cannot balance past that.
fn first_unsupported(values: &[f64]) -> Option<usize> {
    values.iter().position(|v| !v.is_finite() || *v < 1e-12)
}

impl Ranker for OffenseDefense {
    type Data = PairwiseDataset;
    type Model = OffenseDefenseModel;

    fn fit_opts(
        &self,
        data: &PairwiseDataset,
        opts: &FitOptions<'_>,
    ) -> Result<OffenseDefenseModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = data.n_entities();

        // a[(i, j)] = total points j scored against i: a row (s, t, x) means
        // "s scored x on t", which lands in a[(t, s)].
        let mut a: HashMap<(u32, u32), f64> = HashMap::new();
        for (s, t, x) in data.rows() {
            *a.entry((t, s)).or_default() += f64::from(x);
        }

        // Fixed summation order: HashMap iteration would make the float
        // accumulation (and thus the output bytes) run-dependent.
        let mut pairs: Vec<((u32, u32), f64)> = a.into_iter().collect();
        pairs.sort_unstable_by_key(|&(key, _)| key);

        let progress = opts.progress;
        progress.start("offense-defense sweeps", Some(self.iterations as u64));

        let mut o = vec![0.0f64; n];
        let mut d = vec![1.0f64; n];
        let mut converged = false;

        for it in 0..self.iterations {
            let mut o_next = vec![0.0f64; n];
            for &((i, j), pts) in &pairs {
                o_next[j as usize] += pts / d[i as usize];
            }

            if let Some(bad) = first_unsupported(&o_next) {
                progress.finish();
                return Err(Error::Numeric(format!(
                    "offense-defense lost support: {:?} never scores, so its \
                     offensive rating collapses to zero; perturb the data or \
                     drop the entity",
                    data.interner().resolve(bad as u32)
                )));
            }

            // First sweep has no previous o to compare against.
            let rel = if it == 0 {
                f64::INFINITY
            } else {
                o.iter()
                    .zip(&o_next)
                    .map(|(prev, next)| (next - prev).abs() / prev)
                    .fold(0.0, f64::max)
            };
            o = o_next;

            d.iter_mut().for_each(|v| *v = 0.0);
            for &((i, j), pts) in &pairs {
                d[i as usize] += pts / o[j as usize];
            }

            if let Some(bad) = first_unsupported(&d) {
                progress.finish();
                return Err(Error::Numeric(format!(
                    "offense-defense lost support: {:?} never concedes, so its \
                     defensive rating collapses to zero; perturb the data or \
                     drop the entity",
                    data.interner().resolve(bad as u32)
                )));
            }

            progress.update(it as u64 + 1);

            if rel < self.tolerance {
                converged = true;
                break;
            }
        }

        progress.finish();

        if !converged {
            log::warn!(
                "offense-defense stopped at the {}-sweep cap before reaching tolerance {:e}",
                self.iterations,
                self.tolerance
            );
        }

        let scores: Vec<f64> = o.iter().zip(&d).map(|(o, d)| o / d).collect();

        Ok(OffenseDefenseModel {
            params: *self,
            names: data.interner().clone(),
            offense: o,
            defense: d,
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows ("a","b",6) and ("b","a",3) give a[(b,a)] = 6 (a scored 6 on b)
    /// and a[(a,b)] = 3 (b scored 3 on a). From d = (1, 1):
    ///
    ///   sweep 1: o_a = a[(b,a)]/d_b = 6,  o_b = a[(a,b)]/d_a = 3,
    ///            d_a = a[(a,b)]/o_b = 3/3 = 1,  d_b = a[(b,a)]/o_a = 6/6 = 1
    ///   sweep 2: recomputes the identical o — relative change 0, converged.
    ///
    /// So the fixed point is o = (6, 3), d = (1, 1), s = (6, 3) and
    /// s_a/s_b = 2 (any 2-entity case fixes d after the first sweep, since
    /// d_a ← a_ab/(a_ab/d_a) = d_a).
    #[test]
    fn two_entity_fixed_point_is_exact() {
        let mut data = PairwiseDataset::new();
        data.push("a", "b", 6.0);
        data.push("b", "a", 3.0);
        let m = OffenseDefense::default().fit(&data).unwrap();

        let o: std::collections::HashMap<_, _> = m.offense().collect();
        let d: std::collections::HashMap<_, _> = m.defense().collect();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((o["a"] - 6.0).abs() < 1e-6, "o_a {}", o["a"]);
        assert!((o["b"] - 3.0).abs() < 1e-6, "o_b {}", o["b"]);
        assert!((d["a"] - 1.0).abs() < 1e-6, "d_a {}", d["a"]);
        assert!((d["b"] - 1.0).abs() < 1e-6, "d_b {}", d["b"]);
        assert!((s["a"] - 6.0).abs() < 1e-6, "s_a {}", s["a"]);
        assert!((s["b"] - 3.0).abs() < 1e-6, "s_b {}", s["b"]);
        assert!((s["a"] / s["b"] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn symmetric_scores_tie() {
        let mut data = PairwiseDataset::new();
        data.push("a", "b", 7.0);
        data.push("b", "a", 7.0);
        let m = OffenseDefense::default().fit(&data).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - s["b"]).abs() < 1e-12);
    }

    /// Scores a lot, concedes little → ranks first on s.
    #[test]
    fn dominant_team_ranks_first() {
        let mut data = PairwiseDataset::new();
        data.push("a", "b", 30.0);
        data.push("b", "a", 5.0);
        data.push("a", "c", 28.0);
        data.push("c", "a", 4.0);
        data.push("b", "c", 15.0);
        data.push("c", "b", 14.0);
        let m = OffenseDefense::default().fit(&data).unwrap();
        assert_eq!(m.sorted_scores()[0].0, "a");
    }

    #[test]
    fn zero_scorer_is_a_numeric_error() {
        let mut data = PairwiseDataset::new();
        data.push("a", "b", 6.0);
        match OffenseDefense::default().fit(&data) {
            Err(Error::Numeric(msg)) => assert!(msg.contains("\"b\""), "{msg}"),
            other => panic!("expected Numeric error, got {other:?}"),
        }
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let data = PairwiseDataset::new();
        assert!(matches!(
            OffenseDefense::default().fit(&data),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut data = PairwiseDataset::new();
        data.push("a", "b", 30.0);
        data.push("b", "a", 5.0);
        data.push("a", "c", 28.0);
        data.push("c", "a", 4.0);
        data.push("b", "c", 15.0);
        data.push("c", "b", 14.0);
        let m = OffenseDefense::default().fit(&data).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = OffenseDefenseModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}
