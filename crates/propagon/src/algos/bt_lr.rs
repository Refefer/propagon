//! Bradley-Terry via logistic SGD (`docs/algorithms.md` §1.1, estimation
//! variant). Bradley-Terry is exactly logistic regression on outcome
//! indicators, so plain gradient descent fits it — this is the
//! streaming-friendly, warm-startable estimator (and the offline version of
//! what Elo tracks online).

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, RankModel as _, Ranker};

/// Bradley-Terry SGD parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BradleyTerryLR {
    /// Full passes over each period.
    pub passes: usize,
    /// Learning rate.
    pub alpha: f64,
    /// L2 shrinkage applied after each pass.
    pub decay: f64,
    /// Sequential in-place updates (v1's `--thrifty`): lower memory, slightly
    /// different trajectories than the batched default.
    pub thrifty: bool,
}

impl Default for BradleyTerryLR {
    fn default() -> Self {
        Self {
            passes: 10,
            alpha: 1.0,
            decay: 1e-5,
            thrifty: false,
        }
    }
}

/// Fitted Bradley-Terry SGD scores (log-strengths, zero-ish centered).
#[derive(Debug, Clone)]
pub struct BtmLrModel {
    params: BradleyTerryLR,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(BtmLrModel, "btm-lr");

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Ranker for BradleyTerryLR {
    type Data = PairwiseDataset;
    type Model = BtmLrModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<BtmLrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let mut scores = vec![0.0f64; data.n_entities()];
        self.run(data, &mut scores, opts);
        Ok(BtmLrModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }

    fn fit_warm_opts(
        &self,
        data: &PairwiseDataset,
        init: &BtmLrModel,
        opts: &FitOptions<'_>,
    ) -> Result<BtmLrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let mut scores = vec![0.0f64; data.n_entities()];
        for (name, s) in init.scores() {
            if let Some(id) = data.interner().get(name) {
                scores[id as usize] = s;
            }
        }
        self.run(data, &mut scores, opts);
        Ok(BtmLrModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }
}

impl BradleyTerryLR {
    fn run(&self, data: &PairwiseDataset, scores: &mut [f64], opts: &FitOptions<'_>) {
        let progress = opts.progress();
        let total = (self.passes * data.n_periods()) as u64;
        progress.start("sgd passes", Some(total));
        let mut done = 0;

        // v1 processed each blank-line batch as its own training set.
        for period in data.periods() {
            let rows: Vec<(u32, u32, f64)> = data
                .period_rows(period)
                .map(|(w, l, x)| (w, l, f64::from(x)))
                .collect();
            let weight_sum: f64 = rows.iter().map(|r| r.2).sum();
            if weight_sum == 0.0 {
                continue;
            }

            for _ in 0..self.passes {
                if self.thrifty {
                    // Immediate sequential updates.
                    for &(w, l, x) in &rows {
                        let y_hat = sigmoid(scores[w as usize] - scores[l as usize]);
                        let g = self.alpha * x * (y_hat - 1.0) / weight_sum;
                        scores[w as usize] -= g;
                        scores[l as usize] += g;
                    }
                } else {
                    // Batched: gradients computed against the pass-start
                    // scores (parallel), then accumulated deterministically.
                    let frozen: &[f64] = scores;
                    let grads = parallel::par_map_indexed(rows.len(), |i| {
                        let (w, l, x) = rows[i];
                        let y_hat = sigmoid(frozen[w as usize] - frozen[l as usize]);
                        let g = self.alpha * x * (y_hat - 1.0) / weight_sum;
                        (w, l, g)
                    });
                    for (w, l, g) in grads {
                        scores[w as usize] -= g;
                        scores[l as usize] += g;
                    }
                }

                // L2 shrinkage (v1 `norm`).
                let norm = scores.iter().map(|s| s * s).sum::<f64>().sqrt();
                if norm > 0.0 {
                    let decay = self.decay;
                    parallel::par_for_each_mut(scores, |_, s| *s -= decay * *s / norm);
                }
                done += 1;
                progress.update(done);
            }
        }
        progress.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn data() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        for _ in 0..5 {
            d.push("a", "b", 1.0);
            d.push("b", "c", 1.0);
        }
        d.push("c", "a", 1.0); // one upset
        d
    }

    #[test]
    fn recovers_order_and_batch_matches_thrifty_direction() {
        let batch = BradleyTerryLR::default().fit(&data()).unwrap();
        let order: Vec<&str> = batch.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        let thrifty = BradleyTerryLR {
            thrifty: true,
            ..Default::default()
        }
        .fit(&data())
        .unwrap();
        let order: Vec<&str> = thrifty.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn warm_start_continues_from_previous_scores() {
        let algo = BradleyTerryLR::default();
        let first = algo.fit(&data()).unwrap();

        // Warm-started single pass should already be ordered correctly,
        // because it begins from the converged scores.
        let one_pass = BradleyTerryLR {
            passes: 1,
            ..Default::default()
        };
        let warm = one_pass.fit_warm(&data(), &first).unwrap();
        let order: Vec<&str> = warm.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        let warm_top = warm.sorted_scores()[0].1;
        let cold = one_pass.fit(&data()).unwrap();
        let cold_top = cold.sorted_scores()[0].1;
        assert!(
            warm_top > cold_top,
            "warm start retains accumulated strength"
        );
    }

    #[test]
    fn round_trip() {
        let m = BradleyTerryLR::default().fit(&data()).unwrap();
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = BtmLrModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}
