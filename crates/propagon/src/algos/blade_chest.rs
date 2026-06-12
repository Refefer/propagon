//! Blade-Chest matchup model (`docs/algorithms.md` §9.1; Chen & Joachims,
//! WSDM 2016).
//!
//! Each entity gets a scalar strength γ plus two `dims`-dimensional vectors —
//! a *blade* (how it attacks) and a *chest* (where it is vulnerable) — and
//! `P(a ≻ b) = σ(m(a, b))` where the matchup score depends on how `a`'s
//! blade aligns with `b`'s chest:
//!
//! - [`BladeChestVariant::Inner`]:  `m(a,b) = b_a·c_b − b_b·c_a + γ_a − γ_b`
//! - [`BladeChestVariant::Dist`]:   `m(a,b) = ‖b_b − c_a‖² − ‖b_a − c_b‖² + γ_a − γ_b`
//!
//! The vector terms are antisymmetric in `(a, b)`, so the model represents
//! rock-paper-scissors cycles exactly while γ carries the transitive part
//! ([`RankModel::scores`](crate::RankModel::scores) reports γ; the full
//! matchup probability is [`BladeChestModel::matchup`]).
//!
//! Fitting is batch SGD over the log-likelihood with per-epoch Fisher-Yates
//! reshuffles from a seeded RNG. It is sequential by design: each row's
//! update immediately feeds the next row's gradient through the shared
//! vectors, so parallel execution would either race or require frozen
//! batches — a different trajectory — and would break fixed-seed
//! determinism. Periods are ignored (batch fitter); `epochs = 0` legally
//! returns the initialization (γ = 0, seeded random vectors).
//!
//! Gotcha: L2 shrinkage `(1 − lr·l2)` applies to the four touched vectors at
//! every row visit — γ is never regularized — so heavily-compared entities
//! are shrunk more often, exactly like the reference SGD formulation.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Which matchup score the model uses (the two parameterizations of
/// Chen & Joachims 2016).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BladeChestVariant {
    /// Inner-product matchup: `b_a·c_b − b_b·c_a + γ_a − γ_b`.
    #[default]
    Inner,
    /// Squared-distance matchup: `‖b_b − c_a‖² − ‖b_a − c_b‖² + γ_a − γ_b`.
    Dist,
}

/// Blade-Chest parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BladeChest {
    /// Blade/chest vector dimensionality (≥ 1).
    pub dims: usize,
    /// Matchup-score parameterization.
    pub variant: BladeChestVariant,
    /// SGD step size.
    pub lr: f64,
    /// Full passes over the shuffled rows.
    pub epochs: usize,
    /// L2 strength on the vectors (γ is unregularized).
    pub l2: f64,
    /// Std-dev of the `N(0, init_scale)` vector initialization.
    pub init_scale: f64,
    /// Seeds vector initialization and the per-epoch shuffles.
    pub seed: u64,
}

impl Default for BladeChest {
    fn default() -> Self {
        Self {
            dims: 8,
            variant: BladeChestVariant::default(),
            lr: 0.05,
            epochs: 50,
            l2: 1e-4,
            init_scale: 0.1,
            seed: 2016,
        }
    }
}

/// One entity line: transitive strength `s` (= γ), blade `b`, chest `c`.
#[derive(Debug, Serialize, Deserialize)]
struct EntityLine {
    id: String,
    s: f64,
    b: Vec<f64>,
    c: Vec<f64>,
}

/// Fitted Blade-Chest model.
#[derive(Debug, Clone)]
pub struct BladeChestModel {
    params: BladeChest,
    names: Interner,
    gamma: Vec<f64>,
    /// Flat `n × dims` blade vectors, entity-major.
    blade: Vec<f64>,
    /// Flat `n × dims` chest vectors, entity-major.
    chest: Vec<f64>,
}

impl BladeChestModel {
    /// Full matchup probability `P(a ≻ b) = σ(m(a, b))`, vector terms
    /// included. `None` when either entity is unseen.
    pub fn matchup(&self, a: &str, b: &str) -> Option<f64> {
        let ai = self.names.get(a)? as usize;
        let bi = self.names.get(b)? as usize;
        let m = matchup_score(
            self.params.variant,
            self.vec_at(&self.blade, ai),
            self.vec_at(&self.chest, ai),
            self.vec_at(&self.blade, bi),
            self.vec_at(&self.chest, bi),
            self.gamma[ai],
            self.gamma[bi],
        );
        Some(sigmoid(m))
    }

    fn vec_at<'a>(&self, flat: &'a [f64], i: usize) -> &'a [f64] {
        &flat[i * self.params.dims..(i + 1) * self.params.dims]
    }
}

impl RankModel for BladeChestModel {
    fn algorithm(&self) -> &'static str {
        "blade-chest"
    }

    /// The transitive strengths γ only; matchup effects live in
    /// [`BladeChestModel::matchup`].
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.gamma.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<EntityLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| EntityLine {
                id: id.to_string(),
                s: self.gamma[i],
                b: self.vec_at(&self.blade, i).to_vec(),
                c: self.vec_at(&self.chest, i).to_vec(),
            })
            .collect();
        state::save_model(w, "blade-chest", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (BladeChest, Vec<EntityLine>) = state::load_model(r, "blade-chest")?;

        let mut blade = Vec::with_capacity(lines.len() * params.dims);
        let mut chest = Vec::with_capacity(lines.len() * params.dims);
        for line in &lines {
            if line.b.len() != params.dims || line.c.len() != params.dims {
                return Err(Error::State(format!(
                    "entity {:?} has {}-dim blade / {}-dim chest; dims = {}",
                    line.id,
                    line.b.len(),
                    line.c.len(),
                    params.dims
                )));
            }
            blade.extend_from_slice(&line.b);
            chest.extend_from_slice(&line.c);
        }

        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            gamma: lines.iter().map(|l| l.s).collect(),
            blade,
            chest,
        })
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The matchup score `m(a, b)` for either variant, from `a`'s perspective.
fn matchup_score(
    variant: BladeChestVariant,
    ba: &[f64],
    ca: &[f64],
    bb: &[f64],
    cb: &[f64],
    ga: f64,
    gb: f64,
) -> f64 {
    let vector_term = match variant {
        BladeChestVariant::Inner => dot(ba, cb) - dot(bb, ca),
        BladeChestVariant::Dist => sq_dist(bb, ca) - sq_dist(ba, cb),
    };
    vector_term + ga - gb
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

impl Ranker for BladeChest {
    type Data = PairwiseDataset;
    type Model = BladeChestModel;

    /// Maximizes the log-likelihood by sequential SGD. Per observed row
    /// `(winner w, loser l, weight x)`, with `δ = x·(1 − σ(m(w, l)))` and
    /// frozen pre-row copies of the four touched vectors:
    ///
    /// ```text
    /// γ_w += lr·δ        γ_l −= lr·δ
    /// Inner: b_w += lr·δ·c_l    c_l += lr·δ·b_w
    ///        b_l −= lr·δ·c_w    c_w −= lr·δ·b_l
    /// Dist:  b_l += lr·δ·2(b_l − c_w)    c_w −= lr·δ·2(b_l − c_w)
    ///        b_w −= lr·δ·2(b_w − c_l)    c_l += lr·δ·2(b_w − c_l)
    /// ```
    ///
    /// then the four vectors (never γ) shrink by `(1 − lr·l2)`.
    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<BladeChestModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if self.dims == 0 {
            return Err(Error::InvalidInput(
                "blade-chest needs dims >= 1 (a 0-dim model is just logistic γ — \
                 use btm-lr instead)"
                    .into(),
            ));
        }

        let init = Normal::new(0.0, self.init_scale).map_err(|e| {
            Error::InvalidInput(format!(
                "init_scale {} is not a valid normal std-dev: {e}",
                self.init_scale
            ))
        })?;

        let n = data.n_entities();
        let dims = self.dims;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut gamma = vec![0.0f64; n];
        let mut blade = Vec::with_capacity(n * dims);
        let mut chest = Vec::with_capacity(n * dims);

        // Entities in id order, blade before chest, one shared RNG: the
        // whole initialization is a fixed function of (seed, n, dims).
        for _ in 0..n {
            for _ in 0..dims {
                blade.push(init.sample(&mut rng));
            }
            for _ in 0..dims {
                chest.push(init.sample(&mut rng));
            }
        }

        let rows: Vec<(usize, usize, f64)> = data
            .rows()
            .map(|(w, l, x)| (w as usize, l as usize, f64::from(x)))
            .collect();
        let mut order: Vec<usize> = (0..rows.len()).collect();

        // Frozen pre-row copies of the four touched vectors.
        let mut bw = vec![0.0; dims];
        let mut cw = vec![0.0; dims];
        let mut bl = vec![0.0; dims];
        let mut cl = vec![0.0; dims];

        let progress = opts.progress;
        progress.start("blade-chest epochs", Some(self.epochs as u64));

        for epoch in 0..self.epochs {
            order.shuffle(&mut rng);

            for &row in &order {
                let (w, l, x) = rows[row];
                bw.copy_from_slice(&blade[w * dims..(w + 1) * dims]);
                cw.copy_from_slice(&chest[w * dims..(w + 1) * dims]);
                bl.copy_from_slice(&blade[l * dims..(l + 1) * dims]);
                cl.copy_from_slice(&chest[l * dims..(l + 1) * dims]);

                let m = matchup_score(self.variant, &bw, &cw, &bl, &cl, gamma[w], gamma[l]);
                let delta = x * (1.0 - sigmoid(m));
                let step = self.lr * delta;
                gamma[w] += step;
                gamma[l] -= step;

                match self.variant {
                    BladeChestVariant::Inner => {
                        for d in 0..dims {
                            blade[w * dims + d] += step * cl[d];
                            chest[l * dims + d] += step * bw[d];
                            blade[l * dims + d] -= step * cw[d];
                            chest[w * dims + d] -= step * bl[d];
                        }
                    }
                    BladeChestVariant::Dist => {
                        for d in 0..dims {
                            let dl = bl[d] - cw[d];
                            let dw = bw[d] - cl[d];
                            blade[l * dims + d] += step * 2.0 * dl;
                            chest[w * dims + d] -= step * 2.0 * dl;
                            blade[w * dims + d] -= step * 2.0 * dw;
                            chest[l * dims + d] += step * 2.0 * dw;
                        }
                    }
                }

                let shrink = 1.0 - self.lr * self.l2;
                for d in 0..dims {
                    blade[w * dims + d] *= shrink;
                    chest[w * dims + d] *= shrink;
                    blade[l * dims + d] *= shrink;
                    chest[l * dims + d] *= shrink;
                }
            }
            progress.update(epoch as u64 + 1);
        }
        progress.finish();

        Ok(BladeChestModel {
            params: *self,
            names: data.interner().clone(),
            gamma,
            blade,
            chest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::BradleyTerryMM;

    fn rps(rounds: usize) -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        for _ in 0..rounds {
            d.push("a", "b", 1.0);
            d.push("b", "c", 1.0);
            d.push("c", "a", 1.0);
        }
        d
    }

    fn assert_recovers_cycle(variant: BladeChestVariant) {
        let algo = BladeChest {
            dims: 2,
            variant,
            lr: 0.1,
            epochs: 500,
            l2: 1e-4,
            init_scale: 0.5,
            seed: 2016,
        };
        let m = algo.fit(&rps(5)).unwrap();

        let p_ab = m.matchup("a", "b").unwrap();
        let p_bc = m.matchup("b", "c").unwrap();
        let p_ac = m.matchup("a", "c").unwrap();
        assert!(p_ab > 0.7, "{variant:?}: P(a≻b) = {p_ab}");
        assert!(p_bc > 0.7, "{variant:?}: P(b≻c) = {p_bc}");
        assert!(p_ac < 0.3, "{variant:?}: P(a≻c) = {p_ac}");

        // γ stays near-flat: the cycle lives in the vectors.
        let gs: Vec<f64> = m.scores().map(|(_, g)| g).collect();
        let spread = gs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - gs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let cyclic_effect = (p_ab / (1.0 - p_ab)).ln();
        assert!(
            spread < 0.25 * cyclic_effect,
            "{variant:?}: γ spread {spread} vs cyclic logit {cyclic_effect}"
        );

        // matchup is symmetric: P(b≻a) = 1 − P(a≻b).
        let p_ba = m.matchup("b", "a").unwrap();
        assert!((p_ab + p_ba - 1.0).abs() < 1e-12);
        assert_eq!(m.matchup("a", "zzz"), None);
    }

    #[test]
    fn inner_variant_recovers_rps() {
        assert_recovers_cycle(BladeChestVariant::Inner);
    }

    #[test]
    fn dist_variant_recovers_rps() {
        assert_recovers_cycle(BladeChestVariant::Dist);
    }

    #[test]
    fn transitive_gamma_matches_bradley_terry_order() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 8.0);
        d.push("b", "a", 2.0);
        d.push("b", "c", 8.0);
        d.push("c", "b", 2.0);
        d.push("a", "c", 9.0);
        d.push("c", "a", 1.0);

        let bc = BladeChest {
            epochs: 200,
            ..BladeChest::default()
        }
        .fit(&d)
        .unwrap();
        let bt = BradleyTerryMM::default().fit(&d).unwrap();

        let bc_order: Vec<&str> = bc.sorted_scores().iter().map(|e| e.0).collect();
        let bt_order: Vec<&str> = bt.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(bc_order, bt_order);
    }

    #[test]
    fn deterministic_bitwise_at_fixed_seed() {
        for variant in [BladeChestVariant::Inner, BladeChestVariant::Dist] {
            let algo = BladeChest {
                variant,
                epochs: 20,
                ..BladeChest::default()
            };
            let m1 = algo.fit(&rps(3)).unwrap();
            let m2 = algo.fit(&rps(3)).unwrap();

            let mut b1 = Vec::new();
            m1.save_jsonl(&mut b1).unwrap();
            let mut b2 = Vec::new();
            m2.save_jsonl(&mut b2).unwrap();
            assert_eq!(b1, b2, "{variant:?} runs diverged at a fixed seed");
        }
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let algo = BladeChest {
            variant: BladeChestVariant::Dist,
            epochs: 5,
            ..BladeChest::default()
        };
        let m = algo.fit(&rps(2)).unwrap();

        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = BladeChestModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
        assert_eq!(m.matchup("a", "b"), m2.matchup("a", "b"));
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let zero_dims = BladeChest {
            dims: 0,
            ..BladeChest::default()
        };
        assert!(matches!(
            zero_dims.fit(&rps(1)),
            Err(Error::InvalidInput(_))
        ));

        assert!(matches!(
            BladeChest::default().fit(&PairwiseDataset::new()),
            Err(Error::EmptyDataset)
        ));

        // epochs = 0 legally returns the initialization: γ identically 0.
        let init_only = BladeChest {
            epochs: 0,
            ..BladeChest::default()
        };
        let m = init_only.fit(&rps(1)).unwrap();
        assert!(m.scores().all(|(_, g)| g == 0.0));
    }
}
