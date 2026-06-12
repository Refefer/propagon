//! Multidimensional Elo, mElo₂ₖ (`docs/algorithms.md` §9.2; Balduzzi, Tuyls,
//! Pérolat & Graepel, NeurIPS 2018).
//!
//! Elo plus a learned low-rank skew-symmetric correction: each entity carries
//! a scalar rating `r` and a `2k`-dimensional cyclic vector `c`, and
//! `P(a ≻ b) = σ(r_a − r_b + c_aᵀ Ω c_b)` where `Ω` is the block-diagonal
//! matrix of `k` copies of the 2×2 rotation `[[0, 1], [−1, 0]]`. The `cᵀΩc`
//! term captures rock-paper-scissors structure that no scalar rating can
//! represent; [`RankModel::scores`](crate::RankModel::scores) reports only
//! the transitive `r`, while [`MEloModel::predict`] exposes the full matchup
//! probability — the cyclic part is the method's point.
//!
//! Online SGD with Elo's exact conventions: order-dependent, 1v1 games only
//! (multi-player sides are [`Error::InvalidInput`]), ties score `S = ½`,
//! margins ignored, repeat weights scale the step. Ratings live on the
//! natural-log scale: `k = 0` is legal and *is* Elo with `K = lr_rating` on
//! that scale (a `1.0` rating gap means `e : 1` odds, not `10^(1/400) : 1`).
//!
//! Gotchas: a new entity's cyclic vector is drawn `N(0, init_scale)` from an
//! RNG stream derived from `seed` and a persisted draw counter, so
//! save → load → update is byte-identical to an uninterrupted run. Self-play
//! games (reachable only via [`GamesDataset::from_pairwise`], which skips
//! duplicate-player validation) leave `r` untouched but perturb the entity's
//! own cyclic vector; they are processed as the update equations dictate, not
//! specially cased.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::{GameOutcome, GamesDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// mElo₂ₖ parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MElo {
    /// Number of 2-dimensional cyclic blocks (the model is rank `2k`).
    /// `k = 0` degenerates to natural-log-scale Elo.
    pub k: usize,
    /// Step size for the transitive rating (natural-log scale).
    pub lr_rating: f64,
    /// Step size for the cyclic vectors.
    pub lr_vector: f64,
    /// Rating assigned to unseen entities.
    pub initial_rating: f64,
    /// Std-dev of the `N(0, init_scale)` cyclic-vector initialization.
    pub init_scale: f64,
    /// Seeds the per-entity initialization streams.
    pub seed: u64,
}

impl Default for MElo {
    fn default() -> Self {
        Self {
            k: 1,
            lr_rating: 16.0,
            lr_vector: 1.0,
            initial_rating: 0.0,
            init_scale: 0.1,
            seed: 2018,
        }
    }
}

/// What `save_jsonl` writes as the header `params`: the algorithm params plus
/// the draw counter (state, persisted so resumed runs initialize new
/// entities exactly as an uninterrupted run would).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PersistedParams {
    k: usize,
    lr_rating: f64,
    lr_vector: f64,
    initial_rating: f64,
    init_scale: f64,
    seed: u64,
    draws: u64,
}

/// One entity line in the state file: transitive rating `s` plus the
/// `2k`-dimensional cyclic vector.
#[derive(Debug, Serialize, Deserialize)]
struct EntityLine {
    id: String,
    s: f64,
    c: Vec<f64>,
}

/// mElo state: per-entity rating and cyclic vector, plus the draw counter.
#[derive(Debug, Clone)]
pub struct MEloModel {
    params: MElo,
    names: Interner,
    r: Vec<f64>,
    /// Flat `n × 2k` cyclic vectors, entity-major.
    c: Vec<f64>,
    draws: u64,
}

impl MEloModel {
    /// Full matchup probability `P(a ≻ b) = σ(r_a − r_b + c_aᵀ Ω c_b)`,
    /// cyclic term included. `None` when either entity is unseen.
    pub fn predict(&self, a: &str, b: &str) -> Option<f64> {
        let ai = self.names.get(a)? as usize;
        let bi = self.names.get(b)? as usize;
        let m = self.r[ai] - self.r[bi] + omega_form(self.cvec(ai), self.cvec(bi));
        Some(sigmoid(m))
    }

    /// Entity `i`'s cyclic vector as a slice of the flat storage.
    fn cvec(&self, i: usize) -> &[f64] {
        let k2 = 2 * self.params.k;
        &self.c[i * k2..(i + 1) * k2]
    }

    /// Dense index for `name`, initializing a new entity's rating and cyclic
    /// vector on first sight. Each initialization burns one persisted draw:
    /// the vector comes from a fresh RNG seeded by `seed ^ (draws · φ64)`,
    /// the bandits' resumable-stream pattern, so a model saved and reloaded
    /// mid-stream initializes later entities identically.
    fn idx(&mut self, name: &str, init: Normal<f64>) -> usize {
        let idx = self.names.intern(name) as usize;

        if idx == self.r.len() {
            self.r.push(self.params.initial_rating);
            let stream = self.params.seed ^ self.draws.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            self.draws += 1;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(stream);
            for _ in 0..2 * self.params.k {
                self.c.push(init.sample(&mut rng));
            }
        }
        idx
    }
}

impl RankModel for MEloModel {
    fn algorithm(&self) -> &'static str {
        "melo"
    }

    /// The transitive component `r` only; cyclic effects live in
    /// [`MEloModel::predict`].
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.r.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            k: self.params.k,
            lr_rating: self.params.lr_rating,
            lr_vector: self.params.lr_vector,
            initial_rating: self.params.initial_rating,
            init_scale: self.params.init_scale,
            seed: self.params.seed,
            draws: self.draws,
        };

        let lines: Vec<EntityLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| EntityLine {
                id: id.to_string(),
                s: self.r[i],
                c: self.cvec(i).to_vec(),
            })
            .collect();
        state::save_model(w, "melo", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<EntityLine>) = state::load_model(r, "melo")?;

        let mut c = Vec::with_capacity(lines.len() * 2 * params.k);
        for line in &lines {
            if line.c.len() != 2 * params.k {
                return Err(Error::State(format!(
                    "entity {:?} has a {}-dim cyclic vector; k = {} needs {}",
                    line.id,
                    line.c.len(),
                    params.k,
                    2 * params.k
                )));
            }
            c.extend_from_slice(&line.c);
        }

        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params: MElo {
                k: params.k,
                lr_rating: params.lr_rating,
                lr_vector: params.lr_vector,
                initial_rating: params.initial_rating,
                init_scale: params.init_scale,
                seed: params.seed,
            },
            names,
            r: lines.iter().map(|l| l.s).collect(),
            c,
            draws: params.draws,
        })
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The bilinear form `c_aᵀ Ω c_b = Σ_j (c_a[2j]·c_b[2j+1] − c_a[2j+1]·c_b[2j])`
/// — the sum of 2-D cross products over the k blocks. Antisymmetric in its
/// arguments, so `omega_form(c, c) = 0`.
fn omega_form(ca: &[f64], cb: &[f64]) -> f64 {
    ca.chunks_exact(2)
        .zip(cb.chunks_exact(2))
        .map(|(a, b)| a[0] * b[1] - a[1] * b[0])
        .sum()
}

impl OnlineRanker for MElo {
    type Data = GamesDataset;
    type Model = MEloModel;

    fn init(&self) -> MEloModel {
        MEloModel {
            params: *self,
            names: Interner::new(),
            r: Vec::new(),
            c: Vec::new(),
            draws: 0,
        }
    }

    /// Folds each game in insertion order. With `Ω` as in the module docs and
    /// frozen pre-update copies `c_a`, `c_b`:
    ///
    /// ```text
    /// p̂  = σ(r_a − r_b + c_aᵀ Ω c_b)        δ = weight · (s₁ − p̂)
    /// r_a += lr_rating · δ                   r_b −= lr_rating · δ
    /// c_a += lr_vector · δ · (Ω c_b)         c_b −= lr_vector · δ · (Ωᵀ c_a)
    /// ```
    ///
    /// where per 2-block `Ω(x, y) = (y, −x)` and `Ωᵀ(x, y) = (−y, x)`.
    fn update_opts(
        &self,
        model: &mut MEloModel,
        data: &GamesDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        let init = Normal::new(0.0, self.init_scale).map_err(|e| {
            Error::InvalidInput(format!(
                "init_scale {} is not a valid normal std-dev: {e}",
                self.init_scale
            ))
        })?;

        let k2 = 2 * self.k;
        let mut ca = vec![0.0; k2];
        let mut cb = vec![0.0; k2];

        for (g, view) in data.games().enumerate() {
            let (&[a], &[b]) = (view.side1, view.side2) else {
                return Err(Error::InvalidInput(format!(
                    "game {g} has a multi-player side; melo rates 1v1 games — \
                     use weng-lin on a matchups dataset for teams"
                )));
            };
            // Score of side 1: win 1, tie ½, loss 0; margins ignored.
            let s1 = match view.outcome {
                GameOutcome::Side1Win(_) => 1.0,
                GameOutcome::Tie => 0.5,
                GameOutcome::Side2Win(_) => 0.0,
            };

            let an = data.interner().resolve(a);
            let bn = data.interner().resolve(b);
            let ai = model.idx(an, init);
            let bi = model.idx(bn, init);

            ca.copy_from_slice(model.cvec(ai));
            cb.copy_from_slice(model.cvec(bi));
            let p = sigmoid(model.r[ai] - model.r[bi] + omega_form(&ca, &cb));
            let delta = f64::from(view.weight) * (s1 - p);

            model.r[ai] += self.lr_rating * delta;
            model.r[bi] -= self.lr_rating * delta;

            for j in 0..self.k {
                let (xa, ya) = (ca[2 * j], ca[2 * j + 1]);
                let (xb, yb) = (cb[2 * j], cb[2 * j + 1]);
                model.c[ai * k2 + 2 * j] += self.lr_vector * delta * yb;
                model.c[ai * k2 + 2 * j + 1] += self.lr_vector * delta * (-xb);
                model.c[bi * k2 + 2 * j] -= self.lr_vector * delta * (-ya);
                model.c[bi * k2 + 2 * j + 1] -= self.lr_vector * delta * xa;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::Elo;

    /// One aggregated stochastic rock-paper-scissors round: each edge of the
    /// cycle won `w` times forward and once in reverse (pushed as weights).
    fn rps(w: f32) -> GamesDataset {
        let mut d = GamesDataset::new();
        for (x, y) in [("a", "b"), ("b", "c"), ("c", "a")] {
            d.push_pair(x, y, w).unwrap();
            d.push_pair(y, x, 1.0).unwrap();
        }
        d
    }

    #[test]
    fn recovers_the_rps_cycle_with_flat_ratings() {
        let algo = MElo {
            k: 1,
            lr_rating: 0.05,
            lr_vector: 0.1,
            initial_rating: 0.0,
            init_scale: 0.5,
            seed: 2018,
        };
        let mut m = algo.init();
        let d = rps(4.0);

        for _ in 0..200 {
            algo.update(&mut m, &d).unwrap();
        }

        let p_ab = m.predict("a", "b").unwrap();
        let p_bc = m.predict("b", "c").unwrap();
        let p_ac = m.predict("a", "c").unwrap();
        assert!(p_ab > 0.7, "P(a≻b) = {p_ab}");
        assert!(p_bc > 0.7, "P(b≻c) = {p_bc}");
        assert!(p_ac < 0.3, "P(a≻c) = {p_ac}");

        // The transitive component stays near-flat: the cycle is carried by
        // the vectors, not by r. Repeated epochs over a deterministic cycle
        // drive the c-norms into the saturation regime (σ pins at 0/1), so
        // the cyclic logit is measured on the bilinear form directly rather
        // than back-derived from a saturated probability.
        let rs: Vec<f64> = m.scores().map(|(_, r)| r).collect();
        let spread = rs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - rs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let f_ab = omega_form(m.cvec(0), m.cvec(1));
        let f_bc = omega_form(m.cvec(1), m.cvec(2));
        let f_ca = omega_form(m.cvec(2), m.cvec(0));
        assert!(
            f_ab > 0.0 && f_bc > 0.0 && f_ca > 0.0,
            "cyclic forms must follow the cycle: {f_ab} {f_bc} {f_ca}"
        );
        let cyclic_effect = f_ab.min(f_bc).min(f_ca);
        assert!(
            spread < 0.25 * cyclic_effect,
            "r spread {spread} vs cyclic logit {cyclic_effect}"
        );

        // predict is symmetric: P(b≻a) = 1 − P(a≻b).
        let p_ba = m.predict("b", "a").unwrap();
        assert!((p_ab + p_ba - 1.0).abs() < 1e-12);
        assert_eq!(m.predict("a", "zzz"), None);
    }

    #[test]
    fn transitive_data_matches_elo_order() {
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 8.0).unwrap();
        d.push_pair("b", "a", 2.0).unwrap();
        d.push_pair("b", "c", 8.0).unwrap();
        d.push_pair("c", "b", 2.0).unwrap();
        d.push_pair("a", "c", 9.0).unwrap();
        d.push_pair("c", "a", 1.0).unwrap();

        let melo = MElo {
            k: 1,
            lr_rating: 0.1,
            lr_vector: 0.05,
            ..MElo::default()
        };
        let mut mm = melo.init();
        melo.update(&mut mm, &d).unwrap();

        let elo = Elo::default();
        let mut em = elo.init();
        elo.update(&mut em, &d).unwrap();

        let melo_order: Vec<&str> = mm.sorted_scores().iter().map(|e| e.0).collect();
        let elo_order: Vec<&str> = em.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(melo_order, elo_order);
        assert_eq!(melo_order, vec!["a", "b", "c"]);
    }

    #[test]
    fn k_zero_is_natural_log_elo() {
        let melo = MElo {
            k: 0,
            lr_rating: 4.0,
            ..MElo::default()
        };
        // Elo computing σ(r_a − r_b): scale = ln 10 cancels the 10^x base.
        let elo = Elo {
            k: 4.0,
            initial_rating: 0.0,
            scale: std::f64::consts::LN_10,
        };

        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 2.0).unwrap();

        let mut mm = melo.init();
        melo.update(&mut mm, &d).unwrap();
        let mut em = elo.init();
        elo.update(&mut em, &d).unwrap();

        let ms: std::collections::HashMap<_, _> = mm.scores().collect();
        let es: std::collections::HashMap<_, _> = em.scores().collect();
        assert!((ms["a"] - es["a"]).abs() < 1e-12);
        assert!((ms["b"] - es["b"]).abs() < 1e-12);
        assert!(
            (mm.predict("a", "b").unwrap() - elo.expected_score(es["a"], es["b"])).abs() < 1e-12
        );
    }

    #[test]
    fn save_load_mid_stream_equals_uninterrupted() {
        let algo = MElo {
            k: 2,
            lr_rating: 0.5,
            lr_vector: 0.3,
            ..MElo::default()
        };
        let mut d1 = GamesDataset::new();
        d1.push_pair("a", "b", 1.0).unwrap();
        d1.push_pair("b", "a", 2.0).unwrap();
        // d2 introduces new entities, so the persisted draw counter must put
        // the resumed model on the same initialization stream.
        let mut d2 = GamesDataset::new();
        d2.push_pair("c", "a", 1.0).unwrap();
        d2.push_pair("d", "c", 1.0).unwrap();
        d2.push_pair("b", "d", 3.0).unwrap();

        let mut uninterrupted = algo.init();
        algo.update(&mut uninterrupted, &d1).unwrap();

        let mut buf = Vec::new();
        uninterrupted.save_jsonl(&mut buf).unwrap();
        let mut resumed = MEloModel::load_jsonl(buf.as_slice()).unwrap();

        algo.update(&mut uninterrupted, &d2).unwrap();
        algo.update(&mut resumed, &d2).unwrap();

        let mut full = Vec::new();
        uninterrupted.save_jsonl(&mut full).unwrap();
        let mut split = Vec::new();
        resumed.save_jsonl(&mut split).unwrap();
        assert_eq!(full, split, "save → load mid-stream diverged");
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let algo = MElo::default();
        let mut m = algo.init();
        algo.update(&mut m, &rps(2.0)).unwrap();

        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = MEloModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn multiplayer_sides_are_rejected() {
        let algo = MElo::default();
        let mut m = algo.init();
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        assert!(matches!(
            algo.update(&mut m, &d),
            Err(Error::InvalidInput(_))
        ));
    }
}
