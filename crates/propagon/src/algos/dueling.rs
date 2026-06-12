//! Dueling bandits — preference-feedback arm selection
//! (`docs/algorithms.md` §8.2): RUCB (Zoghi, Whiteson, Munos & de Rijke,
//! ICML 2014, Algorithm 1) and Double Thompson Sampling (Wu & Liu,
//! NIPS 2016, Algorithm 1).
//!
//! The model accumulates order-independent sufficient statistics — sparse
//! directed win weights per arm pair — from [`PairwiseDataset`] batches, and
//! [`DuelingModel::select_pair`] proposes the next duel `(champion,
//! challenger)`. Selection is deterministic given the `seed` and the model
//! state: a persisted draw counter derives each round's RNG (the same stream
//! pattern as [`Bandit`](crate::algos::Bandit)), so save → load →
//! `select_pair` is indistinguishable from an uninterrupted run.
//!
//! Because the statistics are sufficient, [`DuelingModel::merge`] of two
//! state files equals processing the concatenated logs (RUCB's
//! hypothesized-best set `B` is selection-loop state, not a statistic; the
//! merge keeps the intersection).
//!
//! Library-only by design: there is no CLI for live duel loops, and a future
//! stdin/stdout streaming mode is explicitly out of scope for now. The
//! driving loop lives in the host application:
//!
//! ```ignore
//! let algo = DuelingBandit { policy: DuelingPolicy::Rucb { alpha: 0.51 }, seed: 7 };
//! let mut model = algo.init();
//! model.add_arm("a"); model.add_arm("b"); model.add_arm("c");
//! loop {
//!     let (x, y) = model.select_pair()?;
//!     // ...run the duel, observe that x beat y...
//!     let mut batch = PairwiseDataset::new();
//!     batch.push(&x, &y, 1.0);
//!     algo.update(&mut model, &batch)?;
//! }
//! ```
//!
//! Gotchas: self-duel rows (`winner == loser`) are rejected as invalid
//! input; RUCB's challenger is `argmax_{j≠c}` and D-TS's second arm skips
//! `a1`, so `select_pair` never returns `(x, x)`.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// Dueling-bandit exploration policy. Serialized into state files; mixing
/// states produced under different policies is a parameter mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum DuelingPolicy {
    /// Relative UCB: optimism on the pairwise win fractions; duels the
    /// sampled candidate champion against its most threatening challenger.
    Rucb {
        /// Confidence-width constant on the win-fraction upper bound.
        alpha: f64,
    },
    /// Double Thompson Sampling: a sampled-Copeland champion from Beta
    /// posteriors, then a resampled most-informative challenger.
    DoubleThompson {
        /// Confidence-width constant gating which pairs stay in play.
        alpha: f64,
    },
}

impl DuelingPolicy {
    /// Confidence-width constant; theory wants `alpha > ½`, and both papers
    /// run their experiments at 0.51.
    pub const DEFAULT_ALPHA: f64 = 0.51;
}

impl Default for DuelingPolicy {
    fn default() -> Self {
        DuelingPolicy::Rucb {
            alpha: Self::DEFAULT_ALPHA,
        }
    }
}

/// Dueling-bandit algorithm parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DuelingBandit {
    /// Exploration policy used to propose each duel.
    pub policy: DuelingPolicy,
    /// Seeds the per-round selection RNG stream.
    pub seed: u64,
}

impl Default for DuelingBandit {
    fn default() -> Self {
        Self {
            policy: DuelingPolicy::default(),
            seed: 42,
        }
    }
}

/// What `save_jsonl` writes as the header `params`: the algorithm params
/// plus the selection-loop state (draw counter and RUCB's hypothesized-best
/// set, by name).
#[derive(Serialize, Deserialize)]
struct PersistedParams {
    policy: DuelingPolicy,
    seed: u64,
    draws: u64,
    b_set: Vec<String>,
}

/// One state-file line: either an arm (vocab) or one directed pair count —
/// the same discriminated-lines pattern as crowd-bt.
#[derive(Serialize, Deserialize)]
#[serde(tag = "k", rename_all = "kebab-case")]
enum DuelLine {
    Arm { id: String },
    Pair { a: String, b: String, w: f64 },
}

/// Accumulating dueling-bandit state: sparse directed win weights plus the
/// selection-loop state (draw counter, RUCB's hypothesized-best set).
#[derive(Debug, Clone)]
pub struct DuelingModel {
    params: DuelingBandit,
    names: Interner,
    /// `wins[(i, j)]` = total weight of duels `i` won over `j`.
    wins: HashMap<(u32, u32), f64>,
    /// Selection rounds played; persisted so the RNG stream resumes exactly.
    draws: u64,
    /// RUCB's hypothesized-best set `B`, kept sorted (≤ 1 entry by
    /// construction; loaded files could carry more, which is tolerated).
    b_set: Vec<u32>,
}

impl DuelingModel {
    /// Number of registered arms.
    pub fn n_arms(&self) -> usize {
        self.names.len()
    }

    /// Registers an arm with no data yet — required before the first
    /// `select_pair` since duel logs only mention arms that already played.
    pub fn add_arm(&mut self, name: &str) {
        self.names.intern(name);
    }

    /// Total duel weight observed across all pairs.
    pub fn total_duels(&self) -> f64 {
        self.wins.values().sum()
    }

    fn w(&self, i: u32, j: u32) -> f64 {
        self.wins.get(&(i, j)).copied().unwrap_or(0.0)
    }

    /// The k×k RUCB upper-bound matrix at `t = total duels + 1`:
    /// `U_ij = w_ij/(w_ij+w_ji) + sqrt(alpha·ln t/(w_ij+w_ji))`, 1 for
    /// unplayed pairs, ½ on the diagonal.
    fn u_matrix(&self, alpha: f64) -> Vec<f64> {
        let k = self.n_arms();
        let ln_t = (self.total_duels() + 1.0).ln();
        let mut u = vec![0.5; k * k];

        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue;
                }
                let wij = self.w(i as u32, j as u32);
                let n = wij + self.w(j as u32, i as u32);
                u[i * k + j] = if n == 0.0 {
                    1.0
                } else {
                    wij / n + (alpha * ln_t / n).sqrt()
                };
            }
        }
        u
    }

    /// Advances the persisted draw counter and returns this round's RNG
    /// (the bandits stream pattern: one derived stream per round).
    fn next_rng(&mut self) -> Xoshiro256PlusPlus {
        let stream = self.params.seed ^ self.draws.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.draws += 1;
        Xoshiro256PlusPlus::seed_from_u64(stream)
    }

    /// Proposes the next duel `(champion, challenger)` under the configured
    /// policy. Needs at least two registered arms; advances the draw
    /// counter, so streams resume exactly across save/load.
    pub fn select_pair(&mut self) -> Result<(String, String)> {
        if self.n_arms() < 2 {
            return Err(Error::InvalidInput(format!(
                "dueling bandit needs at least 2 arms, has {}",
                self.n_arms()
            )));
        }
        let (DuelingPolicy::Rucb { alpha } | DuelingPolicy::DoubleThompson { alpha }) =
            self.params.policy;
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(Error::InvalidInput(format!(
                "alpha must be positive and finite, got {alpha}"
            )));
        }

        let (x, y) = match self.params.policy {
            DuelingPolicy::Rucb { alpha } => self.rucb_round(alpha)?,
            DuelingPolicy::DoubleThompson { alpha } => self.dts_round(alpha)?,
        };
        Ok((
            self.names.resolve(x).to_string(),
            self.names.resolve(y).to_string(),
        ))
    }

    /// One RUCB round (Zoghi et al. 2014, Algorithm 1): candidate set
    /// `C = {i : U_ij ≥ ½ ∀j≠i}`; `B ← B ∩ C`, reset to `C` when `|C| = 1`;
    /// champion sampled from `C` (the hypothesized best, if any, gets
    /// probability ½); challenger `argmax_{j≠c} U_jc`, ties to the smaller
    /// id.
    fn rucb_round(&mut self, alpha: f64) -> Result<(u32, u32)> {
        let mut rng = self.next_rng();
        let k = self.n_arms() as u32;
        let u = self.u_matrix(alpha);
        let uij = |i: u32, j: u32| u[(i * k + j) as usize];

        let c: Vec<u32> = (0..k)
            .filter(|&i| (0..k).all(|j| j == i || uij(i, j) >= 0.5))
            .collect();
        self.b_set.retain(|b| c.contains(b));

        if c.len() == 1 {
            self.b_set = c.clone();
        }

        let champion = if c.is_empty() {
            // Optimism filtered everyone out — the paper falls back to a
            // uniformly random champion.
            rng.random_range(0..k)
        } else if let [b] = self.b_set[..] {
            // B ⊆ C after the intersection above: the hypothesized best gets
            // probability ½, the rest of C shares the other half uniformly.
            let others: Vec<u32> = c.iter().copied().filter(|&i| i != b).collect();
            if others.is_empty() || rng.random::<f64>() < 0.5 {
                b
            } else {
                others[rng.random_range(0..others.len())]
            }
        } else {
            c[rng.random_range(0..c.len())]
        };

        let mut challenger: Option<(u32, f64)> = None;
        for j in 0..k {
            if j == champion {
                continue;
            }
            let v = uij(j, champion);
            challenger = match challenger {
                Some((_, best)) if v <= best => challenger,
                _ => Some((j, v)),
            };
        }
        match challenger {
            Some((d, _)) => Ok((champion, d)),
            // Unreachable: select_pair guarantees k ≥ 2.
            None => Err(Error::InvalidInput(
                "dueling bandit needs at least 2 arms".into(),
            )),
        }
    }

    /// One D-TS round (Wu & Liu 2016, Algorithm 1): sample
    /// `θ_ij ~ Beta(w_ij+1, w_ji+1)` for `i < j` (ascending — the RNG call
    /// order is fixed), `θ_ji = 1−θ_ij`; `a1` is the Copeland winner of the
    /// sampled matrix restricted to pairs the upper bound has not ruled out
    /// (`U_ij ≥ ½`), ties broken uniformly at random; `a2` is the resampled
    /// `argmax θ'_{i,a1}` over the wait-list `{i≠a1 : U_{i,a1} > ½}`, or
    /// `argmax_{i≠a1} U_{i,a1}` when the wait-list is empty.
    fn dts_round(&mut self, alpha: f64) -> Result<(u32, u32)> {
        let mut rng = self.next_rng();
        let k = self.n_arms();
        let u = self.u_matrix(alpha);

        let mut theta = vec![0.5; k * k];
        for i in 0..k {
            for j in (i + 1)..k {
                let s = self.beta_sample(i as u32, j as u32, &mut rng)?;
                theta[i * k + j] = s;
                theta[j * k + i] = 1.0 - s;
            }
        }

        let mut cope = vec![0usize; k];
        for i in 0..k {
            for j in 0..k {
                if i != j && u[i * k + j] >= 0.5 && theta[i * k + j] > 0.5 {
                    cope[i] += 1;
                }
            }
        }
        let best = cope.iter().max().copied().unwrap_or(0);
        let winners: Vec<usize> = (0..k).filter(|&i| cope[i] == best).collect();
        let a1 = winners[rng.random_range(0..winners.len())];

        let mut a2: Option<(usize, f64)> = None;
        for i in 0..k {
            if i == a1 || u[i * k + a1] <= 0.5 {
                continue;
            }
            let s = self.beta_sample(i as u32, a1 as u32, &mut rng)?;
            a2 = match a2 {
                Some((_, top)) if s <= top => a2,
                _ => Some((i, s)),
            };
        }

        let a2 = match a2 {
            Some((i, _)) => i,
            None => {
                // Empty wait-list: duel a1 against its strongest challenger
                // by upper bound, ties to the smaller id.
                let mut fallback: Option<(usize, f64)> = None;
                for i in (0..k).filter(|&i| i != a1) {
                    let v = u[i * k + a1];
                    fallback = match fallback {
                        Some((_, top)) if v <= top => fallback,
                        _ => Some((i, v)),
                    };
                }
                match fallback {
                    Some((i, _)) => i,
                    // Unreachable: select_pair guarantees k ≥ 2.
                    None => {
                        return Err(Error::InvalidInput(
                            "dueling bandit needs at least 2 arms".into(),
                        ));
                    }
                }
            }
        };
        Ok((a1 as u32, a2 as u32))
    }

    /// Draws `θ ~ Beta(w_ij + 1, w_ji + 1)`.
    fn beta_sample(&self, i: u32, j: u32, rng: &mut Xoshiro256PlusPlus) -> Result<f64> {
        let dist = Beta::new(self.w(i, j) + 1.0, self.w(j, i) + 1.0)
            .map_err(|e| Error::Numeric(format!("beta posterior for pair ({i},{j}): {e}")))?;
        Ok(dist.sample(rng))
    }

    /// Folds another state file into this one. The pair statistics are
    /// sufficient, so the merged counts equal processing the concatenation
    /// of both logs; draw counters add (mirroring
    /// [`BanditModel::merge`](crate::algos::BanditModel::merge)) and the
    /// hypothesized-best set keeps only arms both sides agree on.
    pub fn merge(&mut self, other: &DuelingModel) -> Result<()> {
        if self.params != other.params {
            return Err(Error::ParamMismatch(format!(
                "cannot merge dueling states with different params: {:?} vs {:?}",
                self.params, other.params
            )));
        }
        let map: Vec<u32> = other.names.names().map(|n| self.names.intern(n)).collect();

        for (&(i, j), &w) in &other.wins {
            *self
                .wins
                .entry((map[i as usize], map[j as usize]))
                .or_default() += w;
        }
        self.draws += other.draws;

        let other_b: Vec<&str> = other
            .b_set
            .iter()
            .map(|&b| other.names.resolve(b))
            .collect();
        self.b_set
            .retain(|&b| other_b.contains(&self.names.resolve(b)));
        Ok(())
    }
}

impl RankModel for DuelingModel {
    fn algorithm(&self) -> &'static str {
        "dueling-bandit"
    }

    /// Copeland fraction from the empirical majorities: the share of an
    /// arm's *played* opponents it beats by strict majority of duel weight.
    /// Arms with no duels score 0.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        let k = self.n_arms();
        let vals: Vec<f64> = (0..k as u32)
            .map(|i| {
                let mut played = 0usize;
                let mut beaten = 0usize;
                for j in 0..k as u32 {
                    if i == j {
                        continue;
                    }
                    let wij = self.w(i, j);
                    let wji = self.w(j, i);
                    if wij + wji > 0.0 {
                        played += 1;
                        if wij > wji {
                            beaten += 1;
                        }
                    }
                }
                if played == 0 {
                    0.0
                } else {
                    beaten as f64 / played as f64
                }
            })
            .collect();
        self.names.names().zip(vals)
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            policy: self.params.policy,
            seed: self.params.seed,
            draws: self.draws,
            b_set: self
                .b_set
                .iter()
                .map(|&b| self.names.resolve(b).to_string())
                .collect(),
        };

        let mut pairs: Vec<(u32, u32)> = self.wins.keys().copied().collect();
        pairs.sort_unstable();
        let lines: Vec<DuelLine> = self
            .names
            .names()
            .map(|id| DuelLine::Arm { id: id.to_string() })
            .chain(pairs.into_iter().map(|(i, j)| DuelLine::Pair {
                a: self.names.resolve(i).to_string(),
                b: self.names.resolve(j).to_string(),
                w: self.w(i, j),
            }))
            .collect();
        state::save_model(w, "dueling-bandit", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<DuelLine>) =
            state::load_model(r, "dueling-bandit")?;
        let names = Interner::from_names(lines.iter().filter_map(|l| match l {
            DuelLine::Arm { id } => Some(id.as_str()),
            DuelLine::Pair { .. } => None,
        }))?;

        let mut wins = HashMap::new();
        for line in &lines {
            let DuelLine::Pair { a, b, w } = line else {
                continue;
            };
            let (Some(i), Some(j)) = (names.get(a), names.get(b)) else {
                return Err(Error::State(format!(
                    "pair line ({a:?}, {b:?}) references an arm missing from the vocab"
                )));
            };
            if i == j {
                return Err(Error::State(format!("self-duel pair line for {a:?}")));
            }
            *wins.entry((i, j)).or_default() += w;
        }

        let mut b_set = params
            .b_set
            .iter()
            .map(|n| {
                names
                    .get(n)
                    .ok_or_else(|| Error::State(format!("b_set arm {n:?} missing from the vocab")))
            })
            .collect::<Result<Vec<u32>>>()?;
        b_set.sort_unstable();

        Ok(Self {
            params: DuelingBandit {
                policy: params.policy,
                seed: params.seed,
            },
            names,
            wins,
            draws: params.draws,
            b_set,
        })
    }
}

impl OnlineRanker for DuelingBandit {
    type Data = PairwiseDataset;
    type Model = DuelingModel;

    fn init(&self) -> DuelingModel {
        DuelingModel {
            params: *self,
            names: Interner::new(),
            wins: HashMap::new(),
            draws: 0,
            b_set: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut DuelingModel,
        data: &PairwiseDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        for (w, l, x) in data.rows() {
            let wi = model.names.intern(data.interner().resolve(w));
            let li = model.names.intern(data.interner().resolve(l));

            if wi == li {
                return Err(Error::InvalidInput(format!(
                    "self-duel row for {:?}",
                    model.names.resolve(wi)
                )));
            }
            *model.wins.entry((wi, li)).or_default() += f64::from(x);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn duels(rows: &[(&str, &str, f32)]) -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        for (w, l, x) in rows {
            d.push(w, l, *x);
        }
        d
    }

    fn policies() -> [DuelingPolicy; 2] {
        [
            DuelingPolicy::Rucb {
                alpha: DuelingPolicy::DEFAULT_ALPHA,
            },
            DuelingPolicy::DoubleThompson {
                alpha: DuelingPolicy::DEFAULT_ALPHA,
            },
        ]
    }

    /// Runs `rounds` of select → duel → update with a deterministic outcome
    /// rule (the lexicographically smaller name always wins), returning the
    /// selected pairs.
    fn drive(
        algo: &DuelingBandit,
        model: &mut DuelingModel,
        rounds: usize,
    ) -> Vec<(String, String)> {
        (0..rounds)
            .map(|_| {
                let (x, y) = model.select_pair().unwrap();
                let (w, l) = if x <= y { (&x, &y) } else { (&y, &x) };
                let mut batch = PairwiseDataset::new();
                batch.push(w, l, 1.0);
                algo.update(model, &batch).unwrap();
                (x, y)
            })
            .collect()
    }

    #[test]
    fn select_stream_is_seed_deterministic_and_resumable() {
        for policy in policies() {
            let algo = DuelingBandit { policy, seed: 7 };
            let fresh = || {
                let mut m = algo.init();
                for a in ["a", "b", "c"] {
                    m.add_arm(a);
                }
                m
            };

            let mut m1 = fresh();
            let mut m2 = fresh();
            let s1 = drive(&algo, &mut m1, 20);
            let s2 = drive(&algo, &mut m2, 20);
            assert_eq!(s1, s2, "{policy:?}: same seed + state => same stream");

            // save mid-stream, resume, and the stream continues identically
            let mut m3 = fresh();
            let head = drive(&algo, &mut m3, 10);
            assert_eq!(head, s1[..10], "{policy:?}");
            let mut buf = Vec::new();
            m3.save_jsonl(&mut buf).unwrap();
            let mut m4 = DuelingModel::load_jsonl(buf.as_slice()).unwrap();
            let tail3 = drive(&algo, &mut m3, 10);
            let tail4 = drive(&algo, &mut m4, 10);
            assert_eq!(tail3, tail4, "{policy:?}: save -> load resumes exactly");
            assert_eq!(tail3, s1[10..], "{policy:?}");
        }
    }

    /// Synthetic preference matrix with a clear Condorcet winner: a beats
    /// b and c 8:2, b beats c 8:2. After 500 driven rounds both policies
    /// should duel a as champion in at least 80 of the last 100 rounds, and
    /// the Copeland-fraction scores should rank a first.
    #[test]
    fn condorcet_winner_emerges() {
        for policy in policies() {
            let algo = DuelingBandit { policy, seed: 11 };
            let mut m = algo.init();
            for a in ["a", "b", "c"] {
                m.add_arm(a);
            }
            let mut sim = Xoshiro256PlusPlus::seed_from_u64(99);
            let p_first = |x: &str, y: &str| match (x, y) {
                ("a", _) => 0.8,
                (_, "a") => 0.2,
                ("b", "c") => 0.8,
                _ => 0.2,
            };

            let mut champion_a = 0;
            for round in 0..500 {
                let (x, y) = m.select_pair().unwrap();
                assert_ne!(x, y, "{policy:?}: self-duel proposed");
                if round >= 400 && x == "a" {
                    champion_a += 1;
                }
                let first_wins = sim.random::<f64>() < p_first(&x, &y);
                let (w, l) = if first_wins { (&x, &y) } else { (&y, &x) };
                let mut batch = PairwiseDataset::new();
                batch.push(w, l, 1.0);
                algo.update(&mut m, &batch).unwrap();
            }
            assert!(
                champion_a >= 80,
                "{policy:?}: champion was a in only {champion_a}/100 late rounds"
            );
            assert_eq!(m.sorted_scores()[0].0, "a", "{policy:?}");
        }
    }

    #[test]
    fn merge_equals_concatenated_logs() {
        let algo = DuelingBandit {
            policy: DuelingPolicy::default(),
            seed: 3,
        };
        let log1 = [("a", "b", 1.0f32), ("b", "c", 2.0)];
        let log2 = [("c", "a", 1.0f32), ("a", "b", 1.0), ("b", "a", 0.5)];

        let mut split_a = algo.init();
        algo.update(&mut split_a, &duels(&log1)).unwrap();
        let mut split_b = algo.init();
        algo.update(&mut split_b, &duels(&log2)).unwrap();
        split_a.merge(&split_b).unwrap();

        let mut joint = algo.init();
        let mut all = log1.to_vec();
        all.extend_from_slice(&log2);
        algo.update(&mut joint, &duels(&all)).unwrap();

        let mut buf1 = Vec::new();
        split_a.save_jsonl(&mut buf1).unwrap();
        let mut buf2 = Vec::new();
        joint.save_jsonl(&mut buf2).unwrap();
        assert_eq!(
            buf1, buf2,
            "merged state file == concatenated-log state file"
        );

        // Identical state and draw counters select identically too.
        assert_eq!(split_a.select_pair().unwrap(), joint.select_pair().unwrap());

        // Parameter mismatches are rejected.
        let other = DuelingBandit {
            policy: DuelingPolicy::DoubleThompson {
                alpha: DuelingPolicy::DEFAULT_ALPHA,
            },
            seed: 3,
        }
        .init();
        assert!(matches!(
            split_a.merge(&other),
            Err(Error::ParamMismatch(_))
        ));
    }

    /// Loose D-TS/RUCB sanity: across many driven rounds the proposed pairs
    /// are valid registered arms and never self-duels (the champion and the
    /// challenger are distinct by construction in both policies).
    #[test]
    fn pairs_are_valid_and_distinct() {
        for policy in policies() {
            let algo = DuelingBandit { policy, seed: 5 };
            let mut m = algo.init();
            for a in ["a", "b", "c", "d"] {
                m.add_arm(a);
            }
            let pairs = drive(&algo, &mut m, 30);

            for (x, y) in pairs {
                assert_ne!(x, y, "{policy:?}");
                assert!(["a", "b", "c", "d"].contains(&x.as_str()));
                assert!(["a", "b", "c", "d"].contains(&y.as_str()));
            }
        }
    }

    #[test]
    fn round_trip_is_byte_identical_and_small_models_are_rejected() {
        for policy in policies() {
            let algo = DuelingBandit { policy, seed: 9 };
            let mut m = algo.init();
            m.add_arm("a");
            assert!(matches!(m.select_pair(), Err(Error::InvalidInput(_))));

            m.add_arm("b");
            m.add_arm("c");
            algo.update(
                &mut m,
                &duels(&[
                    ("a", "b", 1.0),
                    ("a", "c", 1.0),
                    ("b", "c", 1.0),
                    ("c", "b", 2.0),
                ]),
            )
            .unwrap();
            // Pairs: a>b 1:0, a>c 1:0, b:c 1:2 → Copeland fractions
            // a = 2/2, b = 0/2, c = 1/2.
            let s: std::collections::HashMap<_, _> =
                m.scores().map(|(n, v)| (n.to_string(), v)).collect();
            assert_eq!(s["a"], 1.0);
            assert_eq!(s["b"], 0.0);
            assert_eq!(s["c"], 0.5);

            // Advance the stream so draws and (for RUCB) the B set persist.
            let _ = m.select_pair().unwrap();

            let mut first = Vec::new();
            m.save_jsonl(&mut first).unwrap();
            let mut loaded = DuelingModel::load_jsonl(first.as_slice()).unwrap();
            let mut second = Vec::new();
            loaded.save_jsonl(&mut second).unwrap();
            assert_eq!(first, second, "{policy:?}: byte-identical round trip");
            assert_eq!(m.select_pair().unwrap(), loaded.select_pair().unwrap());
        }

        let mut empty = DuelingBandit::default().init();
        assert!(matches!(empty.select_pair(), Err(Error::InvalidInput(_))));

        // Self-duel rows are invalid input.
        let algo = DuelingBandit::default();
        let mut m = algo.init();
        assert!(matches!(
            algo.update(&mut m, &duels(&[("a", "a", 1.0)])),
            Err(Error::InvalidInput(_))
        ));
    }
}
