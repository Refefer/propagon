//! Mallows φ-model estimation (`docs/algorithms.md` §1.7; Mallows 1957;
//! Marden 1995).
//!
//! `P(σ) ∝ φ^{d(σ, σ₀)}` with `d` the Kendall tau distance: one true order
//! σ₀, ballots falling off exponentially in distance from it. The σ₀ MLE
//! under Kendall distance *is* the Kemeny consensus (Young 1988), so the
//! combinatorial search is delegated wholesale to [`Kemeny`]'s insertion
//! heuristic; this module adds the sufficient statistic (mean Kendall
//! distance to σ₀) and the one-dimensional φ MLE on top.
//!
//! Assumes every ballot ranks the *same complete* item set — the Kendall
//! statistic is only comparable across full permutations. Mismatched or
//! truncated ballots are rejected up front; use Plackett-Luce for partial
//! data.
//!
//! Gotchas: φ̂ = 0 (all ballots identical) sits on the boundary of the
//! parameter space and is returned exactly by convention; a mean distance
//! at or beyond the uniform-noise expectation n(n−1)/4 has no interior MLE
//! either and is clamped to φ → 1 with a warning. The Kemeny heuristic can
//! return a non-optimal σ₀ on hard instances, and φ̂ inherits that
//! approximation — distances are measured against the σ₀ actually found.

use serde::{Deserialize, Serialize};

use crate::algos::{Kemeny, KemenyAlgo, KemenyPasses};
use crate::dataset::RankingsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Mallows parameters. The consensus search delegates to [`Kemeny`]'s
/// insertion heuristic, so the budget reuses [`KemenyPasses`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mallows {
    /// Insertion-pass budget forwarded to the Kemeny search.
    pub passes: KemenyPasses,
    /// Seed forwarded to the Kemeny search (inert under the insertion
    /// heuristic, but kept so a future search swap stays reproducible).
    pub seed: u64,
}

impl Default for Mallows {
    fn default() -> Self {
        Self {
            passes: KemenyPasses::Auto,
            // Mallows' publication year, in the spirit of Kemeny's 2020.
            seed: 1957,
        }
    }
}

impl Mallows {
    /// Rejects datasets where any ballot is not a permutation of ballot 0's
    /// item set — the Kendall sufficient statistic needs complete rankings.
    /// Comparing sorted id multisets also catches repeated items.
    fn validate_complete(data: &RankingsDataset) -> Result<()> {
        let expected = {
            let mut v = data.ranking(0).to_vec();
            v.sort_unstable();
            v
        };

        if expected.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::InvalidInput(
                "mallows needs complete rankings: ballot 0 lists an item twice".into(),
            ));
        }

        for (idx, ballot) in data.rankings().enumerate() {
            if ballot.len() != expected.len() {
                return Err(Error::InvalidInput(format!(
                    "mallows needs every ballot to rank the same item set: \
                     ballot {idx} has {} items, expected {}",
                    ballot.len(),
                    expected.len()
                )));
            }

            let mut sorted = ballot.to_vec();
            sorted.sort_unstable();

            if sorted != expected {
                return Err(Error::InvalidInput(format!(
                    "mallows needs every ballot to rank the same item set: \
                     ballot {idx} ranks different items than ballot 0"
                )));
            }
        }
        Ok(())
    }
}

impl Ranker for Mallows {
    type Data = RankingsDataset;
    type Model = MallowsModel;

    fn fit_opts(&self, data: &RankingsDataset, opts: &FitOptions<'_>) -> Result<MallowsModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::validate_complete(data)?;

        // σ₀ via Kemeny over the ballots' implied pairwise preferences —
        // the same library path the CLI's `rankings kemeny` uses.
        let consensus = Kemeny {
            passes: self.passes,
            min_obs: 1,
            algo: KemenyAlgo::Insertion,
            seed: self.seed,
        }
        .fit_opts(&data.to_pairwise(), opts)?;

        let n = data.n_entities();
        let order: Vec<u32> = consensus
            .order()
            .map(|name| {
                data.interner().get(name).ok_or_else(|| {
                    Error::Numeric(format!("kemeny consensus invented item {name:?}"))
                })
            })
            .collect::<Result<_>>()?;

        // min_obs = 1 keeps every item that appears in any comparison, and
        // each item appears in every ballot here, so a shortfall means the
        // consensus and the dataset disagree about the universe.
        if order.len() != n {
            return Err(Error::Numeric(format!(
                "kemeny consensus ranked {} of {n} items",
                order.len()
            )));
        }

        let mut sigma = vec![0usize; n];
        for (pos, &id) in order.iter().enumerate() {
            sigma[id as usize] = pos;
        }

        // Kendall distance to σ₀ = inversions of the ballot mapped into
        // σ₀-positions.
        let distances: Vec<u64> = data
            .rankings()
            .map(|ballot| {
                let mut seq: Vec<usize> = ballot.iter().map(|&id| sigma[id as usize]).collect();
                count_inversions(&mut seq)
            })
            .collect();

        let mean_distance = distances.iter().sum::<u64>() as f64 / distances.len() as f64;
        let phi = solve_phi(mean_distance, n);

        Ok(MallowsModel {
            params: *self,
            names: data.interner().clone(),
            order,
            distances,
            phi,
            mean_distance,
        })
    }
}

/// Counts inversions — pairs `i < j` with `seq[i] > seq[j]` — by an
/// iterative bottom-up merge sort in O(m log m): when the right run's head
/// wins a merge, everything left in the left run is an inversion against
/// it. Sorts `seq` in place as a side effect.
fn count_inversions(seq: &mut [usize]) -> u64 {
    let n = seq.len();
    let mut scratch = vec![0usize; n];
    let mut inversions = 0u64;
    let mut width = 1usize;

    while width < n {
        let mut lo = 0usize;

        while lo + width < n {
            let mid = lo + width;
            let hi = (lo + 2 * width).min(n);
            let (mut i, mut j, mut k) = (lo, mid, lo);

            while i < mid && j < hi {
                if seq[i] <= seq[j] {
                    scratch[k] = seq[i];
                    i += 1;
                } else {
                    inversions += (mid - i) as u64;
                    scratch[k] = seq[j];
                    j += 1;
                }
                k += 1;
            }

            while i < mid {
                scratch[k] = seq[i];
                i += 1;
                k += 1;
            }

            while j < hi {
                scratch[k] = seq[j];
                j += 1;
                k += 1;
            }

            seq[lo..hi].copy_from_slice(&scratch[lo..hi]);
            lo += 2 * width;
        }
        width *= 2;
    }
    inversions
}

/// φ MLE from the mean Kendall distance: solves `E_θ[D] = mean_d` for
/// `θ = −ln φ` by bisection (E is strictly decreasing in θ), then maps
/// back to φ. Boundary cases have no interior MLE: `mean_d == 0` returns
/// the point-mass convention φ = 0, and `mean_d` at or beyond the θ → 0
/// uniform limit n(n−1)/4 clamps to φ → 1 with a warning.
fn solve_phi(mean_d: f64, n: usize) -> f64 {
    if mean_d == 0.0 {
        return 0.0;
    }

    let uniform_limit = (n * (n - 1)) as f64 / 4.0;

    if mean_d >= uniform_limit {
        log::warn!(
            "mallows: mean Kendall distance {mean_d} reaches the uniform-noise \
             expectation {uniform_limit}; clamping phi to 1"
        );
        return 1.0 - 1e-12;
    }

    // 100 fixed halvings shrink [1e-12, 50] far below the 1e-12 target and
    // keep the solve deterministic. A mean distance below E_{θ=50}[D]
    // converges onto the bracket edge, i.e. φ ≈ e⁻⁵⁰ ≈ 0.
    let (mut lo, mut hi) = (1e-12_f64, 50.0_f64);

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);

        if expected_kendall(mid, n) > mean_d {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (-0.5 * (lo + hi)).exp()
}

/// `E_θ[D]` under Mallows with φ = e^{−θ}: the Fligner-Verducci closed
/// form `n·r(θ) − Σ_{j=1}^{n} j·r(jθ)` with `r(t) = e^{−t}/(1−e^{−t})`.
/// Strictly decreasing in θ, from n(n−1)/4 (θ → 0, uniform) to 0 (point
/// mass).
fn expected_kendall(theta: f64, n: usize) -> f64 {
    let head = n as f64 * geometric_ratio(theta);
    let tail: f64 = (1..=n)
        .map(|j| j as f64 * geometric_ratio(j as f64 * theta))
        .sum();
    head - tail
}

/// `e^{−t} / (1 − e^{−t})`, via `exp_m1` so t → 0 doesn't cancel.
fn geometric_ratio(t: f64) -> f64 {
    let em1 = (-t).exp_m1();
    (1.0 + em1) / -em1
}

/// What `save_jsonl` writes as the header `params`: the algorithm params
/// plus the fitted dispersion and its sufficient statistic.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PersistedParams {
    passes: KemenyPasses,
    seed: u64,
    phi: f64,
    mean_distance: f64,
}

/// One state-file line: consensus items and per-ballot distances share the
/// file, discriminated by `k`.
#[derive(Debug, Serialize, Deserialize)]
struct MallowsLine {
    /// Item name for `k = "item"`; decimal ballot index for `k = "ballot"`.
    id: String,
    k: LineKind,
    /// Rank `n..1` (higher is better) for items; Kendall distance for
    /// ballots.
    v: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum LineKind {
    Item,
    Ballot,
}

/// Fitted Mallows model: consensus order σ₀, per-ballot Kendall distances,
/// and the dispersion φ.
#[derive(Debug, Clone)]
pub struct MallowsModel {
    params: Mallows,
    names: Interner,
    /// Entity ids best-first (the fitted σ₀).
    order: Vec<u32>,
    /// Kendall distance from each ballot to σ₀, in dataset order.
    distances: Vec<u64>,
    phi: f64,
    mean_distance: f64,
}

impl MallowsModel {
    /// The consensus ranking σ₀, best first.
    pub fn order(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(|&id| self.names.resolve(id))
    }

    /// Fitted dispersion φ ∈ [0, 1): 0 = point mass on σ₀, → 1 = uniform
    /// noise.
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Mean Kendall distance from the ballots to σ₀ (φ's sufficient
    /// statistic).
    pub fn mean_distance(&self) -> f64 {
        self.mean_distance
    }

    /// Per-ballot Kendall distances to σ₀, in dataset order.
    pub fn distances(&self) -> &[u64] {
        &self.distances
    }
}

impl RankModel for MallowsModel {
    fn algorithm(&self) -> &'static str {
        "mallows"
    }

    /// Rank positions as scores: best entity gets `n`, worst gets `1` (the
    /// Kemeny output convention; see the [`RankModel::scores`] carve-out).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        let n = self.order.len();
        self.order
            .iter()
            .enumerate()
            .map(move |(pos, &id)| (self.names.resolve(id), (n - pos) as f64))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let n = self.order.len();
        let lines: Vec<MallowsLine> = self
            .order
            .iter()
            .enumerate()
            .map(|(pos, &id)| MallowsLine {
                id: self.names.resolve(id).to_string(),
                k: LineKind::Item,
                v: (n - pos) as f64,
            })
            .chain(
                self.distances
                    .iter()
                    .enumerate()
                    .map(|(b, &d)| MallowsLine {
                        id: b.to_string(),
                        k: LineKind::Ballot,
                        v: d as f64,
                    }),
            )
            .collect();

        let params = PersistedParams {
            passes: self.params.passes,
            seed: self.params.seed,
            phi: self.phi,
            mean_distance: self.mean_distance,
        };

        state::save_model(w, "mallows", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<MallowsLine>) = state::load_model(r, "mallows")?;
        let mut items: Vec<(String, f64)> = Vec::new();
        let mut ballots: Vec<(usize, u64)> = Vec::new();

        for line in lines {
            match line.k {
                LineKind::Item => items.push((line.id, line.v)),
                LineKind::Ballot => {
                    let idx = line
                        .id
                        .parse::<usize>()
                        .map_err(|_| Error::State(format!("bad ballot index {:?}", line.id)))?;
                    ballots.push((idx, line.v as u64));
                }
            }
        }

        // Ranks are distinct integers n..1, so descending sort restores
        // the best-first order regardless of on-disk ordering.
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        ballots.sort_unstable_by_key(|&(idx, _)| idx);

        let mut names = Interner::new();
        let order = items.iter().map(|(id, _)| names.intern(id)).collect();

        Ok(Self {
            params: Mallows {
                passes: params.passes,
                seed: params.seed,
            },
            names,
            order,
            distances: ballots.into_iter().map(|(_, d)| d).collect(),
            phi: params.phi,
            mean_distance: params.mean_distance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn ballots(rows: &[&[&str]]) -> RankingsDataset {
        let mut d = RankingsDataset::new();
        for row in rows {
            d.push_ranking(row.iter().copied()).unwrap();
        }
        d
    }

    /// Kendall distance to the identity by direct pair counting.
    fn pair_count_distance(perm: &[usize]) -> u64 {
        perm.iter()
            .enumerate()
            .map(|(i, &a)| perm[i + 1..].iter().filter(|&&b| a > b).count() as u64)
            .sum()
    }

    /// All permutations of `0..n` via Heap's algorithm.
    fn permutations(n: usize) -> Vec<Vec<usize>> {
        let mut perm: Vec<usize> = (0..n).collect();
        let mut out = vec![perm.clone()];
        let mut c = vec![0usize; n];
        let mut i = 0usize;

        while i < n {
            if c[i] < i {
                if i.is_multiple_of(2) {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                out.push(perm.clone());
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        out
    }

    /// The load-bearing pin: the closed form used by the solver must equal
    /// the brute-force expectation `Σ_σ d(σ)·φ^{d(σ)} / Σ_σ φ^{d(σ)}`
    /// enumerated over all 4! permutations.
    #[test]
    fn closed_form_matches_enumerated_expectation() {
        for theta in [0.5_f64, 1.0, 2.0] {
            let phi = (-theta).exp();
            let (mut num, mut den) = (0.0, 0.0);

            for perm in permutations(4) {
                let d = pair_count_distance(&perm) as f64;
                let w = phi.powf(d);
                num += d * w;
                den += w;
            }

            let brute = num / den;
            let closed = expected_kendall(theta, 4);
            assert!(
                (closed - brute).abs() < 1e-10,
                "theta={theta}: closed {closed} vs enumerated {brute}"
            );
        }
    }

    #[test]
    fn identical_ballots_give_phi_zero() {
        let row: &[&str] = &["a", "b", "c"];
        let d = ballots(&[row, row, row]);
        let m = Mallows::default().fit(&d).unwrap();
        assert_eq!(m.phi(), 0.0);
        assert_eq!(m.mean_distance(), 0.0);
        assert_eq!(m.distances(), &[0, 0, 0]);
        assert_eq!(m.order().collect::<Vec<_>>(), vec!["a", "b", "c"]);
    }

    /// n = 2 collapses the closed form to `E_θ[D] = φ/(1+φ)`, so D̄ = ¼
    /// gives φ̂ = D̄/(1−D̄) = ⅓ analytically.
    #[test]
    fn two_item_mle_is_analytic() {
        let d = ballots(&[&["a", "b"], &["a", "b"], &["a", "b"], &["b", "a"]]);
        let m = Mallows::default().fit(&d).unwrap();
        assert_eq!(m.order().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(m.distances(), &[0, 0, 0, 1]);
        assert_eq!(m.mean_distance(), 0.25);
        assert!((m.phi() - 1.0 / 3.0).abs() < 1e-9, "phi {}", m.phi());
    }

    #[test]
    fn incomplete_or_mismatched_ballots_are_rejected() {
        let mixed = ballots(&[&["a", "b", "c"], &["a", "b"]]);
        assert!(matches!(
            Mallows::default().fit(&mixed),
            Err(Error::InvalidInput(_))
        ));

        let disjoint = ballots(&[&["a", "b"], &["a", "c"]]);
        assert!(matches!(
            Mallows::default().fit(&disjoint),
            Err(Error::InvalidInput(_))
        ));
    }

    /// Hand-built 3-item data: the distances and D̄ are pinned exactly, and
    /// the solved φ̂ must reproduce D̄ through E_θ (MLE self-consistency).
    #[test]
    fn hand_built_distances_and_self_consistent_phi() {
        let d = ballots(&[
            &["a", "b", "c"],
            &["a", "b", "c"],
            &["a", "b", "c"],
            &["b", "a", "c"],
            &["a", "c", "b"],
        ]);
        let m = Mallows::default().fit(&d).unwrap();
        assert_eq!(m.order().collect::<Vec<_>>(), vec!["a", "b", "c"]);
        assert_eq!(m.distances(), &[0, 0, 0, 1, 1]);
        assert_eq!(m.mean_distance(), 0.4);

        let theta = -m.phi().ln();
        let reproduced = expected_kendall(theta, 3);
        assert!(
            (reproduced - 0.4).abs() < 1e-9,
            "E_theta[D] {reproduced} vs mean 0.4"
        );
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let d = ballots(&[&["a", "b", "c"], &["b", "a", "c"], &["c", "a", "b"]]);
        let m = Mallows::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = MallowsModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
        assert_eq!(loaded.phi(), m.phi());
        assert_eq!(loaded.mean_distance(), m.mean_distance());
        assert_eq!(loaded.distances(), m.distances());
        assert_eq!(
            loaded.order().collect::<Vec<_>>(),
            m.order().collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let d = RankingsDataset::new();
        assert!(matches!(
            Mallows::default().fit(&d),
            Err(Error::EmptyDataset)
        ));
    }
}
