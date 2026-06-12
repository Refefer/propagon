//! Nonparametric bootstrap (Efron 1979) over any batch ranker: per-entity
//! score and rank intervals from refits on resampled data.
//!
//! One fit on the full data gives the point estimates; `replicates` further
//! fits on [`Resample`]d copies give the empirical distributions whose
//! central quantiles become the intervals. Only batch rankers qualify — the
//! `R: Ranker` bound enforces it — because an
//! [`OnlineRanker`](crate::OnlineRanker)'s state depends on presentation
//! order: refitting it on a drawn-with-replacement history changes what the
//! model *means*, not just its sampling noise.
//!
//! Assumes the inner ranker is deterministic given its params and data (the
//! house contract) and that the dataset's resampling unit is exchangeable.
//! Replicate failures are expected — resampling routinely disconnects
//! comparison graphs, which solvers like Massey and Colley reject — and are
//! skipped and counted ([`BootstrapModel::replicates_ok`]); only when fewer
//! than half survive does the whole fit fail, carrying the first error.
//!
//! Determinism: replicate `k` draws from its own stream seeded
//! `seed.wrapping_add(k)`, replicates are aggregated in index order and
//! entities in name order, so results are bit-stable at any thread count.
//! Gotcha: the replicate phase reports `start`/`finish` only, no
//! per-replicate ticks — the parallel map offers no completion hook — and
//! each replicate fits with default (silent) [`FitOptions`]; the point fit
//! alone runs with the caller's options.

use std::collections::HashMap;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::Resample;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx::quantile;
use crate::parallel;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Bootstrap wrapper around an inner batch ranker.
#[derive(Clone, Debug)]
pub struct Bootstrap<R: Ranker> {
    /// The wrapped batch ranker; its params are shared by the point fit and
    /// every replicate.
    pub inner: R,
    /// Number of resampled refits (at least 1).
    pub replicates: usize,
    /// Central interval mass in (0, 1) (0.95 → 2.5%..97.5%).
    pub credible: f64,
    /// Master seed; replicate `k` draws from `seed.wrapping_add(k)`.
    pub seed: u64,
}

impl<R: Ranker> Bootstrap<R> {
    /// Wraps `inner` with the documented defaults: 200 replicates, 0.95
    /// interval mass, seed 26.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            replicates: 200,
            credible: 0.95,
            seed: 26,
        }
    }

    /// Rejects parameter values outside their documented domains (NaN
    /// included — the comparisons are written to fail on it).
    fn validate(&self) -> Result<()> {
        if self.replicates == 0 {
            return Err(Error::InvalidInput(
                "need at least 1 bootstrap replicate".into(),
            ));
        }

        if !(self.credible > 0.0 && self.credible < 1.0) {
            return Err(Error::InvalidInput(format!(
                "credible mass must lie in (0, 1), got {}",
                self.credible
            )));
        }
        Ok(())
    }
}

impl<R> Ranker for Bootstrap<R>
where
    R: Ranker + Sync,
    R::Data: Resample + Sync,
    R::Model: Send,
{
    type Data = R::Data;
    type Model = BootstrapModel;

    /// Fits the inner ranker once on the full data (with the caller's
    /// options; its error fails the whole run), then `replicates` times on
    /// resampled copies in parallel (each with default, silent options and
    /// its own `seed + k` stream), and aggregates per-entity score and rank
    /// quantiles by name. Failed replicates are skipped and counted; fewer
    /// than half surviving is [`Error::Numeric`].
    fn fit_opts(&self, data: &Self::Data, opts: &FitOptions<'_>) -> Result<BootstrapModel> {
        self.validate()?;

        let point = self.inner.fit_opts(data, opts)?;

        let progress = opts.progress;
        progress.start("bootstrap replicates", Some(self.replicates as u64));
        let fits: Vec<Result<R::Model>> = parallel::run_scoped(opts, || {
            parallel::par_map_indexed(self.replicates, |k| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed.wrapping_add(k as u64));
                self.inner
                    .fit_opts(&data.resample(&mut rng), &FitOptions::default())
            })
        });
        progress.finish();

        let mut models = Vec::with_capacity(fits.len());
        let mut first_failure = None;

        for fit in fits {
            match fit {
                Ok(m) => models.push(m),
                Err(e) if first_failure.is_none() => first_failure = Some(e),
                Err(_) => {}
            }
        }

        if models.len() < self.replicates.div_ceil(2) {
            let why = match first_failure {
                Some(e) => e.to_string(),
                // Unreachable: fewer Oks than replicates implies an Err.
                None => "unknown".to_string(),
            };
            return Err(Error::Numeric(format!(
                "only {} of {} bootstrap replicates fitted; the data is too fragile \
                 under resampling for trustworthy intervals (first failure: {why})",
                models.len(),
                self.replicates,
            )));
        }

        aggregate(&point, &models, self.replicates, self.credible, self.seed)
    }
}

/// Collapses the point fit and the surviving replicate fits into per-entity
/// score and rank intervals.
///
/// The entity universe is the point model's, ordered by name. Replicate
/// models are read in index order and matched by name: per entity, score
/// samples come from each replicate's `scores()` and rank samples from its
/// 1-based `sorted_scores` position (score descending, ties broken by name —
/// the positions are distinct, so "dense" here means no gaps). Entities a
/// replicate did not emit contribute nothing to it (`n_rep` records the
/// coverage); replicate-only names with no point row are skipped. Entities
/// no replicate emitted keep `n_rep = 0` and an interval collapsed to the
/// point estimate. Intervals are empirical central quantiles at
/// `(1 ± credible) / 2` over the sorted samples.
fn aggregate<M: RankModel>(
    point: &M,
    models: &[M],
    replicates: usize,
    credible: f64,
    seed: u64,
) -> Result<BootstrapModel> {
    let mut universe: Vec<(&str, f64)> = point.scores().collect();
    universe.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let index: HashMap<&str, usize> = universe
        .iter()
        .enumerate()
        .map(|(i, &(name, _))| (name, i))
        .collect();
    let n = universe.len();
    let score_point: Vec<f64> = universe.iter().map(|&(_, s)| s).collect();

    let mut rank_point = vec![0.0f64; n];

    for (pos, (name, _)) in point.sorted_scores().into_iter().enumerate() {
        if let Some(&i) = index.get(name) {
            rank_point[i] = (pos + 1) as f64;
        }
    }

    let mut score_samples: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut rank_samples: Vec<Vec<f64>> = vec![Vec::new(); n];

    for model in models {
        for (pos, (name, s)) in model.sorted_scores().into_iter().enumerate() {
            if let Some(&i) = index.get(name) {
                score_samples[i].push(s);
                rank_samples[i].push((pos + 1) as f64);
            }
        }
    }

    let tail = (1.0 - credible) / 2.0;
    let mut lo = score_point.clone();
    let mut hi = score_point.clone();
    let mut rank_lo = rank_point.clone();
    let mut rank_hi = rank_point.clone();
    let mut n_rep = vec![0u64; n];

    for (i, samples) in score_samples.iter_mut().enumerate() {
        if samples.is_empty() {
            // No surviving replicate emitted this entity: the interval
            // stays collapsed to the point estimate, n_rep stays 0.
            continue;
        }
        n_rep[i] = samples.len() as u64;
        samples.sort_unstable_by(f64::total_cmp);
        lo[i] = quantile(samples, tail);
        hi[i] = quantile(samples, 1.0 - tail);

        let ranks = &mut rank_samples[i];
        ranks.sort_unstable_by(f64::total_cmp);
        rank_lo[i] = quantile(ranks, tail);
        rank_hi[i] = quantile(ranks, 1.0 - tail);
    }

    Ok(BootstrapModel {
        params: BootstrapParams {
            inner_algorithm: point.algorithm().to_string(),
            replicates,
            replicates_ok: models.len(),
            credible,
            seed,
        },
        names: Interner::from_names(universe.iter().map(|&(name, _)| name))?,
        score_point,
        lo,
        hi,
        rank_point,
        rank_lo,
        rank_hi,
        n_rep,
    })
}

/// What the state file records about the run that produced the model. The
/// wrapped ranker itself does not persist — only its algorithm tag — so a
/// [`BootstrapModel`] loads without knowing the inner ranker's type.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BootstrapParams {
    /// `algorithm()` tag of the wrapped ranker's point model.
    inner_algorithm: String,
    /// Replicates attempted.
    replicates: usize,
    /// Replicates that fitted successfully.
    replicates_ok: usize,
    /// Central interval mass.
    credible: f64,
    /// Master seed.
    seed: u64,
}

/// One entity line: point score `s`, score interval, rank point and
/// interval, and replicate coverage.
#[derive(Debug, Serialize, Deserialize)]
struct BootstrapLine {
    id: String,
    s: f64,
    lo: f64,
    hi: f64,
    rank: f64,
    rank_lo: f64,
    rank_hi: f64,
    n_rep: u64,
}

/// Bootstrap intervals around an inner model's point estimates, entities in
/// name order. Self-contained: names, floats, and the inner algorithm tag
/// are all that persist.
#[derive(Debug, Clone)]
pub struct BootstrapModel {
    params: BootstrapParams,
    names: Interner,
    score_point: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    rank_point: Vec<f64>,
    rank_lo: Vec<f64>,
    rank_hi: Vec<f64>,
    n_rep: Vec<u64>,
}

impl BootstrapModel {
    /// `(name, point score, interval lo, interval hi)` per entity.
    pub fn intervals(&self) -> impl Iterator<Item = (&str, f64, f64, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, name)| (name, self.score_point[i], self.lo[i], self.hi[i]))
    }

    /// `(name, point rank, rank lo, rank hi)` per entity. Ranks are 1-based
    /// `sorted_scores` positions (1 = best, ties broken by name).
    pub fn rank_intervals(&self) -> impl Iterator<Item = (&str, f64, f64, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, name)| (name, self.rank_point[i], self.rank_lo[i], self.rank_hi[i]))
    }

    /// Per-entity coverage: how many surviving replicates emitted the
    /// entity. Zero means the intervals collapsed to the point estimate.
    pub fn coverage(&self) -> impl Iterator<Item = (&str, u64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, name)| (name, self.n_rep[i]))
    }

    /// Replicates that fitted successfully, out of the attempted
    /// `replicates` (failed ones were skipped).
    pub fn replicates_ok(&self) -> usize {
        self.params.replicates_ok
    }

    /// Algorithm tag of the wrapped ranker's point model.
    pub fn inner_algorithm(&self) -> &str {
        &self.params.inner_algorithm
    }
}

impl RankModel for BootstrapModel {
    fn algorithm(&self) -> &'static str {
        "bootstrap"
    }

    /// Point-fit scores (the wrapped model's estimates on the full data).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.score_point.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<BootstrapLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| BootstrapLine {
                id: id.to_string(),
                s: self.score_point[i],
                lo: self.lo[i],
                hi: self.hi[i],
                rank: self.rank_point[i],
                rank_lo: self.rank_lo[i],
                rank_hi: self.rank_hi[i],
                n_rep: self.n_rep[i],
            })
            .collect();
        state::save_model(w, "bootstrap", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (BootstrapParams, Vec<BootstrapLine>) =
            state::load_model(r, "bootstrap")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            score_point: lines.iter().map(|l| l.s).collect(),
            lo: lines.iter().map(|l| l.lo).collect(),
            hi: lines.iter().map(|l| l.hi).collect(),
            rank_point: lines.iter().map(|l| l.rank).collect(),
            rank_lo: lines.iter().map(|l| l.rank_lo).collect(),
            rank_hi: lines.iter().map(|l| l.rank_hi).collect(),
            n_rep: lines.iter().map(|l| l.n_rep).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::{BradleyTerryMM, Massey};
    use crate::dataset::PairwiseDataset;

    /// Minimal test model: a name/score table, never persisted.
    struct TinyModel {
        names: Vec<String>,
        scores: Vec<f64>,
    }

    impl RankModel for TinyModel {
        fn algorithm(&self) -> &'static str {
            "tiny-test"
        }

        fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
            self.names
                .iter()
                .map(String::as_str)
                .zip(self.scores.iter().copied())
        }

        fn save_jsonl<W: std::io::Write>(&self, _w: W) -> Result<()> {
            Err(Error::State("tiny-test models are not persisted".into()))
        }

        fn load_jsonl<R: std::io::BufRead>(_r: R) -> Result<Self> {
            Err(Error::State("tiny-test models are not persisted".into()))
        }
    }

    /// Scores every entity by win count, but errors whenever "c" has no
    /// rows — full control over which resamples "fail".
    struct NeedsC;

    impl Ranker for NeedsC {
        type Data = PairwiseDataset;
        type Model = TinyModel;

        fn fit_opts(&self, data: &PairwiseDataset, _o: &FitOptions<'_>) -> Result<TinyModel> {
            let c = data
                .interner()
                .get("c")
                .ok_or_else(|| Error::InvalidInput("fixture must intern c".into()))?;
            let t = data.tally();

            if t.wins[c as usize].0 + t.losses[c as usize].0 == 0 {
                return Err(Error::Numeric("entity c has no rows".into()));
            }
            Ok(TinyModel {
                names: data.interner().names().map(str::to_string).collect(),
                scores: t.wins.iter().map(|&(wins, _)| wins as f64).collect(),
            })
        }
    }

    /// Position-weighted row checksum: differs for any other draw order or
    /// multiset of rows (deterministically verified by the tests using it).
    fn checksum(d: &PairwiseDataset) -> f64 {
        d.rows()
            .enumerate()
            .map(|(i, (w, l, x))| {
                (i + 1) as f64 * (f64::from(w) * 97.0 + f64::from(l) * 89.0 + f64::from(x))
            })
            .sum()
    }

    /// Succeeds only on the exact row sequence it was built from — the
    /// point fit passes, every resample fails.
    struct OriginalOnly {
        checksum: f64,
    }

    impl Ranker for OriginalOnly {
        type Data = PairwiseDataset;
        type Model = TinyModel;

        fn fit_opts(&self, data: &PairwiseDataset, _o: &FitOptions<'_>) -> Result<TinyModel> {
            if checksum(data) != self.checksum {
                return Err(Error::Numeric(
                    "rows differ from the original sequence".into(),
                ));
            }
            Ok(TinyModel {
                names: vec!["a".into()],
                scores: vec![1.0],
            })
        }
    }

    /// BT comparisons sampled at the exact rates of strengths a=4, b=2, c=1
    /// (the rank_centrality fixture).
    fn bt_fixture() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 4.0);
        d.push("b", "a", 2.0);
        d.push("a", "c", 4.0);
        d.push("c", "a", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "b", 1.0);
        d
    }

    #[test]
    fn identical_rows_give_zero_width_intervals() {
        // Every resample of n identical rows is the identical dataset, so
        // every replicate score is bit-equal to the point fit. credible 0.5
        // over 9 samples puts both quantiles on whole positions (0.25 · 8 =
        // 2), so the asserts below are exact, not approximate.
        let mut d = PairwiseDataset::new();
        for _ in 0..8 {
            d.push("a", "b", 4.0);
        }

        let m = Bootstrap {
            replicates: 9,
            credible: 0.5,
            ..Bootstrap::new(Massey::default())
        }
        .fit(&d)
        .unwrap();

        assert_eq!(m.replicates_ok(), 9);
        assert_eq!(m.inner_algorithm(), "massey");

        for (name, s, lo, hi) in m.intervals() {
            assert_eq!(lo, s, "{name}");
            assert_eq!(hi, s, "{name}");
        }
        // Massey splits the margin: a = +2, b = -2.
        let scores: HashMap<&str, f64> = m.scores().collect();
        assert!((scores["a"] - 2.0).abs() < 1e-8);
        assert!((scores["b"] + 2.0).abs() < 1e-8);

        let ranks: HashMap<&str, (f64, f64, f64)> = m
            .rank_intervals()
            .map(|(name, r, lo, hi)| (name, (r, lo, hi)))
            .collect();
        assert_eq!(ranks["a"], (1.0, 1.0, 1.0));
        assert_eq!(ranks["b"], (2.0, 2.0, 2.0));
        assert!(m.coverage().all(|(_, n_rep)| n_rep == 9));
    }

    #[test]
    fn bt_intervals_are_ordered_and_rank_ci_covers_the_truth() {
        // create_fake_games keeps undefeated entities ranked on the same
        // normalized scale, so replicate scores stay comparable even when a
        // resample makes some entity one-sided.
        let inner = BradleyTerryMM {
            create_fake_games: 0.1,
            ..Default::default()
        };
        let m = Bootstrap {
            replicates: 64,
            ..Bootstrap::new(inner)
        }
        .fit(&bt_fixture())
        .unwrap();

        assert_eq!(m.replicates_ok(), 64);

        for (name, s, lo, hi) in m.intervals() {
            assert!(lo <= s && s <= hi, "{name}: [{lo}, {hi}] vs {s}");
        }

        let ranks: HashMap<&str, (f64, f64, f64)> = m
            .rank_intervals()
            .map(|(name, r, lo, hi)| (name, (r, lo, hi)))
            .collect();
        let (r, lo, hi) = ranks["a"];
        assert_eq!(r, 1.0, "the strongest entity points at rank 1");
        assert!(lo <= 1.0 && 1.0 <= hi, "rank CI [{lo}, {hi}] misses 1");
    }

    #[test]
    fn replicate_failures_are_skipped_and_counted() {
        // c appears in 1 of 3 rows, so a resample misses c with probability
        // (2/3)^3 ≈ 0.30 per replicate. Which of the 16 replicates fail is
        // fully pinned by (seed 26, k); the count below is that fixed
        // outcome, asserted exactly so any determinism regression surfaces.
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);

        let m = Bootstrap {
            replicates: 16,
            ..Bootstrap::new(NeedsC)
        }
        .fit(&d)
        .unwrap();

        assert!(m.replicates_ok() < 16, "some replicates must fail");
        assert!(m.replicates_ok() * 2 >= 16, "a majority must survive");
        assert_eq!(m.replicates_ok(), 9);

        // the model still emits the full universe, with full coverage for c
        // (every surviving replicate has c rows by construction)
        let coverage: HashMap<&str, u64> = m.coverage().collect();
        assert_eq!(coverage.len(), 3);
        assert_eq!(coverage["c"], 9);
        assert_eq!(m.inner_algorithm(), "tiny-test");
    }

    #[test]
    fn majority_failure_is_a_numeric_error_naming_the_cause() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "a", 3.0);

        let inner = OriginalOnly {
            checksum: checksum(&d),
        };
        let err = Bootstrap {
            replicates: 4,
            ..Bootstrap::new(inner)
        }
        .fit(&d)
        .unwrap_err();

        assert!(matches!(err, Error::Numeric(_)), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("of 4 bootstrap replicates"), "{msg}");
        assert!(
            msg.contains("rows differ from the original sequence"),
            "{msg}"
        );
    }

    #[test]
    fn point_fit_errors_propagate() {
        // Massey rejects negative margins on the point fit itself.
        let mut d = PairwiseDataset::new();
        d.push("a", "b", -1.0);
        let err = Bootstrap::new(Massey::default()).fit(&d).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    }

    #[test]
    fn parameters_are_validated() {
        let d = bt_fixture();

        let zero = Bootstrap {
            replicates: 0,
            ..Bootstrap::new(Massey::default())
        };
        assert!(matches!(zero.fit(&d), Err(Error::InvalidInput(_))));

        for credible in [0.0, 1.0, -0.5, f64::NAN] {
            let bad = Bootstrap {
                credible,
                ..Bootstrap::new(Massey::default())
            };
            assert!(
                matches!(bad.fit(&d), Err(Error::InvalidInput(_))),
                "credible {credible} must be rejected"
            );
        }
    }

    #[test]
    fn round_trip_is_byte_identical_and_runs_are_bit_stable() {
        let b = Bootstrap {
            replicates: 32,
            ..Bootstrap::new(BradleyTerryMM {
                create_fake_games: 0.1,
                ..Default::default()
            })
        };
        let d = bt_fixture();

        let bytes = |m: &BootstrapModel| -> Vec<u8> {
            let mut v = Vec::new();
            m.save_jsonl(&mut v).unwrap();
            v
        };

        let first = bytes(&b.fit(&d).unwrap());
        let second = bytes(&b.fit(&d).unwrap());
        assert_eq!(first, second, "repeat fits are bit-stable");

        #[cfg(feature = "parallel")]
        {
            for threads in [1, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let opts = FitOptions {
                    threading: crate::Threading::Dedicated(&pool),
                    ..FitOptions::default()
                };
                assert_eq!(
                    first,
                    bytes(&b.fit_opts(&d, &opts).unwrap()),
                    "bit-stable on a {threads}-thread pool"
                );
            }
        }

        let loaded = BootstrapModel::load_jsonl(first.as_slice()).unwrap();
        assert_eq!(first, bytes(&loaded), "save -> load -> save is identical");
        assert_eq!(loaded.inner_algorithm(), "btm-mm");
        assert_eq!(loaded.replicates_ok(), 32);

        // a foreign algorithm tag is rejected
        let mut massey_file = Vec::new();
        Massey::default()
            .fit(&{
                let mut d = PairwiseDataset::new();
                d.push("a", "b", 1.0);
                d
            })
            .unwrap()
            .save_jsonl(&mut massey_file)
            .unwrap();
        assert!(matches!(
            BootstrapModel::load_jsonl(massey_file.as_slice()),
            Err(Error::AlgorithmMismatch { .. })
        ));
    }
}
