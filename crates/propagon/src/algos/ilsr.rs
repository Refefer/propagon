//! I-LSR — iterative Luce Spectral Ranking (`docs/algorithms.md` §3.2;
//! Maystre & Grossglauser, NIPS 2015).
//!
//! Where `lsr` takes one spectral shot at the Plackett-Luce model, I-LSR
//! iterates it to the exact maximum-likelihood fixed point: each outer pass
//! rebuilds the Markov chain with every choice event's transitions weighted
//! by `1/Σ π_k` over the event's choice set under the *current* estimate,
//! then re-solves the stationary distribution warm-started from the previous
//! one. At the fixed point π is the PL MLE (paper, Theorem 3), so the output
//! agrees with `bradley-terry-model` on pairwise data and with
//! `rankings plackett-luce` on ballots — at spectral speed.
//!
//! Consumes pairwise outcomes via [`Ranker`], or rankings natively via
//! [`ILsr::fit_rankings_opts`] (a ballot of m items is m−1 cascading choice
//! events; likelihood-exact, no pair explosion).
//!
//! Assumes Ford connectivity, like every prior-free PL fitter: an entity
//! that never wins (or never loses) an event has no finite MLE, which here
//! surfaces as a typed error up front rather than silent divergence.
//! Mitigations: priors (`bayesian-bradley-terry`) or the bridging options on
//! `bradley-terry-model`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{PairwiseDataset, RankingsDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, RankModel as _, Ranker};

/// I-LSR parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ILsr {
    /// Maximum outer (chain-rebuild) iterations.
    pub outer: usize,
    /// Power-iteration passes per stationary solve.
    pub inner_steps: usize,
    /// Outer convergence: stop when the L1 change of π drops below this.
    pub tolerance: f64,
}

impl Default for ILsr {
    fn default() -> Self {
        Self {
            outer: 100,
            inner_steps: 50,
            tolerance: 1e-10,
        }
    }
}

/// Fitted PL strengths (log-scale, mean-centered; higher is better).
#[derive(Debug, Clone)]
pub struct ILsrModel {
    params: ILsr,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(ILsrModel, "ilsr");

/// Choice events in flat storage: event `e` has winner `winners[e]`, losers
/// `losers[loser_offsets[e]..loser_offsets[e+1]]`, multiplicity `weights[e]`.
struct Events {
    winners: Vec<u32>,
    losers: Vec<u32>,
    loser_offsets: Vec<usize>,
    weights: Vec<f64>,
    n: usize,
}

impl Events {
    fn from_pairwise(data: &PairwiseDataset) -> Self {
        // Aggregate per ordered pair; deterministic event order via sort.
        let mut agg: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, x) in data.rows() {
            *agg.entry((w, l)).or_default() += f64::from(x);
        }
        let mut pairs: Vec<((u32, u32), f64)> = agg.into_iter().collect();
        pairs.sort_unstable_by_key(|&(k, _)| k);

        let mut ev = Events {
            winners: Vec::with_capacity(pairs.len()),
            losers: Vec::with_capacity(pairs.len()),
            loser_offsets: vec![0],
            weights: Vec::with_capacity(pairs.len()),
            n: data.n_entities(),
        };
        for ((w, l), x) in pairs {
            ev.winners.push(w);
            ev.losers.push(l);
            ev.loser_offsets.push(ev.losers.len());
            ev.weights.push(x);
        }
        ev
    }

    fn from_rankings(data: &RankingsDataset) -> Self {
        let mut ev = Events {
            winners: Vec::new(),
            losers: Vec::new(),
            loser_offsets: vec![0],
            weights: Vec::new(),
            n: data.n_entities(),
        };
        for ballot in data.rankings() {
            // Best-first cascade: position r wins against everything after it.
            for r in 0..ballot.len().saturating_sub(1) {
                ev.winners.push(ballot[r]);
                ev.losers.extend_from_slice(&ballot[r + 1..]);
                ev.loser_offsets.push(ev.losers.len());
                ev.weights.push(1.0);
            }
        }
        ev
    }

    fn losers_of(&self, e: usize) -> &[u32] {
        &self.losers[self.loser_offsets[e]..self.loser_offsets[e + 1]]
    }

    /// Ford pre-check (the cheap necessary half): every entity must win and
    /// lose at least once, else its MLE diverges to ±∞.
    fn check_ford(&self, names: &Interner) -> Result<()> {
        let mut won = vec![false; self.n];
        let mut lost = vec![false; self.n];
        for e in 0..self.winners.len() {
            won[self.winners[e] as usize] = true;
            for &l in self.losers_of(e) {
                lost[l as usize] = true;
            }
        }
        let offenders: Vec<&str> = (0..self.n)
            .filter(|&i| !won[i] || !lost[i])
            .filter_map(|i| names.name(i as u32))
            .take(5)
            .collect();
        if offenders.is_empty() {
            Ok(())
        } else {
            Err(Error::Numeric(format!(
                "PL MLE diverges: {} never win or never lose a choice event \
                 (use bayesian-bradley-terry, or bradley-terry-model's bridging options)",
                offenders.join(", ")
            )))
        }
    }
}

impl ILsr {
    /// Fits PL strengths from full or partial rankings (each ballot is a
    /// cascade of choice events — exact, no pairwise explosion).
    pub fn fit_rankings_opts(
        &self,
        data: &RankingsDataset,
        opts: &FitOptions<'_>,
    ) -> Result<ILsrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let events = Events::from_rankings(data);
        self.solve(events, data.interner().clone(), None, opts)
    }

    /// Convenience wrapper over [`ILsr::fit_rankings_opts`].
    pub fn fit_rankings(&self, data: &RankingsDataset) -> Result<ILsrModel> {
        self.fit_rankings_opts(data, &FitOptions::default())
    }

    fn solve(
        &self,
        events: Events,
        names: Interner,
        init: Option<Vec<f64>>,
        opts: &FitOptions<'_>,
    ) -> Result<ILsrModel> {
        events.check_ford(&names)?;
        let n = events.n;

        let scores = parallel::run_scoped(opts, || {
            let progress = opts.progress;
            progress.start("i-lsr outer passes", Some(self.outer as u64));

            let mut pi = init.unwrap_or_else(|| vec![1.0 / n as f64; n]);
            for pass in 0..self.outer {
                // Rebuild the chain under the current estimate: each event
                // moves mass loser → winner at rate weight / Σ π over the
                // event's choice set.
                let mut incoming: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
                let mut outflow = vec![0.0f64; n];
                for e in 0..events.winners.len() {
                    let w = events.winners[e];
                    let losers = events.losers_of(e);
                    let denom: f64 =
                        pi[w as usize] + losers.iter().map(|&l| pi[l as usize]).sum::<f64>();
                    let rate = events.weights[e] / denom;
                    for &l in losers {
                        incoming[w as usize].push((l, rate));
                        outflow[l as usize] += rate;
                    }
                }
                // Uniformize the continuous-time chain into a discrete one.
                let q_max = outflow.iter().copied().fold(f64::MIN, f64::max).max(1e-300);

                // Stationary solve, warm-started from the previous π.
                let mut stat = pi.clone();
                for _ in 0..self.inner_steps {
                    let frozen = &stat;
                    let next = parallel::par_map_indexed(n, |j| {
                        let inflow: f64 = incoming[j]
                            .iter()
                            .map(|&(i, r)| frozen[i as usize] * r)
                            .sum();
                        frozen[j] * (1.0 - outflow[j] / q_max) + inflow / q_max
                    });
                    stat = next;
                }
                let total: f64 = stat.iter().sum();
                stat.iter_mut().for_each(|v| *v /= total);

                let delta: f64 = pi.iter().zip(&stat).map(|(a, b)| (a - b).abs()).sum();
                pi = stat;
                progress.update(pass as u64 + 1);
                if delta < self.tolerance {
                    break;
                }
            }
            progress.finish();

            // Log scale, mean-centered (the family's output convention).
            let mut s: Vec<f64> = pi.iter().map(|&p| p.max(1e-300).ln()).collect();
            let avg = s.iter().sum::<f64>() / n as f64;
            s.iter_mut().for_each(|v| *v -= avg);
            s
        });

        if scores.iter().any(|s| !s.is_finite()) {
            return Err(Error::Numeric(
                "i-lsr produced non-finite strengths; the comparison graph is \
                 likely disconnected"
                    .into(),
            ));
        }

        Ok(ILsrModel {
            params: *self,
            names,
            scores,
        })
    }
}

impl Ranker for ILsr {
    type Data = PairwiseDataset;
    type Model = ILsrModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<ILsrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let events = Events::from_pairwise(data);
        self.solve(events, data.interner().clone(), None, opts)
    }

    fn fit_warm_opts(
        &self,
        data: &PairwiseDataset,
        init: &ILsrModel,
        opts: &FitOptions<'_>,
    ) -> Result<ILsrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = data.n_entities();
        let mut pi = vec![1.0 / n as f64; n];
        for (name, s) in init.scores() {
            if let Some(id) = data.interner().get(name) {
                pi[id as usize] = s.exp();
            }
        }
        let total: f64 = pi.iter().sum();
        pi.iter_mut().for_each(|v| *v /= total);

        let events = Events::from_pairwise(data);
        self.solve(events, data.interner().clone(), Some(pi), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;
    use crate::algos::{BradleyTerryMM, PlackettLuce};

    /// Two entities, a beats b 3:1: the PL/BT MLE has π_a/π_b = 3 exactly.
    #[test]
    fn two_entity_closed_form() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        let m = ILsr::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!(
            ((s["a"] - s["b"]).exp() - 3.0).abs() < 1e-6,
            "ratio {}",
            (s["a"] - s["b"]).exp()
        );
    }

    /// I-LSR converges to the same MLE as Hunter's MM on pairwise data
    /// (both maximize the identical likelihood).
    ///
    /// Every played pair is bidirectional because `btm-mm` preserves a v1
    /// denominator quirk (sums over *beaten* opponents only) that shifts
    /// its fixed point off the true MLE when an entity never beats someone
    /// it played.
    #[test]
    fn agrees_with_bradley_terry_mm() {
        let mut d = PairwiseDataset::new();
        for (w, l, x) in [
            ("a", "b", 5.0),
            ("b", "a", 2.0),
            ("a", "c", 4.0),
            ("c", "a", 3.0),
            ("b", "c", 6.0),
            ("c", "b", 2.0),
            ("c", "d", 3.0),
            ("d", "c", 1.0),
            ("d", "a", 1.0),
            ("a", "d", 2.0),
            ("b", "d", 2.0),
            ("d", "b", 1.0),
        ] {
            d.push(w, l, x);
        }
        let mm = BradleyTerryMM {
            iterations: 20_000,
            tolerance: 1e-13,
            ..BradleyTerryMM::default()
        }
        .fit(&d)
        .unwrap();
        let ilsr = ILsr::default().fit(&d).unwrap();

        let mm_s: std::collections::HashMap<_, _> = mm.scores().collect();
        let il_s: std::collections::HashMap<_, _> = ilsr.scores().collect();
        // MM emits linear strengths π, I-LSR log strengths; compare log gaps.
        for pair in [("a", "b"), ("a", "c"), ("b", "d")] {
            let gap_mm = (mm_s[pair.0] / mm_s[pair.1]).ln();
            let gap_il = il_s[pair.0] - il_s[pair.1];
            assert!(
                (gap_mm - gap_il).abs() < 1e-4,
                "{pair:?}: mm {gap_mm} vs ilsr {gap_il}"
            );
        }
    }

    /// On ballots, I-LSR matches the Plackett-Luce MM fit (same MLE).
    #[test]
    fn agrees_with_plackett_luce_on_rankings() {
        let mut d = RankingsDataset::new();
        for ballot in [
            vec!["a", "b", "c"],
            vec!["a", "c", "b"],
            vec!["b", "a", "c"],
            vec!["c", "b", "a"],
            vec!["a", "b", "c"],
            vec!["b", "c", "a"],
        ] {
            d.push_ranking(ballot.iter().copied()).unwrap();
        }
        let pl = PlackettLuce {
            iterations: 20_000,
            tolerance: 1e-13,
        }
        .fit(&d)
        .unwrap();
        let ilsr = ILsr::default().fit_rankings(&d).unwrap();

        let pl_s: std::collections::HashMap<_, _> = pl.scores().collect();
        let il_s: std::collections::HashMap<_, _> = ilsr.scores().collect();
        for pair in [("a", "b"), ("b", "c")] {
            // PL MM emits strengths π (linear scale); compare log gaps.
            let gap_pl = (pl_s[pair.0] / pl_s[pair.1]).ln();
            let gap_il = il_s[pair.0] - il_s[pair.1];
            assert!(
                (gap_pl - gap_il).abs() < 1e-4,
                "{pair:?}: pl {gap_pl} vs ilsr {gap_il}"
            );
        }
    }

    #[test]
    fn never_winning_entity_is_a_typed_error() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "a", 1.0);
        d.push("a", "doormat", 3.0);
        let err = ILsr::default().fit(&d).unwrap_err();
        assert!(matches!(err, Error::Numeric(m) if m.contains("doormat")));
    }

    #[test]
    fn empty_dataset_is_an_error() {
        assert!(matches!(
            ILsr::default().fit(&PairwiseDataset::new()),
            Err(Error::EmptyDataset)
        ));
        assert!(matches!(
            ILsr::default().fit_rankings(&RankingsDataset::new()),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn deterministic_across_runs() {
        let mut d = PairwiseDataset::new();
        for i in 0..20u32 {
            for j in 0..20u32 {
                if i != j {
                    d.push(&i.to_string(), &j.to_string(), 1.0 + (i % 3) as f32);
                }
            }
        }
        let a = ILsr::default().fit(&d).unwrap();
        let b = ILsr::default().fit(&d).unwrap();
        let sa: Vec<f64> = a.scores().map(|(_, s)| s).collect();
        let sb: Vec<f64> = b.scores().map(|(_, s)| s).collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn warm_start_matches_cold_fit() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "a", 1.0);
        let cold = ILsr::default().fit(&d).unwrap();
        let warm = ILsr::default().fit_warm(&d, &cold).unwrap();
        for ((n1, s1), (n2, s2)) in cold.sorted_scores().iter().zip(warm.sorted_scores()) {
            assert_eq!(*n1, n2);
            assert!((s1 - s2).abs() < 1e-8, "{n1}: {s1} vs {s2}");
        }
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.push("b", "a", 1.0);
        let m = ILsr::default().fit(&d).unwrap();
        let mut buf1 = Vec::new();
        m.save_jsonl(&mut buf1).unwrap();
        let m2 = ILsrModel::load_jsonl(buf1.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf1, buf2);
    }
}
