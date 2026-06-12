//! Whole-History Rating (`docs/algorithms.md` §2.6; Coulom 2008,
//! "Whole-History Rating: A Bayesian Rating System for Players of
//! Time-Varying Strength", incl. Appendix B).
//!
//! Bradley-Terry likelihood at every game plus a Wiener-process prior along
//! each player's timeline; the joint MAP over all (player, day) ratings is
//! found by Newton sweeps that solve each player's tridiagonal system exactly
//! (LU forward elimination + back substitution, Appendix B). Dataset
//! **periods** are the time axis: period index = integer time `t`, a player's
//! "days" are the periods it appears in, and the Wiener variance between
//! consecutive days is `w2 · Δt`. Row weight = game multiplicity.
//!
//! Ratings are fitted in natural log-odds units and reported on the Elo
//! display scale (Elo ≈ 173.7178 · natural, so the default `w2 = 0.0006`
//! is a drift of about (4.25 Elo)² per period). The paper leaves the global
//! translation unspecified; following the reference implementations, each
//! player's first active day carries one virtual **draw** (weight
//! `prior_games`) against a fixed rating-0 anchor, which pins the gauge and
//! keeps undefeated/winless players finite.
//!
//! Gotchas: the per-day σ comes from each player's own tridiagonal Hessian
//! block, i.e. it is conditional on opponents' ratings being held fixed —
//! Coulom's convention, not the full joint posterior. Entities interned in
//! the dataset without any rows get no timeline and are absent from the
//! model. A single-period dataset degrades to a static anchored
//! Bradley-Terry fit (no Wiener terms).

use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Elo display points per natural log-odds unit (400 / ln 10, the same
/// constant the Glicko family uses).
const ELO_SCALE: f64 = 173.7178;

/// Coulom's diagonal stabilizer: keeps the Hessian strictly negative even on
/// days whose game terms contribute numerically zero curvature.
const HESSIAN_EPSILON: f64 = 1e-3;

/// Whole-History Rating parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Whr {
    /// Wiener-process variance per period step, in squared natural
    /// (log-odds) units. Elo ≈ 173.7178 · natural, so the default `0.0006`
    /// allows a drift of about 4.25 Elo per √period.
    pub w2: f64,
    /// Weight of the single virtual draw against a fixed rating-0 anchor
    /// attached to each player's first active period. Pins the global
    /// translation and keeps undefeated/winless players finite (the
    /// reference implementations' convention — the paper leaves anchoring
    /// unspecified).
    pub prior_games: f64,
    /// Maximum Newton sweeps over all players.
    pub iterations: usize,
    /// Stop when the largest |Δ rating| applied in a sweep drops below this
    /// (natural units).
    pub tolerance: f64,
}

impl Default for Whr {
    fn default() -> Self {
        Self {
            w2: 0.0006,
            prior_games: 1.0,
            iterations: 200,
            tolerance: 1e-5,
        }
    }
}

impl Whr {
    /// Rejects parameter values the fitter cannot work with.
    fn validate(&self) -> Result<()> {
        if !self.w2.is_finite() || self.w2 <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "whr needs finite w2 > 0, got {}",
                self.w2
            )));
        }
        if !self.prior_games.is_finite() || self.prior_games < 0.0 {
            return Err(Error::InvalidInput(format!(
                "whr needs finite prior_games >= 0, got {}",
                self.prior_games
            )));
        }
        Ok(())
    }

    /// Shared fit path: builds the timeline layout, seeds ratings (warm or
    /// zero), runs Newton sweeps to the tolerance, then extracts per-day
    /// standard deviations and assembles the Elo-scale model.
    fn fit_inner(
        &self,
        data: &PairwiseDataset,
        warm: Option<&WhrModel>,
        opts: &FitOptions<'_>,
    ) -> Result<WhrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        self.validate()?;

        let layout = Layout::build(data, self.prior_games);
        let mut ratings = vec![0.0f64; layout.n_days()];

        if let Some(init) = warm {
            seed_from(init, data, &layout, &mut ratings);
        }

        let progress = opts.progress;
        progress.start("whr sweeps", Some(self.iterations as u64));
        let scratch = layout.max_days();
        let mut diag = vec![0.0f64; scratch];
        let mut rhs = vec![0.0f64; scratch];

        for it in 0..self.iterations {
            let delta = layout.sweep(self.w2, &mut ratings, &mut diag, &mut rhs);
            progress.update(it as u64 + 1);

            if it % 10 == 0 {
                progress.message(&format!("max |Δr| {delta:0.3e}"));
            }

            if delta < self.tolerance {
                break;
            }
        }
        progress.finish();

        let sds = layout.day_sds(self.w2, &ratings);

        let mut names = Interner::new();
        let mut offsets = vec![0usize];
        let mut periods = Vec::with_capacity(layout.n_days());
        let mut r_elo = Vec::with_capacity(layout.n_days());
        let mut sd_elo = Vec::with_capacity(layout.n_days());

        for p in 0..layout.n_players() {
            let (lo, len) = layout.player_span(p);
            if len == 0 {
                // interned but never compared (e.g. after filter_min_count):
                // no timeline, so no model line.
                continue;
            }
            names.intern(data.interner().resolve(p as u32));
            periods.extend_from_slice(&layout.day_periods[lo..lo + len]);
            r_elo.extend(ratings[lo..lo + len].iter().map(|r| r * ELO_SCALE));
            sd_elo.extend(sds[lo..lo + len].iter().map(|s| s * ELO_SCALE));
            offsets.push(periods.len());
        }

        Ok(WhrModel {
            params: *self,
            names,
            offsets,
            periods,
            ratings: r_elo,
            sds: sd_elo,
        })
    }
}

impl Ranker for Whr {
    type Data = PairwiseDataset;
    type Model = WhrModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<WhrModel> {
        self.fit_inner(data, None, opts)
    }

    /// Seeds each (player, period) rating from `init`'s timelines matched by
    /// name: matching periods copy directly, new periods copy the nearest old
    /// day, unseen players start at 0. The normal sweeps then run to the same
    /// tolerance, so the result is never worse than a cold fit.
    fn fit_warm_opts(
        &self,
        data: &PairwiseDataset,
        init: &WhrModel,
        opts: &FitOptions<'_>,
    ) -> Result<WhrModel> {
        self.fit_inner(data, Some(init), opts)
    }
}

/// Whose rating a game term is evaluated against.
#[derive(Clone, Copy, Debug)]
enum Opponent {
    /// The virtual rating-0 anchor (the prior draw on a first day).
    Anchor,
    /// A real opponent at one of its own timeline days.
    Player { player: u32, day: u32 },
}

/// One game's contribution to a (player, day) log-likelihood.
#[derive(Clone, Copy, Debug)]
struct GameTerm {
    opponent: Opponent,
    /// 1.0 = the owning player won, 0.0 = lost, 0.5 = the anchor draw.
    share: f64,
    weight: f64,
}

/// Flat per-player timeline layout: every (player, active period) pair is a
/// "day" with a global index, and each day owns the game terms evaluated at
/// it (each dataset row is stored on **both** participants' timelines).
struct Layout {
    /// Player `p`'s days are global indices `day_offsets[p]..day_offsets[p+1]`.
    day_offsets: Vec<usize>,
    /// Period index of each global day, ascending within a player.
    day_periods: Vec<u32>,
    /// Global day `g`'s terms are `terms[term_offsets[g]..term_offsets[g+1]]`.
    term_offsets: Vec<usize>,
    terms: Vec<GameTerm>,
}

impl Layout {
    /// Two-pass construction: pass 1 collects each player's sorted active
    /// periods; pass 2 attaches the anchor draw to every first day and each
    /// row to both participants' (player, period) days.
    fn build(data: &PairwiseDataset, prior_games: f64) -> Layout {
        let n = data.n_entities();
        let mut periods_per: Vec<Vec<u32>> = vec![Vec::new(); n];

        for (t, range) in data.periods().enumerate() {
            let t = t as u32;
            for (w, l, _) in data.period_rows(range) {
                for p in [w, l] {
                    let active = &mut periods_per[p as usize];
                    if active.last() != Some(&t) {
                        active.push(t);
                    }
                }
            }
        }

        let mut day_offsets = Vec::with_capacity(n + 1);
        day_offsets.push(0usize);
        let mut day_periods = Vec::new();

        for active in &periods_per {
            day_periods.extend_from_slice(active);
            day_offsets.push(day_periods.len());
        }

        let mut terms_per: Vec<Vec<GameTerm>> = vec![Vec::new(); day_periods.len()];

        for p in 0..n {
            if day_offsets[p] < day_offsets[p + 1] {
                terms_per[day_offsets[p]].push(GameTerm {
                    opponent: Opponent::Anchor,
                    share: 0.5,
                    weight: prior_games,
                });
            }
        }

        // Err is unreachable: pass 1 recorded t for every row endpoint.
        let day_of = |p: u32, t: u32| -> u32 {
            periods_per[p as usize]
                .binary_search(&t)
                .unwrap_or_else(|i| i) as u32
        };

        for (t, range) in data.periods().enumerate() {
            let t = t as u32;
            for (w, l, x) in data.period_rows(range) {
                let (dw, dl) = (day_of(w, t), day_of(l, t));
                let weight = f64::from(x);
                terms_per[day_offsets[w as usize] + dw as usize].push(GameTerm {
                    opponent: Opponent::Player { player: l, day: dl },
                    share: 1.0,
                    weight,
                });
                terms_per[day_offsets[l as usize] + dl as usize].push(GameTerm {
                    opponent: Opponent::Player { player: w, day: dw },
                    share: 0.0,
                    weight,
                });
            }
        }

        let mut term_offsets = Vec::with_capacity(day_periods.len() + 1);
        term_offsets.push(0usize);
        let mut terms = Vec::new();

        for day_terms in terms_per {
            terms.extend(day_terms);
            term_offsets.push(terms.len());
        }

        Layout {
            day_offsets,
            day_periods,
            term_offsets,
            terms,
        }
    }

    fn n_players(&self) -> usize {
        self.day_offsets.len() - 1
    }

    fn n_days(&self) -> usize {
        self.day_periods.len()
    }

    /// `(first global day, day count)` for player `p`.
    fn player_span(&self, p: usize) -> (usize, usize) {
        let lo = self.day_offsets[p];
        (lo, self.day_offsets[p + 1] - lo)
    }

    fn max_days(&self) -> usize {
        (0..self.n_players())
            .map(|p| self.player_span(p).1)
            .max()
            .unwrap_or(0)
    }

    /// Period gap between global day `g` and `g + 1` of the same player.
    fn gap(&self, g: usize) -> f64 {
        f64::from(self.day_periods[g + 1] - self.day_periods[g])
    }

    /// Gradient and Hessian diagonal of the log-posterior at local day `k`
    /// of the player whose span starts at `lo` with `len` days: game terms
    /// `Σ w·(share − σ)` / `−Σ w·σ(1−σ)`, Wiener pulls toward the
    /// neighbouring days, and the `−ε` stabilizer on the diagonal.
    fn day_grad_diag(
        &self,
        ratings: &[f64],
        w2: f64,
        lo: usize,
        len: usize,
        k: usize,
    ) -> (f64, f64) {
        let g_idx = lo + k;
        let r = ratings[g_idx];
        let mut grad = 0.0;
        let mut diag = -HESSIAN_EPSILON;

        for term in &self.terms[self.term_offsets[g_idx]..self.term_offsets[g_idx + 1]] {
            let r_opp = match term.opponent {
                Opponent::Anchor => 0.0,
                Opponent::Player { player, day } => {
                    ratings[self.day_offsets[player as usize] + day as usize]
                }
            };
            let s = sigmoid(r - r_opp);
            grad += term.weight * (term.share - s);
            diag -= term.weight * s * (1.0 - s);
        }

        if k > 0 {
            let v = w2 * self.gap(g_idx - 1);
            grad -= (r - ratings[g_idx - 1]) / v;
            diag -= 1.0 / v;
        }

        if k + 1 < len {
            let v = w2 * self.gap(g_idx);
            grad -= (r - ratings[g_idx + 1]) / v;
            diag -= 1.0 / v;
        }

        (grad, diag)
    }

    /// One Gauss-Seidel pass: for each player in id order, solves its
    /// tridiagonal Newton system `H·Δ = −g` exactly (LU forward elimination
    /// then back substitution, Coulom Appendix B) and applies `Δ` in place.
    /// `diag`/`rhs` are caller-provided scratch sized to the longest
    /// timeline. Returns the largest |Δ| applied.
    fn sweep(&self, w2: f64, ratings: &mut [f64], diag: &mut [f64], rhs: &mut [f64]) -> f64 {
        let mut max_delta = 0.0f64;

        for p in 0..self.n_players() {
            let (lo, len) = self.player_span(p);
            if len == 0 {
                continue;
            }

            for k in 0..len {
                let (g, h) = self.day_grad_diag(ratings, w2, lo, len, k);
                rhs[k] = -g;
                diag[k] = h;
            }

            // Forward elimination of the subdiagonal: diag becomes U's
            // diagonal, rhs becomes y in L·y = −g.
            for k in 1..len {
                let b = 1.0 / (w2 * self.gap(lo + k - 1));
                let a = b / diag[k - 1];
                diag[k] -= a * b;
                rhs[k] -= a * rhs[k - 1];
            }

            // Back substitution against U: rhs becomes Δ.
            rhs[len - 1] /= diag[len - 1];
            for k in (0..len - 1).rev() {
                let b = 1.0 / (w2 * self.gap(lo + k));
                rhs[k] = (rhs[k] - b * rhs[k + 1]) / diag[k];
            }

            for k in 0..len {
                ratings[lo + k] += rhs[k];
                max_delta = max_delta.max(rhs[k].abs());
            }
        }

        max_delta
    }

    /// Per-day posterior standard deviations (natural scale): the diagonal
    /// of |H⁻¹| per player, from Appendix B's pair of recurrences — a
    /// forward LU pass (`fwd`) and a backward UL pass (`bwd`) over the same
    /// tridiagonal Hessian, combined as
    /// `var_k = bwd[k+1] / (b_k² − fwd[k]·bwd[k+1])` (last day: `−1/fwd`).
    fn day_sds(&self, w2: f64, ratings: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n_days()];
        let mut diag: Vec<f64> = Vec::new();
        let mut fwd: Vec<f64> = Vec::new();
        let mut bwd: Vec<f64> = Vec::new();

        for p in 0..self.n_players() {
            let (lo, len) = self.player_span(p);
            if len == 0 {
                continue;
            }

            diag.clear();
            diag.extend((0..len).map(|k| self.day_grad_diag(ratings, w2, lo, len, k).1));

            fwd.clear();
            fwd.extend_from_slice(&diag);
            for k in 1..len {
                let b = 1.0 / (w2 * self.gap(lo + k - 1));
                fwd[k] = diag[k] - b * b / fwd[k - 1];
            }

            bwd.clear();
            bwd.extend_from_slice(&diag);
            for k in (0..len - 1).rev() {
                let b = 1.0 / (w2 * self.gap(lo + k));
                bwd[k] = diag[k] - b * b / bwd[k + 1];
            }

            for k in 0..len {
                let var = if k + 1 < len {
                    let b = 1.0 / (w2 * self.gap(lo + k));
                    bwd[k + 1] / (b * b - fwd[k] * bwd[k + 1])
                } else {
                    -1.0 / fwd[k]
                };
                out[lo + k] = var.abs().sqrt();
            }
        }

        out
    }
}

/// Copies natural-scale seed ratings out of a previous model, matched by
/// (name, period): exact period matches copy directly, new periods copy the
/// nearest old day (ties prefer the earlier one), unknown players stay 0.
fn seed_from(init: &WhrModel, data: &PairwiseDataset, layout: &Layout, ratings: &mut [f64]) {
    for p in 0..layout.n_players() {
        let (lo, len) = layout.player_span(p);
        if len == 0 {
            continue;
        }

        let Some(name) = data.interner().name(p as u32) else {
            continue;
        };
        let Some((old_t, old_r, _)) = init.timeline(name) else {
            continue;
        };
        if old_t.is_empty() {
            // Defensive: timelines are non-empty by construction.
            continue;
        }

        let span = lo..lo + len;
        for (slot, &t) in ratings[span.clone()]
            .iter_mut()
            .zip(&layout.day_periods[span])
        {
            let idx = match old_t.binary_search(&t) {
                Ok(i) => i,
                Err(0) => 0,
                Err(i) if i == old_t.len() => old_t.len() - 1,
                Err(i) => {
                    // Strictly between two old days; t > old_t[i-1] holds.
                    if t - old_t[i - 1] <= old_t[i] - t {
                        i - 1
                    } else {
                        i
                    }
                }
            };
            *slot = old_r[idx] / ELO_SCALE;
        }
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One player's timeline in the state file.
#[derive(Debug, Serialize, Deserialize)]
struct TimelineLine {
    id: String,
    /// Period indices of the active days, strictly ascending.
    t: Vec<u32>,
    /// Ratings per day, Elo scale.
    r: Vec<f64>,
    /// Standard deviations per day, Elo scale.
    sd: Vec<f64>,
}

/// Fitted whole-history timelines on the Elo display scale.
#[derive(Debug, Clone)]
pub struct WhrModel {
    params: Whr,
    names: Interner,
    /// Player `i`'s days live at `offsets[i]..offsets[i+1]` in the three
    /// parallel vecs below; every player has at least one day.
    offsets: Vec<usize>,
    periods: Vec<u32>,
    ratings: Vec<f64>,
    sds: Vec<f64>,
}

impl WhrModel {
    /// A player's full timeline as `(periods, ratings, standard deviations)`
    /// — parallel slices in ascending period order, Elo scale. `None` for a
    /// player the model has never seen.
    pub fn timeline(&self, name: &str) -> Option<(&[u32], &[f64], &[f64])> {
        let id = self.names.get(name)? as usize;
        let (lo, hi) = (self.offsets[id], self.offsets[id + 1]);
        Some((
            &self.periods[lo..hi],
            &self.ratings[lo..hi],
            &self.sds[lo..hi],
        ))
    }
}

impl RankModel for WhrModel {
    fn algorithm(&self) -> &'static str {
        "whr"
    }

    /// Each player's rating at its **last** active period (Elo scale).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, n)| (n, self.ratings[self.offsets[i + 1] - 1]))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<TimelineLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| {
                let (lo, hi) = (self.offsets[i], self.offsets[i + 1]);
                TimelineLine {
                    id: id.to_string(),
                    t: self.periods[lo..hi].to_vec(),
                    r: self.ratings[lo..hi].to_vec(),
                    sd: self.sds[lo..hi].to_vec(),
                }
            })
            .collect();
        state::save_model(w, "whr", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (Whr, Vec<TimelineLine>) = state::load_model(r, "whr")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        let mut offsets = vec![0usize];
        let mut periods = Vec::new();
        let mut ratings = Vec::new();
        let mut sds = Vec::new();

        for line in &lines {
            if line.t.is_empty() || line.t.len() != line.r.len() || line.t.len() != line.sd.len() {
                return Err(Error::State(format!(
                    "timeline for {:?} needs equal, non-zero t/r/sd lengths",
                    line.id
                )));
            }
            if line.t.windows(2).any(|w| w[0] >= w[1]) {
                return Err(Error::State(format!(
                    "timeline periods for {:?} must be strictly ascending",
                    line.id
                )));
            }
            periods.extend_from_slice(&line.t);
            ratings.extend_from_slice(&line.r);
            sds.extend_from_slice(&line.sd);
            offsets.push(periods.len());
        }

        Ok(Self {
            params,
            names,
            offsets,
            periods,
            ratings,
            sds,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::algos::{BradleyTerryMM, SectionKind};

    /// Natural-scale timeline of one player: (periods, ratings).
    fn natural(m: &WhrModel, name: &str) -> (Vec<u32>, Vec<f64>) {
        let (t, r, _) = m.timeline(name).unwrap();
        (t.to_vec(), r.iter().map(|x| x / ELO_SCALE).collect())
    }

    /// Independent gradient recomputation (games + anchor + Wiener) at the
    /// fitted timelines, straight from the model output and the dataset —
    /// no fitter internals.
    fn joint_gradients(
        m: &WhrModel,
        d: &PairwiseDataset,
        algo: &Whr,
        players: &[&str],
    ) -> Vec<f64> {
        let tl: HashMap<&str, (Vec<u32>, Vec<f64>)> =
            players.iter().map(|&n| (n, natural(m, n))).collect();
        let rating_at = |n: &str, t: u32| -> f64 {
            let (ts, rs) = &tl[n];
            rs[ts.binary_search(&t).unwrap()]
        };

        let mut grads = Vec::new();
        for &n in players {
            let (ts, rs) = &tl[n];
            for (k, &t) in ts.iter().enumerate() {
                let r = rs[k];
                let mut g = 0.0;

                for (ti, range) in d.periods().enumerate() {
                    if ti as u32 != t {
                        continue;
                    }
                    for (w, l, x) in d.period_rows(range) {
                        let wn = d.interner().name(w).unwrap();
                        let ln = d.interner().name(l).unwrap();
                        if wn == n {
                            g += f64::from(x) * (1.0 - sigmoid(r - rating_at(ln, t)));
                        }
                        if ln == n {
                            g -= f64::from(x) * sigmoid(r - rating_at(wn, t));
                        }
                    }
                }

                if k == 0 {
                    g += algo.prior_games * (0.5 - sigmoid(r));
                }
                if k > 0 {
                    g -= (r - rs[k - 1]) / (algo.w2 * f64::from(t - ts[k - 1]));
                }
                if k + 1 < ts.len() {
                    g -= (r - rs[k + 1]) / (algo.w2 * f64::from(ts[k + 1] - t));
                }
                grads.push(g);
            }
        }
        grads
    }

    /// Single game A ≻ B, one period: the fitted point must zero the exact
    /// gradient, and match an independent 1-D bisection of the symmetric
    /// fixed-point equation 1 − σ(2r) + ½ − σ(r) = 0.
    #[test]
    fn single_game_matches_hand_solved_fixed_point() {
        let mut d = PairwiseDataset::new();
        d.push("A", "B", 1.0);
        let algo = Whr {
            tolerance: 1e-12,
            iterations: 1000,
            ..Default::default()
        };
        let m = algo.fit(&d).unwrap();

        let (_, ra) = natural(&m, "A");
        let (_, rb) = natural(&m, "B");
        let (ra, rb) = (ra[0], rb[0]);

        // Joint stationarity, no symmetry assumption.
        let g_a = (1.0 - sigmoid(ra - rb)) + (0.5 - sigmoid(ra));
        let g_b = -sigmoid(rb - ra) + (0.5 - sigmoid(rb));
        assert!(g_a.abs() < 1e-8, "g_a = {g_a:e}");
        assert!(g_b.abs() < 1e-8, "g_b = {g_b:e}");

        // Independent solve of the symmetric reduction by bisection.
        let g = |r: f64| (1.0 - sigmoid(2.0 * r)) + (0.5 - sigmoid(r));
        let (mut lo, mut hi) = (0.0f64, 5.0f64);
        assert!(g(lo) > 0.0 && g(hi) < 0.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if g(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let root = 0.5 * (lo + hi);
        assert!((ra - root).abs() < 1e-6, "r_A = {ra}, root = {root}");
        assert!((rb + root).abs() < 1e-6, "r_B = {rb}, root = {root}");
    }

    /// w2 → ∞ decouples periods (day signs follow each period's result);
    /// w2 → 0 rigidifies the timeline into the pooled static fit.
    #[test]
    fn w2_limits_decouple_or_pool() {
        let mut d = PairwiseDataset::new();
        d.push("p", "q", 1.0);
        d.new_period();
        d.push("q", "p", 1.0);

        let loose = Whr {
            w2: 1e6,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let (_, rp) = natural(&loose, "p");
        let (_, rq) = natural(&loose, "q");
        assert!(rp[0] > 0.0 && rp[1] < 0.0, "p timeline {rp:?}");
        assert!(rq[0] < 0.0 && rq[1] > 0.0, "q timeline {rq:?}");

        let tight = Whr {
            w2: 1e-12,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let (_, rp) = natural(&tight, "p");
        assert!((rp[0] - rp[1]).abs() < 1e-5, "rigid timeline {rp:?}");

        // Pooled static fit: same games in one period; by symmetry the MAP
        // is exactly 0, and the rigid timeline must agree.
        let mut pooled = PairwiseDataset::new();
        pooled.push("p", "q", 1.0);
        pooled.push("q", "p", 1.0);
        let stat = Whr::default().fit(&pooled).unwrap();
        let (_, rs) = natural(&stat, "p");
        assert!(rs[0].abs() < 1e-4, "pooled rating {rs:?}");
        assert!((rp[0] - rs[0]).abs() < 1e-3, "{} vs {}", rp[0], rs[0]);
    }

    /// On a single period WHR is anchored static BT: the rating order
    /// matches Bradley-Terry MM on a connected fixture.
    #[test]
    fn single_period_order_matches_bt_mm() {
        let mut d = PairwiseDataset::new();
        for _ in 0..3 {
            d.push("a", "b", 1.0);
            d.push("b", "c", 1.0);
        }
        d.push("b", "a", 1.0);
        d.push("c", "b", 1.0);
        d.push("a", "c", 1.0);
        d.push("c", "a", 1.0);

        let whr = Whr::default().fit(&d).unwrap();
        let whr_order: Vec<&str> = whr.sorted_scores().into_iter().map(|(n, _)| n).collect();

        let bt = BradleyTerryMM::default().fit(&d).unwrap();
        let ranked = &bt.sections()[0];
        assert_eq!(ranked.kind, SectionKind::Ranked);
        let bt_order: Vec<&str> = ranked
            .entries
            .iter()
            .map(|&(id, _)| bt.name(id).unwrap())
            .collect();

        assert_eq!(whr_order, bt_order);
    }

    /// The load-bearing test: on a 3-player, 2-period fixture the full joint
    /// gradient (games + anchor + Wiener), recomputed independently from the
    /// model output, vanishes at every (player, day).
    #[test]
    fn joint_gradient_vanishes_at_fit() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.push("b", "c", 1.0);
        d.push("a", "c", 1.0);
        d.new_period();
        d.push("b", "a", 1.0);
        d.push("c", "b", 3.0);

        let algo = Whr {
            tolerance: 1e-12,
            iterations: 2000,
            ..Default::default()
        };
        let m = algo.fit(&d).unwrap();

        for g in joint_gradients(&m, &d, &algo, &["a", "b", "c"]) {
            assert!(g.abs() < 1e-6, "stationarity violated: g = {g:e}");
        }
    }

    /// Warm-starting from a converged fit reproduces it.
    #[test]
    fn warm_start_reproduces_converged_fit() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.push("b", "c", 1.0);
        d.new_period();
        d.push("b", "a", 1.0);
        d.push("c", "b", 3.0);
        d.push("a", "c", 1.0);

        let algo = Whr {
            tolerance: 1e-12,
            iterations: 1000,
            ..Default::default()
        };
        let cold = algo.fit(&d).unwrap();
        let warm = algo.fit_warm(&d, &cold).unwrap();

        for name in ["a", "b", "c"] {
            let (ct, cr, _) = cold.timeline(name).unwrap();
            let (wt, wr, _) = warm.timeline(name).unwrap();
            assert_eq!(ct, wt);
            for (c, w) in cr.iter().zip(wr) {
                assert!((c - w).abs() < 1e-9, "{name}: {c} vs {w}");
            }
        }
    }

    /// σ sanity: positive everywhere, and a heavily-played day is more
    /// certain than a sparsely-played one.
    #[test]
    fn sd_positive_and_shrinks_with_games() {
        let mut d = PairwiseDataset::new();
        for _ in 0..10 {
            d.push("h", "x", 1.0);
            d.push("x", "h", 1.0);
        }
        d.new_period();
        d.push("h", "x", 1.0);

        // Large w2 so the two days are weakly coupled and the per-day game
        // counts dominate the uncertainty.
        let m = Whr {
            w2: 1.0,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();

        for name in ["h", "x"] {
            let (_, _, sd) = m.timeline(name).unwrap();
            assert!(sd.iter().all(|s| *s > 0.0), "{name} sd {sd:?}");
        }
        let (_, _, sd) = m.timeline("h").unwrap();
        assert!(sd[0] < sd[1], "heavy day sd {} vs sparse {}", sd[0], sd[1]);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.new_period();
        d.push("b", "a", 1.0);
        d.push("b", "c", 1.0);

        let m = Whr::default().fit(&d).unwrap();
        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = WhrModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        let empty = PairwiseDataset::new();
        assert!(matches!(
            Whr::default().fit(&empty),
            Err(Error::EmptyDataset)
        ));

        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        let zero = Whr {
            w2: 0.0,
            ..Default::default()
        };
        assert!(matches!(zero.fit(&d), Err(Error::InvalidInput(_))));
        let nan = Whr {
            w2: f64::NAN,
            ..Default::default()
        };
        assert!(matches!(nan.fit(&d), Err(Error::InvalidInput(_))));
    }
}
