//! Differential evolution: the gradient-free optimizer behind Kemeny's
//! `DiffEvo` mode (faithful v1 port on the modern rand API).

use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand_distr::Normal;

use crate::parallel;

/// Objective to maximize.
pub trait Fitness: Send + Sync {
    /// Scores one candidate vector; the optimizer maximizes this.
    fn score(&self, candidate: &[f32]) -> f32;
}

/// DE hyperparameters (see field docs in v1; defaults chosen by callers).
#[derive(Clone, Copy, Debug)]
pub struct DifferentialEvolution {
    /// Dimensionality of each candidate vector.
    pub dims: usize,
    /// Population size.
    pub lambda: usize,
    /// Dithered learning-rate range F.
    pub f: (f32, f32),
    /// Crossover probability.
    pub cr: f32,
    /// Probability of random perturbation around the best candidate.
    pub m: f32,
    /// Expansion radius for those perturbations.
    pub exp: f32,
    /// Rebuild the population around the best after this many stale rounds
    /// (0 = off).
    pub polish_on_stale: usize,
    /// Full restart after this many stale rounds (0 = off).
    pub restart_on_stale: usize,
    /// Initial population range.
    pub range: f32,
}

impl DifferentialEvolution {
    /// Runs until `total_fns` fitness evaluations are spent; returns the best
    /// (fitness, candidate). `callback(best_fit, fns_remaining)` reports
    /// progress.
    pub fn fit<F: Fitness>(
        &self,
        fit_fn: &F,
        total_fns: usize,
        seed: u64,
        x_in: Option<&[f32]>,
        mut callback: impl FnMut(f32, usize),
    ) -> crate::Result<(f32, Vec<f32>)> {
        if self.dims == 0 || self.lambda < 2 || self.range <= 0.0 || self.f.0 >= self.f.1 {
            return Err(crate::Error::InvalidInput(format!(
                "differential evolution needs dims > 0, lambda >= 2, range > 0, f.0 < f.1; \
                 got dims {}, lambda {}, range {}, f {:?}",
                self.dims, self.lambda, self.range, self.f
            )));
        }

        let mut fns = 0;
        let mut best_fit = f32::NEG_INFINITY;
        let mut best_cand = vec![0.0; self.dims];

        while fns < total_fns {
            let (used, fit, cand) = self.run_pass(
                fit_fn,
                total_fns,
                seed + fns as u64,
                x_in,
                self.restart_on_stale,
                &mut callback,
            )?;
            fns += used;

            if fit > best_fit {
                best_fit = fit;
                best_cand = cand;
            }
        }

        Ok((best_fit, best_cand))
    }

    fn run_pass<F: Fitness>(
        &self,
        fit_fn: &F,
        total_fns: usize,
        seed: u64,
        x_in: Option<&[f32]>,
        early_terminate: usize,
        callback: &mut impl FnMut(f32, usize),
    ) -> crate::Result<(usize, f32, Vec<f32>)> {
        let bad_dist =
            |e: rand::distr::uniform::Error| crate::Error::InvalidInput(format!("de bounds: {e}"));
        let mut rng = StdRng::seed_from_u64(seed);
        let init = Uniform::new(-self.range, self.range).map_err(bad_dist)?;

        // Loop-invariant distributions, validated once (`fit` checked the
        // parameter preconditions already).
        let f_dist = Uniform::new(self.f.0, self.f.1).map_err(bad_dist)?;
        let unit = Uniform::new(0.0f32, 1.0).map_err(bad_dist)?;
        let norm = Normal::new(0.0f32, 1.0)
            .map_err(|e| crate::Error::Numeric(format!("unit normal: {e}")))?;

        let mut pop: Vec<Vec<f32>> = (0..self.lambda)
            .map(|_| (0..self.dims).map(|_| init.sample(&mut rng)).collect())
            .collect();
        if let Some(x) = x_in {
            for (i, p) in pop.iter_mut().enumerate() {
                for (pi, vi) in p.iter_mut().zip(x) {
                    if i == 0 {
                        *pi = *vi;
                    } else {
                        *pi += vi;
                    }
                }
            }
        }

        let mut fits: Vec<f32> = pop.iter().map(|p| fit_fn.score(p)).collect();
        let mut fns = self.lambda;

        let mut best_fit = f32::NEG_INFINITY;
        let mut stale_len = 0;
        let mut last_update = 0;

        while fns < total_fns {
            let best_idx = arg_max(&fits);
            if fits[best_idx] > best_fit {
                best_fit = fits[best_idx];
                stale_len = 0;
                last_update = 0;
            }
            if early_terminate > 0 && last_update == early_terminate {
                break;
            }

            if self.polish_on_stale > 0 && stale_len == self.polish_on_stale {
                let best = pop[best_idx].clone();
                for (i, p) in pop.iter_mut().enumerate() {
                    if i != best_idx {
                        for (vi, bi) in p.iter_mut().zip(&best) {
                            *vi = bi + init.sample(&mut rng);
                        }
                    }
                }
                for (i, f) in fits.iter_mut().enumerate() {
                    if i != best_idx {
                        *f = fit_fn.score(&pop[i]);
                    }
                }
                fns += pop.len() - 1;
                stale_len = 0;
                continue;
            }
            stale_len += 1;
            last_update += 1;

            let f_lr = f_dist.sample(&mut rng);

            // Each candidate's proposal is computed independently with a
            // per-index seeded RNG (v1 pattern) — deterministic and parallel.
            let pop_ref = &pop;
            let fits_ref = &fits;
            let proposals: Vec<Option<(Vec<f32>, f32)>> =
                parallel::par_map_indexed(self.lambda, |idx| {
                    let orig_x = &pop_ref[idx];
                    let best = &pop_ref[best_idx];
                    let mut local = StdRng::seed_from_u64(seed + (idx + fns) as u64);

                    if idx != best_idx && unit.sample(&mut local) < self.m {
                        // Random perturbation around the best candidate.
                        let orig_mag = l2norm(
                            &orig_x
                                .iter()
                                .zip(best)
                                .map(|(a, b)| a - b)
                                .collect::<Vec<_>>(),
                        );
                        let mut x: Vec<f32> =
                            (0..self.dims).map(|_| norm.sample(&mut local)).collect();
                        let mag = l2norm(&x);
                        for (i, xi) in x.iter_mut().enumerate() {
                            *xi = best[i] + orig_mag * self.exp * (*xi) / mag;
                        }
                        let f = fit_fn.score(&x);
                        f.is_finite().then_some((x, f))
                    } else {
                        // DE/best/1 crossover. `fit` guarantees lambda >= 2,
                        // so the misses below are unreachable; degrade to
                        // "keep the original candidate" rather than panic.
                        let mut chosen = pop_ref.choose_multiple(&mut local, 2);
                        let (Some(a), Some(b)) = (chosen.next(), chosen.next()) else {
                            return None;
                        };
                        let mut x = vec![0.0f32; self.dims];
                        for i in 0..self.dims {
                            x[i] = if unit.sample(&mut local) < self.cr {
                                best[i] + f_lr * (a[i] - b[i])
                            } else {
                                orig_x[i]
                            };
                        }
                        let f = fit_fn.score(&x);
                        (f.is_finite() && f > fits_ref[idx]).then_some((x, f))
                    }
                });

            for (idx, proposal) in proposals.into_iter().enumerate() {
                if let Some((x, f)) = proposal {
                    pop[idx] = x;
                    fits[idx] = f;
                }
            }
            fns += self.lambda;

            let best_now = fits[arg_max(&fits)];
            callback(best_now, total_fns.saturating_sub(fns));
        }

        let best_idx = arg_max(&fits);
        Ok((fns, fits[best_idx], pop.swap_remove(best_idx)))
    }
}

fn arg_max(fits: &[f32]) -> usize {
    let mut best = 0;
    for (i, f) in fits.iter().enumerate() {
        if f.total_cmp(&fits[best]).is_gt() {
            best = i;
        }
    }
    best
}

fn l2norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v1 `test_matyas`: minimum of the Matyas function, offset to (10, -10).
    struct Matyas(f32, f32);
    impl Fitness for Matyas {
        fn score(&self, c: &[f32]) -> f32 {
            let x = c[0] + self.0;
            let y = c[1] + self.1;
            -(0.26 * (x * x + y * y) - 0.48 * x * y)
        }
    }

    #[test]
    fn finds_matyas_optimum() {
        let de = DifferentialEvolution {
            dims: 2,
            lambda: 30,
            f: (0.1, 1.0),
            cr: 0.9,
            m: 0.1,
            exp: 3.0,
            polish_on_stale: 0,
            restart_on_stale: 0,
            range: 1.0,
        };
        let (fit, result) = de
            .fit(&Matyas(-10.0, 10.0), 10_000, 2020, None, |_, _| {})
            .unwrap();
        assert!(fit > -1e-6, "fitness {fit}");
        assert!((result[0] - 10.0).abs() < 1e-2, "x = {}", result[0]);
        assert!((result[1] + 10.0).abs() < 1e-2, "y = {}", result[1]);
    }
}
