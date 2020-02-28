extern crate rand;
extern crate rayon;
extern crate float_ord;

use float_ord::FloatOrd;

use rand::prelude::*;
use rand::distributions::{Normal,Uniform};
use rayon::prelude::*;

pub trait Fitness: Send + Sync {
    fn score(&self, candidate: &[f32]) -> f32;
}

pub struct DifferentialEvolution {
    /// Dims
    dims: usize,

    /// Population size.
    lambda: usize,

    /// Dithered learning rate, F.  A good default is (0.1, 0.9)
    f: (f32, f32),

    /// Mutation rate: typically set to either 0.1 or 0.9
    cr: f32,

    /// Likelihood of replacing a candidate with a randomly perturb best.  0.1 is a good
    /// default
    m: f32,

    /// Expansion constant for random perturbation.  A good value is between 2 to 5
    exp: f32
}

impl DifferentialEvolution {

    fn train<F: Fitness>(&self, fit_fn: &F, total_fns: usize, seed: u64) -> Vec<f32> {

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Initialize population
        let dist = Uniform::new(-1., 1.);
        let mut pop: Vec<_> = (0..self.lambda).map(|_| {
            let mut v = vec![0.; self.dims];
            v.iter_mut().for_each(|vi| *vi = dist.sample(&mut rng));
            v
        }).collect();

        let mut tmp_pop = pop.clone();

        // Generate initial fitnesses
        let mut fits: Vec<_> = pop.iter().map(|p| fit_fn.score(&p)).collect();

        // Initial function counts
        let mut fns = self.lambda;

        let norm_dist = Normal::new(0., 1.);

        while fns < total_fns {
            // Get the best candidate
            let best_idx = (0..self.lambda).max_by_key(|i| FloatOrd(fits[*i]))
                .expect("Should never be empty!");

            let best = &pop[best_idx];

            let f_lr = Uniform::new(self.f.0, self.f.1).sample(&mut rng);

            // Generate mutation vector
            let dist = Uniform::new(0., 1.);
            tmp_pop.par_iter_mut().zip(fits.par_iter_mut())
                    .enumerate().for_each(|(idx, (x, f))| {

                let orig_x = &pop[idx];
                let mut local_rng = rand::rngs::StdRng::seed_from_u64((idx + fns) as u64);

                // Randomize a candidate in the population
                if idx != best_idx && dist.sample(&mut local_rng) < self.m {

                    // Figure out magnitude between current x and the best
                    x.iter_mut().zip(orig_x.iter()).enumerate().for_each(|(i, (xi, oxi))| {
                        *xi = oxi - best[i];
                    });
                    let orig_mag = l2norm(&x);

                    // Ok, generate new vector
                    x.iter_mut().for_each(|xi| {
                        *xi = norm_dist.sample(&mut local_rng) as f32;
                    });

                    // Normalize it
                    let v_mag = l2norm(&x);

                    // Expand it out within the expansion radius
                    x.iter_mut().enumerate().for_each(|(i, xi)| {
                        *xi = best[i] + orig_mag * self.exp * (*xi) / v_mag;
                    });

                    // Just override the fitness
                    *f = fit_fn.score(&x);

                } else {
                    // x_b +  F * (a - b)

                    // Select two candidates for the combination
                    let mut slice = pop.choose_multiple(&mut local_rng, 2);
                    let a = slice.next().expect("Number of candidates to low!");
                    let b = slice.next().expect("Number of candidates to low!");

                    // Generate new vector
                    (0..self.dims).for_each(|i| {
                        // If we are mutating
                        if dist.sample(&mut local_rng) < self.cr {
                            x[i] = best[i] + f_lr * (a[i] - b[i]);
                        } else {
                            x[i] = orig_x[i];
                        }
                    });

                    // Score the new individual
                    let new_fitness = fit_fn.score(&x);
                    if new_fitness > *f {
                        *f = new_fitness;
                    } else {
                        // Copy over the original x
                        x.iter_mut().zip(orig_x.iter()).for_each(|(xi, oxi)| {
                            *xi = *oxi;
                        });
                    }
                }
            });

            // Swap tmp with orig
            std::mem::swap(&mut pop, &mut tmp_pop);
            fns += self.lambda;
        }


        // Get the best idx!
        let best_idx = (0..self.lambda).max_by_key(|i| FloatOrd(fits[*i]))
            .expect("Should never be empty!");
        pop.swap_remove(best_idx)
    }
}

fn l2norm(v: &[f32]) -> f32 {
    v.iter()
        .map(|vi| vi.powi(2))
        .sum::<f32>()
        .powf(0.5)
}

fn euc_dist(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter())
        .map(|(v1i, v2i)| (v1i - v2i).powi(2))
        .sum::<f32>()
        .powf(0.5)
}
