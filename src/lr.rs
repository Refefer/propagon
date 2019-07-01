extern crate random;
extern crate rayon;
extern crate hashbrown;

use rayon::prelude::*;

use hashbrown::HashMap;

use super::Games;


pub struct BtmLr {
    pub scores: HashMap<u32, f32>,
    pub passes: usize,
    pub alpha: f32,
    pub decay: f32,
    pub thrifty: bool
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-x))
}

impl BtmLr {

    pub fn new(passes: usize, alpha: f32, decay: f32, thrifty: bool) -> Self {
        BtmLr { 
            scores: HashMap::new(), 
            passes: passes, 
            alpha: alpha, 
            decay: decay,
            thrifty: thrifty
        }
    }

    fn norm(&mut self) {
        let norm = self.scores.values().map(|x| x.powi(2)).sum::<f32>().powf(0.5);
        for wi in self.scores.values_mut() {
            *wi = self.decay * *wi / norm;
        }
    }

    #[inline]
    fn gen_gradient(&self, w: &u32, l: &u32, weight: &f32, weights: &f32) -> (u32, f32, u32, f32) {
        let w_x = self.scores.get(w).unwrap_or(&0.);
        let l_x = self.scores.get(l).unwrap_or(&0.);
        let y_hat = sigmoid(w_x - l_x);
        let denom = self.alpha * weight * (y_hat - 1.0);
        (*w, denom / weights, *l, -denom / weights)
    }

    #[inline]
    fn update_grads(&mut self, w: u32, w_g: f32, l: u32, l_g: f32) {
        let e = self.scores.entry(w).or_insert(0.);
        *e -= w_g;
        let e = self.scores.entry(l).or_insert(0.);
        *e -= l_g;
    }

    pub fn update(&mut self, games: &Games) {

        let weights: f32 = games.par_iter().map(|(_,_,w)| w).sum();
        let mut grads = Vec::new();
        for it in 0..self.passes {
            eprintln!("Iteration: {}", it);

            if self.thrifty {
                // No parallel updates; compute weights on the fly, which
                // will return different results compared to the non-thrifty
                // variant
                for (w, l, weight) in games.iter() {
                    let (w, w_g, l, l_g) = self.gen_gradient(w, l, weight, &weights);
                    self.update_grads(w, w_g, l, l_g);
                }

            } else {
                // Compute all the gradients, in parallel, and update them all
                // at once
                games.par_iter().map(|(w, l, weight)| {
                    self.gen_gradient(w, l, weight, &weights)
                }).collect_into_vec(&mut grads);

                // Update games
                for (w, g, l, g2) in grads.drain(0..) {
                    self.update_grads(w, g, l, g2);
                }
            }
        }
        // Normalize the weights
        self.norm();
    }
}
