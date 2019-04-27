extern crate random;
extern crate rayon;

use rayon::prelude::*;

use std::collections::HashMap;

use super::Games;


pub struct BtmLr {
    pub scores: HashMap<u32, f32>,
    pub passes: usize,
    pub alpha: f32,
    pub decay: f32
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-x))
}

impl BtmLr {

    pub fn new(passes: usize, alpha: f32, decay: f32) -> Self {
        BtmLr { 
            scores: HashMap::new(), 
            passes: passes, 
            alpha: alpha, 
            decay: decay 
        }
    }

    fn norm(&mut self) {
        let norm = self.scores.values().map(|x| x.powi(2)).sum::<f32>().powf(0.5);
        for wi in self.scores.values_mut() {
            *wi = self.decay * *wi / norm;
        }
    }

    pub fn update(&mut self, games: &Games) {

        let weights: f32 = games.par_iter().map(|(_,_,w)| w).sum();
        let mut grads = Vec::new();
        for it in 0..self.passes {
            eprintln!("Iteration: {}", it);

            games.par_iter().map(|(w, l, weight)| {
                let w_x = self.scores.get(w).unwrap_or(&0.);
                let l_x = self.scores.get(l).unwrap_or(&0.);
                let y_hat = sigmoid(w_x - l_x);
                let denom = self.alpha * weight * (y_hat - 1.0);
                (w, denom / weights, l, -denom / weights)
            }).collect_into_vec(&mut grads);

            // Update games
            for (w, g, l, g2) in grads.drain(0..) {
                let e = self.scores.entry(*w).or_insert(0.);
                *e -= g;
                let e = self.scores.entry(*l).or_insert(0.);
                *e -= g2;
            }
        }
        // Normalize the weights
        self.norm();
    }
}
