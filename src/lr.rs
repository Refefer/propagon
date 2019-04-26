extern crate random;
extern crate rayon;

use rayon::prelude::*;

use std::collections::HashMap;

use super::Games;


pub struct BtmLr {
    pub scores: HashMap<u32, f32>
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-x))
}

impl BtmLr {

    pub fn new() -> Self {
        BtmLr { scores: HashMap::new() }
    }

    fn norm(&mut self) {
        let norm = self.scores.values().map(|x| x.powi(2)).sum::<f32>().powf(0.5);
        for w in self.scores.values_mut() {
            *w /= norm;
        }
    }

    pub fn update(&mut self, games: &Games, passes: usize, alpha: f32) {

        // We need to remap all values to a vector
        for (w, l, _) in games {
            for id in &[*w,*l] {
                if !self.scores.contains_key(id) {
                    self.scores.insert(*id, 0f32);
                }
            }
        }

        let weights: f32 = games.par_iter()
            .map(|(_,_,w)| w).sum();
        let mut grads = Vec::new();
        for _i in 0..passes {

            games.par_iter().map(|(w, l, weight)| {
                let w_x = self.scores[w];
                let l_x = self.scores[l];
                let y_hat = sigmoid(w_x - l_x);
                let denom = alpha * weight * (y_hat - 1.0);
                (w, denom, l, -denom)
            }).collect_into_vec(&mut grads);

            // Update games
            for (w, g, l, g2) in grads.drain(0..) {
                let e = self.scores.entry(*w).or_insert(0.);
                *e -= g / weights;
                let e = self.scores.entry(*l).or_insert(0.);
                *e -= g2 / weights;
            }
        }
        // Normalize the weights
        self.norm();
    }
}
