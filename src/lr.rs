extern crate random;
use super::Games;

use std::collections::HashMap;

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
        // Normalize the weights
        self.norm();
        // We need to remap all values to a vector
        for (w, l, _) in games {
            for id in &[*w,*l] {
                if !self.scores.contains_key(id) {
                    self.scores.insert(*id, 0f32);
                }
            }
        }

        let mut grads = Vec::with_capacity(games.len() * 2);
        for i in 0..passes {
            let mut weights = 0.;
            for (j, (w, l, weight)) in games.iter().enumerate() {
                let w_x = self.scores[w];
                let l_x = self.scores[l];
                let y_hat = sigmoid(w_x - l_x);
                weights += weight;
                let denom = alpha * weight * (y_hat - 1.0);
                grads.push((w, denom));
                grads.push((l, -denom));
            }

            // Update games
            for (idx, g) in grads.drain(0..) {
                let e = self.scores.entry(*idx).or_insert(0.);
                *e -= g / weights;
            }
        }
    }
}
