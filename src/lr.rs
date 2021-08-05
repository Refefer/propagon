extern crate random;
extern crate rayon;
extern crate hashbrown;

use rayon::prelude::*;
use indicatif::{ProgressBar,ProgressStyle};

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
        let norm = self.scores.par_values().map(|x| x.powi(2)).sum::<f32>().powf(0.5);
        let decay = self.decay;
        self.scores.par_values_mut().for_each(|wi| {
            *wi -= decay * *wi / norm;
        })
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

    fn create_pb(&self, total_work: u64) -> ProgressBar {
        let pb = ProgressBar::new(total_work);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
        pb
    }

    pub fn update(&mut self, games: &Games) {

        let weights: f32 = games.par_iter().map(|(_,_,w)| w).sum();
        let mut grads = vec![(0u32, 0f32); games.len() * 2];
        let pb = self.create_pb(self.passes as u64);
        let mut msg = "Pass: 0".into();
        for it in 0..self.passes {
            msg = format!("Pass: {}/{}", it+1, self.passes);
            pb.set_message(&msg);

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
                grads.par_chunks_mut(2).zip(games.par_iter()).for_each(|(arr, game)| {
                    let (w, l, weight) = game;
                    let (winner, w_w, loser, l_w) = self.gen_gradient(w, l, weight, &weights);
                    arr[0].0 = winner;
                    arr[0].1 = w_w;
                    arr[1].0 = loser;
                    arr[1].1 = l_w;
                });

                // Sort the gradients so we can avoid multiple hash look ups
                grads.par_sort_by(|x, y| x.0.cmp(&y.0));

                // Update games
                let mut entry_key = grads[0].0;
                let mut entry = self.scores.entry(entry_key).or_insert(0.);
                for (idx, g) in grads.iter() {
                    if *idx != entry_key {
                        entry_key = *idx;
                        entry = self.scores.entry(entry_key).or_insert(0.);
                    }
                    *entry -= g;
                }
            }
            pb.inc(1);

            // Normalize the weights
            self.norm();
        }
        pb.finish();
    }
}
