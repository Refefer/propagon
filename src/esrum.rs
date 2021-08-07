extern crate rand;
extern crate rayon;
extern crate hashbrown;
extern crate statrs;

use std::hash::Hash;
use std::fmt::Write;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize,Ordering};

use indicatif::{ProgressBar,ProgressStyle};
use rayon::prelude::*;
use hashbrown::HashMap;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution as Dist,Gamma,Normal,Beta};
use float_ord::FloatOrd;

use statrs::distribution::{Normal as SNormal};
use statrs::distribution::ContinuousCDF;

#[derive(Debug)]
pub enum Distribution {
    Gaussian,
    FixedNormal
}

impl Distribution {
    fn sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    fn bound(&self, a: &mut f32, b: &mut f32) {
        match self {
            Distribution::Gaussian => {
                *b = Distribution::sigmoid(*b).max(1e-5);
            },
            Distribution::FixedNormal => {
                *a = a.abs().max(1e-5);
                *b = 1f32;
            }
        }
    }
}

pub struct EsRum {
    
    // Distribution to model RUM after
    pub distribution: Distribution,

    // Number of draws per distribution
    pub k: usize,

    // Number of updates to run
    pub passes: usize, 
   
    // Number of gradients to evaluate per run
    pub gradients: usize, 
    
    // Number of children to use for offspring
    pub children: usize, 

    // Learning rate for the gradient
    pub alpha: f32,
    
    // Regularization for distributions
    pub gamma: f32,

    // Smoothing
    pub smoothing: usize,

    // Random Seed
    pub seed: u64
}

impl EsRum {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self, 
        graph_iter: impl Iterator<Item=(K,K,f32)>
    ) -> HashMap<K, [f32; 2]> {

        // Load up graph
        let mut graph = HashMap::new();
        let mut vocab = HashMap::new();
        let mut n_vocab = 0usize;
        graph_iter.for_each(|(winner, loser, margin)| {
            // Numbers are better
            let w_idx = *vocab.entry(winner).or_insert_with(||{ n_vocab += 1; n_vocab - 1 });
            let l_idx = *vocab.entry(loser).or_insert_with(|| { n_vocab += 1; n_vocab - 1 });

            let margin = margin as usize;
            // Add bi-directional
            {
                let w = graph.entry(w_idx.clone()).or_insert_with(|| HashMap::new());
                let s = w.entry(l_idx.clone()).or_insert((0, 0));

                s.0 += margin;
                s.1 += margin;
            }

            // Add loser relationship
            {
                let l = graph.entry(l_idx).or_insert_with(|| HashMap::new());
                let s = l.entry(w_idx).or_insert((0, 0));

                s.1 += margin;
            }
        });
        eprintln!("Total nodes: {}", n_vocab);

        // Flatten
        let mut n_edges = 0usize;
        let graph: HashMap<_,_> = graph.into_iter().map(|(w, sg)| {
            let arr = sg.into_iter()
                .map(|(l, (w, n))| (l, (w + self.smoothing, n + self.smoothing * 2)))
                .collect::<Vec<_>>();
            n_edges += arr.len();
            (w, arr)
        }).collect();

        eprintln!("Total comparisons: {}", n_edges);

        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        
        // Create initial policy
        let mut policy = vec![[0f32, 0f32]; n_vocab];

        policy.iter_mut().for_each(|pi| {
            let mut mu = rng.gen();
            let mut sigma = rng.gen();
            self.distribution.bound(&mut mu, &mut sigma);
            pi[0] = mu;
            pi[1] = sigma;
        });

        // Compute the decay weights for each child
        let weights = {
            let mut weights = vec![0f32; self.children];
            weights.iter_mut().enumerate().for_each(|(idx, wi)| {
                *wi = 1. / ((idx + 2) as f32).ln();
            });
            let s = weights.iter().sum::<f32>();
            weights.iter_mut().for_each(|wi| { *wi /= s});
            weights
        };

        let pb = self.create_pb((self.passes * n_vocab) as u64);

        let mut msg = format!("Pass: 0, Alpha:{}, Fitness: inf", self.alpha);
        pb.set_message(&msg);

        let decay_rate = ((1e-1f32).ln() / (self.passes as f32)).exp();
        let mut last_loss = f32::INFINITY;
        let mut sigma = self.alpha;
        for it in 0..self.passes {
            let updated = AtomicUsize::new(0);

            // Fill the samples
            let normal = Normal::new(0f32, sigma).unwrap();
            let new_policy: Vec<_> = policy.par_iter().enumerate().map(|(c_idx, arr)| {
                let mut grads = vec![(0f32, [0f32, 0f32]); self.gradients];
                let mut new_policy = arr.clone();
                let seed = self.seed + (c_idx + (it + 1) * self.gradients) as u64;
                
                // Create search gradients and compute fitness scores for each
                let comps = &graph[&c_idx].as_slice();
                grads.par_iter_mut().enumerate().for_each(|(g_idx, (fit, g))| {
                    let mut rng = XorShiftRng::seed_from_u64(seed + g_idx as u64);
                    g.iter_mut().zip(new_policy.iter()).for_each(|(vi, pi)| { 
                        *vi = normal.sample(&mut rng) + *pi; 
                    });
                    *fit = self.score(g as &[f32], comps, policy.as_slice());
                });

                // Fitness sort them in ascending order
                grads.sort_by_key(|(f, _g)| FloatOrd(*f));

                // Combine into new policy
                new_policy.par_iter_mut().enumerate().for_each(|(p_idx, pi)| {
                    *pi += grads.iter().take(self.children).zip(weights.iter()).map(|((_, g), wi)| {
                        wi * (g[p_idx] - *pi)
                    }).sum::<f32>();
                });

                // Re-bound distribution
                let [mut n_mu, mut n_sigma] = new_policy;
                self.distribution.bound(&mut n_mu, &mut n_sigma);
                new_policy = [n_mu, n_sigma];

                let old_score = self.score(arr as &[f32], comps, policy.as_slice());
                let new_score = self.score(&new_policy as &[f32], comps, policy.as_slice());

                pb.inc(1);
                if new_score < old_score {
                    updated.fetch_add(1, Ordering::SeqCst);
                    new_policy
                } else {
                    arr.clone()
                }
            }).collect();

            policy = new_policy;
            
            // Evaluate new fitness
            let s = policy.par_iter().enumerate().map(|(idx, stats)| {
                self.score(stats as &[f32], &graph[&idx].as_slice(), policy.as_slice())
            }).sum::<f32>() / n_vocab as f32;
            
            if s < last_loss {
                last_loss = s;
            } else {
                sigma *= decay_rate;
            }

            msg.clear();
            write!(msg, "Pass: {}, Updated: {}, Sigma: {:.3}, Fitness: {:.5}", it + 1, updated.load(Ordering::SeqCst), sigma, s)
                .expect("Oh my, shouldn't have failed!");
            pb.set_message(&msg);
        }
        pb.finish();

        vocab.into_iter().map(|(k, idx)| {
            let [mut a, mut b] = policy[idx];
            self.distribution.bound(&mut a, &mut b);
            (k, [a, b])
        }).collect()
    }

    fn n_kemeny_loss(&self, o_wins: usize, o_n: usize, s_rate: f32) -> f32 {
        // Add inversion loss
        let o_rate = o_wins as f32 / o_n as f32;

        if s_rate > 0.5 && o_rate > 0.5 {
            0f32
        } else if s_rate < 0.5 && o_rate < 0.5 {
            0f32
        } else {
            1f32
        }
    }

    fn score(&self, dist: &[f32], graph: &[(usize, (usize, usize))], dists: &[[f32; 2]]) -> f32 {
        let mut s_mu    = dist[0];
        let mut s_sigma = dist[1];
        self.distribution.bound(&mut s_mu, &mut s_sigma);
        let score = graph.par_iter().map(|(o_idx, (wins, n))| {
            let [e_mu, e_sigma] = dists[*o_idx];
            let s_rate = SNormal::new((e_mu - s_mu) as f64, (s_sigma + e_sigma) as f64)
                .unwrap()
                .cdf(0.) as f32;

            // Wasserstein Distance
            let fitness = (s_rate - *wins as f32 / *n as f32).abs();

            // penalize the distance from 0
            fitness + self.n_kemeny_loss(*wins, *n, s_rate) / graph.len() as f32
        }).sum::<f32>() / graph.len() as f32;

        // Minimize mu, maximize sigma?
        score + self.gamma * (s_mu.powf(2.) + s_sigma.powf(2.)).powf(0.5)
    }

    fn create_pb(&self, total_work: u64) -> ProgressBar {
        let pb = ProgressBar::new(total_work);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
        pb
    }

}

