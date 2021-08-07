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
                *b = b.max(1e-5);
            },
            Distribution::FixedNormal => {
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
        
        let mut policy = self.create_initial_policy(&graph);
        let n_gradients = (n_vocab as f32).powf(0.7).max(10.) as usize;
        let n_children = (n_gradients as f32 / 10.).max(3.) as usize;
        
        // Compute the decay weights for each child
        let weights = {
            let mut weights = vec![0f32; n_children];
            weights.iter_mut().enumerate().for_each(|(idx, wi)| {
                *wi = 1. / ((idx + 2) as f32).ln();
            });
            let s = weights.iter().sum::<f32>();
            weights.iter_mut().for_each(|wi| { *wi /= s});
            weights
        };


        let pb = self.create_pb((self.passes * n_gradients) as u64);

        let mut msg = format!("Pass: 0, Alpha:{}, Fitness: inf", self.alpha);
        pb.set_message(&msg);

        // Create Gradients
        let mut gradients: Vec<_> = (0..n_gradients)
            .map(|_| (0f32, policy.clone()))
            .collect();

        //let decay_rate = ((self.alpha / 100.).ln() / (self.passes as f32)).exp();
        let mut last_loss = self.score_all(policy.as_slice(), &graph);
        let mut sigma = self.alpha;

        let mut new_policy = policy.clone();
        for it in 0..self.passes {
            let normal = Normal::new(0f32, sigma).unwrap();

            new_policy.par_iter_mut().for_each(|arr| {
                *arr= [0., 0.];
            });
            
            // For each gradient, compute its score
            gradients.par_iter_mut().enumerate().for_each(|(idx, (fitness, grad))| {
                
                // Add noise to the gradient
                let seed = self.seed + (idx + it * n_gradients) as u64;
                let mut rng = XorShiftRng::seed_from_u64(seed);
                grad.iter_mut().zip(policy.iter()).enumerate().for_each(|(idx, (arr, p_arr))| {
                    let mut n_mu    = p_arr[0] + normal.sample(&mut rng);
                    let mut n_sigma = p_arr[1] + normal.sample(&mut rng);
                    *arr = [n_mu, n_sigma];
                });

                // Score the gradient
                *fitness = self.score_all(grad.as_slice(), &graph);
                pb.inc(1);
            });
            
            // Fitness sort them in ascending order since we are minimizing the objective
            gradients.sort_by_key(|(f, _g)| FloatOrd(*f));

            // Combine into new policy
            new_policy.par_iter_mut().enumerate().for_each(|(p_idx, arr)| {
                let p_arr = policy[p_idx];
                let mut n_mu = p_arr[0];
                let mut n_sigma = p_arr[1];
                gradients.iter().take(n_children).zip(weights.iter()).for_each(|((_, g), wi)| {
                    n_mu    += wi * (g[p_idx][0] - p_arr[0]);
                    n_sigma += wi * (g[p_idx][1] - p_arr[1]);
                });
                *arr = [n_mu, n_sigma];
            });
            
            // Compare old policy to new policy.
            let new_score = self.score_all(new_policy.as_slice(), &graph);

            if new_score < last_loss {
                std::mem::swap(&mut policy, &mut new_policy);
                last_loss = new_score;
            } else {
                sigma = (sigma / 2.).max(self.alpha / 100.);
            }

            msg.clear();
            write!(msg, "Pass: {}, Sigma: {:.3}, Best Fitness: {:.5}, Last Fitness: {:.5}", it + 1, sigma, last_loss, new_score)
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

    fn create_initial_policy(&self, graph: &HashMap<usize, Vec<(usize, (usize, usize))>>) -> Vec<[f32; 2]> {
        // To create the initial policy, we first order each alternative by
        // their average win/loss rates in all comparisons.
        let mut rates = vec![(0usize, 0f32); graph.len()];

        rates.iter_mut().enumerate().for_each(|(idx, (v_idx, v_rate))| {
            let comps = &graph[&idx];
            *v_rate = comps.iter()
                .map(|(_, (w, n))| *w as f32 / *n as f32)
                .sum::<f32>() / comps.len() as f32;
            *v_idx = idx;
        });

        rates.sort_by_key(|(idx, rate)| FloatOrd(*rate));
        
        // Create initial policy
        let mut policy = vec![[0f32, 0f32]; graph.len()];
        let normal = SNormal::new(0., 1.0).unwrap();
        rates.into_iter().enumerate().for_each(|(idx, (v_idx, _))| {
            policy[v_idx] = [
                normal.inverse_cdf((idx + 1) as f64 / (policy.len() + 2) as f64) as f32,
                1e-5];
        });

        policy
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

    fn score_all(&self, policy: &[[f32; 2]], graph: &HashMap<usize, Vec<(usize, (usize, usize))>>) -> f32 {
        policy.par_iter().enumerate().map(|(idx, arr)| {
            let comps = &graph[&idx].as_slice();
            self.score(arr, comps, policy)
        }).sum::<f32>()
    }

    fn score(&self, dist: &[f32], graph: &[(usize, (usize, usize))], dists: &[[f32; 2]]) -> f32 {
        let mut s_mu    = dist[0];
        let mut s_sigma = dist[1];
        self.distribution.bound(&mut s_mu, &mut s_sigma);
        let score = graph.par_iter().map(|(o_idx, (wins, n))| {
            let [mut e_mu, mut e_sigma] = dists[*o_idx];
            self.distribution.bound(&mut e_mu, &mut e_sigma);
            let s_rate = SNormal::new((e_mu - s_mu) as f64, (s_sigma + e_sigma) as f64)
                .unwrap()
                .cdf(0.) as f32;

            // Wasserstein Distance
            let fitness = *n as f32 * (s_rate - *wins as f32 / *n as f32).abs();

            // penalize the distance from 0
            fitness //+ self.n_kemeny_loss(*wins, *n, s_rate)
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

