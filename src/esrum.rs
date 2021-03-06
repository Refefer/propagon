extern crate rand;
extern crate rayon;
extern crate hashbrown;

use std::hash::Hash;
use std::boxed::Box;
use std::fmt::Write;

use indicatif::{ProgressBar,ProgressStyle};
use rayon::prelude::*;
use hashbrown::HashMap;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution as Dist,Gamma,Normal,Beta};
use float_ord::FloatOrd;

use super::Games;

pub enum Distribution {
    Gaussian,
    Gamma,
    Beta
}

enum Sampler {
    Normal(Normal<f32>),
    Gamma(Gamma<f32>),
    Beta(Beta<f32>)
}

impl Sampler {
    fn sample<R: Rng>(&self, rng: &mut R) -> f32 {
        match self {
            Sampler::Normal(d) => rng.sample(d),
            Sampler::Gamma(d)  => rng.sample(d),
            Sampler::Beta(d)   => rng.sample(d)
        }
    }
}

impl Distribution {
    // Fills a vector with samples from the given distribution
    fn create(&self, a: f32, b: f32) -> Sampler {
        match self {
            Distribution::Gaussian => {
                Sampler::Normal(Normal::new(a, b.max(0.)).unwrap())
            },
            Distribution::Gamma => {
                Sampler::Gamma(Gamma::new(a.max(1e-5), b.max(1e-5)).unwrap())
            },
            Distribution::Beta => {
                Sampler::Beta(Beta::new(a.max(1e-5), b.max(1e-5)).unwrap())
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

    // Random Seed
    pub seed: u64,
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
            let mut w_idx = *vocab.entry(winner).or_insert_with(||{ n_vocab += 1; n_vocab - 1 });
            let mut l_idx = *vocab.entry(loser).or_insert_with(||{  n_vocab += 1; n_vocab - 1 });

            let margin = margin as usize;
            let is_win = if w_idx > l_idx {
                true
            } else {
                std::mem::swap(&mut l_idx, &mut w_idx);
                false
            };
            let w = graph.entry(w_idx).or_insert_with(|| HashMap::new());
            let s = w.entry(l_idx).or_insert((0, 0));
            if is_win {
               s.0 += margin;
            }
            s.1 += margin;
        });

        // Flatten
        let mut edges = Vec::new();
        graph.into_iter().for_each(|(w, sg)| {
            sg.into_iter().for_each(|(l, s)| {
                edges.push((w.clone(), l, s));
            });
        });

        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        
        // Create initial policy
        let mut policy = vec![0f32; n_vocab * 2];
        policy.iter_mut().for_each(|pi| *pi = rng.gen());

        // Gradients
        let mut gradients = vec![(0f32, policy.clone()); self.gradients];

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

        let pb = self.create_pb((self.passes * (self.gradients + 1)) as u64);

        let mut msg = format!("Pass: 0, Fitness: inf");
        pb.set_message(&msg);
        for it in 0..self.passes {

            // Randomize gradients and evaluate
            gradients.par_iter_mut().enumerate().for_each(|(idx, (fit, g))| {
                let mut rng = XorShiftRng::seed_from_u64(self.seed + (idx + (it + 1) * self.gradients) as u64);
                let normal = Normal::new(0f32, 1f32).unwrap();
                g.iter_mut().zip(policy.iter()).for_each(|(vi, pi)| { 
                    *vi = normal.sample(&mut rng) + *pi; 
                });
                *fit = self.score(edges.as_slice(), g.as_slice(), it);
                pb.inc(1);
            });

            // Fitness sort them in ascending order
            gradients.sort_by_key(|(f, _g)| FloatOrd(*f));

            // Combine into new policy
            policy.par_iter_mut().enumerate().for_each(|(p_idx, pi)| {
                *pi += self.alpha * gradients[..self.children].iter().zip(weights.iter()).map(|((_, g), wi)| {
                    wi * (g[p_idx] - *pi)
                }).sum::<f32>();
            });

            // Evaluate new fitness
            let s = self.score(edges.as_slice(), policy.as_slice(), self.passes + 1);

            pb.inc(1);
            msg.clear();
            write!(msg, "Pass: {}, Fitness: {:.5}", it + 1, s);
            pb.set_message(&msg);
        }
        pb.finish();

        std::mem::drop(gradients);

        vocab.into_iter().map(|(k, idx)| {
            (k, [policy[2*idx], policy[2*idx+1]])
        }).collect()
    }

    fn score(&self, rels: &[(usize, usize, (usize, usize))], dist: &[f32], pass: usize) -> f32 {
        rels.par_iter().enumerate().map(|(idx, (w_idx, l_idx, (wins, n)))| {
            let wd = self.distribution.create(dist[2*w_idx],dist[2*w_idx + 1]);
            let ld = self.distribution.create(dist[2*l_idx],dist[2*l_idx + 1]);
            let mut rng = XorShiftRng::seed_from_u64(self.seed + idx as u64 + pass as u64);
            let s_wins = (0..self.k).map(|_| {
                (wd.sample(&mut rng) > ld.sample(&mut rng)) as usize
            }).sum::<usize>();
            (s_wins as f32 / self.k as f32 - *wins as f32 / *n as f32).powi(2)
        }).sum::<f32>() / rels.len() as f32
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
