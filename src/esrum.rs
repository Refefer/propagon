extern crate rand;
extern crate rayon;
extern crate hashbrown;

use std::hash::Hash;
use std::fmt::Write;
use std::cell::RefCell;
use std::sync::Arc;

use indicatif::{ProgressBar,ProgressStyle};
use rayon::prelude::*;
use hashbrown::HashMap;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution as Dist,Gamma,Normal,Beta};
use float_ord::FloatOrd;
use thread_local::ThreadLocal;

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
            // Add bi-directional
            {
                let w = graph.entry(w_idx.clone()).or_insert_with(|| HashMap::new());
                let s = w.entry(l_idx.clone()).or_insert((0, 0));

                if is_win { s.0 += margin; }
                s.1 += margin;
            }

            // Add loser relationship
            {
                let l = graph.entry(l_idx).or_insert_with(|| HashMap::new());
                let s = l.entry(w_idx).or_insert((0, 0));

                if !is_win { s.0 += margin; }
                s.1 += margin;
            }
        });

        // Flatten
        let graph: HashMap<_,_> = graph.into_iter().map(|(w, sg)| {
            (w, sg.into_iter().collect::<Vec<_>>())
        }).collect();

        let mut rng = XorShiftRng::seed_from_u64(self.seed);
        
        // Create initial policy
        let mut policy = vec![[0f32, 0f32]; n_vocab];
        policy.iter_mut().for_each(|pi| {
            pi[0] = rng.gen();
            pi[1] = rng.gen();
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

        let mut msg = format!("Pass: 0, Fitness: inf");
        pb.set_message(&msg);

        let tl_grads = Arc::new(ThreadLocal::new());
        for it in 0..self.passes {

            let new_policy: Vec<_> = policy.par_iter().enumerate().map(|(c_idx, arr)| {
                let mut grads = tl_grads.get_or(|| {
                    RefCell::new(vec![(0f32, [0f32, 0f32]); self.gradients])
                }).borrow_mut();

                // Randomize gradients and compute
                let normal = Normal::new(0f32, 1f32).unwrap();
                let mut rng = XorShiftRng::seed_from_u64(self.seed + (c_idx + (it + 1) * self.gradients) as u64);
                grads.iter_mut().enumerate().for_each(|(idx, (fit, g))| {
                    g.iter_mut().zip(arr.iter()).for_each(|(vi, pi)| { 
                        *vi = normal.sample(&mut rng) + *pi; 
                    });

                    *fit = self.score(g as &[f32], &graph[&c_idx].as_slice(), policy.as_slice(), it);
                });

                // Fitness sort them in ascending order
                grads.sort_by_key(|(f, _g)| FloatOrd(*f));

                // Combine into new policy
                let mut new_policy = arr.clone();
                new_policy.iter_mut().enumerate().for_each(|(p_idx, pi)| {
                    *pi += self.alpha * grads[..self.children].iter().zip(weights.iter()).map(|((_, g), wi)| {
                        wi * (g[p_idx] - *pi)
                    }).sum::<f32>();
                });
                pb.inc(1);
                new_policy
            }).collect();

            policy = new_policy;
            // Evaluate new fitness
            let s = policy.par_iter().enumerate().map(|(idx, stats)| {
                self.score(stats as &[f32], &graph[&idx].as_slice(), policy.as_slice(), it)
            }).sum::<f32>() / n_vocab as f32;

            msg.clear();
            write!(msg, "Pass: {}, Fitness: {:.5}", it + 1, s);
            pb.set_message(&msg);
        }
        pb.finish();

        vocab.into_iter().map(|(k, idx)| {
            (k, policy[idx].clone())
        }).collect()
    }

    fn score(&self, dist: &[f32], graph: &[(usize, (usize, usize))], policy: &[[f32; 2]], pass: usize) -> f32 {
        let cd = self.distribution.create(dist[0], dist[1]);
        let mut rng = XorShiftRng::seed_from_u64(self.seed + pass as u64);
        graph.iter().enumerate().map(|(idx, (o_idx, (wins, n)))| {
            let policy_dist = policy[*o_idx];
            let od = self.distribution.create(policy_dist[0], policy_dist[1]);
            let s_wins = (0..self.k).map(|_| {
                (cd.sample(&mut rng) > od.sample(&mut rng)) as usize
            }).sum::<usize>();
            (s_wins as f32 / self.k as f32 - *wins as f32 / *n as f32).powi(2)
        }).sum::<f32>() / self.gradients as f32
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
