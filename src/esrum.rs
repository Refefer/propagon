extern crate rand;
extern crate rayon;
extern crate hashbrown;

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

#[derive(Debug)]
pub enum Distribution {
    Gaussian,
    FixedNormal,
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
    fn create(&self, mut a: f32, mut b: f32) -> Sampler {
        self.bound(&mut a, &mut b);
        match self {
            Distribution::Gaussian => {
                Sampler::Normal(Normal::new(a, b).unwrap())
            },
            Distribution::FixedNormal => {
                Sampler::Normal(Normal::new(a, b).unwrap())
            },
            Distribution::Gamma => {
                Sampler::Gamma(Gamma::new(a, b).unwrap())
            },
            Distribution::Beta => {
                Sampler::Beta(Beta::new(a, b).unwrap())
            }
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    fn bound(&self, a: &mut f32, b: &mut f32) {
        match self {
            Distribution::Gaussian => {
                *b = Distribution::sigmoid(*b);
            },
            Distribution::FixedNormal => {
                *a = a.abs();
                *b = 1f32;
            },
            Distribution::Gamma | Distribution::Beta=> {
                *a = a.abs().max(1e-5);
                *b = b.abs().max(1e-5);
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

        let mut msg = format!("Pass: 0, Alpha:{}, Fitness: inf", self.alpha);
        pb.set_message(&msg);

        let mut samples = vec![vec![0f32; self.k]; n_vocab];

        for it in 0..self.passes {
            let sigma = self.alpha * 0.99f32.powf(it as f32);
            let updated = AtomicUsize::new(0);

            // Fill the samples
            samples.par_iter_mut().zip(policy.par_iter()).enumerate().for_each(|(idx, (pulls, pi))| {
                let mut rng = XorShiftRng::seed_from_u64(self.seed + (it + 2 * idx) as u64);
                let cd = self.distribution.create(pi[0], pi[1]);
                pulls.iter_mut().for_each(|pi| *pi = cd.sample(&mut rng));
            });

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
                    *fit = self.score(g as &[f32], comps, samples.as_slice(), it + c_idx);
                });

                // Fitness sort them in ascending order
                grads.sort_by_key(|(f, _g)| FloatOrd(*f));

                // Combine into new policy
                new_policy.par_iter_mut().enumerate().for_each(|(p_idx, pi)| {
                    *pi += grads.iter().take(self.children).zip(weights.iter()).map(|((_, g), wi)| {
                        wi * (g[p_idx] - *pi)
                    }).sum::<f32>();
                });

                let old_score = self.score(arr as &[f32], comps, samples.as_slice(), it + c_idx);
                let new_score = self.score(&new_policy as &[f32], comps, samples.as_slice(), it + c_idx);

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
                self.score(stats as &[f32], &graph[&idx].as_slice(), samples.as_slice(), it)
            }).sum::<f32>() / n_vocab as f32;

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

    fn kemeny_loss(&self, o_wins: usize, o_n: usize, s_wins: usize, s_n: usize) -> f32 {
        // Add inversion loss
        let o_rate = o_wins as f32 / o_n as f32;
        let s_rate = s_wins as f32 / s_n as f32;

        if s_rate > 0.5 && o_rate > 0.5 {
            (o_n - o_wins) as f32
        } else if s_rate < 0.5 && o_rate < 0.5 {
            o_wins as f32
        } else {
            (o_wins).max(o_n - o_wins) as f32
        }
    }

    fn score(&self, dist: &[f32], graph: &[(usize, (usize, usize))], samples: &[Vec<f32>], pass: usize) -> f32 {
        let cd = self.distribution.create(dist[0], dist[1]);
        let mut rng = XorShiftRng::seed_from_u64(self.seed + pass as u64);
        let cd_dist: Vec<_> = (0..self.k).map(|_| cd.sample(&mut rng)).collect();
        graph.par_iter().map(|(o_idx, (wins, n))| {
            let s_wins = cd_dist.iter().zip(samples[*o_idx].iter())
                .map(|(w,l)| (w > l) as usize)
                .sum::<usize>();

            // Wasserstein
            let fitness = (s_wins as f32 / self.k as f32 - *wins as f32 / *n as f32).abs();

        
            // penalize the distance from 0
            let smooth_fit = fitness + self.gamma * samples[*o_idx][0].powf(2.);

            smooth_fit + self.kemeny_loss(*wins, *n, s_wins, self.k)
        }).sum::<f32>() / graph.len() as f32
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

