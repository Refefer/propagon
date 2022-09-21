extern crate hashbrown;

use std::sync::atomic::{AtomicUsize, Ordering};
use core::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::thread;

use atomic_float::AtomicF64;
use float_ord::FloatOrd;
use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use super::Games;

static EPS: f64 = 1e-8;

type AdjList = Vec<(usize, f32)>;
type MarkovChain = Vec<NormedAdjList>;
type Vocab = HashMap<u32, usize>;
type Weights = Vec<f64>;

#[derive(Debug, Clone)]
struct NormedAdjList {
    degree: f32,
    adj_list: AdjList
}

#[derive(Debug, Clone)]
pub enum Estimator {
    MonteCarlo,
    PowerMethod
}

pub struct LSR {
    pub steps: usize,
    pub estimator: Estimator,
    pub seed: u64
}

impl LSR {

    fn construct_markov_chain(games: Games) -> (Vocab, MarkovChain) {
        
        // Build sparse markov chain.  We use hashmaps to start to make accumulations
        // easier but ultimately convert it into an adjacency list format.
        let mut sparse_chain: Vec<HashMap<usize, f32>> = Vec::new();
        let mut vocab = HashMap::new();
        let mut n_vocab = 0usize;
        for (winner, loser, amt) in games.into_iter() {
            let winner_idx = vocab.entry(winner).or_insert_with(|| {
                n_vocab += 1; 
                sparse_chain.push(HashMap::new());
                n_vocab - 1
            }).clone();

            let loser_idx = vocab.entry(loser).or_insert_with(||{
                n_vocab += 1; 
                sparse_chain.push(HashMap::new());
                n_vocab - 1
            }).clone();

            // This isn't the iterative version, so we assume a uniform prior on 'pi'
            *sparse_chain[loser_idx].entry(winner_idx).or_insert(0.) += amt / 2f32;
        }

        // Convert to more efficient format.  We compute the cumulative sum to make weighted
        // sampling faster (using bisection) and preserve the original degree weight to ensure
        // we can debias popularity.
        let sparse_chain: Vec<NormedAdjList> = sparse_chain.into_par_iter().map(|v| {
            // We L1 norm each node to ensure we have a proper transition matrix
            let degree = v.iter().map(|(_k, v)| *v).sum::<f32>();
            let mut c = 0.;
            let adj_list = v.into_iter().map(|(k, v)| {
                c += v;
                (k, c / degree)
            }).collect();
            NormedAdjList { degree, adj_list }
        }).collect();

        (vocab, sparse_chain)

    }

    fn atomic_matmul(b: &[f64], b_prime: &[AtomicF64], sparse_chain: &MarkovChain) {
        // Zero out destination
        b_prime.par_iter().for_each(|f| f.store(0., Relaxed));

        sparse_chain.par_iter().enumerate().for_each(|(i, adj_list)| {
            let mut cur = 0.;
            adj_list.adj_list.iter().for_each(|(idx, v)| {
                let f = *v - cur;
                cur = *v;
                let w = b[i] * f as f64;
                b_prime[*idx].fetch_add(w, Relaxed);
            });
        });
    }

    fn power_method_est(&self, sparse_chain: &MarkovChain) -> (f64, Weights) {

        // Compute stationary distribution via 
        let ni = 1. / (sparse_chain.len() as f64);
        let mut pi: Vec<_> = vec![ni; sparse_chain.len()];
        let mut pi_prime: Vec<_> = (0..sparse_chain.len()).map(|_| AtomicF64::new(0.)).collect();
        
        for pass in 0..self.steps {
            LSR::atomic_matmul(&pi, &pi_prime, sparse_chain);

            // L1 Norm and copy over to pi
            let mut denom = AtomicF64::new(0.);
            pi.par_iter_mut().zip(pi_prime.par_iter()).for_each(|(pi_i, pi_prime_i)| {
                let f = pi_prime_i.load(Relaxed) + EPS;
                denom.fetch_add(f, Relaxed);
                *pi_i = f;
            });

            let denom = denom.load(Relaxed);
            pi.par_iter_mut().for_each(|v| *v /= denom);

            let err = LSR::compute_error(&pi, sparse_chain);
            eprintln!("Pass: {}, Error: {:.3e}", pass, err);
        }
        
        let err = LSR::compute_error(pi.as_slice(), sparse_chain);

        // Discount degree impact
        pi.iter_mut().enumerate()
            .for_each(|(i, x)| *x = (*x / sparse_chain[i].degree.powf(0.5) as f64).ln() );

        // Log norm results
        let avg = pi.iter().cloned().sum::<f64>() / pi.len() as f64;
        pi.iter_mut().for_each(|v| *v -= avg); 
        (err, pi)
    }

    fn monte_carlo_est(&self, sparse_chain: &MarkovChain) -> (f64, Weights) {

        // Compute stationary distribution via 
        let pi: Vec<_> = (0..sparse_chain.len()).map(|_| AtomicUsize::new(0)).collect();

        // We randomly sample walks from each node to estimate the likelihood
        // of transitioning to a node (the hit rate).  
        let dist = Uniform::new(0usize, sparse_chain.len());
        (0..sparse_chain.len()).into_par_iter().for_each(|mut cur_node| {
            let mut rng = XorShiftRng::seed_from_u64(self.seed + cur_node as u64);
            for n in 0..self.steps {
                pi[cur_node].fetch_add(1, Ordering::Relaxed);
                let adj_list = &sparse_chain[cur_node].adj_list;
                // Need to teleport to a random node, uniformly, when reaching a state without 
                // transitions out (aka undefeated)
                if adj_list.is_empty() {
                    cur_node = rng.sample(dist);
                } else {
                    let p: f32 = rng.gen();
                    let new_idx = adj_list
                        .binary_search_by_key(&FloatOrd(p), |(_idx, weight)| FloatOrd(*weight))
                        .unwrap_or_else(|idx| idx);
                    cur_node = adj_list[new_idx].0;
                }
            }
        });

        // L1 normalize the atomics into f32s
        let mut s = 0.;
        let mut pi: Vec<f64> = pi.into_iter().enumerate()
            .map(|(i, x)| {
                let f = x.into_inner() as f64;
                s += f;
                f
            })
            .collect();

        pi.par_iter_mut().for_each(|v| *v /= s);

        let err = LSR::compute_error(pi.as_slice(), sparse_chain);

        // Discount degree impact
        pi.iter_mut().enumerate()
            .for_each(|(i, x)| *x = (*x / sparse_chain[i].degree.powf(0.5) as f64).ln() );

        // Log norm results
        let avg = pi.iter().cloned().sum::<f64>() / pi.len() as f64;
        pi.iter_mut().for_each(|v| *v -= avg); 
        (err, pi)
    }

    fn compute_error(pi: &[f64], sparse_chain: &MarkovChain) -> f64 {
        let mut pi_est = vec![0f64; pi.len()];
        sparse_chain.iter().enumerate().for_each(|(i, adj_list)| {
            let mut cur = 0.;
            adj_list.adj_list.iter().for_each(|(idx, v)| {
                let f = *v - cur;
                cur = *v;
                pi_est[*idx] += pi[i] * f as f64;
            });
        });

        pi.iter().zip(pi_est.iter())
            .map(|(pi_1, pi_2)| (*pi_1 - *pi_2).powf(2.))
            .sum::<f64>()
            .sqrt()
    }

    pub fn fit(&self, games: Games) -> Vec<(u32, f64)> {
        
        // Create the comparison graph
        let (vocab, sparse_chain) = LSR::construct_markov_chain(games);
        
        let (err, pi) = match self.estimator {
            Estimator::PowerMethod => self.power_method_est(&sparse_chain),
            Estimator::MonteCarlo  => self.monte_carlo_est(&sparse_chain),
        };

        eprintln!("Error: {:.3e}", err);

        // Reconstruct from vocab
        vocab.into_iter()
            .map(|(v, idx)| (v, pi[idx]))
            .collect()

    }
}

#[cfg(test)]
mod test_lsr {
    use super::*;

    #[test]
    fn test_creating_tournament_graph() {
        let matches = vec![
            (1, 2, 1.),
            (2, 3, 2.),
            (3, 1, 1.)
        ];

        let (vocab, mc) = LSR::construct_markov_chain(matches);
        let assumed_mc = vec![
            NormedAdjList {degree: 0.5, adj_list: vec![(2, 1.0)]},
            NormedAdjList {degree: 0.5, adj_list: vec![(0, 1.0)]},
            NormedAdjList {degree: 1., adj_list: vec![(1, 1.0)]},
        ];
        
        // Deeply frustrating that rust doesn't implement f32 Eq or Ord
        for (a_nadj, r_nadj) in assumed_mc.into_iter().zip(mc.into_iter()) {
            assert_eq!(a_nadj.degree, r_nadj.degree);
            assert_eq!(a_nadj.adj_list, r_nadj.adj_list);
        }
    }

    #[test]
    fn test_error() {
        let matches = vec![
            (1, 2, 1.),
            (1, 3, 10.),
            (2, 3, 3.),
            (2, 1, 1.),
            (3, 1, 1.),
            (3, 2, 2.),
            (4, 1, 1.),
            (4, 3, 2.),
            (2, 4, 2.),
        ];

        let (vocab, mc) = LSR::construct_markov_chain(matches);

        let lsr = LSR {
            steps: 1_000_000,
            estimator: Estimator::MonteCarlo,
            seed: 20202
        };

        let (act_err, pi) = lsr.power_method_est(&mc);
        println!("err: {}", act_err);
        assert!(act_err < 1e-3);
        panic!()
    }


}
