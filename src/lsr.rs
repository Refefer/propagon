extern crate hashbrown;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use float_ord::FloatOrd;
use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rand_distr::{Distribution,Uniform};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use super::Games;

type AdjList = Vec<(usize, f32)>;
type MarkovChain = Vec<NormedAdjList>;
type Vocab = HashMap<u32, usize>;
type Weights = Vec<f32>;

#[derive(Debug)]
struct NormedAdjList {
    degree: f32,
    adj_list: AdjList
}

pub struct LSR {
    pub stationary_steps: usize,
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

    fn compute_stationary_distribution(&self, sparse_chain: MarkovChain) -> Weights {

        // Compute stationary distribution via 
        let pi: Vec<_> = (0..sparse_chain.len()).map(|_| AtomicUsize::new(0)).collect();

        let dist = Uniform::new(0usize, sparse_chain.len());
        (0..sparse_chain.len()).into_par_iter().for_each(|mut cur_node| {
            let mut rng = XorShiftRng::seed_from_u64(self.seed + cur_node as u64);
            for n in 0..self.stationary_steps {
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

        // Discount degree impact
        let mut pi: Vec<f32> = pi.into_iter().enumerate()
            .map(|(i, x)| (x.into_inner() as f32 / sparse_chain[i].degree.powf(0.5)).ln() )
            .collect();

        // Log norm results
        let avg = pi.iter().cloned().sum::<f32>() / pi.len() as f32;
        pi.iter_mut().for_each(|v| *v -= avg); 
        pi


    }

    pub fn fit(&self, games: Games) -> Vec<(u32, f32)> {
        
        // Create the comparison graph
        let (vocab, sparse_chain) = LSR::construct_markov_chain(games);
        
        let pi = self.compute_stationary_distribution(sparse_chain);

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

}
