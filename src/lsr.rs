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

    pub fn fit(&self, games: Games) -> Vec<(u32, f32)> {
        
        // Build sparse markov chain
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

            *sparse_chain[loser_idx].entry(winner_idx).or_insert(0.) += amt / 2f32;
        }

        // Convert to more efficient format.  We compute the cumulative sum to make weighted
        // sampling faster (using bisection) and preserve the original degree weight to ensure
        // we can debias popularity.
        let sparse_chain: Vec<NormedAdjList> = sparse_chain.into_par_iter().map(|v| {
            let mut adj_list: Vec<_> = v.into_iter().collect();
            let mut degree = adj_list.iter().map(|(_k, v)| *v).sum::<f32>();
            let mut c = 0.;
            adj_list.iter_mut().for_each(|(k, v)| {
                c += *v;
                *v = c / degree;
            });
            NormedAdjList { degree, adj_list }
        }).collect();

        // Threads, naturally
        let sparse_chain = Arc::new(sparse_chain);

        // Compute stationary distribution via 
        let pi: Arc<Vec<_>> = Arc::new((0..sparse_chain.len()).map(|_|AtomicUsize::new(1)).collect());
        let n_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let work_per_thread = self.stationary_steps / n_threads;

        let handles: Vec<_> = (0..n_threads).map(|i| {
            let seed = self.seed + i as u64;
            let pi = pi.clone();
            let sparse_chain = sparse_chain.clone();
            let work_per_thread = work_per_thread.clone();
            thread::spawn(move || {
                let mut rng = XorShiftRng::seed_from_u64(seed);
                let dist = Uniform::new(0usize, sparse_chain.len());
                let mut cur_node = rng.sample(dist);
                for n in 0..work_per_thread {
                    pi[cur_node].fetch_add(1, Ordering::Relaxed);
                    // If we end at a node without a loss, uniformly random
                    // transition
                    let adj_list = &sparse_chain[cur_node].adj_list;
                    if adj_list.is_empty() || n % 1_000 == 0 {
                        cur_node = rng.sample(dist);
                    } else {
                        let p: f32 = rng.gen();
                        let new_idx = adj_list
                            .binary_search_by_key(&FloatOrd(p), |(_idx, weight)| FloatOrd(*weight))
                            .unwrap_or_else(|idx| idx);
                        cur_node = adj_list[new_idx].0;
                    }
                }
            })
        }).collect();
        
        // Wait until all threads are finished
        handles.into_iter()
            .for_each(|h| {
                h.join().expect("Worker thread errored out!");
            });

        let pi = Arc::try_unwrap(pi).expect("Not all worker threads dead!");
        let sparse_chain = Arc::try_unwrap(sparse_chain).expect("Not all worker threads dead!");

        // Discount degree impact
        let mut pi: Vec<f32> = pi.into_iter().enumerate()
            .map(|(i, x)| (x.into_inner() as f32 / sparse_chain[i].degree.powf(0.5)).ln() )
            .collect();

        // Log norm results
        let avg = pi.iter().cloned().sum::<f32>() / pi.len() as f32;
        pi.iter_mut().for_each(|v| *v -= avg);

        // Reconstruct from vocab
        vocab.into_iter()
            .map(|(v, idx)| (v, pi[idx]))
            .collect()
                

    }
}

#[cfg(test)]
mod test_page_rank {
    use super::*;

    #[test]
    fn test_example() {
        let matches = vec![
            (1, 2, 1.),
            (3, 2, 1.),
            (1, 3, 1.),
            (1, 4, 1.),
            (2, 4, 1.),
            (3, 4, 1.),
        ];

        let pr = PageRank::new(0.85, 1, Sink::None);
        let results = pr.compute(matches);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 - 0.427083 < 1e-4);

        assert_eq!(results[1].0, 3);
        assert!(results[1].1 - 0.214583 < 1e-4);

        assert_eq!(results[2].0, 2);
        assert!(results[2].1 - 0.108333 < 1e-4);

        assert_eq!(results[3].0, 4);
        assert!(results[3].1 - 0.0375 < 1e-4);
    }

    #[test]
    fn test_reverse() {
        let matches = vec![
            (1, 2, 1.),
            (3, 2, 1.),
            (1, 3, 1.),
            (1, 4, 1.),
            (2, 4, 1.),
            (3, 4, 1.),
        ];

        let pr = PageRank::new(0.85, 10, Sink::Reverse);
        let results = pr.compute(matches);

        assert_eq!(results[0].0, 1);
        assert!(results[0].1 - 0.39064 < 1e-4);

        assert_eq!(results[1].0, 3);
        assert!(results[1].1 - 0.27099 < 1e-4);

        assert_eq!(results[2].0, 2);
        assert!(results[2].1 - 0.190172 < 1e-4);

        assert_eq!(results[3].0, 4);
        assert!(results[3].1 - 0.14818 < 1e-4);

        assert!((results.into_iter().map(|(_, v)| v).sum::<f32>() -  1f32).abs() < 1e-5);
    }

    #[test]
    fn test_all_links() {
        let matches = vec![
            (1, 2, 1.),
            (3, 2, 1.),
            (1, 3, 1.),
            (1, 4, 1.),
            (2, 4, 1.),
            (3, 4, 1.),
        ];

        let pr = PageRank::new(0.85, 10, Sink::All);
        let results = pr.compute(matches);

        assert_eq!(results[0].0, 1);
        assert!(results[0].1 - 0.39064 < 1e-4);

        assert_eq!(results[1].0, 3);
        assert!(results[1].1 - 0.27099 < 1e-4);

        assert_eq!(results[2].0, 2);
        assert!(results[2].1 - 0.190172 < 1e-4);

        assert_eq!(results[3].0, 4);
        assert!(results[3].1 - 0.14818 < 1e-4);

        assert!((results.into_iter().map(|(_, v)| v).sum::<f32>() - 1f32).abs() < 1e-5);
    }

}
