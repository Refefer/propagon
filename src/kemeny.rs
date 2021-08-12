extern crate rayon;
extern crate hashbrown;

use std::hash::Hash;
use std::cmp::Ordering;

use float_ord::FloatOrd;
use rayon::prelude::*;
use hashbrown::HashMap;
use indicatif::{ProgressBar,ProgressStyle};

pub struct Kemeny {
    pub passes: usize
}

impl Kemeny {

//    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
//        &self, 
//        graph_iter: impl Iterator<Item=(K,K,f32)>
//    ) -> Vec<K> {
    pub fn fit(
        &self, 
        graph_iter: impl Iterator<Item=(u32,u32,f32)>
    ) -> Vec<u32> {


        // Load up graph
        let mut graph = HashMap::new();
        let mut vocab = HashMap::new();
        let mut idx_to_vocab = Vec::new();
        let mut n_vocab = 0usize;
        graph_iter.for_each(|(winner, loser, margin)| {
            // Numbers are better
            let w_idx = *vocab.entry(winner.clone()).or_insert_with(||{ 
                idx_to_vocab.push(winner);
                n_vocab += 1; 
                n_vocab - 1 
            });

            let l_idx = *vocab.entry(loser.clone()).or_insert_with(|| { 
                idx_to_vocab.push(loser);
                n_vocab += 1; 
                n_vocab - 1 
            });

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
            let arr = sg.into_iter().collect::<HashMap<_,_>>();
            n_edges += arr.len();
            (w, arr)
        }).collect();

        eprintln!("Total comparisons: {}", n_edges);

        // Compute scores
        let compute_score = |g: &HashMap<usize, (usize, usize)>, other: usize, other_to_left:bool| -> usize {
            if let Some((wins, n)) = g.get(&other) {
                if 2 * wins != *n {
                    if other_to_left {
                        *wins
                    } else {
                        *n - *wins
                    }
                } else {
                    0
                }
            } else {
                0
            }
        };

        let pb = ProgressBar::new(self.passes as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(self.passes as u64 / 1000);

        // Insertion sort?
        let mut old_policy: Vec<_> = (0..n_vocab).collect();
        let mut policy: Vec<_> = Vec::with_capacity(n_vocab);
        for pass in 0..self.passes {
            policy.clear();
            for idx in old_policy.iter().cloned() {
                let g = &graph[&idx];

                // The way this algo works is we sum all the scores once, and while
                // we scan only have to compute the previous location and its swap.
                let mut total_score = policy.iter()
                    .map(|idx| compute_score(g, *idx, false) )
                    .sum::<usize>();

                let mut best_idx = 0;
                let mut best_kem = total_score;

                for cur_pos in 1..(policy.len() + 1) {
                    let other = policy[cur_pos - 1];
                    let orig_order = compute_score(g, other, false);
                    let swap_order = compute_score(g, other, true);

                    total_score = total_score - orig_order + swap_order;

                    if total_score > best_kem {
                        best_idx = cur_pos;
                        best_kem = total_score;
                    }

                }

                policy.insert(best_idx, idx);

            }

            pb.inc(1);
            std::mem::swap(&mut old_policy, &mut policy);
            if old_policy.iter().zip(policy.iter()).all(|(left, right)| left == right) {
                break
            } 

        }
        pb.finish();

        old_policy.into_iter().rev().map(|idx| idx_to_vocab[idx].clone()).collect()
    }

}

#[cfg(test)]
mod test_kemeny {
    use super::*;

    #[test]
    fn test_simple() {
        let kemeny = Kemeny { passes: 1};
        let it = vec![
            (1u32, 0u32, 1f32),
            (2u32, 1u32, 1f32),
            (3u32, 2u32, 1f32)
        ];
        let results = kemeny.fit(it.into_iter());
        let expected = vec![3,2,1,0];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_conflict() {
        let kemeny = Kemeny { passes: 1};
        let it = vec![
            (1u32, 0u32, 1f32),
            (0u32, 1u32, 2f32),
            (2u32, 1u32, 1f32),
            (3u32, 2u32, 1f32)
        ];
        let results = kemeny.fit(it.into_iter());
        let expected = vec![0,3,2,1];
        assert_eq!(results, expected);
    }
}
