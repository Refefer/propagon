extern crate rayon;
extern crate hashbrown;

use std::hash::Hash;
use std::cmp::Ordering;

use float_ord::FloatOrd;
use rayon::prelude::*;
use hashbrown::HashMap;

pub struct Kemeny {
    pub passes: usize
}

impl Kemeny {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self, 
        graph_iter: impl Iterator<Item=(K,K,f32)>
    ) -> Vec<K> {

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

        // Insertion sort?
        let mut old_policy: Vec<_> = (0..n_vocab).collect();
        let mut policy: Vec<_> = Vec::with_capacity(n_vocab);
        for pass in 0..self.passes {
            eprintln!("Pass: {}", pass);
            policy.clear();
            for idx in old_policy.iter().cloned() {
                let mut best_idx = 0;
                let mut best_kem: usize = usize::MIN;
                let g = &graph[&idx];
                for cur_pos in 0..(policy.len() + 1) {
                    // Score the item at position
                    let mut score = (0..cur_pos).map(|i| {
                        if let Some((wins, n)) = g.get(&policy[i]) {
                            // If `idx` has won more games than left hand side
                            // return the number of games it has won
                            // otherwise
                            // Ok, we got it right.  return the number of wins
                            if 2 * *wins > *n {
                                // 3 wins, 5 games, which means item idx should be to the left of i and
                                //   we got it wrong. 
                                *wins
                            } else if 2 * *wins < *n {
                                // 2 wins, 5 games
                                *wins 
                            } else {
                                0
                            }
                        } else {
                            // No comparisons, so nothing to add
                            0
                        }
                    }).sum::<usize>();

                    // Score the items "greater" than current item
                    score += (cur_pos..policy.len()).map(|i| {
                      if let Some((wins, n)) = g.get(&policy[i]) {
                            if 2 * *wins > *n {
                                *n - *wins
                            } else if 2 * *wins < *n {
                                *n - *wins
                            } else {
                                0
                            }
                        } else {
                            0
                        }
                    }).sum::<usize>();

                    if score > best_kem {
                        best_idx = cur_pos;
                        best_kem = score;
                    }

                }

                policy.insert(best_idx, idx);

            }
            if old_policy.iter().zip(policy.iter()).all(|(left, right)| left == right) {
                break
            }
            std::mem::swap(&mut old_policy, &mut policy);
        }

        policy.into_iter().rev().map(|idx| idx_to_vocab[idx].clone()).collect()
    }

}

