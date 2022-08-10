extern crate rayon;
extern crate hashbrown;

use std::hash::Hash;
use std::cmp::Ordering;
use std::fmt::Write;

use crate::de::{DifferentialEvolution,Fitness};

use float_ord::FloatOrd;
use rayon::prelude::*;
use hashbrown::HashMap;
use indicatif::{ProgressBar,ProgressStyle};

pub enum Algorithm {
    Insertion,
    DiffEvo
}

pub struct Kemeny {
    pub passes: usize,
    pub min_obs: usize,
    pub algo: Algorithm
}

impl Kemeny {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self, 
        graph_iter: impl Iterator<Item=(K,K,f32)>
    ) -> Vec<K> {


        // Load up graph
        let mut graph: Vec<HashMap<usize, (usize, usize)>> = Vec::new();
        let mut vocab = HashMap::new();
        let mut idx_to_vocab = Vec::new();
        let mut n_vocab = 0usize;
        
        graph_iter.for_each(|(winner, loser, margin)| {
            // Generalize
            let mut get_idx = |node: K| -> usize{
                *vocab.entry(node.clone()).or_insert_with(||{ 
                    idx_to_vocab.push(node);
                    graph.push(HashMap::new());
                    n_vocab += 1; 
                    n_vocab - 1 
                })
            };

            let w_idx = get_idx(winner);
            let l_idx = get_idx(loser);

            let margin = margin as usize;
            // Add bi-directional
            {
                let s = &mut graph[w_idx].entry(l_idx.clone()).or_insert((0, 0));

                s.0 += margin;
                s.1 += margin;
            }

            // Add loser relationship
            {
                let s = &mut graph[l_idx].entry(w_idx).or_insert((0, 0));

                s.1 += margin;
            }
        });

        eprintln!("Total nodes: {}", n_vocab);

        // Flatten
        let mut n_edges = 0;
        let mut total_pairs = 0;
        graph.iter().for_each(|sg| {
            total_pairs += sg.iter().map(|(_, (_, g))| *g).sum::<usize>();
            n_edges += sg.len();
        });
        total_pairs /= 2;

        eprintln!("Total comparisons: {}", n_edges);

        match self.algo {
            Algorithm::Insertion => {
                self.insertion_kem(graph, n_vocab, total_pairs)
                    .map(|idx| idx_to_vocab[idx].clone()).collect()
            },
            Algorithm::DiffEvo => { 
                self.de_kem(graph, n_vocab, total_pairs)
                    .map(|idx| idx_to_vocab[idx].clone()).collect()
            }
        }
    }

    // Runs the insertion sort version of kemeny.
    fn insertion_kem(
        &self,
        graph: Vec<HashMap<usize, (usize, usize)>>, 
        n_vocab: usize,
        total_pairs: usize
    ) -> impl Iterator<Item=usize> {
        
        // Compute scores
        let compute_score = |g: &HashMap<usize, (usize, usize)>, other: usize, other_to_left:bool| -> usize {
            if let Some((wins, n)) = g.get(&other) {
                if other_to_left {
                    *wins
                } else {
                    *n - *wins
                }
            } else {
                0
            }
        };

        let pb = ProgressBar::new(self.passes as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(self.passes as u64 / 1000);
        let mut msg = String::new();

        // Initial policy is based on sorting alternatives by the total number
        // of matchups it has won.
        let mut old_policy: Vec<_> = {
            let mut scores: Vec<_> = graph.iter().enumerate().map(|(idx, sg)|{
                let wins = sg.iter()
                    .map(|(_, (wins, games))| if *wins * 2 > *games { 1 } else {0})
                    .sum::<usize>();
                (idx, wins)
            }).collect();
            scores.sort_by_key(|(_idx, wins)| *wins);
            scores.into_iter().rev().map(|(idx, _)| idx).collect()
        };

        let mut policy: Vec<_> = Vec::with_capacity(n_vocab);
        // O(N^2) Insertion Sort-ish approach
        let mut last_score = 0;
        for pass in 0..self.passes {
            policy.clear();
            let mut cur_score = 0;
            for idx in old_policy.iter().cloned() {
                let g = &graph[idx];

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

                cur_score += best_kem;
                policy.insert(best_idx, idx);

            }

            msg.clear();
            write!(msg, "Kem: {}/{}", cur_score, total_pairs).expect("Should work");
            pb.set_message(&msg);

            pb.inc(1);
            std::mem::swap(&mut old_policy, &mut policy);
            if last_score == cur_score { break } 
            last_score = cur_score

        }
        pb.finish();

        let min_obs = self.min_obs;
        old_policy.into_iter().rev()
            .filter(move |idx| {
                graph[*idx].iter().map(|(_, (_, n))| n).sum::<usize>() >= min_obs
            })
    }

    fn de_kem(
        &self,
        graph: Vec<HashMap<usize, (usize, usize)>>, 
        n_vocab: usize,
        total_pairs: usize
    ) -> impl Iterator<Item=usize> {
        let graph: Vec<Vec<_>> = graph.into_iter()
            .map(|hm| hm.into_iter().collect())
            .collect();

        let de = DifferentialEvolution {
            dims: n_vocab,
            lambda: (n_vocab as f32).powf(0.7) as usize,
            f: (0.1, 0.9),
            cr: 0.9,
            m: 0.1,
            exp: 3.,
            polish_on_stale: 10000,
            restart_on_stale: 0,
            range: 1.
        };

        let fit_fn = KemenyFit { graph: &graph };
        let pb = ProgressBar::new(self.passes as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(self.passes as u64 / 1000);

        let mut msg = String::new();

        let mut rem = self.passes;
        let (fit, results) = de.fit(&fit_fn, self.passes, 2020, None, |best_fit, fns_remaining| {
            msg.clear();
            write!(msg, "Kem: {}/{}", best_fit, total_pairs).expect("Should work");
            pb.set_message(&msg);

            pb.inc((rem - fns_remaining) as u64);
            rem = fns_remaining;
        });
        pb.finish();

        let min_obs = self.min_obs;
        let mut results: Vec<_> = results.into_iter().enumerate()
            .filter(move |(idx, w)| {
                graph[*idx].iter().map(|(_, (_, n))| n).sum::<usize>() >= min_obs
            }).collect();

        results.sort_by_key(|(_idx, w)| FloatOrd(-*w));
        results.into_iter().map(|(idx, _)| idx)
    }
}

struct KemenyFit<'a> {
    graph: &'a [Vec<(usize, (usize, usize))>]
}

impl <'a> Fitness for KemenyFit<'a> {
    fn score(&self, candidate: &[f32]) -> f32 {
        self.graph.iter().enumerate()
            .map(|(winner_idx, losers)| {
                let w_score = candidate[winner_idx];
                losers.iter().map(|(loser_idx, (wins, n))| {
                    let l_score = candidate[*loser_idx];
                    if *loser_idx < winner_idx {
                        0f32
                    } else if w_score > l_score {
                        *wins as f32 - if *wins * 2 < *n {
                            (w_score - l_score).abs()
                        } else {
                            0.
                        }
                    } else {
                        (*n - *wins) as f32 - if *wins * 2 > *n {
                            (w_score - l_score).abs()
                        } else {
                            0.
                        }
                    }
                }).sum::<f32>()
            }).sum::<f32>()
    }
}

#[cfg(test)]
mod test_kemeny {
    use super::*;

    #[test]
    fn test_simple() {
        let kemeny = Kemeny { passes: 1, 
            min_obs: 1, 
            algo: Algorithm::Insertion};

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
        let kemeny = Kemeny { passes: 10, 
            min_obs: 1, 
            algo: Algorithm::Insertion};

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
