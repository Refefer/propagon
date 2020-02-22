extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::hash::Hash;
use std::collections::VecDeque;

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::{HashMap, HashSet};
use rand::prelude::*;
use rayon::prelude::*;

use crate::utils;
use crate::chashmap::CHashMap;

#[derive(PartialEq,Eq,Clone,Copy)]
pub enum Distance {
    Uniform,
    Weighted
}

pub struct ECTRW {
    pub dims: usize,
    pub distance: Distance,
    pub chunks: usize,
    pub l2norm: bool,
    pub seed: u64
}

impl ECTRW {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> HashMap<K, Vec<f32>> {

        // Create graph
        let mut edges = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));
        }

        eprintln!("Number of Vertices: {}", edges.len());

        // Setup initial embeddings
        let keys: Vec<_> = edges.keys().map(|k| k).collect();
        let it = keys.iter()
            .map(|key| {
                ((*key).clone(), vec![0.; self.dims])
            });

        let embeddings = CHashMap::new(self.chunks);
        let embeddings = embeddings.extend(it);

        // Progress bar time
        let total_work = self.dims;
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));

        // We randomly choose a node each time pass
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);

        let start_nodes: Vec<_> = keys.as_slice()
            .choose_multiple(&mut rng, self.dims)
            .collect();

        // Do it!
        (0..self.dims).into_par_iter().for_each(|idx| {
            let new_distances = if self.distance == Distance::Weighted {
                weighted_walk_distance(&edges, &start_nodes[idx])
            } else {
                unweighted_walk_distance(&edges, &start_nodes[idx])
            };

            for (k, dist) in new_distances {
                let mut v = embeddings.get_map(k).write().unwrap();
                let emb = v.get_mut(k).expect("Should never be empty");
                emb[idx] = dist;
            }
            pb.inc(1);
        });

        pb.finish();

        embeddings.into_inner().into_iter().flat_map(|hm| {
            hm.into_iter().map(|(k, mut v)| {
                if self.l2norm { 
                    utils::l2_norm(&mut v); 
                }
                (k, v)
            })
        }).collect()
    }
}

fn unweighted_walk_distance<'a, K: Hash + Eq>(
    edges: &'a HashMap<K, Vec<(K,f32)>>,
    start_node: &'a K
) -> HashMap<&'a K, f32> {
    let mut distance = HashMap::new();
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();

    seen.insert(start_node);
    queue.push_back((start_node, 0.));

    while let Some((vert, cur_dist)) = queue.pop_front() {
        distance.insert(vert, cur_dist);

        for (out_edge, _) in edges[vert].iter() {
            if !seen.contains(&out_edge) {
                seen.insert(&out_edge);
                queue.push_back((&out_edge, cur_dist + 1.));
            }
        }
    }

    distance
}

fn weighted_walk_distance<'a, K: Hash + Eq>(
    edges: &'a HashMap<K, Vec<(K,f32)>>,
    start_node: &'a K
) -> HashMap<&'a K, f32> {
    let mut distance = HashMap::new();
    distance.insert(start_node, 0f32);
    let mut queue = VecDeque::new();

    queue.push_back(start_node);

    while let Some(vert) = queue.pop_front() {
        let cur_dist = distance[vert];

        let degrees = (1. + edges[vert].len() as f32).ln();
        for (out_edge, wi) in edges[vert].iter() {
            let new_dist = cur_dist + degrees / (1. + wi).ln();
            let out_dist = *distance.get(&out_edge).unwrap_or(&std::f32::INFINITY);
            if new_dist < out_dist {
                distance.insert(&out_edge, new_dist);
                queue.push_back(&out_edge);
            }
        }
    }

    distance
}

#[cfg(test)]
mod test_ect_rw {
    use super::*;

    #[test]
    fn test_unweighted_walk() {
        let hm: HashMap<_,_> = vec![
            (0usize, vec![(1usize, 1.), (2usize, 1.)]),
            (1usize, vec![(0usize, 1.), (4usize, 1.)]),
            (2usize, vec![(0usize, 1.), (3usize, 1.)]),
            (3usize, vec![(2usize, 1.), (5usize, 1.)]),
            (4usize, vec![(1usize, 1.)]),
            (5usize, vec![(3usize, 1.)])
        ].into_iter().collect();

        let start_node = 2;
        let distances = unweighted_walk_distance(&hm, &start_node);

        assert_eq!(distances[&0], 1.);
        assert_eq!(distances[&1], 2.);
        assert_eq!(distances[&2], 0.);
        assert_eq!(distances[&3], 1.);
        assert_eq!(distances[&4], 3.);
        assert_eq!(distances[&5], 2.);

    }
}
