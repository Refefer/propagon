extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::fmt::Write;
use std::hash::Hash;
use std::collections::{VecDeque,BinaryHeap};
use std::cmp::{Reverse,Ordering};

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::{HashMap, HashSet};
use rand::prelude::*;
use rayon::prelude::*;

use crate::utils;
use crate::chashmap::CHashMap;
use crate::de::{Fitness,DifferentialEvolution};

#[derive(Debug)]
struct BestDegree<K>(K, usize);

impl <K> Ord for BestDegree<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.cmp(&other.1)
    }
}

impl <K> PartialOrd for BestDegree<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl <K> Eq for BestDegree<K> {}

impl <K> PartialEq for BestDegree<K> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

#[derive(PartialEq,Eq,Clone,Copy)]
pub enum Distance {

    // Distance is measured in hops rather than edge weights
    Uniform,

    // Distance is measured as Degrees / Edge Weight, biasing against walks
    // through large out-degree vertices.
    DegreeWeighted,

    // Distance is measured as 1. / Edge Weight, biasing toward walks through
    // stronger connections.
    EdgeWeighted
}

#[derive(PartialEq,Eq,Clone,Copy)]
pub enum LandmarkSelection {
    // Selects nodes randomly
    Random,

    // Selects nodes with the highest out-degrees
    Degree
}

pub struct ECTRW {
    pub landmarks: usize,
    pub dims: usize,
    pub global_fns: usize,
    pub local_fns: usize,
    pub distance: Distance,
    pub selection: LandmarkSelection,
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

        let (mut distances, landmarks) = self.compute_landmark_distances(&edges);

        eprintln!("Computed walks, globally embedding landmarks");

        // Gather the landmarks
        let landmark_walks = landmarks.iter().map(|l| {
            distances[l].as_slice()
        }).collect();

        // Map the landmarks into a different dimensional space, attempting
        // to preserve distance measurements
        let embedded_landmarks = self.global_opt(landmark_walks);

        // locally embed each of the points
        let emb_slice = embedded_landmarks.iter().map(|v| v.as_slice()).collect();

        eprintln!("Computed distances, embedding local points...");
        let pb = ProgressBar::new(distances.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));

        distances.par_drain().map(|(k, v)| {
            pb.inc(1);
            (k, self.local_emb(&emb_slice, v, 137))
        }).collect()

    }

    fn global_opt(&self, landmark_dists: Vec<&[f32]>) -> Vec<Vec<f32>> {

        let fitness = GlobalLandmarkEmbedding(self.dims, &landmark_dists);

        let total_dims = self.dims * self.landmarks;
        let lambda = 30.max((total_dims as f32).powf(0.5) as usize);
        let de = DifferentialEvolution {
            dims: self.dims * self.landmarks,
            lambda: lambda,
            f: (0.1, 1.5),
            cr: 0.9,
            m: 0.1,
            exp: 3.
        };

        let pb = ProgressBar::new(self.global_fns as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} {eta_precise}"));

        pb.inc(lambda as u64);
        let mut msg = String::new();
        let results = de.fit(&fitness, self.global_fns, self.seed + 2, |best_fit, rem| {
            msg.clear();
            write!(msg, "Error: {:.4}", -best_fit).unwrap();
            pb.set_message(&msg);
            pb.inc(lambda as u64)
        });
        pb.finish();
            
        results.chunks(self.dims).map(|chunks| chunks.to_vec()).collect()

    }

    // Locall embed each point by landmarks
    fn local_emb(&self, emb_landmarks: &Vec<&[f32]>, dist: Vec<f32>, idx: usize) -> Vec<f32> {

        let fitness = LocalLandmarkEmbedding(emb_landmarks, dist.as_slice());

        let de = DifferentialEvolution {
            dims: self.dims,
            lambda: 30,
            f: (0.1, 0.9),
            cr: 0.9,
            m: 0.1,
            exp: 3.
        };

        de.fit(&fitness, self.local_fns, self.seed + idx as u64, |_best_fit, _rem| { })
            
    }

    // computes the walk distances
    fn compute_landmark_distances<K: Hash + Eq + Clone + Send + Sync>(
        &self, 
        edges: &HashMap<K, Vec<(K, f32)>>

    ) -> (HashMap<K, Vec<f32>>, Vec<K>) {
        // Setup initial embeddings
        let keys: Vec<_> = edges.keys().collect();
        let it = keys.iter()
            .map(|key| {
                ((*key).clone(), vec![0.; self.landmarks])
            });

        let embeddings = CHashMap::new(self.chunks);
        let embeddings = embeddings.extend(it);

        // Progress bar time
        let total_work = self.landmarks;
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));

        // We randomly choose a node each time pass
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);

        let start_nodes: Vec<_> = if self.selection == LandmarkSelection::Random {
            keys.as_slice()
                .choose_multiple(&mut rng, self.landmarks)
                .collect()
        } else {
            top_k_nodes(&keys, &edges, self.landmarks)
        };

        // Compute walks for all 
        (0..self.landmarks).into_par_iter().for_each(|idx| {
            let new_distances = if self.distance == Distance::Uniform {
                unweighted_walk_distance(&edges, &start_nodes[idx])
            } else {
                let degree_weighted = self.distance == Distance::DegreeWeighted;
                weighted_walk_distance(&edges, &start_nodes[idx], degree_weighted)
            };

            for (k, dist) in new_distances {
                let mut v = embeddings.get_map(k).write().unwrap();
                let emb = v.get_mut(k).expect("Should never be empty");
                emb[idx] = dist;
            }
            pb.inc(1);
        });

        pb.finish();

        let es = embeddings.into_inner().into_iter().flat_map(|hm| {
            hm.into_iter()
        }).collect();

        (es, start_nodes.into_iter().map(|v| (*v).clone()).collect())
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
    start_node: &'a K,
    degree_weighted: bool
) -> HashMap<&'a K, f32> {
    let mut distance = HashMap::new();
    distance.insert(start_node, 0f32);
    let mut queue = VecDeque::new();

    queue.push_back(start_node);

    while let Some(vert) = queue.pop_front() {
        let cur_dist = distance[vert];

        let degrees = if degree_weighted {
            (1. + edges[vert].len() as f32).ln()
        } else {
            1.
        };
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

fn top_k_nodes<'a, 'b, K: Hash + Eq>(
    keys: &'b Vec<&'a K>,
    edges: &HashMap<K, Vec<(K, f32)>>,
    dims: usize
) -> Vec<&'b &'a K> {
    let mut bh = BinaryHeap::with_capacity(dims + 1);
    for k in keys.iter() {
        let degrees = edges[k].len();
        bh.push(Reverse(BestDegree(k, degrees)));
        if bh.len() > dims {
            bh.pop();
        }
    }
    bh.into_iter().map(|Reverse(BestDegree(k, _))| k).collect()
}

struct GlobalLandmarkEmbedding<'a>(usize, &'a Vec<&'a [f32]>);

impl <'a> Fitness for GlobalLandmarkEmbedding<'a> {

    fn score(&self, candidate: &[f32]) -> f32 {
        let n_cands = self.1.len();
        let dims = self.0;
        let mut err = 0.;
        for i in 0..n_cands {
            let i_start = i * dims;
            let v1 = &candidate[i_start..i_start + dims];
            for j in (i+1)..n_cands {
                let j_start = j * dims;
                let v2 = &candidate[j_start..j_start + dims];
                err += (euc_dist(v1, v2) - self.1[i][j]).abs()
            }
        }

        -err / (n_cands as f32 * (n_cands as f32- 1.) / 2.)
    }
}

struct LocalLandmarkEmbedding<'a>(&'a Vec<&'a [f32]>, &'a [f32]);

impl <'a> Fitness for LocalLandmarkEmbedding<'a> {

    fn score(&self, candidate: &[f32]) -> f32 {
        let n_cands = self.0.len();
        let mut err = 0.;
        for i in 0..n_cands {
            err += (euc_dist(candidate, self.0[i]) - self.1[i]).abs();
        }

        -err / (n_cands as f32)
    }
}


fn euc_dist(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter())
        .map(|(v1i, v2i)| (v1i - v2i).powi(2))
        .sum::<f32>()
        .powf(0.5)
}


#[cfg(test)]
mod test_ect_rw {
    use super::*;

    #[test]
    fn test_top_nodes() {
        let edges: HashMap<_,_> = vec![
            (0usize, vec![(1usize, 1.), (2usize, 1.)]),
            (1usize, vec![(0usize, 1.), (4usize, 1.)]),
            (2usize, vec![(0usize, 1.), (3usize, 1.), (5usize, 1.)]),
            (3usize, vec![(2usize, 1.), (5usize, 1.), (4usize, 1.)]),
            (4usize, vec![(1usize, 1.), (3usize, 2.)]),
            (5usize, vec![(3usize, 1.), (2usize, 1.)])
        ].into_iter().collect();

        let keys = edges.keys().collect();
        let mut v = top_k_nodes(&keys, &edges, 2);
        v.sort();
        assert_eq!(v[0], &&2);
        assert_eq!(v[1], &&3);

    }

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
