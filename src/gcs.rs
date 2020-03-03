extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::fmt::Write;
use std::hash::Hash;
use std::collections::{VecDeque,BinaryHeap};
use std::cmp::{Reverse,Ordering};
use std::sync::{Arc,Mutex};

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::{HashMap, HashSet};
use rand::prelude::*;
use rayon::prelude::*;

use crate::utils;
use crate::metric::Metric;
use crate::chashmap::CHashMap;
use crate::de::{Fitness,DifferentialEvolution};

#[derive(Ord,Eq,PartialEq,PartialOrd)]
struct BestDegree<K: Ord + Eq + PartialEq + PartialOrd>(usize, K);

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

pub struct GCS<M> {
    pub metric: M,
    pub landmarks: usize,
    pub only_walks: bool,
    pub dims: usize,
    pub global_fns: usize,
    pub local_fns: usize,
    pub neighbor_fns: usize,
    pub distance: Distance,
    pub selection: LandmarkSelection,
    pub local_stablization: Option<f32>,
    pub stable_passes: usize, 
    pub chunks: usize,
    pub l2norm: bool,
    pub seed: u64
}

impl <M: Metric> GCS<M> {

    pub fn fit<K: Hash + Eq + Ord + Clone + Send + Sync>(
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

        let (distances, landmarks) = self.compute_landmark_distances(&edges);

        // Early exit if only showing walks
        if self.only_walks {
            return distances
        }

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

        
        // Setup the embeddings for constant use
        let embeddings = CHashMap::new(self.chunks).extend(distances.keys().map(|k| {
            (k.clone(), vec![0f32; self.dims])
        }));

        // We store the running loss in a mutex
        let data = Arc::new(Mutex::new((0f32, 0usize, String::new())));
        for pass in 0..self.stable_passes {
            eprintln!("Pass: {}/{} - Computed distances, embedding local points...", pass + 1, self.stable_passes);

            let pb = ProgressBar::new(distances.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));

            pb.enable_steady_tick(200);
            pb.set_draw_delta(distances.len() as u64 / 1000);

            distances.par_iter().for_each(|(k, v)| {
                let iter = {data.lock().unwrap().1};
                // Get the embedding
                let emb_orig = {
                    embeddings.get_map(k).read().unwrap()
                        .get(k).unwrap()
                        .clone()
                };

                let (loss, new_emb) = self.local_emb(&emb_slice, &distances[k], 
                                                     emb_orig.as_slice(), iter);
                pb.inc(1);
                {
                    let mut pl = data.lock().unwrap();
                    (*pl).0 += -loss;
                    (*pl).1 += 1;
                    pl.2.clear();
                    let rate = pl.0 / pl.1 as f32;
                    write!(pl.2, "Avg Loss: {:.5}", rate).unwrap();
                    pb.set_message(&pl.2);
                };
                embeddings.get_map(k).write().unwrap()
                    .insert((*k).clone(), new_emb);
            });

            pb.finish();

            // Check to see if want to preserve local neighborhoods as well.
            if let Some(global_preserve) = self.local_stablization {
                self.embed_neighborhood(global_preserve, &embeddings, &edges);
            }
        }

        embeddings.into_inner().into_iter().flat_map(|mut hm| {
            hm.par_values_mut().for_each(|v| {
                // Norm if we have to
                self.metric.normalize(v.as_mut_slice());
                if self.l2norm {
                    utils::l2_norm(v);
                }
            });
            hm.into_iter()
        }).collect()

    }

    fn global_opt(&self, landmark_dists: Vec<&[f32]>) -> Vec<Vec<f32>> {

        let fitness = GlobalLandmarkEmbedding(self.dims, &landmark_dists, &self.metric);

        let total_dims = self.dims * self.landmarks;
        let lambda = 30.max((total_dims as f32).powf(0.8) as usize);
        let init = self.metric.component_range(self.dims);
        let de = DifferentialEvolution {
            dims: self.dims * self.landmarks,
            lambda: lambda,
            f: (0.1, 1.5),
            cr: 0.9,
            m: 0.1,
            exp: 3.,
            restart_on_stale: 100,
            range: init
        };

        let pb = ProgressBar::new(self.global_fns as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} {eta_precise}"));

        pb.inc(lambda as u64);
        let mut msg = String::new();
        let (_, results) = de.fit(&fitness, self.global_fns, self.seed + 2, 
                                  None, |best_fit, _rem| {
            msg.clear();
            write!(msg, "Loss: {:.5}", -best_fit).unwrap();
            pb.set_message(&msg);
            pb.inc(lambda as u64)
        });
        pb.finish();
            
        results.chunks(self.dims).map(|chunks| chunks.to_vec()).collect()

    }

    // Locall embed each point by landmarks
    fn local_emb(
        &self, 
        emb_landmarks: &Vec<&[f32]>, 
        dist: &[f32], 
        x_in: &[f32],
        idx: usize
    ) -> (f32, Vec<f32>) {

        let fitness = LocalLandmarkEmbedding{
            landmarks: emb_landmarks, 
            landmarks_dists: dist, 
            metric: &self.metric
        };

        let init = self.metric.component_range(self.dims);
        let de = DifferentialEvolution {
            dims: self.dims,
            lambda: 30,
            f: (0.1, 0.9),
            cr: 0.9,
            m: 0.1,
            exp: 3.,
            restart_on_stale: 10,
            range: init
        };

        de.fit(&fitness, self.local_fns, self.seed + idx as u64, 
               Some(x_in), |_best_fit, _rem| { })
    }

    fn embed_neighborhood<K: Hash + Eq + Clone + Send + Sync>(
        &self, 
        global_preserve: f32,
        embeddings: &CHashMap<K, Vec<f32>>,
        edges: &HashMap<K,Vec<(K, f32)>>
    ) {
        // Load everything into a concurrent hashmap
        let init = self.metric.component_range(self.dims);
        let de = DifferentialEvolution {
            dims: self.dims,
            lambda: 30,
            f: (0.1, 0.9),
            cr: 0.9,
            m: 0.1,
            exp: 3.,
            restart_on_stale: 10,
            range: init
        };

        eprintln!("Computing neighborhoods...");
        let pb = ProgressBar::new((edges.len()) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));

        pb.enable_steady_tick(200);
        pb.set_draw_delta(edges.len() as u64 / 1000);
        // We store the running loss in a mutex

        let fits = Arc::new(Mutex::new((0f32, 0usize, String::new())));

        // Get keys and iterator
        edges.par_iter().for_each(|(k, es)| {

            // Get original embedding
            let emb_orig = {
                embeddings.get_map(k).read().unwrap()
                    .get(k).unwrap()
                    .clone()
            };

            // Get neighbors
            let hms = embeddings.cache(es.iter().map(|(k, _v)| k.clone()));

            // Construct weighted neighbors
            let (neighbor_emb, dists): (Vec<_>, Vec<_>) = es.iter().map(|(k, w)| {
                let n_emb = hms.get(k).expect("Should always exist");
                (n_emb.as_slice(), w)
            }).unzip();

            let fitness = LocalNeighborEmbedding {
                global_preserve: global_preserve,
                orig: emb_orig.as_slice(), 
                neighbors: &neighbor_emb,
                neighbors_dists: dists.as_slice(),
                metric: &self.metric
            };

            // Eh, local seed uses weights
            let local_seed = dists.iter().sum::<f32>() as u64;
            
            let (loss, new_emb) = de.fit(
                &fitness, self.neighbor_fns, self.seed + local_seed, 
                   Some(emb_orig.as_slice()), |_best_fit, _rem| { }
            );

            // Insert the new embedding
            embeddings.get_map(k).write().unwrap()
                .insert((*k).clone(), new_emb);

            pb.inc(1);
            {
                let mut pl = fits.lock().unwrap();
                (*pl).0 += -loss;
                (*pl).1 += 1;
                pl.2.clear();
                let rate = pl.0 / pl.1 as f32;
                write!(pl.2, "Avg Loss: {:.5}", rate).unwrap();
                pb.set_message(&pl.2);
            };

        });
        pb.finish();

    }

    // computes the walk distances
    fn compute_landmark_distances<K: Hash + Eq + Ord + Clone + Send + Sync>(
        &self, 
        edges: &HashMap<K, Vec<(K, f32)>>

    ) -> (HashMap<K, Vec<f32>>, Vec<K>) {
        // Setup initial embeddings
        let mut keys: Vec<_> = edges.keys().collect();
        keys.sort();
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
            let new_dist = if degree_weighted {
                let out_degrees = (1. + edges[out_edge].len() as f32).ln();
                cur_dist + degrees.max(out_degrees) / (1. + wi).ln()
            } else {
                cur_dist + degrees / (1. + wi).ln()
            };

            let out_dist = *distance.get(&out_edge).unwrap_or(&std::f32::INFINITY);
            if new_dist < out_dist {
                distance.insert(&out_edge, new_dist);
                queue.push_back(&out_edge);
            }
        }
    }

    distance
}

fn top_k_nodes<'a, 'b, K: Hash + Eq + Ord>(
    keys: &'b Vec<&'a K>,
    edges: &HashMap<K, Vec<(K, f32)>>,
    dims: usize
) -> Vec<&'b &'a K> {
    let mut bh = BinaryHeap::with_capacity(dims + 1);
    for k in keys.iter() {
        let degrees = edges[k].len();
        bh.push(Reverse(BestDegree(degrees, k)));
        if bh.len() > dims {
            bh.pop();
        }
    }
    bh.into_iter().map(|Reverse(BestDegree(_, k))| k).collect()
}

struct GlobalLandmarkEmbedding<'a, M>(usize, &'a Vec<&'a [f32]>, &'a M);

impl <'a,M: Metric> Fitness for GlobalLandmarkEmbedding<'a, M> {

    fn score(&self, candidate: &[f32]) -> f32 {
        
        let n_cands = self.1.len();
        let dims = self.0;
        let mut err = 0.;
        for i in 0..n_cands {
            let i_start = i * dims;
            let v1 = &candidate[i_start..i_start + dims];

            if !self.2.in_domain(v1) {
                return std::f32::NEG_INFINITY;
            }

            for j in (i+1)..n_cands {
                let j_start = j * dims;
                let v2 = &candidate[j_start..j_start + dims];
                err += (self.2.distance(v1, v2) - self.1[i][j]).powi(2)
            }
        }

        -err.sqrt() / (n_cands as f32 * (n_cands as f32- 1.) / 2.)
    }
}

struct LocalLandmarkEmbedding<'a,M> {
    landmarks: &'a Vec<&'a [f32]>, 
    landmarks_dists: &'a [f32], 
    metric: &'a M
}

impl <'a, M: Metric> Fitness for LocalLandmarkEmbedding<'a, M> {

    fn score(&self, candidate: &[f32]) -> f32 {
        if !self.metric.in_domain(candidate) {
            return std::f32::NEG_INFINITY;
        }

        let n_cands = self.landmarks.len();
        let mut err = 0.;
        for i in 0..n_cands {
            let d = self.metric.distance(candidate, self.landmarks[i]);
            err += (d - self.landmarks_dists[i]).powi(2);
        }

        -err.sqrt() / (n_cands as f32)
    }
}

struct LocalNeighborEmbedding<'a, M> {
    global_preserve: f32,
    orig: &'a [f32], 
    neighbors: &'a Vec<&'a [f32]>, 
    neighbors_dists: &'a [f32],
    metric: &'a M
}

impl <'a, M: Metric> Fitness for LocalNeighborEmbedding<'a,M> {

    fn score(&self, candidate: &[f32]) -> f32 {
        if !self.metric.in_domain(candidate) {
            return std::f32::NEG_INFINITY;
        }
        let global_score = -self.metric.distance(self.orig, candidate);
        let emb = LocalLandmarkEmbedding{
            landmarks: self.neighbors, 
            landmarks_dists: self.neighbors_dists, 
            metric: self.metric
        };

        let neighbor_score = emb.score(candidate);
        (self.global_preserve * global_score) + (1. - self.global_preserve) * neighbor_score
    }
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
