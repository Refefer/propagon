extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::fmt::Write;
use std::hash::Hash;
use std::sync::{Arc,Mutex};

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::HashMap;
use rand::prelude::*;
use rand::distributions::Uniform;
use rayon::prelude::*;

use crate::utils;
use crate::metric::Metric;
use crate::chashmap::CHashMap;
use crate::de::{Fitness,DifferentialEvolution};

#[derive(PartialEq,Eq,Clone,Copy)]
pub enum Distance {

    // Uses the original weights on the graph
    Original,

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
    pub distance: Distance,
    pub selection: LandmarkSelection,
    pub global_bias: f32,
    pub passes: usize, 
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
            // Modify the weights, depending on our distance computation
            let weight = match self.distance {
                Distance::Original     => weight,
                Distance::Uniform      => 1.,
                _ => 1. / (1. + weight as f32).ln()
            };

            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));
        }

        // Need to scale the weights by degree distance
        if self.distance == Distance::DegreeWeighted {
            let new_weights: Vec<_> = edges.iter().map(|(k, out)| {
                let v_degree = (1. + out.len() as f32).ln();
                let xs: Vec<_> = out.iter().map(|(v, w)| {
                    let out_degree = (1. + edges[v].len() as f32).ln();
                    w * v_degree.max(out_degree)
                }).collect();

                (k.clone(), xs)
            }).collect();
            
            // Update weights
            for (v, ws) in new_weights.into_iter() {
                let out = edges.get_mut(&v).unwrap();
                out.iter_mut().zip(ws.into_iter()).for_each(|(edge, nw)| {
                    edge.1 = nw;
                });
            }

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
        let mut emb_rng = rand::rngs::StdRng::seed_from_u64(self.seed + 10);
        let c_range = self.metric.component_range(self.dims);
        let dist = Uniform::new(-c_range, c_range);
        let embeddings = CHashMap::new(self.chunks).extend(distances.keys().map(|k| {
            let mut emb = vec![0f32; self.dims];
            emb.iter_mut().for_each(|vi| *vi = dist.sample(&mut emb_rng));
            (k.clone(), emb)
        }));

        // Many passes
        eprintln!("Computing global and local neighborhoods...");
        
        let pb = ProgressBar::new((self.passes * edges.len()) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));

        pb.enable_steady_tick(200);
        pb.set_draw_delta(edges.len() as u64 / 1000);

        let mut keys: Vec<_> = edges.keys().map(|k| k.clone()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + 1);
        for pass in 0..self.passes {
            keys.shuffle(&mut rng);
            self.global_local_embed(&pb, pass, &emb_slice, &distances, &embeddings, &edges, &keys);
        }
        pb.finish();
        
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
            polish_on_stale: 20,
            restart_on_stale: 50,
            range: init
        };

        let pb = ProgressBar::new(self.global_fns as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} {eta_precise}"));

        pb.inc(lambda as u64);
        let mut msg = String::new();
        let (best_fit, results) = de.fit(&fitness, self.global_fns, self.seed + 2, 
                                  None, |best_fit, _rem| {
            msg.clear();
            write!(msg, "Loss: {:.5}", -best_fit).unwrap();
            pb.set_message(&msg);
            pb.inc(lambda as u64)
        });
        pb.finish();

        eprintln!("Best fit for global optimization: {:.5}", -best_fit);
            
        results.chunks(self.dims).map(|chunks| chunks.to_vec()).collect()

    }

    fn global_local_embed<K: Hash + Eq + Clone + Send + Sync>(
        &self, 
        pb: &ProgressBar,
        pass: usize,
        landmarks: &Vec<&[f32]>,
        distances: &HashMap<K,Vec<f32>>,
        embeddings: &CHashMap<K, Vec<f32>>,
        edges: &HashMap<K,Vec<(K, f32)>>,
        keys: &Vec<K>
    ) {
        // Load everything into a concurrent hashmap
        let init = self.metric.component_range(self.dims);
        let de = DifferentialEvolution {
            dims: self.dims,
            lambda: 30,
            f: (0.1, 1.5),
            cr: 0.9,
            m: 0.1,
            exp: 3.,
            polish_on_stale: 10,
            restart_on_stale: 0,
            range: init
        };

        // We store the running loss in a mutex
        let fits = Arc::new(Mutex::new((0f32, 0usize, String::new())));

        // Get keys and iterator
        keys.par_iter().for_each(|k| {
            let es = &edges[k];

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

            let fitness = GlobalLocalEmbedding {
                metric: &self.metric,
                global: LocalLandmarkEmbedding {
                    landmarks: landmarks,
                    landmarks_dists: &distances[k],
                    metric: &self.metric
                },
                neighborhood: LocalLandmarkEmbedding {
                    landmarks: &neighbor_emb,
                    landmarks_dists: &dists,
                    metric: &self.metric
                },
                blend: self.global_bias 
            };
            
            // Eh, local seed uses weights
            let local_seed = dists.iter().sum::<f32>() as u64;
            
            let (loss, new_emb) = de.fit(
                &fitness, self.local_fns, self.seed + local_seed, 
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
                write!(pl.2, "({}/{}), Avg Loss: {:.5}", pass + 1, 
                       self.passes, rate).unwrap();
                pb.set_message(&pl.2);
            };

        });
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
            utils::top_k_nodes(&keys, &edges, self.landmarks)
        };

        // Compute walks for all 
        (0..self.landmarks).into_par_iter().for_each(|idx| {
            let new_distances = if self.distance == Distance::Uniform {
                utils::unweighted_walk_distance(&edges, &start_nodes[idx])
            } else {
                utils::weighted_walk_distance(&edges, &start_nodes[idx])
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

struct GlobalLandmarkEmbedding<'a, M>(usize, &'a Vec<&'a [f32]>, &'a M);

impl <'a,M: Metric> Fitness for GlobalLandmarkEmbedding<'a, M> {

    fn score(&self, candidate: &[f32]) -> f32 {
        
        let n_cands = self.1.len();
        let dims = self.0;
        let mut err = 0.;

        let sq_dist = self.2.square_distance();
        for i in 0..n_cands {
            let i_start = i * dims;
            let v1 = &candidate[i_start..i_start + dims];

            if !self.2.in_domain(v1) {
                return std::f32::NEG_INFINITY;
            }

            for j in (i+1)..n_cands {
                let j_start = j * dims;
                let v2 = &candidate[j_start..j_start + dims];
                let dist = self.2.distance(v1, v2) - self.1[i][j];
                err += if sq_dist {dist.powi(2)} else {dist.abs()};
            }
        }

        if sq_dist {
            err = err.sqrt();
        }

        -err / (n_cands as f32 * (n_cands as f32 - 1.) / 2.)
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
        let sq_dist = self.metric.square_distance();
        for i in 0..n_cands {
            let d = self.metric.distance(candidate, self.landmarks[i]);
            let dist = d - self.landmarks_dists[i];
            err += if sq_dist {dist.powi(2)} else {dist.abs()};
        }

        if sq_dist {
            err = err.sqrt();
        }
        -err / (n_cands as f32)
    }
}

struct GlobalLocalEmbedding<'a, M> {
    metric: &'a M,
    global: LocalLandmarkEmbedding<'a, M>,
    neighborhood: LocalLandmarkEmbedding<'a, M>,
    blend: f32
}

impl <'a, M: Metric> Fitness for GlobalLocalEmbedding<'a, M> {

    fn score(&self, candidate: &[f32]) -> f32 {
        if !self.metric.in_domain(candidate) {
            return std::f32::NEG_INFINITY;
        }

        let mut score = 0.;
        if self.blend > 0. {
            let global_score = self.global.score(candidate);
            score += self.blend * global_score;
        }

        if self.blend < 1. {
            let neighbor_score = self.neighborhood.score(candidate);
            score += (1. - self.blend) * neighbor_score;
        }
        score
    }
}



