extern crate heap;
extern crate hashbrown;
extern crate rand;
extern crate rayon;

use std::hash::Hash;

use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;
use super::vp::{Embedding};
use super::utils;

pub struct VecWalk {
    pub n_iters: usize,
    pub error: f32,
    pub max_terms: usize,
    pub alpha: f32,
    pub chunks: usize,
    pub walk_len: usize,
    pub context_window: usize,
    pub negative_sample: usize,
    pub seed: u64
}

struct ThreadTemp<F: Hash + Eq + Send + Sync> {
    features: HashMap<F, f32>,
    indices: Vec<usize>,
    rng: StdRng
}

impl VecWalk {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync, F: Hash + Eq + Clone + Ord + Send + Sync>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>, 
        prior: &HashMap<K,Embedding<F>>
    ) -> HashMap<K, Embedding<F>> {

        // Create graph
        let mut edges = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node).or_insert_with(|| vec![]);
            e.push((f_node, weight));
        }

        eprintln!("Number of Vertices: {}", edges.len());

        // Setup initial embeddings
        let mut keys: Vec<_> = edges.keys().map(|k| k.clone()).collect();
        let mut verts: HashMap<_,_> = keys.iter()
            .map(|key| {
                let e = if let Some(emb) = prior.get(key) {
                    let mut e = emb.clone();
                    if emb.0.len() < self.max_terms {
                        e.0.reserve(self.max_terms - emb.0.len());
                    }
                    e
                } else {
                    Embedding::with_capacity(self.max_terms)
                };
                (key.clone(), e)
            }).collect();

        // We randomly sort our keys each pass in the same style as label embeddings
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut temp_space: Vec<ThreadTemp<F>> = (0..self.chunks)
            .map(|i| {
                ThreadTemp {
                    features: HashMap::with_capacity(
                        (2 * self.context_window + 1) * self.max_terms),
                    indices: (0..self.walk_len).collect(),
                    rng: rand::rngs::StdRng::seed_from_u64(self.seed + 1 + i as u64)
                }
            }).collect();

        for n_iter in 0..self.n_iters {
            eprintln!("Iteration: {}", n_iter);
            keys.shuffle(&mut rng);

            // We go over the keys in chunks so that we can parallelize them
            for key_subset in keys.as_slice().chunks(self.chunks) {

                let it = key_subset.par_iter().zip(temp_space.par_iter_mut());
                let maps: Vec<_> = it.map(|(key, ts)| {
                    let mut new_map = HashMap::new();

                    // Generate Random Walk
                    let walk = gen_walk(key, &edges, &mut ts.rng, self.walk_len);

                    // We randomize the indices to break up linear percolation
                    ts.indices.as_mut_slice().shuffle(&mut ts.rng);

                    let mut features = &mut ts.features;
                    
                    for &i in ts.indices.iter() {
                        features.clear();
                        let get = |x| {
                            new_map.get(x).or_else(|| verts.get(x))
                                .expect("verts should have the context, always")
                        };
                        let ctx = walk[i];

                        // If we've already processed it this walk, move along.
                        if new_map.contains_key(ctx) {continue}

                        // Get the window framing
                        let (left, right) = get_bounds(i, self.context_window, walk.len());

                        // Compute the sum of the embeddings in the walk
                        for vert in walk[left..right].iter() {
                            for (f, v) in get(vert).iter() {
                                if let Some(nv) = features.get_mut(f) {
                                    *nv += *v;
                                } else {
                                    features.insert(f.clone(), *v);
                                }
                            }
                        }

                        // Subtract random negative samples from the mean
                        for _ in 0..self.negative_sample {
                            let random_negative = keys.as_slice()
                                .choose(&mut ts.rng)
                                .expect("Should always have at least 1 key");

                            for (f, v) in get(random_negative).iter() {
                                if let Some(nv) = features.get_mut(f) {
                                    *nv = (*nv - *v).max(0.);
                                } 
                            }
                        }

                        // L2 norm the embedding
                        utils::l2_norm_hm(&mut features);

                        // Check if there is a prior, adding blending it if so
                        if self.alpha < 1. {
                            utils::update_prior(
                                features, ctx, &prior, self.alpha, true);
                        }

                        // Clean up data
                        let mut t_emb = Vec::with_capacity(self.max_terms);
                        utils::clean_map(features, &mut t_emb, self.error, self.max_terms);
                        new_map.insert(ctx.clone(), Embedding(t_emb));
                    }
                    new_map
                }).collect();

                // Swap in the new embeddings
                for map in maps.into_iter() {
                    for (k, new_e) in map.into_iter() {
                        verts.insert(k, new_e);
                    }
                }

            }

        }

        // L2 normalize on the way out
        verts.par_values_mut().for_each(|v| {
            utils::l2_normalize(&mut (v.0));
        });
        verts
    }
}


fn gen_walk<'a, K: Hash + Eq + Clone, R: Rng>(
    mut key: &'a K, 
    edges: &'a HashMap<K, Vec<(K, f32)>>, 
    mut rng: &mut R,
    walk_len: usize
) -> Vec<&'a K> {
    let mut walk = Vec::with_capacity(walk_len+1);
    walk.push(key);
    for _ in 0..walk_len {
        key = &edges[key]
            .choose_weighted(&mut rng, |(_, w)| *w)
            .expect("Should never be empty!").0;
        walk.push(key);
    }
    walk
}

fn get_bounds(i: usize, context_window: usize, walk_len: usize) -> (usize, usize) {
    // Get the window framing
    let left = if i < context_window {
        0 
    } else {
        i - context_window
    };
    let right = (i + context_window + 1).min(walk_len);
    (left, right)
}

#[cfg(test)]
mod test_vec_walk {
    use super::*;

    #[test]
    fn test_bounds() {
        let walk_len = 5;
        assert_eq!(get_bounds(0, 3, walk_len), (0, 4));
        assert_eq!(get_bounds(4, 3, walk_len), (1, 5));
        assert_eq!(get_bounds(2, 3, walk_len), (0, 5));
    }
}
