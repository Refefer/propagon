extern crate heap;
extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash,Hasher};
use std::thread;
use std::cell::RefCell;
use std::sync::Arc;
use std::ops::DerefMut;

use thread_local::ThreadLocal;
use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;
use super::vp::{Embedding};
use super::utils;
use crate::chashmap::CHashMap;

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
        let it = keys.iter()
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
            });

        let verts = CHashMap::new(self.chunks);
        let verts = verts.extend(it);

        // We randomly sort our keys each pass in the same style as label embeddings
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let ts_rng = Arc::new(ThreadLocal::new());
        let ts_feats = Arc::new(ThreadLocal::new());
        let ts_inds = Arc::new(ThreadLocal::new());

        for n_iter in 0..self.n_iters {
            eprintln!("Iteration: {}", n_iter);
            keys.shuffle(&mut rng);

            // We go over the keys in chunks so that we can parallelize them
            keys.par_iter().for_each(|key| {

                // Create the thread local storage pieces
                let mut rng = ts_rng.get_or(|| {
                   let mut hasher = DefaultHasher::new();
                    thread::current().id().hash(&mut hasher);
                    let seed = hasher.finish();
                    RefCell::new(rand::rngs::StdRng::seed_from_u64(self.seed ^ seed))
                }).borrow_mut();

                let mut features = ts_feats.get_or(|| {
                    RefCell::new(HashMap::with_capacity(
                            (2 * self.context_window + 1) * self.max_terms))
                }).borrow_mut();

                let mut indices = ts_inds.get_or(|| {
                   RefCell::new((0..self.walk_len).collect::<Vec<_>>())
                }).borrow_mut();

                // Generate Random Walk
                let walk = gen_walk(key, &edges, rng.deref_mut(), self.walk_len);

                // We randomize the indices to break up linear percolation
                indices.as_mut_slice().shuffle(rng.deref_mut());

                for &i in indices.iter() {
                    features.clear();
                    let ctx = walk[i];

                    // Get the window framing
                    let (left, right) = get_bounds(i, self.context_window, walk.len());

                    // Compute the sum of the embeddings in the walk
                    for vert in walk[left..right].iter() {
                        let v = verts.get_map(vert).read().unwrap();
                        for (f, v) in v.get(vert).unwrap().iter() {
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
                            .choose(rng.deref_mut())
                            .expect("Should always have at least 1 key");

                        let v = verts.get_map(random_negative).read().unwrap();
                        for (f, v) in v.get(random_negative).unwrap().iter() {
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
                            &mut features, ctx, &prior, self.alpha, true);
                    }

                    // Clean up data
                    let mut t_emb = Vec::with_capacity(self.max_terms);
                    utils::clean_map(&mut features, &mut t_emb, self.error, self.max_terms);

                    let mut v = verts.get_map(ctx).write().unwrap();
                    v.insert(ctx.clone(), Embedding(t_emb));
                }
            });

        }

        // L2 normalize on the way out
        verts.into_inner().into_iter().flat_map(|mut m| {
            m.par_values_mut().for_each(|v| {
                utils::l2_normalize(&mut (v.0));
            });
            m.into_iter()
        }).collect()
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
