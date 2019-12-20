extern crate heap;
extern crate hashbrown;
extern crate rand;
extern crate rayon;

use std::hash::Hash;

use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;
use super::vp::{Embedding,l2_normalize};

pub struct VecWalk {
    pub n_iters: usize,
    pub error: f32,
    pub max_terms: usize,
    pub alpha: f32,
    pub chunks: usize,
    pub walk_len: usize,
    pub context_window: usize,
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
            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));
        }

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
        let mut rngs: Vec<_> = (0..self.chunks)
            .map(|i| rand::rngs::StdRng::seed_from_u64(self.seed + 1 + i as u64))
            .collect();

        for n_iter in 0..self.n_iters {
            eprintln!("Iteration: {}", n_iter);
            keys.shuffle(&mut rng);

            // We go over the keys in chunks so that we can parallelize them
            for key_subset in keys.as_slice().chunks(self.chunks) {

                let it = key_subset.par_iter().zip(rngs.par_iter_mut());
                let maps: Vec<_> = it.map(|(key, mut rng)| {
                    let mut new_map = HashMap::new();
                    // Generate Random Walk
                    let walk = gen_walk(key, &edges, &mut rng, self.walk_len);
                    for i in 0..walk.len() {
                        let left = if i < self.context_window {
                            0 
                        } else {
                            i - self.context_window
                        };
                        let right = (i + self.context_window + 1).min(walk.len());
                        let ctx = walk[i];

                        let mut features = HashMap::with_capacity(
                            (right - left) * self.max_terms);

                        // Compute the sum of the embeddings in the walk
                        for vert in walk[left..right].iter() {
                            for (f, v) in (&verts[vert].0).iter() {
                                if let Some(nv) = features.get_mut(f) {
                                    *nv += *v;
                                } else {
                                    features.insert(f.clone(), *v);
                                }
                            }
                        }

                        // L2 norm the embedding
                        l2_norm_hm(&mut features);

                        // Check if there is a prior, adding blending it if so
                        if let Some(p) = prior.get(ctx) {

                            // Scale the data by alpha
                            features.values_mut().for_each(|v| {
                                *v *= self.alpha;
                            });

                            // add the prior
                            for (k, v) in (p.0).iter() {
                                let nv = (1. - self.alpha) * (*v);
                                if features.contains_key(k) {
                                    if let Some(v) = features.get_mut(k) {
                                        *v += nv;
                                    }
                                } else {
                                    features.insert(k.clone(), nv);
                                }
                            }
                        }


                        // Normalize
                        let sum = features.values()
                            .map(|v| (*v).powi(2)).sum::<f32>().powf(0.5);
                        features.values_mut().for_each(|v| *v /= sum);

                        // Clean up data
                        let mut t_emb = Vec::with_capacity(self.max_terms);
                        for (k, f) in features.drain() {
                            // Ignore features smaller than error rate
                            if f.abs() > self.error {
                                // Add items to the heap until it's full
                                if t_emb.len() < self.max_terms {
                                    t_emb.push((k, f));
                                    if t_emb.len() == self.max_terms {
                                        heap::build(t_emb.len(), 
                                            |a,b| a.1 < b.1, 
                                            t_emb.as_mut_slice());
                                    }
                                } else if t_emb[0].1 < f {
                                    // Found a bigger item, replace the smallest item
                                    // with the big one
                                    heap::replace_root(
                                        t_emb.len(), 
                                        |a,b| a.1 < b.1, 
                                        t_emb.as_mut_slice(),
                                        (k, f));
                                    
                                }
                            }
                        }
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
            l2_normalize(&mut (v.0));
        });
        verts
    }
}

fn l2_norm_hm<F: Hash>(features: &mut HashMap<F, f32>) {
    let sum = features.values()
        .map(|v| (*v).powi(2)).sum::<f32>().powf(0.5);
    features.values_mut().for_each(|v| *v /= sum);
}


fn gen_walk<'a, K: Hash + Eq + Clone, R: Rng>(
    key: &'a K, 
    edges: &'a HashMap<K, Vec<(K, f32)>>, 
    mut rng: &mut R,
    walk_len: usize
) -> Vec<&'a K> {
    let mut walk = Vec::with_capacity(walk_len+1);
    walk.push(key);
    for _ in 0..walk_len {
        let (key, _w) = edges[key]
            .choose_weighted(&mut rng, |(_, w)| *w)
            .expect("Should never be empty!");
        walk.push(key);
    }
    walk
}
