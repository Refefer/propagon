extern crate heap;
extern crate hashbrown;
extern crate rand;
extern crate rayon;

use std::hash::Hash;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Embedding<F>(pub Vec<(F, f32)>);

impl <F: Ord> Embedding<F> {
    pub fn new(feats: Vec<(F, f32)>) -> Self {
        Embedding(feats)
    }
}

impl <F> Embedding<F> {

    pub fn with_capacity(size: usize) -> Self {
        Embedding(Vec::with_capacity(size))
    }

    pub fn swap(&mut self, rep: &mut Vec<(F, f32)>) {
        std::mem::swap(&mut self.0, rep);
    }
}

#[derive(PartialEq,Eq,Clone,Copy)]
pub enum Regularizer {
    L1,
    L2,
    Symmetric
}

// Used to track records and accumulated weight
pub struct Vertex<F>(pub Embedding<F>, f32);

pub struct VecProp {
    pub n_iters: usize,
    pub regularizer: Regularizer,
    pub error: f32,
    pub max_terms: usize,
    pub alpha: f32,
    pub chunks: usize,
    pub normalize: bool,
    pub seed: u64
}

impl VecProp {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync, F: Hash + Eq + Clone + Ord + Send + Sync>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>, 
        prior: &HashMap<K,Embedding<F>>
    ) -> HashMap<K, Vertex<F>> {

        // Create graph
        let mut edges = HashMap::new();
        let mut weights = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));

            // If we are performing symmetric normalization, track the aggregate
            // weights
            if self.regularizer == Regularizer::Symmetric {
                let w = weights.entry(f_node).or_insert(0.);
                *w += weight;
                let w = weights.entry(t_node).or_insert(0.);
                *w += weight;
            }
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
                let w = weights.get(key).unwrap_or(&0.).clone().powf(0.5);
                (key.clone(), Vertex(e, w))
            }).collect();

        std::mem::drop(weights);

        // We randomly sort our keys each pass in the same style as label embeddings
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        //let mut feature_sets = vec![HashMap::new(); self.chunks];
        let mut tmp_embeddings: Vec<Vec<(F,f32)>> = 
            vec![Vec::with_capacity(self.max_terms); self.chunks];
        for n_iter in 0..self.n_iters {
            eprintln!("Iteration: {}", n_iter);
            keys.shuffle(&mut rng);

            // We go over the keys in chunks so that we can parallelize them
            for key_subset in keys.as_slice().chunks(self.chunks) {

                //let it = key_subset.par_iter()
                //    .zip(feature_sets.par_iter_mut()
                //         .zip(tmp_embeddings.par_iter_mut()));

                let it = key_subset.par_iter().zip(tmp_embeddings.par_iter_mut());

                //it.for_each(|(key, (features, t_emb))| {
                it.for_each(|(key, t_emb)| {
                    let t_edges = &edges[key];
                    let mut features = HashMap::with_capacity(t_edges.len() * self.max_terms);
                    let d1 = verts[key].1;
                    for (t_node, wi) in t_edges {
                        let vert = &verts[&t_node];

                        // We pull out the weights in cases where we are using 
                        // symmetric normalization
                        let scale = if self.regularizer == Regularizer::Symmetric {
                            d1 * vert.1
                        } else {
                            0.
                        };

                        // Compute weighted sum of inbound
                        for (f, v) in (vert.0).0.iter() {
                            let weight = match self.regularizer {
                                Regularizer::Symmetric => wi / scale,
                                _                      => *wi
                            };
                            if let Some(nv) = features.get_mut(f) {
                                *nv += v * weight;
                            } else {
                                features.insert(f.clone(), v * weight);
                            }
                        }
                    }

                    // Scale
                    if self.regularizer != Regularizer::Symmetric {
                        let total_weight: f32 = edges[key].iter().map(|(_, w)| w).sum();
                        features.values_mut().for_each(|v| *v /= total_weight);
                    }

                    // Check for the prior
                    if let Some(p) = prior.get(key) {

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
                    match self.regularizer {
                        Regularizer::L1 => {
                            let sum: f32 = features.values().map(|v| (*v).abs()).sum();
                            features.values_mut().for_each(|v| *v /= sum);
                        },
                        Regularizer::L2 => {
                            let sum: f32 = features.values().map(|v| (*v).powi(2)).sum();
                            features.values_mut().for_each(|v| *v /= sum.powf(0.5));
                        },
                        Regularizer::Symmetric => ()
                    }

                    // Clean up data
                    t_emb.clear();
                    for p in features.drain() {
                        // Ignore features smaller than error rate
                        if p.1.abs() > self.error {
                            // Add items to the heap until it's full
                            if t_emb.len() < self.max_terms {
                                t_emb.push(p);
                                if t_emb.len() == self.max_terms {
                                    heap::build(t_emb.len(), 
                                        |a,b| a.1 < b.1, 
                                        t_emb.as_mut_slice());
                                }
                            } else if t_emb[0].1 < p.1 {
                                // Found a bigger item, replace the smallest item
                                // with the big one
                                heap::replace_root(
                                    t_emb.len(), 
                                    |a,b| a.1 < b.1, 
                                    t_emb.as_mut_slice(),
                                    p);
                                
                            }
                        }
                    }
                });

                // Swap in the new embeddings
                for (k, new_e) in key_subset.iter().zip(tmp_embeddings.iter_mut()) {
                    if let Some(e) = verts.get_mut(k) {
                        e.0.swap(new_e);
                    }
                }
            }

        }

        if self.regularizer == Regularizer::Symmetric {
            // L2 normalize on the way out
            verts.par_values_mut().for_each(|v| {
                l2_normalize(&mut (v.0).0);
            });
        }
        verts
    }
}

#[inline]
fn l2_normalize<A>(vec: &mut Vec<(A, f32)>) {
    let sum: f32 = vec.iter().map(|(_, v)| (*v).powi(2)).sum();
    let sqr = sum.powf(0.5);
    vec.iter_mut().for_each(|p| (*p).1 /= sqr);
}

pub fn load_priors(path: &str) -> (HashMap<u32,Embedding<usize>>, HashMap<usize,String>) {
    let f = File::open(path).expect("Error opening priors file");
    let br = BufReader::new(f);

    let mut prior = HashMap::new();
    let mut vocab_to_index = HashMap::new();

    for line in br.lines() {
        let line = line
            .expect("Failed to read line!");

        if let Some(idx) = line.find(" ") {
            let id: u32 = line.as_str()[0..idx].parse().unwrap();

            let mut features = HashMap::new();
            for token in line.as_str()[idx..].split_whitespace() {
                if !vocab_to_index.contains_key(token) {
                    vocab_to_index.insert(token.to_string(), vocab_to_index.len());
                }
                features.insert(vocab_to_index[token].clone(), 1f32);
            }
            // Normalize
            let size = features.len() as f32;
            let feats = features.into_iter()
                .map(|(k, v)| (k, v / size)).collect();
            prior.insert(id, Embedding::new(feats));
        }

    }
    (prior, vocab_to_index.into_iter().map(|(k, v)| (v, k)).collect())
}
