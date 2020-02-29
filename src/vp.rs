extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::ops::Deref;
use std::hash::Hash;
use std::sync::Arc;
use std::cell::RefCell;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use indicatif::{ProgressBar,ProgressStyle};
use thread_local::ThreadLocal;
use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;

use crate::chashmap::CHashMap;
use crate::utils;

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

impl <F> Deref for Embedding<F> {
    type Target = Vec<(F, f32)>;

    fn deref(&self) -> &Self::Target {
        &self.0
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
                let w = weights.get(key).unwrap_or(&0.).clone().powf(0.5);
                (key.clone(), Vertex(e, w))
            });

        let verts = CHashMap::new(self.chunks);
        let verts = verts.extend(it);

        std::mem::drop(weights);

        // We randomly sort our keys each pass in the same style as label embeddings
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);

        let tl_emb = Arc::new(ThreadLocal::new());

        // Progress bar time
        let total_work = edges.len() * self.n_iters;
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);

        for _n_iter in 0..self.n_iters {
            keys.shuffle(&mut rng);

            // We go over the keys in chunks so that we can parallelize them
            keys.par_iter().for_each(|key| {

                let mut t_emb = tl_emb.get_or(|| {
                    RefCell::new(Vec::with_capacity(self.max_terms))
                }).borrow_mut();

                let t_edges = &edges[key];
                let mut features = HashMap::with_capacity(t_edges.len() * self.max_terms);

                let d1 = {
                    let v = verts.get_map(key).read().unwrap();
                    v.get(key).unwrap().1
                };

                // Combine neighbor nodes
                for (t_node, wi) in t_edges {
                    let v = verts.get_map(t_node).read().unwrap();
                    let vert = v.get(t_node).unwrap();

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
                utils::update_prior(&mut features, key, &prior, self.alpha, false);

                // Normalize
                match self.regularizer {
                    Regularizer::L1 => {
                        let sum: f32 = features.values().map(|v| (*v).abs()).sum();
                        features.values_mut().for_each(|v| *v /= sum);
                    },
                    Regularizer::L2 => utils::l2_norm_hm(&mut features),
                    Regularizer::Symmetric => ()
                }

                // Clean up data
                t_emb.clear();
                utils::clean_map(&mut features, &mut t_emb, self.error, self.max_terms);

                let mut vm = verts.get_map(key).write().unwrap();
                if let Some(e) = vm.get_mut(key) {
                    e.0.swap(&mut t_emb);
                }

                pb.inc(1);
            });

        }
        pb.finish();

        verts.into_inner().into_iter().flat_map(|mut m| {
            if self.regularizer == Regularizer::Symmetric {
                m.par_values_mut().for_each(|v| {
                    utils::l2_normalize(&mut (v.0).0);
                });
            }
            m.into_iter()
        }).collect()

    }
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
