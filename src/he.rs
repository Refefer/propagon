extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;
extern crate rand_xorshift;

use std::cmp::Ord;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use ahash::{AHasher};
use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::HashMap;
use rand::prelude::*;
use rand::distributions::Uniform;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

pub enum Sampler {
    MetropolisHastings,
    RandomWalk
}

pub enum Norm {
    L1,
    L2,
    None
}

pub struct HashEmbeddings {
    pub dims: usize,
    pub max_steps: usize,
    pub restarts: f32,
    pub hashes: usize,
    pub sparse_walks: bool,
    pub sampler: Sampler,
    pub norm: Norm,
    pub b: f32,
    pub seed: u64
}

impl HashEmbeddings {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> Vec<(K, Vec<f32>)> {

        // Create graph
        let mut vocab = HashMap::new();
        let mut edges = Vec::new();
        let mut n_nodes = 0;
        for (f_node, t_node, weight) in graph.into_iter() {
            for n in vec![&f_node, &t_node] {
                if !vocab.contains_key(n) {
                    vocab.insert((*n).clone(), n_nodes);
                    edges.push(vec![]);
                    n_nodes += 1;
                }
            }

            let t_idx = vocab[&t_node];
            let f_idx = vocab[&f_node];

            edges[t_idx].push((f_idx, weight));
            edges[f_idx].push((t_idx, weight));
        }

        let mut names: Vec<_> = vocab.into_iter().collect();
        names.sort_by_key(|(name, idx)| *idx);
        let names: Vec<_> = names.into_iter().map(|(name, _)| name).collect();
        eprintln!("Number of Vertices: {}", n_nodes);

        self.generate_embeddings(&edges, &names)
    }

    #[inline(always)]
    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = AHasher::default();
        t.hash(&mut s);
        s.finish()
    }

    fn generate_embeddings<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        edges: &Vec<Vec<(usize, f32)>>,
        names: &Vec<K>
    ) -> Vec<(K, Vec<f32>)> {

        // total edges is 2m
        let total_edges = edges.iter().map(|v| v.len() as f32).sum::<f32>();
        
        // Progress bar time
        eprintln!("Generating random walks...");
        let pb = self.create_pb(edges.len() as u64);
        let mut rng = XorShiftRng::seed_from_u64(self.seed - 1);

        let hashes: Vec<usize> = vec![0; self.hashes].into_iter()
            .map(|_| rng.sample(Uniform::new(0usize, std::usize::MAX)))
            .collect();

        let embeddings: Vec<_> = edges.par_iter().enumerate().map(|(key, _)| {
            let mut rng = XorShiftRng::seed_from_u64(self.seed + key as u64);
            let mut emb = vec![0f32; self.dims];
            
            // Compute random walk counts to generate embeddings
            let mut step = 0;
            let mut u = &key;
            let mut terminated = false;
            while step < self.max_steps {
                step += 1;
                if terminated {
                    u = &key;
                    terminated = false;
                }

                // Check for a restart
                if rng.sample(Uniform::new(0f32, 1f32)) < self.restarts {
                    terminated = true;
                } else {
                    
                    // Update our step count and get our next proposed edge
                    let out = &edges[*u];
                    let v = &out[rng.sample(Uniform::new(0usize, out.len()))].0;

                    match self.sampler {
                        // Always accept
                        Sampler::RandomWalk => u = v,

                        // We scale the acceptance based on node degree
                        Sampler::MetropolisHastings => {
                            let acceptance = edges[*v].len() as f32 / edges[*u].len() as f32;
                            if acceptance > rng.sample(Uniform::new(0f32, 1f32)) {
                                u = v;
                            }
                        }
                    };
                }
                    
                // Hash items, scaling by global beta and node prominance in the graph
                if !self.sparse_walks || terminated {
                    //let mut weight = (edges[*u].len() as f32 / total_edges).powf(self.b);
                    let weight = 1.;
                    for h in &hashes {
                        let hash = HashEmbeddings::calculate_hash(&(h, u)) as usize;
                        let sign = hash & 1;
                        let idx = (hash >> 1) % self.dims;
                        emb[idx] += if sign == 1 { weight } else { -weight };
                    }
                }
            }

            // Normalize embeddings by the overall weight in the graph
            let total_nodes = edges.len() as f32;
            emb.iter_mut().for_each(|wi| {
                let sign = if wi.is_sign_negative() { -1. } else { 1.};

                // Scale item by log
                *wi = sign * ((wi.abs() / self.max_steps as f32) * total_nodes).ln().max(0.);
            });

            // Normalize if necessary
            match self.norm {
                Norm::L1 => {
                    let norm = emb.iter().map(|v| (*v).abs()).sum::<f32>();
                    emb.iter_mut().for_each(|e| { *e /= norm });
                },
                Norm::L2 => {
                    let norm = emb.iter().map(|v| (*v).powf(2.)).sum::<f32>().powf(0.5);
                    emb.iter_mut().for_each(|e| { *e /= norm });
                },
                Norm::None => {}
            }

            pb.inc(1);
            (names[key].clone(), emb)
        }).collect();

        pb.finish();
        embeddings
    }

    fn create_pb(&self, total_work: u64) -> ProgressBar {
        let pb = ProgressBar::new(total_work);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
        pb
    }

}

