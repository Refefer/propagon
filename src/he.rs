extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;
extern crate rand_xorshift;

use std::cmp::Ord;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use std::cell::RefCell;
use thread_local::ThreadLocal;
use ahash::{AHasher};
use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::HashMap;
use rand::prelude::*;
use rand::distributions::Uniform;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

#[derive(PartialEq,Eq)]
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
    pub dims: u16,
    pub max_steps: usize,
    pub restarts: f32,
    pub hashes: usize,
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
    fn calculate_hash<T: Hash>(t: T) -> u64 {
        let mut s = AHasher::default();
        t.hash(&mut s);
        s.finish()
    }

    /*
    fn hash_node(&self, hashes: &[usize], u: usize, emb: &mut [i32]) {
        // Hash items, scaling by global beta and node prominance in the graph
        for h in hashes {
            let hash = HashEmbeddings::calculate_hash((h, u)) as usize;
            let sign = (hash & 1) as i32;
            let idx = (hash >> 1) % self.dims;
            let w = unsafe {
                emb.get_unchecked_mut(idx)
            };
            *w += 2 * sign - 1;
        }
    }
    */

    fn norm_embedding(&self, total_nodes: f32, emb: &mut [i32]) -> Vec<f32> {
        // Normalize embeddings by the overall weight in the graph
        let mut emb: Vec<_> = emb.iter().map(|wi| {
            let sign = if wi.is_negative() { -1. } else { 1.};

            // Scale item by log
            sign * ((wi.abs() as f32 / self.max_steps as f32) * total_nodes).ln().max(0.)
        }).collect();

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
        emb
    }

    fn generate_embeddings<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        edges: &Vec<Vec<(usize, f32)>>,
        names: &Vec<K>
    ) -> Vec<(K, Vec<f32>)> {

        // total edges is 2m
        let total_edges = edges.iter().map(|v| v.len() as f32).sum::<f32>();
        
        // Progress bar time
        let mut rng = XorShiftRng::seed_from_u64(self.seed - 1);

        // haha
        eprintln!("Precomputing hashes...");
        let mut hash_table = vec![(true, 0u16); edges.len() * self.hashes];
        eprintln!("Table Size: {}", std::mem::size_of::<(bool, u16)>() * self.hashes * edges.len());

        let hashes: Vec<usize> = vec![0; self.hashes].into_iter()
            .map(|_| rng.sample(Uniform::new(0usize, std::usize::MAX)))
            .collect();

        // Fill the hashtable
        let d = self.dims as usize;
        hash_table.par_chunks_mut(self.hashes).enumerate().for_each(|(u, slice)| {
            for (i, h) in hashes.iter().enumerate() {
                let hash = HashEmbeddings::calculate_hash((h, u)) as usize;
                let sign = (hash & 1) == 1;
                let idx = (hash >> 1) % d;
                slice[i] = (sign, idx as u16);
            }
        });

        eprintln!("Generating random walks...");
        let pb = self.create_pb(edges.len() as u64);

        let counts = Arc::new(ThreadLocal::new());
        let uni_dist = Uniform::new(0f32, 1f32);
        let embeddings: Vec<_> = edges.par_iter().enumerate().map(|(key, _)| {
            let mut rng = XorShiftRng::seed_from_u64(self.seed + key as u64);

            let mut emb = counts.get_or(|| {
                RefCell::new(vec![0i32; self.dims as usize])
            }).borrow_mut();

            emb.iter_mut().for_each(|ei| {
                *ei = 0;
            });

            // Compute random walk counts to generate embeddings
            let mut u = &key;
            for _ in 0..self.max_steps {
                
                // Check for a restart
                if rng.sample(uni_dist) < self.restarts {
                    u = &key;
                } else {
                    
                    // Update our step count and get our next proposed edge
                    let v = unsafe {
                        let out = &edges.get_unchecked(*u);
                        &out.get_unchecked(rng.sample(Uniform::new(0usize, out.len()))).0
                    };

                    if self.sampler == Sampler::RandomWalk {
                        // Always accept
                        u = v
                    } else {
                        // We scale the acceptance based on node degree
                        let (v_len, u_len) = unsafe {
                            (edges.get_unchecked(*v).len() as f32,  edges.get_unchecked(*u).len() as f32)
                        };
                        if v_len > u_len || (v_len / u_len) > rng.sample(Uniform::new(0f32, 1f32)) {
                            u = v;
                        }
                    }
                }

                // Hash
                let start = *u * self.hashes;
                let end = (*u + 1) * self.hashes;
                for (ref pos, ref idx) in &hash_table[start..end] {
                    emb[*idx as usize] += 2 * (*pos as i32) - 1;
                }
                
                //self.hash_node(&hashes, *u, &mut emb);
            }

            
            let total_nodes = edges.len() as f32;
            let emb = self.norm_embedding(total_nodes, &mut emb);

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

