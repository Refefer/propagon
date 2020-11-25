extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;
extern crate rand_xorshift;

use std::cmp::Ord;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

pub struct HashEmbeddings {
    pub dims: usize,
    pub max_steps: usize,
    pub restarts: f32,
    pub hashes: usize,
    pub sampler: Sampler,
    pub b: f32,
    pub seed: u64
}

impl HashEmbeddings {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> Vec<(K, Vec<f32>)> {

        // Create graph
        let mut edges = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));
        }

        eprintln!("Number of Vertices: {}", edges.len());

        self.generate_embeddings(&edges)
    }

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    fn generate_embeddings<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        edges: &HashMap<K, Vec<(K, f32)>>
    ) -> Vec<(K, Vec<f32>)> {

        let total_edges = edges.values().map(|v| v.len() as f32).sum::<f32>();
        
        // Sort keys for consistency
        let mut keys: Vec<_> = edges.keys().map(|k| k.clone()).collect();
        keys.sort();
        
        // Progress bar time
        eprintln!("Generating random walks...");
        let pb = self.create_pb(edges.len() as u64);

        let embeddings: Vec<_> = keys.into_par_iter().enumerate().map(|(i, key)| {
            let mut rng = XorShiftRng::seed_from_u64(self.seed + i as u64);
            let mut emb = vec![0f32; self.dims];
            
            // Compute random walk counts to generate embeddings
            let mut step = 0;
            while step < self.max_steps {
                let mut u = &key;
                let mut i = 0;
                step += 1;

                // Check for a restart
                while rng.sample(Uniform::new(0f32, 1f32)) > self.restarts {
                    
                    // Update our step count and get our next proposed edge
                    i += 1;
                    step += 1;
                    let v = &edges[u]
                        .choose(&mut rng)
                        .expect("Should never be empty!").0;

                    match self.sampler {
                        Sampler::RandomWalk => u = v,
                        Sampler::MetropolisHastings => {
                            let acceptance = edges[v].len() as f32 / edges[u].len() as f32;
                            if acceptance > rng.sample(Uniform::new(0f32, 1f32)) {
                                u = v;
                            }
                        }
                    };
                    
                    // Hash items
                    let mut weight = (edges[u].len() as f32 / total_edges).powf(self.b);
                    for i in 0..self.hashes {
                        let hash = HashEmbeddings::calculate_hash(&(i, u)) as usize;
                        let sign = (hash & 1);
                        let idx = (hash >> 1) % self.dims;
                        emb[idx] += if sign == 1 { weight } else { -weight };
                    }
                }
            }

            let norm = emb.iter().map(|v| (*v).abs()).sum::<f32>();
            emb.iter_mut().for_each(|e| { *e /= norm });
            pb.inc(1);
            (key, emb)
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

