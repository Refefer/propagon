extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::hash::Hash;

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;

use crate::walker::RandomWalk;

pub struct MCCluster {
    pub num_walks: usize,
    pub walk_len: usize,
    pub max_terms: usize,
    pub biased_walk: bool,
    pub seed: u64
}

impl MCCluster {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> HashMap<K, Vec<(K, f32)>> {

        // Create graph
        let mut edges = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));
        }

        eprintln!("Number of Vertices: {}", edges.len());

        let embeddings = self.generate_markov_embeddings(&edges);
        embeddings
    }

    fn generate_markov_embeddings<K: Hash + Eq + Clone + Send + Sync>(
        &self,
        edges: &HashMap<K, Vec<(K, f32)>>
    ) -> HashMap<K, Vec<(K, f32)>> {
        // Setup initial embeddings
        let keys: Vec<_> = edges.keys().map(|k| k.clone()).collect();
        
        // Progress bar time
        let total_work = edges.len();
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
    
        let embeddings: HashMap<_,_> = keys.into_par_iter().enumerate().map(|(i, key)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + i as u64);
            let rw = RandomWalk::new(&edges);
            let mut counts: HashMap<K, usize> = HashMap::new();
            let mut buffer = Vec::with_capacity(self.walk_len + 1);
            // Compute random walk counts
            for _walk_num in 0..self.num_walks {
                if self.biased_walk {
                    rw.gen_biased_walk_with_buff(&key, &mut rng, self.walk_len, &mut buffer);
                } else {
                    rw.gen_uniform_walk_with_buff(&key, &mut rng, self.walk_len, &mut buffer);
                }
                buffer.iter().skip(1).for_each(|k| {
                    let e = counts.entry((*k).clone()).or_insert(0);
                    *e += 1;
                });
            }
            let mut items: Vec<_> = counts.into_iter().collect();
            items.sort_by_key(|(_, v)| *v);
            items.reverse();
            let emb: Vec<(K, f32)> = items.into_iter()
                .take(self.max_terms)
                .map(|(k, count)| {
                    let score: f32 = count as f32 / (self.walk_len * self.num_walks) as f32;
                    (k, score)
                })
                .collect();
            pb.inc(1);
            (key, emb)
        }).collect();
        pb.finish();

        embeddings
    }
}

