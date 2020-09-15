extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::hash::Hash;
use std::cmp::Ord;

use indicatif::{ProgressBar,ProgressStyle};
use thread_local::ThreadLocal;
use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rayon::prelude::*;

use crate::walker::RandomWalk;

pub struct MCCluster {
    pub num_walks: usize,
    pub walk_len: usize,
    pub max_terms: usize,
    pub biased_walk: bool,
    pub threshold: f32,
    pub min_cluster_size: usize,
    pub seed: u64
}

impl MCCluster {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord + std::fmt::Debug>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> Vec<(K, usize)> {

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
        self.generate_clusters(edges, embeddings).into_iter()
            .enumerate()
            .flat_map(|(idx, cluster)| {
                cluster.into_iter().map(move |s| (s, idx))
            }).collect()
    }

    fn generate_markov_embeddings<K: Hash + Eq + Clone + Send + Sync + Ord + std::fmt::Debug>(
        &self,
        edges: &HashMap<K, Vec<(K, f32)>>
    ) -> HashMap<K, HashMap<K, f32>> {
        // Setup initial embeddings
        let mut keys: Vec<_> = edges.keys().map(|k| k.clone()).collect();
        keys.sort();
        
        // Progress bar time
        eprintln!("Generating random walks...");
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
            
            // Compute random walk counts to generate embeddings
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

            // Take the top K terms by frequency
            let mut items: Vec<_> = counts.into_iter().collect();
            items.sort_by_key(|(_, v)| *v);
            let mut emb: HashMap<_,f32> = items.into_iter()
                .rev()
                .take(self.max_terms)
                .map(|(k, count)| {
                    let score: f32 = count as f32 / (self.walk_len * self.num_walks) as f32;
                    (k, score)
                })
                .collect();

            let norm = emb.values().map(|v| v.powi(2)).sum::<f32>().sqrt();
            emb.values_mut().for_each(|v| {
                *v /= norm;
            });

            pb.inc(1);
            (key, emb)
        }).collect();

        pb.finish();
        embeddings

    }

    fn generate_clusters<K: Hash + Eq + Clone + Send + Sync + Ord + std::fmt::Debug>(
        &self,
        edges: HashMap<K, Vec<(K, f32)>>,
        embeddings: HashMap<K, HashMap<K, f32>>
    ) -> Vec<Vec<K>> {
        let adj_graph: HashMap<_, Vec<_>> = edges.into_iter()
            .map(|(k, ls)| (k, ls.into_iter().map(|(k, _)| k).collect()))
            .collect();

        let mut seen_keys = HashSet::new();
        let mut clusters = Vec::new();

        eprintln!("Cosine clustering...");
        let total_work = adj_graph.len();
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);

        for key in adj_graph.keys() {
            if seen_keys.contains(key) { continue }

            seen_keys.insert(key);

            let mut cur_set = vec![key.clone()];
            let mut stack = vec![key];
            while let Some(node) = stack.pop() {
                let emb = &embeddings[node];
                // Look at immediate edges.  If the similarity between two edges is greater than
                // the threshold, add it to the current set and explore queue.
                for neighbor in adj_graph[node].iter() {
                    if seen_keys.contains(&neighbor) {
                        continue
                    }

                    if cosine(emb, &embeddings[&neighbor]) > self.threshold {
                        stack.push(&neighbor);
                        seen_keys.insert(&neighbor);
                        cur_set.push(neighbor.clone());
                    }
                }
            }

            pb.inc(cur_set.len() as u64);
            if cur_set.len() > self.min_cluster_size {
                cur_set.sort();
                clusters.push(cur_set);
            }
        }
        pb.finish();
        
        clusters.sort_by_key(|s| s.len());
        clusters.reverse();
        clusters
    }
}

fn cosine<K: Hash + Eq + Clone + Ord>(e1: &HashMap<K, f32>, e2: &HashMap<K, f32>) -> f32 {
    let mut sum = 0.;
    for (k, v) in e1 {
        if let Some(v2) = e2.get(k) {
            sum += v * v2;
        }
    }

    sum
}
