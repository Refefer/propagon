extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::hash::Hash;
use std::cmp::Ord;
use std::fs::File;
use std::fmt::{Display,Write};

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rand::distributions::Uniform;
use rayon::prelude::*;

pub enum Sampler {
    MetropolisHastings,
    RandomWalk
}

pub enum Similarity {
    Cosine,
    Jaccard,
    Overlap
}

pub struct MCCluster {
    pub max_steps: usize,
    pub restarts: f32,
    pub max_terms: usize,
    pub sampler: Sampler,
    pub similarity: Similarity,
    pub threshold: f32,
    pub min_cluster_size: usize,
    pub emb_path: Option<String>,
    pub seed: u64
}

impl MCCluster {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord + Display + std::fmt::Debug>(
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

        if let Some(ref path) = self.emb_path {
            self.write_out_embeddings(path, &embeddings);
        }

        self.generate_clusters(edges, embeddings).into_iter()
            .enumerate()
            .flat_map(|(idx, cluster)| {
                cluster.into_iter().map(move |s| (s, idx))
            }).collect()
    }

    // Write out the embeddings to a separate file
    fn write_out_embeddings<K: Hash + Eq + Display>(&self, fname: &str, embeddings: &HashMap<K, HashMap<K, f32>>) {
        use std::io::{BufWriter,Write};
        
        let output = File::create(fname)
            .expect("Unable to open file for writing!");

        let mut out = BufWriter::new(output);
        let mut s = String::new();
        for (node, vec) in embeddings {
            s.clear();
            for (i,(k, v)) in vec.iter().enumerate() {
                if i > 0 { s.push_str(", "); }
                write!(&mut s, "{}:{}", k, v).expect("Error writing out!");
            }

            write!(out, "{}: {}\n", node, s).expect("Error writing out!");
        }
    }

    fn generate_markov_embeddings<K: Hash + Eq + Clone + Send + Sync + Ord>(
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
            let mut counts: HashMap<K, usize> = HashMap::new();
            
            // Compute random walk counts to generate embeddings
            let mut step = 0;
            while step < self.max_steps {
                let mut u = &key;

                // Check for a restart
                while rng.sample(Uniform::new(0f32, 1f32)) > self.restarts {
                    
                    // Update our step count and get our next proposed edge
                    step += 1;
                    let v = &edges[u]
                        .choose(&mut rng)
                        .expect("Should never be empty!").0;

                    match self.sampler {
                        Sampler::RandomWalk => u = v,
                        Sampler::MetropolisHastings => {
                            let acceptance = edges[v].len() as f32/ edges[u].len() as f32;
                            if acceptance > rng.sample(Uniform::new(0f32, 1f32)) {
                                u = v;
                            }
                        }
                    };
                    let e = counts.entry((*u).clone()).or_insert(0);
                    *e += 1;
                }
            }

            // Take the top K terms by frequency
            let mut items: Vec<_> = counts.into_iter().collect();
            items.sort_by_key(|(_, v)| *v);
            let mut emb: HashMap<_,f32> = items.into_iter()
                .rev()
                .take(self.max_terms)
                .map(|(k, count)| {
                    let score: f32 = count as f32 / self.max_steps as f32;
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

    fn generate_clusters<K: Hash + Eq + Clone + Send + Sync + Ord>(
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

                    let n_emb = &embeddings[&neighbor];
                    let score = match self.similarity {
                        Similarity::Cosine => cosine(emb, n_emb),
                        Similarity::Jaccard => jaccard(emb, n_emb),
                        Similarity::Overlap => overlap(emb, n_emb),
                    };

                    if score > self.threshold {
                        stack.push(&neighbor);
                        seen_keys.insert(&neighbor);
                        cur_set.push(neighbor.clone());
                    }
                }
            }

            pb.inc(cur_set.len() as u64);
            if cur_set.len() >= self.min_cluster_size {
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

fn jaccard<K: Hash + Eq + Clone + Ord>(e1: &HashMap<K, f32>, e2: &HashMap<K, f32>) -> f32 {
    let mut nom = 0;
    let mut denom = HashSet::new();
    for k in e1.keys() {
        if e2.contains_key(k) {
            nom += 1;
        }
        denom.insert(k);
    }
    denom.extend(e2.keys());

    nom as f32 / denom.len() as f32
}

fn overlap<K: Hash + Eq + Clone + Ord>(e1: &HashMap<K, f32>, e2: &HashMap<K, f32>) -> f32 {
    let mut nom = 0;
    for k in e1.keys() {
        if e2.contains_key(k) {
            nom += 1;
        }
    }
    nom as f32 / e1.len().max(e2.len()) as f32
}
