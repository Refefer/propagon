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
    pub best_only: bool,
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
                let mut i = 0;
                //let e = counts.entry((*u).clone()).or_insert(0);
                //*e += 1;
                //step += 1;

                // Check for a restart
                while i == 0 || rng.sample(Uniform::new(0f32, 1f32)) > self.restarts {
                    
                    // Update our step count and get our next proposed edge
                    i += 1;
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

        eprintln!("Constructing sparse graph...");
        let total_work = adj_graph.len();
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
        
        let keys: Vec<_> = adj_graph.keys().collect();
        let new_edges: Vec<_> = keys.par_iter().enumerate().map(|(i, f_node)| {
            let neighbors = &adj_graph[f_node];
            // Compute scores of the neighborhood
            let emb = &embeddings[&f_node];
            let scores = neighbors.par_iter().map(|n| {
                let n_emb = &embeddings[&n];
                match self.similarity {
                    Similarity::Cosine => cosine(emb, n_emb),
                    Similarity::Jaccard => jaccard(emb, n_emb),
                    Similarity::Overlap => overlap(emb, n_emb),
                }
            }).collect::<Vec<_>>();

            // If you only have one neighbor, choose it
            if scores.len() < 2 {
                pb.inc(1);
                return (*f_node, vec![&neighbors[0]])
            }

            let threshold = if self.best_only {
                
                // Choose the best edge only
                *scores.iter()
                    .max_by_key(|s| float_ord::FloatOrd(**s))
                    .unwrap()

            } else if scores.len() > 30 {
                
                // If you have more than 30, pretend it's normal and do outlier analysis
                let mu = scores.iter().sum::<f32>() / scores.len() as f32;
                let var = scores.iter().map(|s| (s - mu).powi(2)).sum::<f32>() / scores.len() as f32;
                let sigma = var.powf(0.5);
                
                // Use outlier numbers
                let high_score = *scores.iter()
                    .max_by_key(|s| float_ord::FloatOrd(**s))
                    .unwrap();
                (mu + 2.68 * sigma).min(high_score)

            } else {

                // Bootstrap estimate the 99th percentile!
                let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + i as u64);
                let mut mus: Vec<_> = (0..500).map(|_| { 
                    (0..scores.len())
                        .map(|_| scores.choose(&mut rng).unwrap())
                        .sum::<f32>() / scores.len() as f32
                }).collect();
                mus.sort_by_key(|x| float_ord::FloatOrd(*x));
                mus[495]
            };

            // Add the edges with the highest density scores
            let es = scores.iter().zip(neighbors.iter())
                .filter(|(s, _)| **s >= threshold)
                .map(|(_s, n)| n)
                .collect::<Vec<_>>();

            pb.inc(1);
            (*f_node, es)
        }).collect();

        // Generate sparse graph
        let mut sparse_graph = HashMap::with_capacity(keys.len());
        for (f_n, t_ns) in new_edges {
            for t_n in t_ns {
                let e = sparse_graph.entry(f_n).or_insert_with(|| Vec::new());
                e.push(t_n);
                let e = sparse_graph.entry(t_n).or_insert_with(|| Vec::new());
                e.push(f_n);
            }
        }

        // BFS collect graphs
        let mut clusters: Vec<Vec<_>> = Vec::new();
        let mut clustered_nodes = 0;
        for key in keys.into_iter() {
            if !sparse_graph.contains_key(key) {
                continue
            }
            let mut stack = vec![key];
            let mut cur_set = HashSet::new();
            cur_set.insert(key.clone());
            while stack.len() > 0 {
                let q = stack.pop().unwrap();
                for n in sparse_graph[q].iter() {
                    if !cur_set.contains(n) {
                        cur_set.insert((*n).clone());
                        stack.push(n);
                    }
                }
            }
            for n in cur_set.iter() {
                sparse_graph.remove(n);
            }
            if cur_set.len() >= self.min_cluster_size {
                clustered_nodes += cur_set.len();
                clusters.push(cur_set.into_iter().collect());
            }
        }

        pb.finish();

        // If we're testing, compute average inbound/outbound ratios
        eprintln!("Found {} clusters", clusters.len());
        eprintln!("Computing modularity ratios...");
        let total_edges = adj_graph.par_values()
            .map(|v| v.len())
            .sum::<usize>() as f64;

        let modularity = clusters.par_iter().map(|cluster| {
            let in_cluster: HashSet<_> = cluster.iter().collect();
            let mut eii = 0;
            let mut ai = 0;
            for c in cluster.iter() {
                for edge in adj_graph[&c].iter() {
                    if in_cluster.contains(&edge) {
                        eii += 1;
                    } 
                    ai += 1;
                }
            }
            let eii = eii as f64 / total_edges;
            let ai = ai as f64 / total_edges;
            let diff = eii as f64 - ai.powi(2);
            diff
        }).sum::<f64>();

        eprintln!("Modularity: {:.3}", modularity);
        let coverage = clustered_nodes as f64 / adj_graph.len() as f64;
        eprintln!("Coverage: {}/{}({:.3})", clustered_nodes, adj_graph.len(), coverage * 100.);

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
