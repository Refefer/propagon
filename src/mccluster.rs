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
    pub rem_weak_links: bool,
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
        let pb = self.create_pb(edges.len() as u64);

        let embeddings: HashMap<_,_> = keys.into_par_iter().enumerate().map(|(i, key)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + i as u64);
            let mut counts: HashMap<K, usize> = HashMap::new();
            
            // Compute random walk counts to generate embeddings
            let mut step = 0;
            while step < self.max_steps {
                let mut u = &key;
                let mut i = 0;
                let e = counts.entry((*u).clone()).or_insert(0);
                *e += 1;
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

    fn create_pb(&self, total_work: u64) -> ProgressBar {
        let pb = ProgressBar::new(total_work);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
        pb
    }

    fn generate_clusters<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        edges: HashMap<K, Vec<(K, f32)>>,
        embeddings: HashMap<K, HashMap<K, f32>>
    ) -> Vec<Vec<K>> {
        eprintln!("Creating adjacency graph");
        let adj_graph: HashMap<_, Vec<_>> = edges.into_iter()
            .map(|(k, ls)| (k, ls.into_iter().map(|(k, _)| k).collect()))
            .collect();

        eprintln!("Constructing sparse cluster graph...");
        let pb = self.create_pb(adj_graph.len() as u64);
        
        let mut keys: Vec<_> = adj_graph.keys().collect();
        keys.sort();
        let new_edges: Vec<_> = keys.par_iter().enumerate().map(|(i, f_node)| {
            let neighbors = &adj_graph[f_node];

            // If you only have one neighbor, choose it
            if neighbors.len() < 2 {
                pb.inc(1);
                return (*f_node, vec![(&neighbors[0], 1.)])
            }
 
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
           
            let threshold = if self.best_only {
                
                // Choose the best edge only
                *scores.iter()
                    .max_by_key(|s| float_ord::FloatOrd(**s))
                    .unwrap()

            } else if scores.len() > 30 {
                
                // If you have more than 30, pretend it's normal and do outlier analysis
                let (mu, sigma) = self.sample_stats(&scores);
                
                // Use outlier numbers
                let high_score = *scores.iter()
                    .max_by_key(|s| float_ord::FloatOrd(**s))
                    .unwrap();
                Percentile::P995.score(mu, sigma).min(high_score)

            } else {

                let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + i as u64);
                self.bootstrap_ci(&scores, 500, 0.99, &mut rng)
            };

            // Add the edges with the highest density scores
            let es = scores.iter().zip(neighbors.iter())
                .filter(|(s, _)| **s >= threshold)
                .map(|(s, n)| (n, *s))
                .collect::<Vec<_>>();

            pb.inc(1);
            (*f_node, es)
        }).collect();
        pb.finish();

        // Generate sparse adjacency graph
        let mut sparse_graph = HashMap::with_capacity(keys.len());
        for (f_n, t_ns) in new_edges {
            for (t_n, t_s) in t_ns {
                let e = sparse_graph.entry(f_n).or_insert_with(|| Vec::new());
                e.push((t_n, t_s));
                let e = sparse_graph.entry(t_n).or_insert_with(|| Vec::new());
                e.push((f_n, t_s));
            }
        }

        // DFS collect graphs from sparse graph list
        let mut clusters: Vec<Vec<_>> = Vec::new();
        let mut clustered_nodes = 0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + 1000 as u64);
        let mut weak_links = 0;
        let mut pruned = HashSet::new();
        eprintln!("Pruning graph, tracing graph components...");
        let pb = self.create_pb(keys.len() as u64);
        for key in keys.into_iter() {
            if !sparse_graph.contains_key(key) {
                continue
            }

            let cluster = if self.rem_weak_links && !pruned.contains(key) {
                // First Pass: Gather edge weights, figure out weak links
                // and remove them per-graph.
                let (nodes, threshold) = {
                    let mut stack = vec![key];
                    let mut cur_set = HashSet::new();
                    cur_set.insert(key.clone());
                    let mut scores = Vec::new();
                    while stack.len() > 0 {
                        let q = stack.pop().unwrap();
                        for (n, s) in sparse_graph[q].iter() {
                            if !cur_set.contains(n) {
                                cur_set.insert((*n).clone());
                                stack.push(n);
                            }
                            scores.push(*s);
                        }
                    }

                    // Compute score percentile
                    let t = if scores.len() < 30 {
                        self.bootstrap_ci(scores.as_slice(), 500, 0.01, &mut rng)
                    } else {
                        let (mu, sigma) = self.sample_stats(scores.as_slice());
                        Percentile::P01.score(mu, sigma)
                    };
                    (cur_set, t)
                };

                // Remove weak edges
                let mut rem_link = false;
                for n in nodes.iter() {
                    let v = sparse_graph.get_mut(n).unwrap();
                    for idx in (0..v.len()).rev() {
                        if v[idx].1 < threshold {
                            weak_links += 1;
                            rem_link = true;
                            v.swap_remove(idx);
                        }
                    }
                    pruned.insert(n.clone());
                }

                if rem_link {
                    None
                } else {
                    Some(nodes)
                }

            } else {
                None
            };

            let cur_set = if let Some(nodes) = cluster {
                nodes
            } else {
                // Compute twice: Once to gather weights, figure out weak links
                // and remove them per-graph.  Second time to 
                let mut stack = vec![key];
                let mut cur_set = HashSet::new();
                cur_set.insert(key.clone());
                while stack.len() > 0 {
                    let q = stack.pop().unwrap();
                    for (n, _s) in sparse_graph[q].iter() {
                        if !cur_set.contains(n) {
                            cur_set.insert((*n).clone());
                            stack.push(n);
                        }
                    }
                }
                cur_set
            };

            pb.inc(cur_set.len() as u64);

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
        eprintln!("Removed {} weak links", weak_links);
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

    fn bootstrap_ci<R: Rng>(&self, values: &[f32], runs: usize, percentile: f32, mut rng: R) -> f32 {
        // Bootstrap estimate the kth percentile!
        let n = values.len();
        let mut mus: Vec<_> = (0..runs).map(|_| { 
            (0..n)
                .map(|_| values.choose(&mut rng).unwrap())
                .sum::<f32>() / n as f32
        }).collect();
        mus.sort_by_key(|x| float_ord::FloatOrd(*x));
        let idx = (runs as f32 * percentile) as usize;
        mus[idx.min(runs - 1).max(0)]
    }

    fn sample_stats(&self, values: &[f32]) -> (f32, f32) {
        let n     = values.len() as f32;
        let mu    = values.iter().sum::<f32>() / n;
        let var   = values.iter().map(|s| (s - mu).powi(2)).sum::<f32>() / n;
        let sigma = var.powf(0.5);
        (mu, sigma)
    }

}

enum Percentile {
    P995,
    P99,
    P95,
    P90,
    P10,
    P05,
    P01,
    P005
}

impl Percentile {
    fn score(&self, mu: f32, sigma: f32) -> f32 {
        use Percentile::*;
        let zvalue = match self {
            P995 => 2.807,
            P99  => 2.576,
            P95  => 1.960,
            P90  => 1.645,
            P10  => -1.645,
            P05  => -1.960,
            P01  => -2.576,
            P005 => -2.807
        };
        mu + zvalue * sigma
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
