extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate thread_local;
extern crate indicatif;

use std::hash::Hash;
use std::cmp::Ord;

use hashbrown::{HashMap,HashSet};
use rand::prelude::*;
use rayon::prelude::*;
use float_ord::FloatOrd;

use crate::pb::simple_pb;

pub enum Metric {
    Cosine,
    Jaccard,
    Overlap
}

pub struct SimStrategy {
    pub metric: Metric,
    pub best_only: bool,
    pub rem_weak_links: bool,
    pub min_cluster_size: usize,
    pub seed: u64
}

pub struct AttractorStrategy {
    pub num: usize,
    pub min_cluster_size: usize
}

pub enum ClusterStrategy {
    Similarity(SimStrategy),
    Attractors(AttractorStrategy)
}

impl ClusterStrategy {
    pub fn cluster<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        adj_graph: &HashMap<K, Vec<K>>,
        embeddings: HashMap<K, HashMap<K, f32>>
    ) -> Vec<Vec<K>> {
        use ClusterStrategy::*;
        match self {
            Similarity(ss) => ss.cluster(adj_graph, embeddings),
            Attractors(ats) => ats.cluster(adj_graph, embeddings),
        }
    }
}

impl SimStrategy {
    
    fn cluster<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        adj_graph: &HashMap<K, Vec<K>>,
        embeddings: HashMap<K, HashMap<K, f32>>
    ) -> Vec<Vec<K>> {
        let mut keys: Vec<_> = adj_graph.keys().collect();
        keys.sort();

        let new_edges = self.gen_edge_scores(&adj_graph, &embeddings, keys.as_slice());

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
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + 1000 as u64);
        let mut weak_links = 0;
        let mut pruned = HashSet::new();

        eprintln!("Pruning graph, tracing graph components...");
        let pb = simple_pb(keys.len() as u64);
        for key in keys.into_iter() {
            if !sparse_graph.contains_key(key) {
                continue
            }

            // Skip over nodes that have already been pruned
            let cluster = if self.rem_weak_links && !pruned.contains(key) {
                // First Pass: Gather edge weights, figure out weak links
                // and remove them per-graph.
                let (component, threshold) = thresh_dfs(key, &sparse_graph, Percentile::P01, &mut rng);

                // Remove weak edges
                let mut rem_link = false;
                for n in component.iter() {
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
                    Some(component)
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
                clusters.push(cur_set.into_iter().collect());
            }
        }

        eprintln!("Removed {} weak links", weak_links);
        pb.finish();
        clusters
    }

    // Computes similarity scores for each edge in the graph based on the learned embeddings
    fn gen_edge_scores<'a, K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        adj_graph: &'a HashMap<K, Vec<K>>,
        embeddings: &'a HashMap<K, HashMap<K, f32>>,
        keys: &[&'a K]
    ) -> Vec<(&'a K, Vec<(&'a K, f32)>)> {
        let pb = simple_pb(adj_graph.len() as u64);
        let new_edges: Vec<_> = keys.par_iter().enumerate().map(|(i, f_node)| {
            let node = f_node;
            let neighbors = &adj_graph[node];

            // If you only have one neighbor, choose it
            if neighbors.len() < 2 {
                pb.inc(1);
                return (*node, vec![(&neighbors[0], 1.)])
            }
 
            // Compute scores of the neighborhood
            let emb = &embeddings[&node];
            let scores = neighbors.par_iter().map(|n| {
                let n_emb = &embeddings[&n];
                match self.metric {
                    Metric::Cosine => cosine(emb, n_emb),
                    Metric::Jaccard => jaccard(emb, n_emb),
                    Metric::Overlap => overlap(emb, n_emb),
                }
            }).collect::<Vec<_>>();
           
            let threshold = if self.best_only {
                
                // Choose the best edge only
                *scores.iter()
                    .max_by_key(|s| FloatOrd(**s))
                    .unwrap()

            } else if scores.len() > 30 {
                
                // If you have more than 30, pretend it's normal and do outlier analysis
                let (mu, sigma) = sample_stats(&scores);
                
                // Use outlier numbers
                let high_score = *scores.iter()
                    .max_by_key(|s| FloatOrd(**s))
                    .unwrap();

                Percentile::P995.score(mu, sigma).min(high_score)

            } else {

                let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + i as u64);
                bootstrap_ci(&scores, 500, 0.99, &mut rng)
            };

            // Add the edges with the highest density scores
            let es = scores.iter().zip(neighbors.iter())
                .filter(|(s, _)| **s >= threshold)
                .map(|(s, n)| (n, *s))
                .collect::<Vec<_>>();

            pb.inc(1);
            (*node, es)
        }).collect();
        pb.finish();
        new_edges
    }

}

// Finds connected components and computes their high quality
// threshold scores for edge weights.
fn thresh_dfs<K: Hash + Eq + Clone, R: Rng>(
    key: &K, 
    sparse_graph: &HashMap<&K, Vec<(&K, f32)>>,
    ptile: Percentile,
    rng: R
) -> (HashSet<K>, f32) {
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
    let threshold = if scores.len() < 30 {
        bootstrap_ci(scores.as_slice(), 500, 0.01, rng)
    } else {
        let (mu, sigma) = sample_stats(scores.as_slice());
        ptile.score(mu, sigma)
    };

    (cur_set, threshold)
}

// Bootstrap estimate the kth percentile!
fn bootstrap_ci<R: Rng>(values: &[f32], runs: usize, percentile: f32, mut rng: R) -> f32 {
    let n = values.len();
    let mut mus: Vec<_> = (0..runs).map(|_| { 
        (0..n)
            .map(|_| values.choose(&mut rng).unwrap())
            .sum::<f32>() / n as f32
    }).collect();
    mus.sort_by_key(|x| FloatOrd(*x));
    let idx = (runs as f32 * percentile) as usize;
    mus[idx.min(runs - 1).max(0)]
}

fn sample_stats(values: &[f32]) -> (f32, f32) {
    let n     = values.len() as f32;
    let mu    = values.iter().sum::<f32>() / n;
    let var   = values.iter()
        .map(|s| (s - mu).powi(2))
        .sum::<f32>() / n;

    let sigma = var.powf(0.5);
    (mu, sigma)
}

impl AttractorStrategy {

   fn cluster<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self,
        adj_graph: &HashMap<K, Vec<K>>,
        embeddings: HashMap<K, HashMap<K, f32>>
    ) -> Vec<Vec<K>> {

       // Take the top K most influential nodes from each node.
       let it: Vec<_> = adj_graph.par_iter().map(|(k, _vs)| {
            let mut vs: Vec<_> = embeddings[k].iter().collect();
            
            // Take the top K most influential nodes
            vs.sort_by_key(|(_, v)| FloatOrd(-**v));
            let out = vs.into_iter()
                .take(self.num)
                .map(|(ki, _vi)| ki.clone())
                .collect::<Vec<_>>();
            (k.clone(), out)
       }).collect();

       // For each attractor, add the node to the list of items.
       let mut clusters = HashMap::new();
       for (node, cidxs) in it.into_iter() {
           for cidx in cidxs.into_iter() {
               let e = clusters.entry(cidx).or_insert_with(|| Vec::new());
               e.push(node.clone());
           }
       }

       // Filter out clusters which are too small
       clusters.into_iter()
           .filter(|(_k, cluster)| cluster.len() >= self.min_cluster_size)
           .map(|(_, cluster)| cluster)
           .collect()
   }

}

enum Percentile {
    P995,
    //P99,
    //P95,
    //P90,
    //P10,
    //P05,
    P01,
    //P005
}

impl Percentile {
    fn score(&self, mu: f32, sigma: f32) -> f32 {
        use Percentile::*;
        let zvalue = match self {
            P995 => 2.807,
            //P99  => 2.576,
            //P95  => 1.960,
            //P90  => 1.645,
            //P10  => -1.645,
            //P05  => -1.960,
            P01  => -2.576,
            //P005 => -2.807
        };
        mu + zvalue * sigma
    }
}

// Computes the cosine between two embeddings.  Assumes the embeddings are already l2 normalized.
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
