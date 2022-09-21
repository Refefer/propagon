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

use crate::cluster_strat::*;

pub enum Sampler {
    MetropolisHastings,
    RandomWalk
}

pub struct MCCluster {
    pub max_steps: usize,
    pub restarts: f32,
    pub ppr: bool,
    pub max_terms: usize,
    pub sampler: Sampler,
    pub clusterer: ClusterStrategy,
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
            let mut u = &key;
            let mut count = 0;
            for _ in 0..self.max_steps {
                if !self.ppr {
                    let e = counts.entry((*u).clone()).or_insert(0);
                    *e += 1;
                    count += 1;
                }

                // Check for a restart
                if rng.sample(Uniform::new(0f32, 1f32)) > self.restarts {
                    
                    // Update our step count and get our next proposed edge
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
                } else {
                    if self.ppr {
                        let e = counts.entry((*u).clone()).or_insert(0);
                        *e += 1;
                        count += 1;
                    }
                    u = &key;
                }
            }

            // Take the top K terms by frequency
            let mut items: Vec<_> = counts.into_iter().collect();
            items.sort_by_key(|(_, v)| *v);
            let mut emb: HashMap<_,f32> = items.into_iter()
                .rev()
                .take(self.max_terms)
                .map(|(k, c)| {
                    let score: f32 = c as f32 / count as f32;
                    (k, score)
                })
                .collect();

            // L2 Norm the results
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
        let adj_graph: HashMap<_, Vec<_>> = edges.into_iter()
            .map(|(k, ls)| (k, ls.into_iter().map(|(k, _)| k).collect()))
            .collect();

        let mut clusters = self.clusterer.cluster(&adj_graph, embeddings);
 
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

        clusters.sort_by_key(|s| s.len());
        clusters.reverse();
        clusters
    }

}

