extern crate hashbrown;
extern crate rand;
extern crate rayon;
extern crate indicatif;

use std::hash::Hash;
use std::fmt::Write;

use indicatif::{ProgressBar,ProgressStyle};
use hashbrown::HashMap;
use rand::prelude::*;
use rayon::prelude::*;

pub struct LPA {
    pub n_iters: usize,
    pub chunks: usize,
    pub seed: u64
}

impl LPA {
    pub fn prepare_data<K: Hash + Eq + Clone + Ord>(
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> HashMap<K, Vec<K>> {
        // Create graph
        let mut edges = HashMap::new();
        for (f_node, t_node, _weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push(t_node.clone());
            let e = edges.entry(t_node).or_insert_with(|| vec![]);
            e.push(f_node.clone());
        }
        edges

    }

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + Ord>(
        &self, 
        edges: &HashMap<K, Vec<K>>
    ) -> HashMap<K, usize> {

        // We randomly interate over keys
        let mut keys: Vec<_> = edges.keys().map(|k| k.clone()).collect();
        keys.sort();

        // Assign initial IDs to group
        let mut clusters: HashMap<_,_> = keys.iter().enumerate()
            .map(|(i, k)| (k.clone(), i))
            .collect();

        // We randomly sort our keys each pass in the same style as label embeddings.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut n_iter = 0;
        let mut rngs: Vec<_> = (0..self.chunks)
            .map(|i| rand::rngs::StdRng::seed_from_u64(self.seed + 1 + i as u64))
            .collect();

        let total_work = keys.len() * self.n_iters;
        let pb = ProgressBar::new(total_work as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} {eta_precise}"));
        pb.enable_steady_tick(200);
        pb.set_draw_delta(total_work as u64 / 1000);
        let mut msg: String = "Iter 0...".into();
        pb.set_message(&msg);

        loop {
            let mut updated = 0;
            keys.shuffle(&mut rng);
            // Loop over items in parallel.  We rely on randomization to (mostly) ensure
            // that we're not updating adjacent nodes at the same time.
            for key_subset in keys.as_slice().chunks(self.chunks) {
                let it = key_subset.par_iter().zip(rngs.par_iter_mut());
                let new_clusters: Vec<_> = it.map(|(key, mut rng)| {
                    let node_edges = &edges[key];
                    // Count the labels of neighbor nodes
                    let mut counts = HashMap::with_capacity(node_edges.len());
                    for t_node in node_edges.iter() {
                        let e = counts.entry(clusters[&t_node]).or_insert(0);
                        *e += 1;
                    }

                    // Get max label count as new label.  If we have ties, we'll need to tiebreak
                    let mut best_cluster = 0;
                    let mut best_count = 0;
                    let mut ties = false;
                    let mut ties = Vec::new();
                    for (cluster, count) in counts.iter() {
                        if *count > best_count {
                            best_cluster = *cluster;
                            best_count = *count;
                            ties.clear();
                            ties.push(*cluster)
                        } else if *count == best_count {
                            ties.push(*cluster)
                        } 
                    }

                    pb.inc(1);
                    // We tie break by randomly choosing and item
                    if ties.len() > 1 {
                        // Makes LPA deterministic
                        ties.sort();
                        *ties.as_slice()
                            .choose(&mut rng)
                            .expect("If a node has no edges, code bug")
                    } else {
                        ties[0]
                    }
                }).collect();

                for (key, new_cluster) in key_subset.iter().zip(new_clusters.into_iter()) {
                    let e = clusters.get_mut(key)
                        .expect("Code bug!  All keys should exist in cluster map");
                    if new_cluster != *e {
                        updated += 1;
                        *e = new_cluster;
                    }
                }
            }

            n_iter += 1;

            let ratio = 100. * updated as f64 / keys.len() as f64;
            msg.clear();
            write!(msg, "Iter {}/{}, Updated Nodes {}/{} ({:.2}%)", n_iter, self.n_iters, updated, keys.len(), ratio)
                .expect("Failed to update message!");
            pb.set_message(&msg);
            if updated == 0 || self.n_iters <= n_iter {
                break
            }
        }
        pb.finish();

        clusters
    }
}

