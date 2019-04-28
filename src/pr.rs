extern crate hashbrown;

use hashbrown::{HashMap,HashSet};

use super::Games;

pub enum Sink {
    Reverse,
    All,
    None
}

pub struct PageRank {
    damping: f32,
    iterations: usize,
    sink: Sink
}

impl PageRank {

    pub fn new(damping: f32, iterations: usize, sink: Sink) -> Self {
        PageRank { 
            damping: damping,
            iterations: iterations,
            sink: sink
        }
    }

    pub fn compute(&self, games: Games) -> Vec<(u32, f32)> {
        // Build dag
        let mut dag = HashMap::new();
        for (winner, loser, _) in games.into_iter() {
            // Add edge to dag
            let e = dag.entry(loser).or_insert_with(|| HashSet::with_capacity(1));
            e.insert(winner);
        }


        let all_nodes = match self.sink {
            Sink::None => {
                // pass
                None
            },
            // Reverse all sinks which don't have outputs
            Sink::Reverse => {
                let mut reverse = HashMap::new();
                for (root, vs) in dag.iter() {
                    for inbound in vs {
                        if !dag.contains_key(inbound) {
                            let e = reverse.entry(*inbound)
                                .or_insert_with(|| HashSet::with_capacity(1));
                            e.insert(*root);
                        }
                    }
                }

                for (root, outbound) in reverse.into_iter() {
                    dag.insert(root, outbound);
                }

                None
            },
            // All Sinks map back to every node
            Sink::All => {
                let all_nodes = dag.values()
                    .flat_map(|vs| vs.iter()
                        .filter(|v| !dag.contains_key(v))
                        .map(|v| *v)
                    ).collect::<HashSet<u32>>();
                Some(all_nodes)
            }
        };

        eprintln!("Finished building DAG");
        dag.shrink_to_fit();
        let mut policy = HashMap::new();
        for (from_node, to_nodes) in dag.iter_mut() {
            to_nodes.shrink_to_fit();
            // Add initial items to policy
            policy.insert(*from_node, 0f32);
            for node in to_nodes.iter() {
                policy.insert(*node, 0f32);
            }
        }

        // set initial policy parameters
        let n_docs = policy.len();
        for v in policy.values_mut() {
            *v = 1. / n_docs as f32;
        }

        let mut new_policy = HashMap::with_capacity(policy.capacity());
        for iter in 0..self.iterations {
            eprintln!("Iteration: {}", iter);

            // go over DAG and add weight to new policy
            for (key, ob) in dag.iter() {
                let pr_k = policy[key] / ob.len() as f32;
                for inbound in ob {
                    let e = new_policy.entry(*inbound).or_insert(0f32);
                    *e += pr_k;
                }
            }

            // Go over all_nodes and add their probs to each
            if let Some(nodes) = &all_nodes {
                // Get Sum
                let all_sinks_sums = nodes.iter().map(|v| policy[v])
                    .sum::<f32>() / (policy.len() - 1) as f32;

                for to_node in policy.keys() {
                    let e = new_policy.entry(*to_node).or_insert(0f32);
                    *e += all_sinks_sums;
                    // Subtract self since no self references allowed
                    if nodes.contains(to_node) {
                        *e -= policy[to_node] / (policy.len() - 1) as f32;
                    }
                }
            }

            // Go over old policy to ensure we don't miss nodes without weight and
            // apply damping
            for key in policy.keys() {
                let pr_key = new_policy.get(key).unwrap_or(&0f32);
                let damped_value = pr_key * self.damping + (1. - self.damping) / n_docs as f32;
                new_policy.insert(*key, damped_value);
            }

            // swap new policy with old one
            std::mem::swap(&mut policy, &mut new_policy);
            new_policy.clear();
        }
        std::mem::drop(new_policy);

        let mut np: Vec<_> = policy.into_iter().collect();
        np.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());
        np

    }
}

#[cfg(test)]
mod test_page_rank {
    use super::*;

    #[test]
    fn test_example() {
        let matches = vec![
            (1, 2, 1.),
            (3, 2, 1.),
            (1, 3, 1.),
            (1, 4, 1.),
            (2, 4, 1.),
            (3, 4, 1.),
        ];

        let pr = PageRank::new(0.85, 1, Sink::None);
        let results = pr.compute(matches);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 - 0.427083 < 1e-4);

        assert_eq!(results[1].0, 3);
        assert!(results[1].1 - 0.214583 < 1e-4);

        assert_eq!(results[2].0, 2);
        assert!(results[2].1 - 0.108333 < 1e-4);

        assert_eq!(results[3].0, 4);
        assert!(results[3].1 - 0.0375 < 1e-4);
    }

    #[test]
    fn test_reverse() {
        let matches = vec![
            (1, 2, 1.),
            (3, 2, 1.),
            (1, 3, 1.),
            (1, 4, 1.),
            (2, 4, 1.),
            (3, 4, 1.),
        ];

        let pr = PageRank::new(0.85, 10, Sink::Reverse);
        let results = pr.compute(matches);

        assert_eq!(results[0].0, 1);
        assert!(results[0].1 - 0.39064 < 1e-4);

        assert_eq!(results[1].0, 3);
        assert!(results[1].1 - 0.27099 < 1e-4);

        assert_eq!(results[2].0, 2);
        assert!(results[2].1 - 0.190172 < 1e-4);

        assert_eq!(results[3].0, 4);
        assert!(results[3].1 - 0.14818 < 1e-4);

        assert!((results.into_iter().map(|(_, v)| v).sum::<f32>() -  1f32).abs() < 1e-5);
    }

    #[test]
    fn test_all_links() {
        let matches = vec![
            (1, 2, 1.),
            (3, 2, 1.),
            (1, 3, 1.),
            (1, 4, 1.),
            (2, 4, 1.),
            (3, 4, 1.),
        ];

        let pr = PageRank::new(0.85, 10, Sink::All);
        let results = pr.compute(matches);

        assert_eq!(results[0].0, 1);
        assert!(results[0].1 - 0.39064 < 1e-4);

        assert_eq!(results[1].0, 3);
        assert!(results[1].1 - 0.27099 < 1e-4);

        assert_eq!(results[2].0, 2);
        assert!(results[2].1 - 0.190172 < 1e-4);

        assert_eq!(results[3].0, 4);
        assert!(results[3].1 - 0.14818 < 1e-4);

        assert!((results.into_iter().map(|(_, v)| v).sum::<f32>() - 1f32).abs() < 1e-5);
    }

}
