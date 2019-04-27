use std::collections::{HashMap,HashSet};

use super::Games;

pub struct PageRank {
    damping: f32,
    iterations: usize
}

impl PageRank {

    pub fn new(damping: f32, iterations: usize) -> Self {
        PageRank { 
            damping: damping,
            iterations: iterations
        }
    }

    pub fn compute(&self, games: Games) -> Vec<(u32, f32)> {
        // Build dag, set initial policy
        let mut dag = HashMap::new();
        let mut policy = HashMap::new();
        for (winner, loser, _) in games {
            // Add edge to dag
            let e = dag.entry(loser).or_insert_with(|| HashSet::new());
            e.insert(winner);

            // Add initial items to policy
            policy.insert(winner, 0f32);
            policy.insert(loser, 0f32);
        }

        // set initial policy parameters
        let n_docs = policy.len();
        for v in policy.values_mut() {
            *v = 1. / n_docs as f32;
        }

        let mut new_policy = HashMap::new();
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

        let pr = PageRank::new(0.85, 1);
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

}
