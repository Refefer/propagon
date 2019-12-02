extern crate hashbrown;
extern crate rayon;

use std::hash::Hash;

use hashbrown::{HashMap,HashSet};
use rayon::prelude::*;

#[derive(Debug)]
pub struct LabelRankPlus {
    pub n_iters: usize,
    pub inflation: f32,
    pub max_terms: usize,
    pub q: f32,
    pub prior: f32
}

impl LabelRankPlus {

    pub fn fit<K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + Ord>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>
    ) -> HashMap<K, usize> {

        // Create graph
        let mut edges = HashMap::new();
        let mut weights = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));

            let e = edges.entry(t_node.clone()).or_insert_with(|| vec![]);
            e.push((f_node.clone(), weight));

            *weights.entry(f_node).or_insert(0f32) += weight;
            *weights.entry(t_node).or_insert(0f32) += weight;
        }

        // Setup initial embedding
        let mut p: HashMap<_,_> = {
            let mut keys: Vec<_> = edges.keys().collect();
            keys.sort();
            keys.iter().enumerate()
                .map(|(i, k)| {
                    let mut hm = HashMap::with_capacity(self.max_terms);
                    hm.insert(i, 1f32);
                    ((*k).clone(), hm)
                }).collect()
        };

        // We randomly sort our keys each pass in the same style as label embeddings
        for n_iter in 0..self.n_iters {

            // Create new p_prime
            let mut p_prime: HashMap<_,_> = if self.prior > 0f32 {
                // Create new prime.
                p.par_iter().map(|(v, c)| {
                    let mut c = c.clone();
                    let total = weights[v];
                    // Scale labels by the prior
                    for (_label, lw) in c.iter_mut() {
                        *lw = *lw * total * self.prior;
                    }
                    (v.clone(), c)
                }).collect()
            } else {
                p.keys()
                    .map(|k| (k.clone(), HashMap::new()))
                    .collect()
            };

            // Propagate values from neighbors.  Since edges are bi-directional,
            // we can cheaply propagate into the node in parallel
            p_prime.par_iter_mut().for_each(|(v, labels)| {
                for (f_node, edge_weight) in edges[v].iter() {
                    for (label, label_weight) in p[f_node].iter() {
                        let lw = label_weight * edge_weight;
                        if let Some(lv) = labels.get_mut(&label) {
                            *lv += lw;
                        } else {
                            labels.insert(label.clone(), lw);
                        }
                    }
                }
            });

            // Normalize, expand, and then contract.
            p_prime.par_iter_mut().for_each(|(vert, labels)| {
                let vert_weight = weights[vert] * (1. + self.prior);
                let mut denom = 0.;
                for (_label, l_p) in labels.iter_mut() {
                    let norm_l_p = (*l_p / vert_weight).powf(self.inflation);
                    denom += norm_l_p;
                    *l_p = norm_l_p;
                }

                // finish inflation and cutoff at max_terms
                for (_label, l_p) in labels.iter_mut() {
                    *l_p /= denom;
                }

                if self.max_terms < labels.len() {
                    // sad allocation noises
                    let mut label_set: Vec<_> = labels.into_iter().collect();

                    label_set.sort_by(|a, b| 
                           (b.1).partial_cmp(&a.1).expect("Should never blow up!"));
                    *labels = label_set.into_iter()
                        .take(self.max_terms)
                        .map(|(k, w)| (*k, *w)).collect()
                } 

            });

            // conditional termination based on dot product similarity
            let updated: HashSet<_> = p.par_iter().map(|(v, labels)| {
                let sim: f32 = edges[v].iter().map(|(t, w)| {
                    w * hm_dot(labels, &p[t])
                }).sum();

                if sim < weights[v] * self.q * (1. + self.prior) {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();

            eprintln!("{}: Changed: {:.1}%", n_iter, 
                      (100 * updated.len()) as f32/ p.len() as f32);

            for v in updated.into_iter() {
                let old_p = p.get_mut(&v).unwrap();
                let new_p = p_prime.get_mut(&v).unwrap();
                std::mem::swap(old_p, new_p);
            }
        }

        p.into_iter().map(|(v, labels)| {
            let (k, _) = labels.iter()
                .max_by(|&(_, v1), &(_, v2)| (*v1).partial_cmp(v2).unwrap())
                .expect("Should have a label at all times!");
            (v, *k)
        }).collect()
    }
}

fn hm_dot(d1: &HashMap<usize,f32>, d2: &HashMap<usize,f32>) -> f32 {
    d1.iter().map(|(k, v)| {
        *v * *d2.get(k).unwrap_or(&0.)
    }).sum()
}

#[cfg(test)]
mod test_lrank {
    use super::*;

    #[test]
    fn test_hm_dot() {
        let mut hm1 = HashMap::new();
        hm1.insert(0, 0.7);
        hm1.insert(1, 0.3);

        let mut hm2 = HashMap::new();
        hm2.insert(0, 0.3);
        hm2.insert(1, 0.7);

        assert_eq!(hm_dot(&hm1, &hm2), 0.7*0.3*2.);

    }
}
