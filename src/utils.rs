extern crate heap;
extern crate hashbrown;

use std::ops::Deref;
use std::hash::Hash;
use std::collections::{VecDeque,BinaryHeap};
use std::cmp::Reverse;

use hashbrown::{HashMap,HashSet};

#[inline]
pub fn l2_norm_hm<F: Hash>(features: &mut HashMap<F, f32>) {
    let sum = features.values()
        .map(|v| (*v).powi(2)).sum::<f32>().powf(0.5);
    features.values_mut().for_each(|v| *v /= sum);
}

#[inline]
pub fn l2_normalize<A>(vec: &mut Vec<(A, f32)>) {
    let sum: f32 = vec.iter().map(|(_, v)| (*v).powi(2)).sum();
    let sqr = sum.powf(0.5);
    vec.iter_mut().for_each(|p| (*p).1 /= sqr);
}

#[inline]
pub fn l2_norm(vec: &mut Vec<f32>) {
    let sum: f32 = vec.iter().map(|v| (*v).powi(2)).sum();
    let sqr = sum.powf(0.5);
    vec.iter_mut().for_each(|p| (*p) /= sqr);
}

pub fn clean_map<K: Hash>(
    features: &mut HashMap<K,f32>, 
    out: &mut Vec<(K,f32)>,
    error: f32,
    max_terms: usize
) {
    for (k, f) in features.drain() {
        // Ignore features smaller than error rate
        if f.abs() > error {
            // Add items to the heap until it's full
            if out.len() < max_terms {
                out.push((k, f));
                if out.len() == max_terms {
                    heap::build(out.len(), 
                        |a,b| a.1 < b.1, 
                        out.as_mut_slice());
                }
            } else if out[0].1 < f {
                // Found a bigger item, replace the smallest item
                // with the big one
                heap::replace_root(
                    out.len(), 
                    |a,b| a.1 < b.1, 
                    out.as_mut_slice(),
                    (k, f));
            }
        }
    }
}

pub fn interpolate_vecs<'a, F: 'a>(
    mut features: &mut HashMap<F, f32>, 
    orig: impl Iterator<Item=&'a (F, f32)>, 
    alpha: f32, 
    norm: bool
) 
where F: Hash + Eq + Clone {

    // Scale the data by alpha
    features.values_mut().for_each(|v| {
        *v *= alpha;
    });

    // add the prior
    for (k, v) in orig {
        let nv = (1. - alpha) * (*v);
        if features.contains_key(k) {
            if let Some(v) = features.get_mut(k) {
                *v += nv;
            }
        } else {
            features.insert(k.clone(), nv);
        }
    }
    if norm {
        l2_norm_hm(&mut features);
    }


}

pub fn update_prior<K, F, E>(
    features: &mut HashMap<F, f32>, 
    ctx: &K, 
    prior: &HashMap<K, E>, 
    alpha: f32, 
    norm: bool
) 
where K: Hash + Eq,
      F: Hash + Eq + Clone,
      E: Deref<Target=Vec<(F, f32)>> {
    if let Some(p) = prior.get(ctx) {
        interpolate_vecs(features, p.iter(), alpha, norm);
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::vp::Embedding;

    #[test]
    fn test_update_prior() {
        let matches: HashMap<usize, f32> = vec![
            (1, 1.0),
            (2, 0.8),
            (3, 0.3),
        ].into_iter().collect();

        let ctx = 0;

        let prior: HashMap<usize, _> = vec![
            (0, Embedding(vec![(3, 1.), (2, 0.8), (1, 0.3)]))
        ].into_iter().collect();

        let mut t1 = matches.clone();
        update_prior(&mut t1, &ctx, &prior, 0.5, false);

        let expected = vec![(1, 0.65), (2, 0.8), (3, 0.65)].into_iter().collect();

        assert_eq!(t1, expected);

        let mut t2 = matches.clone();
        update_prior(&mut t2, &ctx, &prior, 0.9, false);

        let expected = vec![(1, 1. * 0.9 + 0.3 * (1. - 0.9)), 
                            (2, 0.8), 
                            (3, 0.3*0.9 + 1. * (1. - 0.9))].into_iter().collect();

        assert_eq!(t2, expected);

    }

    #[test]
    fn test_l2_norm_hm() {

        let mut matches: HashMap<usize, f32> = vec![
            (1, 1.0),
            (2, 0.8),
            (3, 0.3),
        ].into_iter().collect();

        let denom = (1f32.powi(2) + (0.8f32).powi(2) + (0.3f32).powi(2)).powf(0.5);
        let expected: HashMap<_,_> = vec![(1, 1.  / denom), 
                            (2, 0.8 / denom), 
                            (3, 0.3 / denom)].into_iter().collect();

        l2_norm_hm(&mut matches);

        for (k, v) in matches {
            let v2 = expected[&k];
            println!("{},{}", v, v2);
            assert!((v - v2).abs() < 1e-5);
        }
    }

}

pub fn unweighted_walk_distance<'a, K: Hash + Eq>(
    edges: &'a HashMap<K, Vec<(K,f32)>>,
    start_node: &'a K
) -> HashMap<&'a K, f32> {
    let mut distance = HashMap::new();
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();

    seen.insert(start_node);
    queue.push_back((start_node, 0.));

    while let Some((vert, cur_dist)) = queue.pop_front() {
        distance.insert(vert, cur_dist);

        for (out_edge, _) in edges[vert].iter() {
            if !seen.contains(&out_edge) {
                seen.insert(&out_edge);
                queue.push_back((&out_edge, cur_dist + 1.));
            }
        }
    }

    distance
}

pub fn weighted_walk_distance<'a, K: Hash + Eq>(
    edges: &'a HashMap<K, Vec<(K,f32)>>,
    start_node: &'a K,
    degree_weighted: bool
) -> HashMap<&'a K, f32> {
    let mut distance = HashMap::new();
    distance.insert(start_node, 0f32);
    let mut queue = VecDeque::new();

    queue.push_back(start_node);

    while let Some(vert) = queue.pop_front() {
        let cur_dist = distance[vert];

        let degrees = if degree_weighted {
            (1. + edges[vert].len() as f32).ln()
        } else {
            1.
        };
        for (out_edge, wi) in edges[vert].iter() {
            let new_dist = if degree_weighted {
                let out_degrees = (1. + edges[out_edge].len() as f32).ln();
                cur_dist + degrees.max(out_degrees) / (1. + wi).ln()
            } else {
                cur_dist + degrees / (1. + wi).ln()
            };

            let out_dist = *distance.get(&out_edge).unwrap_or(&std::f32::INFINITY);
            if new_dist < out_dist {
                distance.insert(&out_edge, new_dist);
                queue.push_back(&out_edge);
            }
        }
    }

    distance
}

#[derive(Ord,Eq,PartialEq,PartialOrd)]
struct BestDegree<K: Ord + Eq + PartialEq + PartialOrd>(usize, K);

pub fn top_k_nodes<'a, 'b, K: Hash + Eq + Ord>(
    keys: &'b Vec<&'a K>,
    edges: &HashMap<K, Vec<(K, f32)>>,
    dims: usize
) -> Vec<&'b &'a K> {
    let mut bh = BinaryHeap::with_capacity(dims + 1);
    for k in keys.iter() {
        let degrees = edges[k].len();
        bh.push(Reverse(BestDegree(degrees, k)));
        if bh.len() > dims {
            bh.pop();
        }
    }
    bh.into_iter().map(|Reverse(BestDegree(_, k))| k).collect()
}

#[cfg(test)]
mod test_utils {
    use super::*;

    #[test]
    fn test_top_nodes() {
        let edges: HashMap<_,_> = vec![
            (0usize, vec![(1usize, 1.), (2usize, 1.)]),
            (1usize, vec![(0usize, 1.), (4usize, 1.)]),
            (2usize, vec![(0usize, 1.), (3usize, 1.), (5usize, 1.)]),
            (3usize, vec![(2usize, 1.), (5usize, 1.), (4usize, 1.)]),
            (4usize, vec![(1usize, 1.), (3usize, 2.)]),
            (5usize, vec![(3usize, 1.), (2usize, 1.)])
        ].into_iter().collect();

        let keys = edges.keys().collect();
        let mut v = top_k_nodes(&keys, &edges, 2);
        v.sort();
        assert_eq!(v[0], &&2);
        assert_eq!(v[1], &&3);

    }

    #[test]
    fn test_unweighted_walk() {
        let hm: HashMap<_,_> = vec![
            (0usize, vec![(1usize, 1.), (2usize, 1.)]),
            (1usize, vec![(0usize, 1.), (4usize, 1.)]),
            (2usize, vec![(0usize, 1.), (3usize, 1.)]),
            (3usize, vec![(2usize, 1.), (5usize, 1.)]),
            (4usize, vec![(1usize, 1.)]),
            (5usize, vec![(3usize, 1.)])
        ].into_iter().collect();

        let start_node = 2;
        let distances = unweighted_walk_distance(&hm, &start_node);

        assert_eq!(distances[&0], 1.);
        assert_eq!(distances[&1], 2.);
        assert_eq!(distances[&2], 0.);
        assert_eq!(distances[&3], 1.);
        assert_eq!(distances[&4], 3.);
        assert_eq!(distances[&5], 2.);

    }

}
