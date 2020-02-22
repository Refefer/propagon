extern crate heap;
extern crate hashbrown;

use std::ops::Deref;
use std::hash::Hash;
use hashbrown::HashMap;

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
