extern crate heap;
extern crate hashbrown;

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


pub fn clean_map<K: Hash>(
    mut features: HashMap<K,f32>, 
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
