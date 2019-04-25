extern crate random;
use super::Games;

use std::collections::HashMap;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-x))
}

pub fn lr(games: &Games, passes: usize, alpha: f32) -> Vec<(u32,f32)> {
    let mut index_map: HashMap<u32,f32> = HashMap::new();
    // We need to remap all values to a vector
    for (w, l, _) in games {
        for id in &[*w,*l] {
            if !index_map.contains_key(id) {
                index_map.insert(*id, 0f32);
            }
        }
    }

    let mut grads = Vec::with_capacity(games.len() * 2);
    for i in 0..passes {
        eprintln!("Processing pass {}", i);
        let mut weights = 0.;
        for (j, (w, l, weight)) in games.iter().enumerate() {
            let w_x = index_map[w];
            let l_x = index_map[l];
            let y_hat = sigmoid(w_x - l_x);
            weights += weight;
            let denom = alpha * weight * (y_hat - 1.0);
            grads.push((w, denom));
            grads.push((l, -denom));
        }

        // Update games
        for (idx, g) in grads.drain(0..) {
            let e = index_map.entry(*idx).or_insert(0.);
            *e -= g / weights;
        }
    }

    index_map.into_iter().collect()
}
