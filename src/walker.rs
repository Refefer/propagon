extern crate rand;
extern crate hashbrown;

use std::hash::Hash;

use hashbrown::HashMap;
use rand::prelude::*;


pub struct RandomWalk<'a, K: Hash + Eq + Clone> {
    edges: &'a HashMap<K, Vec<(K, f32)>>,
}

impl <'a, K: Hash + Eq + Clone> RandomWalk<'a, K> {

    pub fn new(edges: &'a HashMap<K, Vec<(K, f32)>>) -> Self {
        RandomWalk { edges }
    }

    pub fn gen_biased_walk<R: Rng>(
        &'a self,
        mut key: &'a K, 
        mut rng: &mut R,
        walk_len: usize
    ) -> Vec<&'a K> {
        let mut walk = Vec::with_capacity(walk_len+1);
        walk.push(key);
        for _ in 0..walk_len {
            key = &self.edges[key]
                .choose_weighted(&mut rng, |(_, w)| *w)
                .expect("Should never be empty!").0;
            walk.push(key);
        }
        walk
    }

    pub fn gen_uniform_walk<R: Rng>(
        &'a self,
        mut key: &'a K, 
        mut rng: &mut R,
        walk_len: usize
    ) -> Vec<&'a K> {
        let mut walk = Vec::with_capacity(walk_len+1);
        walk.push(key);
        for _ in 0..walk_len {
            key = &self.edges[key]
                .choose(&mut rng)
                .expect("Should never be empty!").0;
            walk.push(key);
        }
        walk
    }


}

