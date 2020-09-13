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
        key: &'a K, 
        rng: &mut R,
        walk_len: usize
    ) -> Vec<&'a K> {
        let mut walk = Vec::with_capacity(walk_len+1);
        self.gen_biased_walk_with_buff(key, rng, walk_len, &mut walk);
        walk
    }

    pub fn gen_biased_walk_with_buff<R: Rng>(
        &'a self,
        mut key: &'a K, 
        mut rng: &mut R,
        walk_len: usize,
        walk: &mut Vec<&'a K>
    ) {
        walk.clear();
        walk.push(key);
        for _ in 0..walk_len {
            key = &self.edges[key]
                .choose_weighted(&mut rng, |(_, w)| *w)
                .expect("Should never be empty!").0;
            walk.push(key);
        }
    }

    pub fn gen_uniform_walk<R: Rng>(
        &'a self,
        key: &'a K, 
        rng: &mut R,
        walk_len: usize
    ) -> Vec<&'a K> {
        let mut walk = Vec::with_capacity(walk_len+1);
        self.gen_uniform_walk_with_buff(key, rng, walk_len, &mut walk);
        walk
    }

    pub fn gen_uniform_walk_with_buff<R: Rng>(
        &'a self,
        mut key: &'a K, 
        mut rng: &mut R,
        walk_len: usize,
        walk: &mut Vec<&'a K>
    ) {
        walk.clear();
        walk.push(key);
        for _ in 0..walk_len {
            key = &self.edges[key]
                .choose(&mut rng)
                .expect("Should never be empty!").0;
            walk.push(key);
        }
    }


}

