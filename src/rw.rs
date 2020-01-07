extern crate hashbrown;
extern crate rand;
extern crate thread_local;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash,Hasher};
use std::thread;
use std::cell::RefCell;
use std::sync::Arc;
use std::ops::DerefMut;

use rayon::prelude::*;
use rand::prelude::*;
use hashbrown::HashMap;
use thread_local::ThreadLocal;

use crate::walker;

pub struct RandomWalkIterator<K: Hash,R> {
    remaining: usize,
    buffer: Vec<Vec<K>>,
    edges: HashMap<K,Vec<(K, f32)>>,
    keys: Vec<K>,
    biased_walk: bool,
    walk_len: usize,
    rng: R
}

impl <K: Hash + Eq + Clone + Send + Sync, R: Rng + Sync> Iterator for RandomWalkIterator<K, R> {
    type Item=Vec<K>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.len() == 0 {
            let rbound = self.remaining;
            let lbound = (self.remaining - self.buffer.capacity()).max(0);
            
            let ts_rng = Arc::new(ThreadLocal::new());
            let seed: u64 = self.rng.gen();
            let mut tmp_buff = Vec::with_capacity(0);
            std::mem::swap(&mut tmp_buff, &mut self.buffer);
            (lbound..rbound).into_par_iter().map(|offset| {
                let mut rng = ts_rng.get_or(|| {
                       let mut hasher = DefaultHasher::new();
                        thread::current().id().hash(&mut hasher);
                        let t_seed = hasher.finish();
                        RefCell::new(rand::rngs::StdRng::seed_from_u64(seed ^ t_seed))
                    }).borrow_mut();

                let key = &self.keys[offset % self.keys.len()];
                let walker = walker::RandomWalk::new(&self.edges);
                let walk = if self.biased_walk {
                    walker.gen_biased_walk(key, rng.deref_mut(), self.walk_len)
                } else {
                    walker.gen_uniform_walk(key, rng.deref_mut(), self.walk_len)
                };

                walk.into_iter().cloned().collect()
            }).collect_into_vec(&mut tmp_buff);
            std::mem::swap(&mut tmp_buff, &mut self.buffer);
        }

        if self.remaining > 0 {
            self.remaining -= 1;
            self.buffer.pop()
        } else {
            None
        }
    }
    
}

pub struct RandomWalk {
    pub iterations: usize,
    pub walk_len: usize,
    pub biased_walk: bool,
    pub buffer_size: usize,
    pub seed: u64
}

impl RandomWalk {
    // Create graph
    pub fn generate<K: Hash + Eq + Clone + Send + Sync>(
        &self, 
        graph: impl Iterator<Item=(K,K,f32)>, 
    ) -> impl Iterator<Item=Vec<K>> {

        let mut edges = HashMap::new();
        for (f_node, t_node, weight) in graph.into_iter() {
            let e = edges.entry(f_node.clone()).or_insert_with(|| vec![]);
            e.push((t_node.clone(), weight));
            let e = edges.entry(t_node).or_insert_with(|| vec![]);
            e.push((f_node, weight));
        }

        //  Need keys for randomization
        let rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let keys = edges.keys().cloned().collect();

        eprintln!("Number of Vertices: {}", edges.len());
        RandomWalkIterator {
            remaining: edges.len() * self.iterations,
            buffer: Vec::with_capacity(self.buffer_size),
            edges,
            keys,
            biased_walk: self.biased_walk,
            walk_len: self.walk_len,
            rng
        }
    }
}
