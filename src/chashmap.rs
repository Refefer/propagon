extern crate hashbrown;

use std::fmt::Debug;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash,Hasher};
use std::sync::RwLock;

use hashbrown::HashMap;

#[derive(Debug)]
pub struct CHashMap<K: Hash + Eq,V> {
    partitions: Vec<RwLock<HashMap<K, V>>>,
    segments: usize
}

impl <K: Hash + Eq, V> CHashMap<K, V> {
    pub fn new(segments: usize) -> Self {
        CHashMap {
            partitions: (0..segments).map(|_| RwLock::new(HashMap::new())).collect(),
            segments: segments
        }
    }

    pub fn extend(mut self, it: impl Iterator<Item=(K, V)>) -> Self {
        let len = self.segments;
        let mut maps: Vec<_> = self.partitions.into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();

        for item in it {
            maps.get_mut(Self::get_idx(&item.0, len))
                .unwrap()
                .insert(item.0, item.1);
        }

        self.partitions = maps.into_iter().map(|m| RwLock::new(m)).collect();
        self
    }

    fn get_idx(key: &K, size: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % size
    }

    pub fn get_map(&self, key: &K) -> &RwLock<HashMap<K, V>> {
        let idx = Self::get_idx(key, self.segments);
        unsafe {
            self.partitions.get_unchecked(idx)
        }
    }

    pub fn into_inner(self) -> Vec<HashMap<K, V>> {
        self.partitions.into_iter().map(|m| m.into_inner().unwrap()).collect()
    }

}

impl <K: Hash + Eq + Clone, V: Clone> CHashMap<K, V> {

    #[allow(dead_code)]
    pub fn update<F: Fn(Option<&V>) -> V>(&self, key: &K, f: F) {
        let mut map = self.get_map(key).write().unwrap();
        if let Some(v) = map.get_mut(key) {
            let new_v = f(Some(&v));
            *v = new_v;
        } else {
            let new_v = f(None);
            map.insert(key.clone(), new_v);
        }
    }

    #[allow(dead_code)]
    pub fn cache(&self, keys: impl Iterator<Item=K>) -> HashMap<K,V> {
        let mut hm = HashMap::new();
        for k in keys {
            if !hm.contains_key(&k) {
                if let Some(v) = self.get_map(&k).read().unwrap().get(&k) {
                    hm.insert(k, v.clone());
                }
            }
        }
        hm
    }
}
