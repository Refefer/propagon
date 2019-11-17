extern crate rayon;
extern crate random;
extern crate hashbrown;

use std::hash::Hash;
use std::fmt::Display;

use super::emit_scores;

use rayon::prelude::*;
use hashbrown::HashMap;
use random::Source;

pub struct Settings {
    pub n_iters: usize,
    pub alpha: f32,
    pub beta: f32,
    pub seed: u64
}

pub struct BiRank<K: Hash> {
    u_vocab: HashMap<K,usize>,
    p_vocab: HashMap<K,usize>,

    u: HashMap<usize,f32>,
    p: HashMap<usize,f32>,

    u_edges: HashMap<usize, Vec<(usize, f32)>>,
    p_edges: HashMap<usize, Vec<(usize, f32)>>,

    d_u: HashMap<usize, f32>,
    d_p: HashMap<usize, f32>
}

impl <K: Hash + Eq> BiRank<K> {
    pub fn build(edges: impl Iterator<Item=(K,K,f32)>) -> Self {
        let mut u_vocab = HashMap::new();
        let mut p_vocab = HashMap::new();
        let mut u       = HashMap::new();
        let mut p       = HashMap::new();
        let mut u_edges = HashMap::new();
        let mut p_edges = HashMap::new();
        let mut d_u     = HashMap::new();
        let mut d_p     = HashMap::new();

        for (u_i, p_j, w_ij) in edges {
            let u_size = u_vocab.len();
            let u_idx = u_vocab.entry(u_i).or_insert(u_size);
            let p_size = p_vocab.len();
            let p_idx = p_vocab.entry(p_j).or_insert(p_size);

            u.insert(*u_idx, 0.);
            p.insert(*p_idx, 0.);

            let u_idx_edge = u_edges.entry(*u_idx).or_insert_with(|| vec![]);
            u_idx_edge.push((*p_idx, w_ij));
            let p_idx_edge = p_edges.entry(*p_idx).or_insert_with(|| vec![]);
            p_idx_edge.push((*u_idx, w_ij));

            *d_u.entry(*u_idx).or_insert(0f32) += w_ij;
            *d_p.entry(*p_idx).or_insert(0f32) += w_ij;
        }

        d_u.par_iter_mut().for_each(|(_, v)| {
            *v = v.powf(0.5);
        });

        d_p.par_iter_mut().for_each(|(_, v)| {
            *v = v.powf(0.5);
        });


        BiRank {
            u_vocab, p_vocab, 
            u, p, 
            u_edges, p_edges,
            d_u, d_p
        }
    }

    pub fn randomize(&mut self, settings: &Settings) {
        let mut source = random::default().seed([1234, settings.seed]);
        for (_, v) in self.u.iter_mut() {
            *v = source.read_f64() as f32;
        }
        for (_, v) in self.p.iter_mut() {
            *v = source.read_f64() as f32;
        }

    }

    pub fn compute(
        &mut self, 
        settings: &Settings, 
        u_0: HashMap<K,f32>, 
        p_0: HashMap<K,f32>
    ) {
        // Convert u_0 and p_0
        let u_0: HashMap<_,_> = u_0.into_iter()
            .map(|(k, v)| (*self.u_vocab.get(&k).unwrap_or(&self.u_vocab.len()), v))
            .collect();

        let p_0: HashMap<_,_> = p_0.into_iter()
            .map(|(k, v)| (*self.p_vocab.get(&k).unwrap_or(&self.p_vocab.len()), v))
            .collect();

        for n_iter in 0..settings.n_iters {
           eprintln!("Pass: {}", n_iter);
           let p_error = iterate( &mut self.p, &self.u, &self.p_edges, 
               &self.d_p, &self.d_u, &p_0, settings.alpha);

           eprintln!("p error: {}", p_error);
           let u_error = iterate( &mut self.u, &self.p, &self.u_edges, 
               &self.d_u, &self.d_p, &u_0, settings.beta);

           eprintln!("u error: {}", u_error);
        }
    }

}

fn iterate(
    left: &mut HashMap<usize, f32>,
    right: &HashMap<usize, f32>,
    edges: &HashMap<usize, Vec<(usize, f32)>>,
    d_l: &HashMap<usize, f32>,
    d_r: &HashMap<usize, f32>,
    l_0: &HashMap<usize, f32>,
    scale: f32
) -> f32 {
    let error = left.par_iter_mut().map(|(l_i, v_i)| {
        let mut s = 0.;
        if let Some(edge_list) = edges.get(&l_i) {
            let d_l_i = d_l[l_i];
            for (r_j, w_ij) in edge_list {
                let denom = d_l_i * d_r[r_j];
                s += (w_ij * right[r_j]) / denom;
            }
        }
        let l_ip = scale * s + (1. - scale) * *l_0.get(l_i).unwrap_or(&s);
        let e = (*v_i - l_ip).abs();
        *v_i = l_ip;
        e
    }).sum::<f32>();
    error / left.len() as f32
}

impl <K: Hash + Display> BiRank<K> {

    pub fn emit(&self) {
       fn emit_v<K: Hash + Display>(vocab: &HashMap<K, usize>, scores: &HashMap<usize, f32>) {
            let mut v = Vec::new();
            for (n, idx) in vocab {
                v.push((n, scores[idx]));
            }
            v.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());
            emit_scores(v.into_iter());
       }
       emit_v(&self.u_vocab, &self.u);
       println!("");
       emit_v(&self.p_vocab, &self.p);
    }

}
