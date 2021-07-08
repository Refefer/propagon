use std::hash::Hash;
use std::fmt::Display;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;

use hashbrown::{HashMap,HashSet};

fn dfs<K: Hash + Eq + Clone>(n: K, graph: &HashMap<K, Vec<(K, f32)>>) -> Vec<K> {
    let mut seen = HashSet::new();
    seen.insert(n.clone());
    let mut stack = vec![n];
    let mut component = Vec::new();
    while !stack.is_empty() {
        let n = stack.pop().unwrap();
        
        for (out_edge, _w) in graph[&n].iter() {
            if !seen.contains(out_edge) {
                seen.insert(out_edge.clone());
                stack.push(out_edge.clone());
                component.push(out_edge.clone());
            }
        }
    }
    component
}

fn write_component<K: Hash + Eq + Display>(path: &str, index: usize, graph: &HashMap<K, Vec<(K, f32)>>, component: Vec<K>) {
    let f = File::create(format!("{}.{}", path, index))
        .expect("Cannot open edges file for writing");
    let mut f = BufWriter::new(f);
    let mut written = HashSet::new();
    for f_node in component.into_iter() {
        for (t_node, w) in graph[&f_node].iter() {
            if !written.contains(t_node) {
                write!(f, "{} {} {}\n", f_node, t_node, w)
                    .expect("Couldn't write out edge!");
            }
        }
        written.insert(f_node);
    }
}

pub fn extract_components<K: Hash + Eq + Clone + Display>(path: &str, min_size: usize, edges: impl Iterator<Item=(K,K,f32)>) {
    let mut graph = HashMap::new();
    for (t,f,w) in edges {
        let e = graph.entry(t.clone()).or_insert_with(|| Vec::new());
        e.push((f.clone(), w.clone()));
        let e = graph.entry(f).or_insert_with(|| Vec::new());
        e.push((t, w));
    }

    // DFS for fully connected components
    let mut seen = HashSet::new();
    let mut i = 0;
    for key in graph.keys() {
        if !seen.contains(key) {
            let component = dfs(key.clone(), &graph);
            seen.extend(component.iter().cloned());
            if component.len() >= min_size {
                write_component(path, i, &graph, component);
                i += 1;
            }
        }
    }
}
