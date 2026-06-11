//! Connected-component extraction (the v1 `extract-components` utility).
//!
//! Splits a graph into its undirected connected components — the standard
//! triage step before fitting rankers that require connectivity
//! (`docs/algorithms.md` §0.2, Ford's condition).

use crate::dataset::{GraphDataset, GraphView};

/// Returns each undirected connected component with at least `min_size`
/// nodes as its own [`GraphDataset`] (original directed edges, input order),
/// largest component first; ties broken by first-seen node for determinism.
pub fn extract_components(view: GraphView<'_>, min_size: usize) -> Vec<GraphDataset> {
    let n = view.n_nodes();
    let adj = view.undirected_adjacency();
    let mut present = vec![false; n];
    for (s, d, _) in view.edges() {
        present[s as usize] = true;
        present[d as usize] = true;
    }

    // Component id per node, discovered in ascending node order.
    let mut component = vec![usize::MAX; n];
    let mut n_components = 0;
    let mut stack = Vec::new();
    for start in 0..n {
        if !present[start] || component[start] != usize::MAX {
            continue;
        }
        component[start] = n_components;
        stack.push(start as u32);
        while let Some(node) = stack.pop() {
            for &(next, _) in &adj[node as usize] {
                if component[next as usize] == usize::MAX {
                    component[next as usize] = n_components;
                    stack.push(next);
                }
            }
        }
        n_components += 1;
    }

    let mut out: Vec<GraphDataset> = vec![GraphDataset::new(); n_components];
    let mut sizes = vec![0usize; n_components];
    for (node, &c) in component.iter().enumerate() {
        if c != usize::MAX {
            sizes[c] += 1;
            let _ = node;
        }
    }
    for (s, d, w) in view.edges() {
        let c = component[s as usize];
        let g = &mut out[c];
        let sn = view.interner.resolve(s);
        let dn = view.interner.resolve(d);
        g.push(sn, dn, w);
    }

    let mut keyed: Vec<(usize, usize, GraphDataset)> = out
        .into_iter()
        .enumerate()
        .map(|(c, g)| (sizes[c], c, g))
        .filter(|(size, _, _)| *size >= min_size)
        .collect();
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, _, g)| g).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_and_filters_components() {
        let mut g = GraphDataset::new();
        // component 1: a-b-c (3 nodes)
        g.push("a", "b", 1.0);
        g.push("b", "c", 2.0);
        // component 2: x-y (2 nodes)
        g.push("x", "y", 1.0);
        // singleton edge to itself-ish pair: p-q
        g.push("p", "q", 1.0);

        let comps = extract_components(g.view(), 1);
        assert_eq!(comps.len(), 3);
        assert_eq!(comps[0].n_nodes(), 3, "largest first");
        assert_eq!(comps[0].len(), 2);

        let comps = extract_components(g.view(), 3);
        assert_eq!(comps.len(), 1);
        let names: Vec<&str> = comps[0].interner().names().collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }
}
