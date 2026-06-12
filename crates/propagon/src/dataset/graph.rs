//! Directed weighted graphs: edges are endorsements (`src` endorses `dst`).
//!
//! Consumed by PageRank, BiRank, Rank Centrality / LSR (via
//! [`PairwiseDataset::as_graph`](super::PairwiseDataset::as_graph)), and
//! component extraction.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Owned columnar edge list.
#[derive(Clone, Debug, Default)]
pub struct GraphDataset {
    interner: Interner,
    src: Vec<u32>,
    dst: Vec<u32>,
    weights: Vec<f32>,
}

impl GraphDataset {
    /// An empty edge list with an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a directed edge `src → dst`, interning both names.
    pub fn push(&mut self, src: &str, dst: &str, weight: f32) {
        let s = self.interner.intern(src);
        let d = self.interner.intern(dst);
        self.src.push(s);
        self.dst.push(d);
        self.weights.push(weight);
    }

    /// Adds a directed edge between already-interned ids.
    pub fn push_ids(&mut self, src: u32, dst: u32, weight: f32) -> Result<()> {
        let n = self.interner.len() as u32;
        if src >= n || dst >= n {
            return Err(Error::InvalidInput(format!(
                "id out of range: ({src}, {dst}) with {n} interned entities"
            )));
        }
        self.src.push(src);
        self.dst.push(dst);
        self.weights.push(weight);
        Ok(())
    }

    /// Same interner (so the same node universe) with no edges — the seed
    /// for resampled copies.
    pub(crate) fn empty_like(&self) -> Self {
        Self {
            interner: self.interner.clone(),
            ..Self::default()
        }
    }

    /// Appends one edge without the [`GraphDataset::push_ids`] range check;
    /// only resampling may use it, where the ids come from `edges()` of a
    /// dataset sharing this interner and so cannot be out of range.
    pub(crate) fn push_edge_unchecked(&mut self, src: u32, dst: u32, weight: f32) {
        self.src.push(src);
        self.dst.push(dst);
        self.weights.push(weight);
    }

    /// Number of edges.
    pub fn len(&self) -> usize {
        self.src.len()
    }

    /// Whether the graph has no edges.
    pub fn is_empty(&self) -> bool {
        self.src.is_empty()
    }

    /// Number of distinct nodes seen by the interner.
    pub fn n_nodes(&self) -> usize {
        self.interner.len()
    }

    /// The interner backing this graph's node ids.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Interns a name ahead of `push_ids` ingestion.
    pub fn intern(&mut self, name: &str) -> u32 {
        self.interner.intern(name)
    }

    /// Borrowed view, the common input type for graph algorithms.
    pub fn view(&self) -> GraphView<'_> {
        GraphView {
            src: &self.src,
            dst: &self.dst,
            weights: &self.weights,
            interner: &self.interner,
        }
    }
}

/// Borrowed columnar edge list — what graph algorithms actually consume.
/// Produced by [`GraphDataset::view`] and
/// [`PairwiseDataset::as_graph`](super::PairwiseDataset::as_graph).
#[derive(Clone, Copy)]
pub struct GraphView<'a> {
    /// Edge source ids (the endorsing node), one per edge.
    pub src: &'a [u32],
    /// Edge destination ids (the endorsed node), one per edge.
    pub dst: &'a [u32],
    /// Edge weights, one per edge.
    pub weights: &'a [f32],
    /// The interner mapping these ids back to node names.
    pub interner: &'a Interner,
}

/// Out-adjacency lists indexed by node id.
pub type Adjacency = Vec<Vec<(u32, f32)>>;

impl<'a> GraphView<'a> {
    /// Number of distinct nodes seen by the interner.
    pub fn n_nodes(&self) -> usize {
        self.interner.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.src.len()
    }

    /// Whether the graph has no edges.
    pub fn is_empty(&self) -> bool {
        self.src.is_empty()
    }

    /// All edges as `(src, dst, weight)` in stored order.
    pub fn edges(&self) -> impl Iterator<Item = (u32, u32, f32)> + 'a {
        self.src
            .iter()
            .zip(self.dst)
            .zip(self.weights)
            .map(|((&s, &d), &w)| (s, d, w))
    }

    /// Builds out-adjacency lists (`adj[src]` holds `(dst, weight)`).
    pub fn out_adjacency(&self) -> Adjacency {
        let mut adj: Adjacency = vec![Vec::new(); self.n_nodes()];
        for (s, d, w) in self.edges() {
            adj[s as usize].push((d, w));
        }
        adj
    }

    /// Builds undirected adjacency lists (each edge appears in both rows).
    pub fn undirected_adjacency(&self) -> Adjacency {
        let mut adj: Adjacency = vec![Vec::new(); self.n_nodes()];
        for (s, d, w) in self.edges() {
            adj[s as usize].push((d, w));
            adj[d as usize].push((s, w));
        }
        adj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_construction() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("a", "c", 2.0);
        g.push("c", "a", 0.5);
        let v = g.view();
        assert_eq!(v.n_nodes(), 3);
        let adj = v.out_adjacency();
        assert_eq!(adj[0], vec![(1, 1.0), (2, 2.0)]);
        assert_eq!(adj[1], vec![]);
        let und = v.undirected_adjacency();
        assert_eq!(und[1], vec![(0, 1.0)]);
        assert_eq!(und[0].len(), 3);
    }
}
