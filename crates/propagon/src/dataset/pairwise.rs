//! Pairwise comparison outcomes: `winner ≻ loser` with a weight.
//!
//! The workhorse dataset — feeds Bradley-Terry (MM/LR), Elo, Glicko-2,
//! LSR/Rank Centrality, ES-RUM, Wilson rate, Borda, Copeland, and Kemeny.
//!
//! **Periods** partition the rows in insertion order (v1's blank-line-separated
//! batches). Rating-period systems (Glicko-2) treat each period as one update;
//! batch fitters ignore periods and read all rows.

use crate::error::{Error, Result};
use crate::interner::Interner;

use super::graph::GraphView;

/// Columnar pairwise-outcome dataset.
#[derive(Clone, Debug, Default)]
pub struct PairwiseDataset {
    interner: Interner,
    winners: Vec<u32>,
    losers: Vec<u32>,
    weights: Vec<f32>,
    /// Row indices where periods after the first begin; strictly increasing.
    period_starts: Vec<usize>,
}

/// Per-entity win/loss tallies, indexed by dense id.
#[derive(Clone, Debug)]
pub struct Tally {
    /// (number of wins, total winning weight) per id.
    pub wins: Vec<(u64, f64)>,
    /// (number of losses, total losing weight) per id.
    pub losses: Vec<(u64, f64)>,
}

impl PairwiseDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `winner ≻ loser`, interning both names.
    pub fn push(&mut self, winner: &str, loser: &str, weight: f32) {
        let w = self.interner.intern(winner);
        let l = self.interner.intern(loser);
        self.winners.push(w);
        self.losers.push(l);
        self.weights.push(weight);
    }

    /// Records an outcome between already-interned ids.
    pub fn push_ids(&mut self, winner: u32, loser: u32, weight: f32) -> Result<()> {
        let n = self.interner.len() as u32;
        if winner >= n || loser >= n {
            return Err(Error::InvalidInput(format!(
                "id out of range: ({winner}, {loser}) with {n} interned entities"
            )));
        }
        self.winners.push(winner);
        self.losers.push(loser);
        self.weights.push(weight);
        Ok(())
    }

    /// Bulk variant of [`PairwiseDataset::push_ids`].
    pub fn push_chunk(&mut self, winners: &[u32], losers: &[u32], weights: &[f32]) -> Result<()> {
        if winners.len() != losers.len() || winners.len() != weights.len() {
            return Err(Error::InvalidInput(format!(
                "chunk length mismatch: {} winners, {} losers, {} weights",
                winners.len(),
                losers.len(),
                weights.len()
            )));
        }
        let n = self.interner.len() as u32;
        if let Some(&max) = winners.iter().chain(losers).max()
            && max >= n
        {
            return Err(Error::InvalidInput(format!(
                "id out of range: {max} with {n} interned entities"
            )));
        }
        self.winners.extend_from_slice(winners);
        self.losers.extend_from_slice(losers);
        self.weights.extend_from_slice(weights);
        Ok(())
    }

    /// Starts a new period at the current end of the dataset. Calling this
    /// with no rows since the previous boundary is a no-op (empty periods
    /// collapse, matching v1's blank-line semantics).
    pub fn new_period(&mut self) {
        let here = self.winners.len();
        if here == 0 {
            return;
        }
        if self.period_starts.last() != Some(&here) {
            self.period_starts.push(here);
        }
    }

    /// Number of comparison rows.
    pub fn len(&self) -> usize {
        self.winners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.winners.is_empty()
    }

    /// Number of distinct entities seen by the interner.
    pub fn n_entities(&self) -> usize {
        self.interner.len()
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Interns a name ahead of bulk `push_chunk` ingestion.
    pub fn intern(&mut self, name: &str) -> u32 {
        self.interner.intern(name)
    }

    /// All rows as `(winner, loser, weight)` in insertion order.
    pub fn rows(&self) -> impl Iterator<Item = (u32, u32, f32)> + '_ {
        self.winners
            .iter()
            .zip(&self.losers)
            .zip(&self.weights)
            .map(|((&w, &l), &x)| (w, l, x))
    }

    /// Number of non-empty periods (0 for an empty dataset, 1 when no
    /// boundaries are set). A boundary at the very end of the data marks the
    /// start of a period that has no rows yet and is not counted.
    pub fn n_periods(&self) -> usize {
        self.periods().count()
    }

    /// Period boundaries as row ranges, in order.
    pub fn periods(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let len = self.len();
        let starts = std::iter::once(0).chain(self.period_starts.iter().copied());
        let ends = self.period_starts.iter().copied().chain(std::iter::once(len));
        starts.zip(ends).filter(|(s, e)| e > s).map(|(s, e)| s..e)
    }

    /// Rows of one period (as produced by [`PairwiseDataset::periods`]).
    pub fn period_rows(
        &self,
        range: std::ops::Range<usize>,
    ) -> impl Iterator<Item = (u32, u32, f32)> + '_ {
        let r2 = range.clone();
        self.winners[range.clone()]
            .iter()
            .zip(&self.losers[r2])
            .zip(&self.weights[range])
            .map(|((&w, &l), &x)| (w, l, x))
    }

    /// Win/loss tallies per entity.
    pub fn tally(&self) -> Tally {
        let n = self.n_entities();
        let mut wins = vec![(0u64, 0f64); n];
        let mut losses = vec![(0u64, 0f64); n];
        for (w, l, x) in self.rows() {
            let e = &mut wins[w as usize];
            e.0 += 1;
            e.1 += f64::from(x);
            let e = &mut losses[l as usize];
            e.0 += 1;
            e.1 += f64::from(x);
        }
        Tally { wins, losses }
    }

    /// Iteratively removes comparisons whose endpoints appear in fewer than
    /// `min_count` rows (v1's `--min-count` semantics). Ids are preserved;
    /// entities may end up with zero rows. Returns a filtered copy.
    pub fn filter_min_count(&self, min_count: usize) -> Self {
        if min_count <= 1 {
            return self.clone();
        }
        let n = self.n_entities();
        let mut keep: Vec<bool> = vec![true; self.len()];
        loop {
            let mut degree = vec![0usize; n];
            for (i, (w, l, _)) in self.rows().enumerate() {
                if keep[i] {
                    degree[w as usize] += 1;
                    degree[l as usize] += 1;
                }
            }
            let mut changed = false;
            for (i, (w, l, _)) in self.rows().enumerate() {
                if keep[i] && (degree[w as usize] < min_count || degree[l as usize] < min_count) {
                    keep[i] = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut out = Self { interner: self.interner.clone(), ..Self::default() };
        let mut boundaries = self.period_starts.iter().copied().peekable();
        for (i, (w, l, x)) in self.rows().enumerate() {
            if boundaries.peek() == Some(&i) {
                boundaries.next();
                out.new_period();
            }
            if keep[i] {
                out.winners.push(w);
                out.losers.push(l);
                out.weights.push(x);
            }
        }
        out
    }

    /// The comparison data as an endorsement graph: each `winner ≻ loser` row
    /// becomes a `loser → winner` edge ("the loser endorses the winner"),
    /// which is the orientation spectral and centrality methods expect.
    pub fn as_graph(&self) -> GraphView<'_> {
        GraphView {
            src: &self.losers,
            dst: &self.winners,
            weights: &self.weights,
            interner: &self.interner,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("a", "c", 2.0);
        d.new_period();
        d.push("b", "c", 1.0);
        d
    }

    #[test]
    fn periods_and_rows() {
        let d = ds();
        assert_eq!(d.len(), 3);
        assert_eq!(d.n_entities(), 3);
        assert_eq!(d.n_periods(), 2);
        let ps: Vec<_> = d.periods().collect();
        assert_eq!(ps, vec![0..2, 2..3]);
        let rows: Vec<_> = d.period_rows(ps[1].clone()).collect();
        assert_eq!(rows, vec![(1, 2, 1.0)]);

        // empty period boundaries collapse
        let mut d2 = ds();
        d2.new_period();
        d2.new_period();
        assert_eq!(d2.n_periods(), 2);
    }

    #[test]
    fn tally_counts_wins_and_losses() {
        let t = ds().tally();
        assert_eq!(t.wins[0], (2, 3.0)); // a won twice, weights 1+2
        assert_eq!(t.losses[0], (0, 0.0));
        assert_eq!(t.wins[2], (0, 0.0)); // c never won
        assert_eq!(t.losses[2], (2, 3.0));
    }

    #[test]
    fn min_count_filter_removes_sparse_endpoints() {
        let mut d = PairwiseDataset::new();
        // stable triangle (every endpoint keeps degree 2) + one pendant row
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        d.push("c", "a", 1.0);
        d.push("x", "a", 1.0);
        let f = d.filter_min_count(2);
        // x has degree 1 -> its row drops; the triangle survives; ids unchanged
        assert_eq!(f.len(), 3);
        assert_eq!(f.n_entities(), 4);
        assert!(f.rows().all(|(w, l, _)| {
            f.interner().name(w) != Some("x") && f.interner().name(l) != Some("x")
        }));

        // removal cascades: a 2-chain fully erodes under min_count=2
        let mut chain = PairwiseDataset::new();
        chain.push("p", "q", 1.0);
        chain.push("q", "r", 1.0);
        assert_eq!(chain.filter_min_count(2).len(), 0);
    }

    #[test]
    fn push_ids_validates_range() {
        let mut d = PairwiseDataset::new();
        let a = d.intern("a");
        let b = d.intern("b");
        d.push_ids(a, b, 1.0).unwrap();
        assert!(d.push_ids(a, 7, 1.0).is_err());
        assert!(d.push_chunk(&[a], &[b, b], &[1.0]).is_err());
    }

    #[test]
    fn graph_view_orientation() {
        let d = ds();
        let g = d.as_graph();
        // first row a≻b becomes b -> a
        assert_eq!(g.src[0], 1);
        assert_eq!(g.dst[0], 0);
    }
}
