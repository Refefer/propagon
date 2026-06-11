//! Full or partial rankings (ordered lists, best first), CSR-packed.
//!
//! Consumed by Borda and Kemeny today; sized for Plackett-Luce multiway
//! input (I-LSR) in v2.x.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Ragged list of rankings: ranking `i` is `items[offsets[i]..offsets[i+1]]`,
/// ordered best-first.
#[derive(Clone, Debug)]
pub struct RankingsDataset {
    interner: Interner,
    items: Vec<u32>,
    offsets: Vec<usize>,
}

impl Default for RankingsDataset {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            items: Vec::new(),
            offsets: vec![0],
        }
    }
}

impl RankingsDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one ranking (best first). Rankings with fewer than two items
    /// carry no preference information and are rejected.
    pub fn push_ranking<'a, I>(&mut self, ranking: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let names: Vec<&str> = ranking.into_iter().collect();
        if names.len() < 2 {
            return Err(Error::InvalidInput(
                "a ranking needs at least two items".into(),
            ));
        }
        for name in names {
            let id = self.interner.intern(name);
            self.items.push(id);
        }
        self.offsets.push(self.items.len());
        Ok(())
    }

    /// Number of rankings.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn n_entities(&self) -> usize {
        self.interner.len()
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Ranking `i` as a slice of ids, best first.
    pub fn ranking(&self, i: usize) -> &[u32] {
        &self.items[self.offsets[i]..self.offsets[i + 1]]
    }

    pub fn rankings(&self) -> impl Iterator<Item = &[u32]> {
        (0..self.len()).map(|i| self.ranking(i))
    }

    /// Explodes every ballot into its implied pairwise outcomes (each
    /// ordered pair contributes one unit-weight comparison) — the standard
    /// rank-breaking bridge to the pairwise rankers (Kemeny, Copeland, …).
    pub fn to_pairwise(&self) -> crate::dataset::PairwiseDataset {
        let mut out = crate::dataset::PairwiseDataset::new();

        for ranking in self.rankings() {
            for (p, &winner) in ranking.iter().enumerate() {
                for &loser in &ranking[p + 1..] {
                    out.push(
                        self.interner.resolve(winner),
                        self.interner.resolve(loser),
                        1.0,
                    );
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_packing() {
        let mut d = RankingsDataset::new();
        d.push_ranking(["a", "b", "c"]).unwrap();
        d.push_ranking(["c", "a"]).unwrap();
        assert!(d.push_ranking(["solo"]).is_err());
        assert_eq!(d.len(), 2);
        assert_eq!(d.ranking(0), &[0, 1, 2]);
        assert_eq!(d.ranking(1), &[2, 0]);
        assert_eq!(d.n_entities(), 3);
    }
}
