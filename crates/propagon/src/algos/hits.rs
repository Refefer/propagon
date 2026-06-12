//! HITS — hubs and authorities (`docs/algorithms.md` §4.5; Kleinberg 1999).
//!
//! Two mutually-recursive scores per node: a good **authority** is pointed
//! to by good hubs, a good **hub** points to good authorities. Iterate
//! `a ← Aᵀh`, `h ← Aa` with L1 normalization — the principal eigenvectors
//! of `AᵀA` and `AAᵀ`.
//!
//! Assumes an unweighted endorsement graph: parallel edges are deduplicated
//! and weights ignored (classic HITS; Langville & Meyer's worked examples).
//! Gotcha (their §3.2): `AᵀA` can be reducible, making the limit depend on
//! the start vector — propagon always starts uniform, which selects a
//! deterministic, repeatable solution; nodes outside the dominant
//! community can legitimately score 0. Also vulnerable to tightly-knit
//! community capture; for bipartite interaction data prefer `birank`.

use serde::{Deserialize, Serialize};

use crate::dataset::GraphDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// HITS parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hits {
    /// Power-iteration budget.
    pub iterations: usize,
    /// L1-change early-exit threshold (sum over both vectors).
    pub tolerance: f64,
}

impl Default for Hits {
    fn default() -> Self {
        Self {
            iterations: 100,
            tolerance: 1e-12,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct HitsLine {
    id: String,
    a: f64,
    h: f64,
}

/// Fitted authority and hub scores (each L1-normalized).
#[derive(Debug, Clone)]
pub struct HitsModel {
    params: Hits,
    names: Interner,
    authority: Vec<f64>,
    hub: Vec<f64>,
}

impl HitsModel {
    /// Per-node authority scores (pointed to by good hubs), L1-normalized.
    pub fn authority_scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.authority.iter().copied())
    }

    /// Per-node hub scores (point to good authorities), L1-normalized.
    pub fn hub_scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.hub.iter().copied())
    }
}

impl RankModel for HitsModel {
    fn algorithm(&self) -> &'static str {
        "hits"
    }

    /// Authority scores (the usual "importance" reading); hubs via
    /// [`HitsModel::hub_scores`].
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.authority_scores()
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<HitsLine> = self
            .names
            .names()
            .zip(self.authority.iter().zip(&self.hub))
            .map(|(id, (&a, &h))| HitsLine {
                id: id.to_string(),
                a,
                h,
            })
            .collect();
        state::save_model(w, "hits", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (Hits, Vec<HitsLine>) = state::load_model(r, "hits")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            authority: lines.iter().map(|l| l.a).collect(),
            hub: lines.iter().map(|l| l.h).collect(),
        })
    }
}

impl Ranker for Hits {
    type Data = GraphDataset;
    type Model = HitsModel;

    fn fit_opts(&self, data: &GraphDataset, _opts: &FitOptions<'_>) -> Result<HitsModel> {
        let g = data.view();
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = g.n_nodes();
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];

        for (s, d, _) in g.edges() {
            out[s as usize].push(d);
        }
        for adj in &mut out {
            adj.sort_unstable();
            adj.dedup();
        }

        let mut authority = vec![1.0 / n as f64; n];
        let mut hub = vec![1.0 / n as f64; n];
        let mut next_a = vec![0.0; n];
        let mut next_h = vec![0.0; n];

        for _ in 0..self.iterations {
            // a ← Aᵀ h, h ← A a (using the refreshed authorities).
            next_a.iter_mut().for_each(|v| *v = 0.0);
            for (src, adj) in out.iter().enumerate() {
                for &dst in adj {
                    next_a[dst as usize] += hub[src];
                }
            }
            normalize_l1(&mut next_a)?;

            for (src, adj) in out.iter().enumerate() {
                next_h[src] = adj.iter().map(|&d| next_a[d as usize]).sum();
            }
            normalize_l1(&mut next_h)?;

            let change: f64 = authority
                .iter()
                .zip(&next_a)
                .chain(hub.iter().zip(&next_h))
                .map(|(a, b)| (a - b).abs())
                .sum();

            authority.copy_from_slice(&next_a);
            hub.copy_from_slice(&next_h);

            if change < self.tolerance {
                break;
            }
        }

        Ok(HitsModel {
            params: *self,
            names: g.interner.clone(),
            authority,
            hub,
        })
    }
}

fn normalize_l1(v: &mut [f64]) -> Result<()> {
    let total: f64 = v.iter().sum();
    if total <= 0.0 || total.is_nan() || total.is_infinite() {
        return Err(Error::Numeric(
            "hits iteration collapsed (no edges reachable?)".into(),
        ));
    }
    v.iter_mut().for_each(|x| *x /= total);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// A star pointing at one node: it is the sole authority, the spokes
    /// are the hubs.
    #[test]
    fn star_graph() {
        let mut g = GraphDataset::new();
        g.push("h1", "center", 1.0);
        g.push("h2", "center", 1.0);
        g.push("h3", "center", 1.0);

        let m = Hits::default().fit(&g).unwrap();
        let a: std::collections::HashMap<&str, f64> = m.authority_scores().collect();
        let h: std::collections::HashMap<&str, f64> = m.hub_scores().collect();

        assert!((a["center"] - 1.0).abs() < 1e-9);
        for hub in ["h1", "h2", "h3"] {
            assert!((h[hub] - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "c", 1.0);
        g.push("a", "c", 1.0);

        let m = Hits::default().fit(&g).unwrap();
        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = HitsModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}
