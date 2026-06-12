//! BiRank (`docs/algorithms.md` §4.7, He et al. 2017).
//!
//! Co-ranks the two sides of a bipartite endorsement graph (users ↔ items)
//! by alternating degree-normalized propagation. The two sides have separate
//! identity spaces (v1 semantics): the same string appearing as both a `src`
//! and a `dst` is two different entities.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::GraphDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// BiRank parameters. `alpha` scales the `dst`-side update and `beta` the
/// `src`-side update (v1 wiring); with no priors configured both default to
/// pure propagation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiRank {
    /// Number of alternating propagation sweeps.
    pub iterations: usize,
    /// Scales the `dst`-side (item) update (v1 wiring).
    pub alpha: f64,
    /// Scales the `src`-side (user) update (v1 wiring).
    pub beta: f64,
    /// Seeds the random initialization.
    pub seed: u64,
}

impl Default for BiRank {
    fn default() -> Self {
        Self {
            iterations: 10,
            alpha: 1.0,
            beta: 1.0,
            seed: 2019,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SideLine {
    id: String,
    s: f64,
    side: char, // 'u' = src side, 'p' = dst side (v1 naming)
}

/// Fitted BiRank scores for both sides.
#[derive(Debug, Clone)]
pub struct BiRankModel {
    params: BiRank,
    src_names: Interner,
    dst_names: Interner,
    src_scores: Vec<f64>,
    dst_scores: Vec<f64>,
}

impl BiRankModel {
    /// Scores for the `src` side (v1's "u" section).
    pub fn src_scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.src_names.names().zip(self.src_scores.iter().copied())
    }

    /// Scores for the `dst` side (v1's "p" section).
    pub fn dst_scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.dst_names.names().zip(self.dst_scores.iter().copied())
    }
}

impl RankModel for BiRankModel {
    fn algorithm(&self) -> &'static str {
        "birank"
    }

    /// Both sides chained (src side first). A name can appear on both sides;
    /// use [`BiRankModel::src_scores`]/[`BiRankModel::dst_scores`] when side
    /// identity matters.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.src_scores().chain(self.dst_scores())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let mut lines: Vec<SideLine> = Vec::new();
        for (id, s) in self.src_scores() {
            lines.push(SideLine {
                id: id.to_string(),
                s,
                side: 'u',
            });
        }
        for (id, s) in self.dst_scores() {
            lines.push(SideLine {
                id: id.to_string(),
                s,
                side: 'p',
            });
        }
        state::save_model(w, "birank", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (BiRank, Vec<SideLine>) = state::load_model(r, "birank")?;
        let mut src_names = Interner::new();
        let mut dst_names = Interner::new();
        let mut src_scores = Vec::new();
        let mut dst_scores = Vec::new();
        for line in lines {
            match line.side {
                'u' => {
                    src_names.intern(&line.id);
                    src_scores.push(line.s);
                }
                'p' => {
                    dst_names.intern(&line.id);
                    dst_scores.push(line.s);
                }
                other => {
                    return Err(Error::State(format!("unknown birank side {other:?}")));
                }
            }
        }
        Ok(Self {
            params,
            src_names,
            dst_names,
            src_scores,
            dst_scores,
        })
    }
}

impl Ranker for BiRank {
    type Data = GraphDataset;
    type Model = BiRankModel;

    fn fit_opts(&self, data: &GraphDataset, opts: &FitOptions<'_>) -> Result<BiRankModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let progress = opts.progress;
        let view = data.view();

        // Side-local identity spaces.
        let mut src_names = Interner::new();
        let mut dst_names = Interner::new();
        let mut src_edges: Vec<Vec<(u32, f64)>> = Vec::new();
        let mut dst_edges: Vec<Vec<(u32, f64)>> = Vec::new();
        let mut d_src: Vec<f64> = Vec::new();
        let mut d_dst: Vec<f64> = Vec::new();

        for (s, d, w) in view.edges() {
            let w = f64::from(w);
            let sname = view.interner.resolve(s);
            let dname = view.interner.resolve(d);
            let si = src_names.intern(sname) as usize;
            if si == src_edges.len() {
                src_edges.push(Vec::new());
                d_src.push(0.0);
            }
            let di = dst_names.intern(dname) as usize;
            if di == dst_edges.len() {
                dst_edges.push(Vec::new());
                d_dst.push(0.0);
            }
            src_edges[si].push((di as u32, w));
            dst_edges[di].push((si as u32, w));
            d_src[si] += w;
            d_dst[di] += w;
        }
        d_src.iter_mut().for_each(|v| *v = v.sqrt());
        d_dst.iter_mut().for_each(|v| *v = v.sqrt());

        // Random initialization (v1 randomize, with the v2 RNG).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut src_scores: Vec<f64> = (0..src_edges.len()).map(|_| rng.random()).collect();
        let mut dst_scores: Vec<f64> = (0..dst_edges.len()).map(|_| rng.random()).collect();

        let sweep = |left: &mut [f64],
                     right: &[f64],
                     edges: &[Vec<(u32, f64)>],
                     d_l: &[f64],
                     d_r: &[f64],
                     scale: f64|
         -> f64 {
            let mut err = 0.0;
            for (i, value) in left.iter_mut().enumerate() {
                let mut s = 0.0;
                for &(j, w) in &edges[i] {
                    s += w * right[j as usize] / (d_l[i] * d_r[j as usize]);
                }
                // No priors configured: the prior term defaults to s itself
                // (v1's `l_0.get(i).unwrap_or(&s)`).
                let next = scale * s + (1.0 - scale) * s;
                err += (*value - next).abs();
                *value = next;
            }
            err / left.len() as f64
        };

        progress.start("birank sweeps", Some(self.iterations as u64));
        for it in 0..self.iterations {
            let p_err = sweep(
                &mut dst_scores,
                &src_scores,
                &dst_edges,
                &d_dst,
                &d_src,
                self.alpha,
            );
            let u_err = sweep(
                &mut src_scores,
                &dst_scores,
                &src_edges,
                &d_src,
                &d_dst,
                self.beta,
            );
            progress.update(it as u64 + 1);
            progress.message(&format!("p err {p_err:0.3e}, u err {u_err:0.3e}"));
        }
        progress.finish();

        Ok(BiRankModel {
            params: *self,
            src_names,
            dst_names,
            src_scores,
            dst_scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> GraphDataset {
        let mut g = GraphDataset::new();
        // user1 interacts with items A (heavily) and B; user2 with B only.
        g.push("user1", "A", 3.0);
        g.push("user1", "B", 1.0);
        g.push("user2", "B", 1.0);
        g.push("user3", "A", 2.0);
        g
    }

    #[test]
    fn popular_item_and_active_user_rank_first() {
        let m = BiRank::default().fit(&graph()).unwrap();
        let dst: Vec<(String, f64)> = m.dst_scores().map(|(n, s)| (n.to_string(), s)).collect();
        let best_item = dst.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(best_item.0, "A");

        let src: Vec<(String, f64)> = m.src_scores().map(|(n, s)| (n.to_string(), s)).collect();
        let best_user = src.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(best_user.0, "user1");
    }

    #[test]
    fn deterministic_given_seed_and_round_trips() {
        let a = BiRank::default().fit(&graph()).unwrap();
        let b = BiRank::default().fit(&graph()).unwrap();
        let sa: Vec<f64> = a.scores().map(|(_, s)| s).collect();
        let sb: Vec<f64> = b.scores().map(|(_, s)| s).collect();
        assert_eq!(sa, sb);

        let mut buf = Vec::new();
        a.save_jsonl(&mut buf).unwrap();
        let a2 = BiRankModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        a2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}
