//! Plackett-Luce fitted by Hunter's MM algorithm (`docs/algorithms.md`
//! §1.4; Hunter, Ann. Statist. 2004, eq. 30).
//!
//! Maximum-likelihood worth parameters γ from full or partial rankings:
//! each ballot is a cascade of choices, and the MM update is
//! `γ_t ← w_t / Σ_ballots Σ_stages [t still in the choice set] / Σ_remaining γ`
//! with `w_t` = the number of ballots ranking `t` above at least one other
//! item. Partial ballots (subsets, top-k truncations) are likelihood-exact.
//!
//! Assumes Ford's connectivity condition per component. Items that are
//! **last in every ballot** (`w_t = 0`) or **first in every ballot** (never
//! beaten) have no finite MLE — Hunter dropped four such NASCAR drivers by
//! hand; here they are stripped iteratively into the `Winless`/`Undefeated`
//! sections (scores = ∓ballot counts) and each remaining co-occurrence
//! component is fitted separately with γ normalized to sum 1 within it.
//! Scores across sections/components are not on one scale.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::bt_mm::{Section, SectionKind};
use crate::dataset::RankingsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Plackett-Luce MM parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlackettLuce {
    /// Maximum MM sweeps (Hunter's NASCAR fit needed ~26).
    pub iterations: usize,
    /// Mean-absolute-change stopping rule (Hunter used ‖Δγ‖ < 1e-9).
    pub tolerance: f64,
}

impl Default for PlackettLuce {
    fn default() -> Self {
        Self {
            iterations: 10_000,
            tolerance: 1e-9,
        }
    }
}

impl PlackettLuce {
    /// One MM pass over `ballots` (component-local ids), updating `gamma`
    /// in place; returns the mean absolute change.
    fn mm_sweep(ballots: &[Vec<usize>], w: &[f64], gamma: &mut [f64]) -> f64 {
        let mut denom = vec![0.0; gamma.len()];
        let mut suffix: Vec<f64> = Vec::new();

        for ballot in ballots {
            let m = ballot.len();
            suffix.clear();
            suffix.resize(m + 1, 0.0);

            for p in (0..m).rev() {
                suffix[p] = suffix[p + 1] + gamma[ballot[p]];
            }

            // The item at position p sat in the choice sets of stages
            // 0..=min(p, m-2); the final stage is deterministic.
            let mut acc = 0.0;
            for (p, &item) in ballot.iter().enumerate() {
                if p + 2 <= m {
                    acc += 1.0 / suffix[p];
                }
                denom[item] += acc;
            }
        }

        let mut next: Vec<f64> = w.iter().zip(&denom).map(|(w, d)| w / d).collect();
        let total: f64 = next.iter().sum();
        next.iter_mut().for_each(|g| *g /= total);

        let change = gamma
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / gamma.len() as f64;
        gamma.copy_from_slice(&next);
        change
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct EntryLine {
    id: String,
    s: f64,
    /// Section group index (component boundaries survive round trips).
    g: usize,
    k: SectionKind,
}

/// Fitted Plackett-Luce worths, sectioned like
/// [`BtmMmModel`](crate::algos::BtmMmModel).
#[derive(Debug, Clone)]
pub struct PlackettLuceModel {
    params: PlackettLuce,
    names: Interner,
    sections: Vec<Section>,
}

impl PlackettLuceModel {
    /// The fitted sections: `Ranked` per component (γ sums to 1 within
    /// each), then `Undefeated` and `Winless` strip-outs.
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Resolves an entity id from [`Section`] entries.
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.name(id)
    }
}

impl RankModel for PlackettLuceModel {
    fn algorithm(&self) -> &'static str {
        "plackett-luce"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.sections
            .iter()
            .flat_map(|sec| sec.entries.iter())
            .map(|&(id, s)| (self.names.resolve(id), s))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<EntryLine> = self
            .sections
            .iter()
            .enumerate()
            .flat_map(|(g, sec)| {
                sec.entries.iter().map(move |&(id, s)| EntryLine {
                    id: self.names.resolve(id).to_string(),
                    s,
                    g,
                    k: sec.kind,
                })
            })
            .collect();
        state::save_model(w, "plackett-luce", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PlackettLuce, Vec<EntryLine>) =
            state::load_model(r, "plackett-luce")?;
        let mut names = Interner::new();
        let mut sections: Vec<Section> = Vec::new();
        let mut last_group = usize::MAX;

        for line in lines {
            let id = names.intern(&line.id);

            if line.g != last_group {
                sections.push(Section {
                    kind: line.k,
                    entries: Vec::new(),
                });
                last_group = line.g;
            }

            if let Some(section) = sections.last_mut() {
                section.entries.push((id, line.s));
            }
        }

        Ok(Self {
            params,
            names,
            sections,
        })
    }
}

impl Ranker for PlackettLuce {
    type Data = RankingsDataset;
    type Model = PlackettLuceModel;

    fn fit_opts(&self, data: &RankingsDataset, opts: &FitOptions<'_>) -> Result<PlackettLuceModel> {
        self.fit_inner(data, None, opts)
    }

    fn fit_warm_opts(
        &self,
        data: &RankingsDataset,
        init: &PlackettLuceModel,
        opts: &FitOptions<'_>,
    ) -> Result<PlackettLuceModel> {
        let warm: HashMap<&str, f64> = init
            .sections
            .iter()
            .filter(|sec| sec.kind == SectionKind::Ranked)
            .flat_map(|sec| sec.entries.iter())
            .map(|&(id, s)| (init.names.resolve(id), s))
            .collect();
        self.fit_inner(data, Some(&warm), opts)
    }
}

impl PlackettLuce {
    fn fit_inner(
        &self,
        data: &RankingsDataset,
        warm: Option<&HashMap<&str, f64>>,
        opts: &FitOptions<'_>,
    ) -> Result<PlackettLuceModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = data.n_entities();
        let mut ballots: Vec<Vec<u32>> = data.rankings().map(<[u32]>::to_vec).collect();
        let mut undefeated: Vec<(u32, f64)> = Vec::new();
        let mut winless: Vec<(u32, f64)> = Vec::new();

        // Strip items with no finite MLE; removals cascade.
        loop {
            let mut appear = vec![0u32; n];
            let mut non_last = vec![0u32; n];
            let mut beaten = vec![0u32; n];

            for ballot in &ballots {
                for (p, &id) in ballot.iter().enumerate() {
                    appear[id as usize] += 1;
                    if p + 1 < ballot.len() {
                        non_last[id as usize] += 1;
                    }
                    if p > 0 {
                        beaten[id as usize] += 1;
                    }
                }
            }

            let mut strip = vec![false; n];
            let mut stripped_any = false;

            for id in 0..n {
                if appear[id] == 0 {
                    continue;
                }

                if non_last[id] == 0 {
                    winless.push((id as u32, -f64::from(appear[id])));
                    strip[id] = true;
                    stripped_any = true;
                } else if beaten[id] == 0 {
                    undefeated.push((id as u32, f64::from(appear[id])));
                    strip[id] = true;
                    stripped_any = true;
                }
            }

            if !stripped_any {
                break;
            }

            for ballot in &mut ballots {
                ballot.retain(|&id| !strip[id as usize]);
            }
            ballots.retain(|b| b.len() >= 2);
        }

        // Union-find over co-occurrence: each surviving ballot lies wholly
        // inside one component.
        let mut parent: Vec<u32> = (0..n as u32).collect();

        fn find(parent: &mut [u32], x: u32) -> u32 {
            let mut root = x;
            while parent[root as usize] != root {
                root = parent[root as usize];
            }
            let mut cur = x;
            while parent[cur as usize] != root {
                let next = parent[cur as usize];
                parent[cur as usize] = root;
                cur = next;
            }
            root
        }

        for ballot in &ballots {
            for pair in ballot.windows(2) {
                let a = find(&mut parent, pair[0]);
                let b = find(&mut parent, pair[1]);
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
        }

        let mut components: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut active = vec![false; n];

        for ballot in &ballots {
            for &id in ballot {
                if !active[id as usize] {
                    active[id as usize] = true;
                    components
                        .entry(find(&mut parent, id))
                        .or_default()
                        .push(id);
                }
            }
        }

        let mut roots: Vec<u32> = components.keys().copied().collect();
        roots.sort_unstable();

        let progress = opts.progress;
        progress.start("mm sweeps", Some(self.iterations as u64));

        let mut sections: Vec<Section> = Vec::new();

        for root in roots {
            let Some(mut members) = components.remove(&root) else {
                continue;
            };
            members.sort_unstable();

            let local: HashMap<u32, usize> =
                members.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let comp_ballots: Vec<Vec<usize>> = ballots
                .iter()
                .filter(|b| find(&mut parent, b[0]) == root)
                .map(|b| b.iter().map(|id| local[id]).collect())
                .collect();

            let mut w = vec![0.0; members.len()];
            for ballot in &comp_ballots {
                for &item in &ballot[..ballot.len() - 1] {
                    w[item] += 1.0;
                }
            }

            let mut gamma = vec![1.0 / members.len() as f64; members.len()];

            if let Some(warm) = warm {
                for (&id, &slot) in &local {
                    if let Some(&g) = warm.get(data.interner().resolve(id))
                        && g > 0.0
                    {
                        gamma[slot] = g;
                    }
                }
                let total: f64 = gamma.iter().sum();
                gamma.iter_mut().for_each(|g| *g /= total);
            }

            for sweep in 0..self.iterations {
                let change = Self::mm_sweep(&comp_ballots, &w, &mut gamma);
                progress.update(sweep as u64 + 1);

                if change < self.tolerance {
                    break;
                }
            }

            let mut entries: Vec<(u32, f64)> =
                members.iter().map(|&id| (id, gamma[local[&id]])).collect();
            entries.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            sections.push(Section {
                kind: SectionKind::Ranked,
                entries,
            });
        }

        progress.finish();

        for (kind, mut entries) in [
            (SectionKind::Undefeated, undefeated),
            (SectionKind::Winless, winless),
        ] {
            if !entries.is_empty() {
                entries.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                sections.push(Section { kind, entries });
            }
        }

        if sections.is_empty() {
            return Err(Error::InvalidInput(
                "no rankable items remain after stripping degenerate ones".into(),
            ));
        }

        Ok(PlackettLuceModel {
            params: *self,
            names: data.interner().clone(),
            sections,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn ballots(rows: &[&[&str]]) -> RankingsDataset {
        let mut d = RankingsDataset::new();
        for row in rows {
            d.push_ranking(row.iter().copied()).unwrap();
        }
        d
    }

    /// Exact PL log-likelihood of the fitted worths.
    fn log_lik(model: &PlackettLuceModel, data: &RankingsDataset) -> f64 {
        let gamma: std::collections::HashMap<&str, f64> = model.scores().collect();
        let mut ll = 0.0;
        for ballot in data.rankings() {
            let g: Vec<f64> = ballot
                .iter()
                .map(|&id| gamma[data.interner().name(id).unwrap()])
                .collect();
            let mut suffix: f64 = g.iter().sum();
            for v in &g[..g.len() - 1] {
                ll += (v / suffix).ln();
                suffix -= v;
            }
        }
        ll
    }

    #[test]
    fn recovers_obvious_order() {
        let d = ballots(&[
            &["a", "b", "c"],
            &["a", "c", "b"],
            &["b", "a", "c"],
            &["a", "b", "c"],
            &["c", "a", "b"],
            &["b", "c", "a"],
        ]);
        let m = PlackettLuce::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    /// Brute-force oracle: dense grid over the 2-simplex must not beat the
    /// MM fit's exact log-likelihood.
    #[test]
    fn mm_attains_grid_maximum() {
        let d = ballots(&[
            &["a", "b", "c"],
            &["b", "a", "c"],
            &["a", "c", "b"],
            &["c", "a"],
            &["b", "c"],
            &["c", "b", "a"],
        ]);
        let m = PlackettLuce::default().fit(&d).unwrap();
        let fitted_ll = log_lik(&m, &d);

        let eval = |ga: f64, gb: f64, gc: f64| -> f64 {
            let g = std::collections::HashMap::from([("a", ga), ("b", gb), ("c", gc)]);
            let mut ll = 0.0;
            for ballot in d.rankings() {
                let vals: Vec<f64> = ballot
                    .iter()
                    .map(|&id| g[d.interner().name(id).unwrap()])
                    .collect();
                let mut suffix: f64 = vals.iter().sum();
                for v in &vals[..vals.len() - 1] {
                    ll += (v / suffix).ln();
                    suffix -= v;
                }
            }
            ll
        };

        let mut best = f64::NEG_INFINITY;
        let steps = 200;
        for i in 1..steps {
            for j in 1..(steps - i) {
                let (ga, gb) = (i as f64 / steps as f64, j as f64 / steps as f64);
                best = best.max(eval(ga, gb, 1.0 - ga - gb));
            }
        }

        assert!(
            fitted_ll >= best - 1e-6,
            "mm {fitted_ll} vs grid best {best}"
        );
    }

    /// Always-first and always-last items strip into sections; the rest fit.
    #[test]
    fn degenerate_items_are_sectioned() {
        let d = ballots(&[
            &["hero", "a", "b", "dud"],
            &["hero", "b", "a", "dud"],
            &["hero", "a", "dud"],
        ]);
        let m = PlackettLuce::default().fit(&d).unwrap();

        let kinds: Vec<SectionKind> = m.sections().iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                SectionKind::Ranked,
                SectionKind::Undefeated,
                SectionKind::Winless
            ]
        );
        let names: Vec<&str> = m.sections()[1]
            .entries
            .iter()
            .map(|&(id, _)| m.name(id).unwrap())
            .collect();
        assert_eq!(names, vec!["hero"]);
    }

    #[test]
    fn warm_start_not_worse() {
        let d = ballots(&[
            &["a", "b", "c"],
            &["b", "a", "c"],
            &["a", "c", "b"],
            &["c", "b", "a"],
        ]);
        let pl = PlackettLuce::default();
        let cold = pl.fit(&d).unwrap();
        let warm = pl.fit_warm(&d, &cold).unwrap();
        assert!(log_lik(&warm, &d) >= log_lik(&cold, &d) - 1e-12);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let d = ballots(&[&["a", "b", "c"], &["b", "c", "a"], &["c", "a", "b"]]);
        let m = PlackettLuce::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = PlackettLuceModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}
