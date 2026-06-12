//! Bradley-Terry via minorization-maximization (`docs/algorithms.md` §1.1).
//!
//! Hunter's (2004) MM iteration over win/loss data, with v1's connectivity
//! workarounds: the Bradley-Terry MLE only exists on strongly connected
//! comparison graphs with no undefeated/winless entities (Ford's condition,
//! survey §0.2), so the fitter either removes such entities (reporting them
//! in separate sections) or patches the data with fake games / random
//! inter-component links. All mitigations operate on an internal copy — the
//! input dataset is never mutated (FR-1.4).
//!
//! Disconnected data produces one ranked section per component (scores are
//! normalized within a component and are **not** comparable across sections).

use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Bradley-Terry MM parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BradleyTerryMM {
    /// Maximum MM sweeps.
    pub iterations: usize,
    /// Early-exit when the mean absolute policy change drops below this.
    pub tolerance: f64,
    /// Components smaller than this are skipped.
    pub min_graph_size: usize,
    /// Also remove never-won entities (not just undefeated ones).
    pub remove_total_losers: bool,
    /// If > 0: instead of removing undefeated/winless entities, add
    /// bidirectional fake games of this weight involving them.
    pub create_fake_games: f64,
    /// If > 0 and the graph is disconnected: add this many rounds of random
    /// bidirectional links between components and fit as one graph.
    pub random_subgraph_links: usize,
    /// Weight of those random links.
    pub random_subgraph_weight: f64,
    /// Seed for the random inter-component links.
    pub seed: u64,
}

impl Default for BradleyTerryMM {
    fn default() -> Self {
        Self {
            iterations: 10_000,
            tolerance: 1e-6,
            min_graph_size: 1,
            remove_total_losers: false,
            create_fake_games: 0.0,
            random_subgraph_links: 0,
            random_subgraph_weight: 1e-3,
            seed: 1221,
        }
    }
}

type Row = (u32, u32, f64);

impl BradleyTerryMM {
    fn prepare(
        &self,
        data: &PairwiseDataset,
    ) -> (Vec<Row>, HashMap<u32, usize>, HashMap<u32, usize>) {
        let mut rows: Vec<Row> = data.rows().map(|(w, l, x)| (w, l, f64::from(x))).collect();

        if self.create_fake_games > 0.0 {
            add_fake_games(&mut rows, self.create_fake_games);
            (rows, HashMap::new(), HashMap::new())
        } else {
            let (rows, undef, winless) = remove_one_sided(rows, self.remove_total_losers);
            (rows, undef, winless)
        }
    }

    fn fit_inner(
        &self,
        data: &PairwiseDataset,
        warm: Option<&HashMap<String, f64>>,
        opts: &FitOptions<'_>,
    ) -> Result<BtmMmModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let progress = opts.progress;
        let (mut rows, undef, winless) = self.prepare(data);

        let mut components = connected_components(&rows, data.n_entities());
        log::debug!(
            "btm-mm: {} rows across {} components after mitigation",
            rows.len(),
            components.len()
        );

        let components = if self.random_subgraph_links > 0 && components.len() > 1 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
            for _ in 0..self.random_subgraph_links {
                link_components(
                    &mut rows,
                    &components,
                    self.random_subgraph_weight,
                    &mut rng,
                );
            }
            vec![components.into_iter().flatten().collect::<Vec<u32>>()]
        } else {
            // Largest component first, deterministic tie-break by first member.
            components.sort_by_key(|c| (std::cmp::Reverse(c.len()), c.first().copied()));
            components
        };

        let mut sections = Vec::new();
        for component in components {
            if component.len() < self.min_graph_size {
                continue;
            }
            let entries = mm_fit(
                &rows,
                &component,
                self.iterations,
                self.tolerance,
                |id| warm.and_then(|w| data.interner().name(id).and_then(|n| w.get(n)).copied()),
                progress,
            );
            sections.push(Section {
                kind: SectionKind::Ranked,
                entries,
            });
        }

        // v1's two trailing sections: undefeated (by win count, descending)
        // and winless (by loss count ascending, negated scores).
        let mut u: Vec<(u32, f64)> = undef.into_iter().map(|(id, c)| (id, c as f64)).collect();
        u.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sections.push(Section {
            kind: SectionKind::Undefeated,
            entries: u,
        });

        let mut w: Vec<(u32, f64)> = winless
            .into_iter()
            .map(|(id, c)| (id, -(c as f64)))
            .collect();
        w.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sections.push(Section {
            kind: SectionKind::Winless,
            entries: w,
        });

        Ok(BtmMmModel {
            params: *self,
            names: data.interner().clone(),
            sections,
        })
    }
}

impl Ranker for BradleyTerryMM {
    type Data = PairwiseDataset;
    type Model = BtmMmModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<BtmMmModel> {
        self.fit_inner(data, None, opts)
    }

    fn fit_warm_opts(
        &self,
        data: &PairwiseDataset,
        init: &BtmMmModel,
        opts: &FitOptions<'_>,
    ) -> Result<BtmMmModel> {
        // Seed strengths by name from the previous model's ranked sections.
        let mut warm: HashMap<String, f64> = HashMap::new();
        for sec in init
            .sections
            .iter()
            .filter(|s| s.kind == SectionKind::Ranked)
        {
            for &(id, s) in &sec.entries {
                warm.insert(init.names.resolve(id).to_string(), s);
            }
        }
        self.fit_inner(data, Some(&warm), opts)
    }
}

/// For each game whose winner appears only as a winner XOR only as a loser,
/// add a bidirectional fake pairing of weight `weight` (v1 semantics).
fn add_fake_games(rows: &mut Vec<Row>, weight: f64) {
    let (wins, losses) = tally(rows);
    let mut fake = Vec::new();
    for &(w, l, _) in rows.iter() {
        if wins.contains_key(&w) ^ losses.contains_key(&w) {
            fake.push((w, l, weight));
            fake.push((l, w, weight));
        }
    }
    log::debug!("btm-mm: created {} fake games", fake.len());
    rows.extend(fake);
}

/// Iteratively strips undefeated entities (and, with `losers_too`, winless
/// ones), returning the surviving rows and the removed entities with their
/// win/loss counts (v1 `remove_undefeated`).
fn remove_one_sided(
    mut rows: Vec<Row>,
    losers_too: bool,
) -> (Vec<Row>, HashMap<u32, usize>, HashMap<u32, usize>) {
    let mut undef = HashMap::new();
    let mut winless = HashMap::new();
    loop {
        let (wins, losses) = tally(&rows);
        let before = rows.len();
        if losers_too {
            let one_sided = |id: &u32| wins.contains_key(id) ^ losses.contains_key(id);
            rows.retain(|(w, l, _)| {
                if one_sided(w) || one_sided(l) {
                    if let Some(&(c, _)) = wins.get(w).filter(|_| one_sided(w)) {
                        undef.insert(*w, c);
                    }
                    if let Some(&(c, _)) = losses.get(l).filter(|_| one_sided(l)) {
                        winless.insert(*l, c);
                    }
                    false
                } else {
                    true
                }
            });
        } else {
            let all_wins = |id: &u32| wins.contains_key(id) && !losses.contains_key(id);
            rows.retain(|(w, _, _)| {
                if all_wins(w) {
                    undef.insert(*w, wins[w].0);
                    false
                } else {
                    true
                }
            });
        }
        if rows.len() == before {
            break;
        }
    }
    (rows, undef, winless)
}

/// `(count, weight sum)` per entity id.
type TallyMap = HashMap<u32, (usize, f64)>;

fn tally(rows: &[Row]) -> (TallyMap, TallyMap) {
    let mut wins: TallyMap = HashMap::new();
    let mut losses: TallyMap = HashMap::new();
    for &(w, l, s) in rows {
        let e = wins.entry(w).or_default();
        e.0 += 1;
        e.1 += s;
        let e = losses.entry(l).or_default();
        e.0 += 1;
        e.1 += s;
    }
    (wins, losses)
}

/// Undirected connected components over the entities present in `rows`,
/// deterministic (ids visited in ascending order).
fn connected_components(rows: &[Row], n_entities: usize) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_entities];
    let mut present = vec![false; n_entities];
    for &(w, l, _) in rows {
        adj[w as usize].push(l);
        adj[l as usize].push(w);
        present[w as usize] = true;
        present[l as usize] = true;
    }
    let mut seen = vec![false; n_entities];
    let mut components = Vec::new();
    let mut stack = Vec::new();
    for start in 0..n_entities {
        if !present[start] || seen[start] {
            continue;
        }
        let mut component = Vec::new();
        stack.push(start as u32);
        seen[start] = true;
        while let Some(node) = stack.pop() {
            component.push(node);
            for &next in &adj[node as usize] {
                if !seen[next as usize] {
                    seen[next as usize] = true;
                    stack.push(next);
                }
            }
        }
        components.push(component);
    }
    components
}

/// One round of random bidirectional links from each component into another.
fn link_components(
    rows: &mut Vec<Row>,
    components: &[Vec<u32>],
    weight: f64,
    rng: &mut Xoshiro256PlusPlus,
) {
    for (i, members) in components.iter().enumerate() {
        let src = members[rng.random_range(0..members.len())];
        let other = loop {
            let g = rng.random_range(0..components.len());
            if g != i {
                break g;
            }
        };
        let dst = components[other][rng.random_range(0..components[other].len())];
        rows.push((src, dst, weight));
        rows.push((dst, src, weight));
    }
}

/// Hunter's MM sweeps over one component:
/// `π_i ← W_i / Σ_{j played} n_ij / (π_i + π_j)`.
///
/// (v1 summed the denominator over opponents the entity had *beaten* rather
/// than all opponents played, which biased the fixed point off the true MLE
/// whenever an entity never beat someone it played; v2 is the textbook
/// iteration.)
fn mm_fit(
    rows: &[Row],
    component: &[u32],
    iterations: usize,
    tolerance: f64,
    warm_score: impl Fn(u32) -> Option<f64>,
    progress: &dyn crate::Progress,
) -> Vec<(u32, f64)> {
    let members: HashSet<u32> = component.iter().copied().collect();

    let mut played: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut total_wins: HashMap<u32, f64> = HashMap::new();
    let mut n_ij: HashMap<(u32, u32), f64> = HashMap::new();

    for &(w, l, s) in rows {
        if !members.contains(&w) {
            continue;
        }
        played.entry(w).or_default().push(l);
        played.entry(l).or_default().push(w);
        *total_wins.entry(w).or_default() += s;
        total_wins.entry(l).or_default();
        *n_ij.entry((w, l)).or_default() += s;
        *n_ij.entry((l, w)).or_default() += s;
    }
    // Sorted, deduplicated adjacency: the denominator accumulates in a
    // fixed order, so refits are bit-stable across runs.
    for adj in played.values_mut() {
        adj.sort_unstable();
        adj.dedup();
    }

    // Deterministic iteration order.
    let mut ids: Vec<u32> = total_wins.keys().copied().collect();
    ids.sort_unstable();

    let uniform = 1.0 / total_wins.len() as f64;
    let mut policy: HashMap<u32, f64> = HashMap::new();
    let mut new_policy: HashMap<u32, f64> = ids
        .iter()
        .map(|&id| (id, warm_score(id).unwrap_or(uniform)))
        .collect();
    // Sum in sorted-id order: HashMap iteration order varies per instance
    // (and per thread), and float accumulation order changes the last ulp.
    let warm_total: f64 = ids.iter().map(|id| new_policy[id]).sum();
    if warm_total > 0.0 {
        for v in new_policy.values_mut() {
            *v /= warm_total;
        }
    }

    progress.start("mm sweeps", Some(iterations as u64));
    for iter in 0..iterations {
        std::mem::swap(&mut policy, &mut new_policy);
        new_policy.clear();

        for &team in &ids {
            let pi = policy[&team];
            let npi = if let Some(opponents) = played.get(&team) {
                let mut denom = 0.0;
                for &other in opponents {
                    denom += n_ij[&(team, other)] / (pi + policy[&other]);
                }
                total_wins[&team] / denom
            } else {
                0.0
            };
            new_policy.insert(team, npi);
        }

        // Sorted-id order again: an unordered sum here made fits differ in
        // the last ulp across thread pools (per-element division is fine).
        let total: f64 = ids.iter().map(|id| new_policy[id]).sum();
        for v in new_policy.values_mut() {
            *v /= total;
        }

        let err = ids
            .iter()
            .map(|id| (policy[id] - new_policy[id]).abs())
            .sum::<f64>()
            / ids.len() as f64;
        progress.update(iter as u64 + 1);
        if iter % 10 == 0 {
            progress.message(&format!("error {err:0.3e}"));
        }
        if err < tolerance {
            break;
        }
    }
    progress.finish();

    let mut out: Vec<(u32, f64)> = new_policy.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// What a model section contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SectionKind {
    /// MM-fitted strengths, normalized to sum to 1 within the section.
    Ranked,
    /// Entities removed as undefeated; score = win count.
    Undefeated,
    /// Entities removed as never-winning; score = −loss count.
    Winless,
}

/// A group of entities sharing one score scale.
#[derive(Clone, Debug)]
pub struct Section {
    /// Which score scale these entries share (ranked, undefeated, winless).
    pub kind: SectionKind,
    /// `(id, score)` sorted as v1 emitted them.
    pub entries: Vec<(u32, f64)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EntryLine {
    id: String,
    s: f64,
    g: usize,
    k: SectionKind,
}

/// Fitted Bradley-Terry strengths, preserving v1's section structure
/// (per-component rankings, then undefeated, then winless).
#[derive(Debug, Clone)]
pub struct BtmMmModel {
    params: BradleyTerryMM,
    names: Interner,
    sections: Vec<Section>,
}

impl BtmMmModel {
    /// Sections in emission order. The trailing two are always
    /// [`SectionKind::Undefeated`] and [`SectionKind::Winless`] (possibly empty).
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Resolves an entity id to its name, or `None` if out of range.
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.name(id)
    }
}

impl RankModel for BtmMmModel {
    fn algorithm(&self) -> &'static str {
        "btm-mm"
    }

    /// Flattened across sections; scores from different sections are on
    /// different scales (see [`SectionKind`]).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.sections.iter().flat_map(|sec| {
            sec.entries
                .iter()
                .map(|&(id, s)| (self.names.resolve(id), s))
        })
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let mut lines = Vec::new();
        for (g, sec) in self.sections.iter().enumerate() {
            for &(id, s) in &sec.entries {
                lines.push(EntryLine {
                    id: self.names.resolve(id).to_string(),
                    s,
                    g,
                    k: sec.kind,
                });
            }
        }
        state::save_model(w, "btm-mm", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (BradleyTerryMM, Vec<EntryLine>) = state::load_model(r, "btm-mm")?;
        let mut names = Interner::new();
        let mut sections: Vec<Section> = Vec::new();
        for line in lines {
            let id = names.intern(&line.id);
            if line.g >= sections.len() {
                sections.resize_with(line.g + 1, || Section {
                    kind: line.k,
                    entries: Vec::new(),
                });
            }
            sections[line.g].kind = line.k;
            sections[line.g].entries.push((id, line.s));
        }
        // Guarantee the two trailing bookkeeping sections exist.
        ensure_trailing_sections(&mut sections);
        Ok(Self {
            params,
            names,
            sections,
        })
    }
}

fn ensure_trailing_sections(sections: &mut Vec<Section>) {
    if !sections.iter().any(|s| s.kind == SectionKind::Undefeated) {
        sections.push(Section {
            kind: SectionKind::Undefeated,
            entries: Vec::new(),
        });
    }
    if !sections.iter().any(|s| s.kind == SectionKind::Winless) {
        sections.push(Section {
            kind: SectionKind::Winless,
            entries: Vec::new(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> PairwiseDataset {
        // a > b > c with mutual games so nobody is one-sided
        let mut d = PairwiseDataset::new();
        for _ in 0..3 {
            d.push("a", "b", 1.0);
            d.push("b", "c", 1.0);
        }
        d.push("b", "a", 1.0);
        d.push("c", "b", 1.0);
        d.push("a", "c", 1.0);
        d.push("c", "a", 1.0);
        d
    }

    #[test]
    fn recovers_obvious_order() {
        let m = BradleyTerryMM::default().fit(&chain()).unwrap();
        let ranked = &m.sections()[0];
        assert_eq!(ranked.kind, SectionKind::Ranked);
        let names: Vec<&str> = ranked
            .entries
            .iter()
            .map(|&(id, _)| m.name(id).unwrap())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        let total: f64 = ranked.entries.iter().map(|e| e.1).sum();
        assert!((total - 1.0).abs() < 1e-9, "strengths normalize to 1");
    }

    #[test]
    fn undefeated_entities_are_sectioned_out() {
        let mut d = chain();
        d.push("champ", "a", 1.0); // champ never loses
        d.push("champ", "b", 1.0);
        let m = BradleyTerryMM::default().fit(&d).unwrap();
        let undef: Vec<_> = m
            .sections()
            .iter()
            .find(|s| s.kind == SectionKind::Undefeated)
            .unwrap()
            .entries
            .iter()
            .map(|&(id, c)| (m.name(id).unwrap().to_string(), c))
            .collect();
        assert_eq!(undef, vec![("champ".to_string(), 2.0)]);
        // and champ is absent from the ranked section
        let ranked = &m.sections()[0];
        assert!(
            ranked
                .entries
                .iter()
                .all(|&(id, _)| m.name(id) != Some("champ"))
        );
    }

    #[test]
    fn fake_games_keep_everyone_ranked() {
        let mut d = chain();
        d.push("champ", "a", 1.0);
        let algo = BradleyTerryMM {
            create_fake_games: 0.1,
            ..Default::default()
        };
        let m = algo.fit(&d).unwrap();
        let ranked = &m.sections()[0];
        assert!(
            ranked
                .entries
                .iter()
                .any(|&(id, _)| m.name(id) == Some("champ"))
        );
    }

    #[test]
    fn disconnected_components_rank_separately_or_linked() {
        let mut d = chain();
        // disjoint pair x <-> y
        d.push("x", "y", 1.0);
        d.push("y", "x", 1.0);

        let m = BradleyTerryMM::default().fit(&d).unwrap();
        let ranked_sections = m
            .sections()
            .iter()
            .filter(|s| s.kind == SectionKind::Ranked)
            .count();
        assert_eq!(ranked_sections, 2);

        let algo = BradleyTerryMM {
            random_subgraph_links: 1,
            ..Default::default()
        };
        let m = algo.fit(&d).unwrap();
        let ranked_sections = m
            .sections()
            .iter()
            .filter(|s| s.kind == SectionKind::Ranked)
            .count();
        assert_eq!(ranked_sections, 1, "linked components fit as one graph");
    }

    #[test]
    fn warm_start_matches_cold_result() {
        let algo = BradleyTerryMM::default();
        let d = chain();
        let cold = algo.fit(&d).unwrap();
        let warm = algo.fit_warm(&d, &cold).unwrap();
        for (a, b) in cold.sections()[0]
            .entries
            .iter()
            .zip(&warm.sections()[0].entries)
        {
            assert_eq!(a.0, b.0, "same order");
            assert!((a.1 - b.1).abs() < 1e-6, "{} vs {}", a.1, b.1);
        }
    }

    #[test]
    fn round_trip() {
        let m = BradleyTerryMM::default().fit(&chain()).unwrap();
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = BtmMmModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}
