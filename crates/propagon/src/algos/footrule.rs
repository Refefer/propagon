//! Footrule-optimal rank aggregation (`docs/algorithms.md` §6.5; Dwork,
//! Kumar, Naor & Sivakumar, WWW 2001; Diaconis & Graham 1977).
//!
//! Finds the consensus order minimizing total Spearman footrule distance —
//! the sum of absolute rank displacements — to the ballots. Unlike Kemeny
//! this is solvable *exactly* in polynomial time: placing item `i` at
//! consensus position `p` costs `Σ_{ballots b ∋ i} |rank_b(i) − p|`
//! independently of where the other items go, so the optimum is a min-cost
//! perfect matching of items onto positions. Footrule and Kendall distance
//! are within a factor of two of each other [Diaconis & Graham 1977], which
//! makes the result a 2-approximation to the NP-hard Kemeny optimum
//! [Dwork et al. 2001] — a guarantee where Kemeny's heuristics have none.
//!
//! Assumes ballots are orderings over a shared item universe; they need not
//! be complete. Partial ballots use the **induced** footrule objective: a
//! ballot charges displacement only for the items it actually ranks, so items
//! it never saw sit wherever the remaining evidence is cheapest. Fully
//! deterministic — no RNG, and matching ties break toward lower position
//! indices.
//!
//! Gotcha: the cost matrix and the matching are dense — O(n²) memory,
//! O(n³) time. Metasearch and leaderboard sizes, not 10⁵-item catalogs.

use serde::{Deserialize, Serialize};

use crate::dataset::RankingsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Footrule parameters — there are none (the method is exact and
/// deterministic); the struct keeps the params plumbing uniform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Footrule {}

impl Ranker for Footrule {
    type Data = RankingsDataset;
    type Model = FootruleModel;

    fn fit_opts(&self, data: &RankingsDataset, _opts: &FitOptions<'_>) -> Result<FootruleModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = data.n_entities();

        // cost[i][p] = Σ_{ballots b ∋ i} |rank_b(i) − p|; ballot ranks and
        // positions are small integers, so i64 sums are exact — no float
        // tie ambiguity inside the matching.
        let mut cost = vec![vec![0i64; n]; n];

        for ballot in data.rankings() {
            for (rank, &item) in ballot.iter().enumerate() {
                for (pos, cell) in cost[item as usize].iter_mut().enumerate() {
                    *cell += rank.abs_diff(pos) as i64;
                }
            }
        }

        let (position_of, total) = solve_assignment(&cost);

        // Invert item → position into a best-first order.
        let mut order = vec![0u32; n];
        for (item, &pos) in position_of.iter().enumerate() {
            order[pos] = item as u32;
        }

        Ok(FootruleModel {
            names: data.interner().clone(),
            order,
            // Displacements are non-negative, so the optimal total is too.
            cost: total as u64,
        })
    }
}

/// Solves the dense n×n assignment problem (rows = items, columns =
/// positions), returning the column assigned to each row and the total cost.
///
/// This is the Jonker-Volgenant shortest-augmenting-path scheme — the
/// standard O(n³) Hungarian variant (Jonker & Volgenant 1987; Burkard,
/// Dell'Amico & Martello, *Assignment Problems*, ch. 4). Rows enter one at
/// a time; a Dijkstra-like scan over columns on reduced costs
/// `cost[i][j] − u[i] − v[j]` grows an alternating tree until it reaches a
/// free column, the duals absorb the minimum slack `delta` so reduced costs
/// stay non-negative, and the augmenting path is flipped. Internally
/// 1-indexed with column 0 as the virtual root holding the entering row
/// (the classic formulation). Deterministic: columns are scanned in index
/// order and strict `<` keeps the first minimizer, so ties break toward
/// lower positions.
fn solve_assignment(cost: &[Vec<i64>]) -> (Vec<usize>, i64) {
    // Effectively infinite, but with headroom so `min_slack[j] - delta`
    // can never wrap.
    const INF: i64 = i64::MAX / 2;

    let n = cost.len();
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; n + 1];
    // matched_row[j] = row currently assigned to column j (0 = free).
    let mut matched_row = vec![0usize; n + 1];
    // prev_col[j] = the tree column from which j's best slack was found.
    let mut prev_col = vec![0usize; n + 1];

    for row in 1..=n {
        matched_row[0] = row;
        let mut j0 = 0usize;
        let mut min_slack = vec![INF; n + 1];
        let mut visited = vec![false; n + 1];

        // Grow the alternating tree until it reaches a free column. Every
        // unvisited column gets a finite slack on the first scan (the
        // matrix is dense), so `j1` always lands on a real column.
        loop {
            visited[j0] = true;
            let i0 = matched_row[j0];
            let mut delta = INF;
            let mut j1 = 0usize;

            for j in 1..=n {
                if visited[j] {
                    continue;
                }

                let reduced = cost[i0 - 1][j - 1] - u[i0] - v[j];

                if reduced < min_slack[j] {
                    min_slack[j] = reduced;
                    prev_col[j] = j0;
                }

                if min_slack[j] < delta {
                    delta = min_slack[j];
                    j1 = j;
                }
            }

            for j in 0..=n {
                if visited[j] {
                    u[matched_row[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_slack[j] -= delta;
                }
            }

            j0 = j1;

            if matched_row[j0] == 0 {
                break;
            }
        }

        // Flip the augmenting path back to the virtual root.
        loop {
            let j1 = prev_col[j0];
            matched_row[j0] = matched_row[j1];
            j0 = j1;

            if j0 == 0 {
                break;
            }
        }
    }

    let mut col_of_row = vec![0usize; n];
    let mut total = 0i64;

    for j in 1..=n {
        let row = matched_row[j];
        col_of_row[row - 1] = j - 1;
        total += cost[row - 1][j - 1];
    }

    (col_of_row, total)
}

#[derive(Debug, Serialize, Deserialize)]
struct RankLine {
    id: String,
    rank: usize, // n..1, higher is better (v1 output convention)
}

/// What `save_jsonl` writes as the header `params`: footrule itself is
/// parameter-free, so only the fitted objective rides along.
#[derive(Debug, Serialize, Deserialize)]
struct PersistedParams {
    cost: u64,
}

/// Consensus order (best first), exposed as descending rank scores.
///
/// [`Footrule`] has no knobs, so the model stores no params; the state
/// file's params object carries the fitted total cost instead.
#[derive(Debug, Clone)]
pub struct FootruleModel {
    names: Interner,
    /// Entity ids best-first.
    order: Vec<u32>,
    /// Total induced footrule distance achieved by the matching.
    cost: u64,
}

impl FootruleModel {
    /// The consensus ranking, best first.
    pub fn order(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(|&id| self.names.resolve(id))
    }

    /// Total induced footrule distance of the consensus — the exact
    /// optimum of the matching objective.
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

impl RankModel for FootruleModel {
    fn algorithm(&self) -> &'static str {
        "footrule"
    }

    /// Rank positions as scores: best entity gets `n`, worst gets `1` (the
    /// Kemeny output convention; see the [`RankModel::scores`] carve-out).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        let n = self.order.len();
        self.order
            .iter()
            .enumerate()
            .map(move |(pos, &id)| (self.names.resolve(id), (n - pos) as f64))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let n = self.order.len();
        let lines: Vec<RankLine> = self
            .order
            .iter()
            .enumerate()
            .map(|(pos, &id)| RankLine {
                id: self.names.resolve(id).to_string(),
                rank: n - pos,
            })
            .collect();

        state::save_model(w, "footrule", &PersistedParams { cost: self.cost }, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, mut lines): (PersistedParams, Vec<RankLine>) =
            state::load_model(r, "footrule")?;
        lines.sort_by_key(|l| std::cmp::Reverse(l.rank));

        let mut names = Interner::new();
        let order = lines.iter().map(|l| names.intern(&l.id)).collect();

        Ok(Self {
            names,
            order,
            cost: params.cost,
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

    /// Induced footrule cost of an assignment (`position_of[item id]`).
    fn induced_cost(d: &RankingsDataset, position_of: &[usize]) -> u64 {
        let mut total = 0u64;
        for ballot in d.rankings() {
            for (rank, &item) in ballot.iter().enumerate() {
                total += rank.abs_diff(position_of[item as usize]) as u64;
            }
        }
        total
    }

    /// Exhaustive oracle: minimum induced cost over all n! assignments,
    /// enumerated with Heap's algorithm.
    fn brute_force_min(d: &RankingsDataset) -> u64 {
        let n = d.n_entities();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = induced_cost(d, &perm);
        let mut c = vec![0usize; n];
        let mut i = 0usize;

        while i < n {
            if c[i] < i {
                if i.is_multiple_of(2) {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                best = best.min(induced_cost(d, &perm));
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        best
    }

    /// The fitted assignment as `position_of[item id]`.
    fn fitted_positions(d: &RankingsDataset, m: &FootruleModel) -> Vec<usize> {
        let mut position_of = vec![0usize; d.n_entities()];
        for (pos, name) in m.order().enumerate() {
            position_of[d.interner().get(name).unwrap() as usize] = pos;
        }
        position_of
    }

    /// The matching is exact, so its cost must *equal* the enumerated
    /// optimum — full, conflicting, and partial ballot sets alike.
    #[test]
    fn matching_attains_the_enumerated_optimum() {
        let cases: &[&[&[&str]]] = &[
            &[
                &["a", "b", "c", "d"],
                &["b", "a", "d", "c"],
                &["d", "a", "b", "c"],
            ],
            &[
                &["a", "b", "c", "d", "e"],
                &["c", "e", "a", "b", "d"],
                &["b", "d", "e", "a", "c"],
                &["e", "d", "c", "b", "a"],
            ],
            &[&["a", "b", "c"], &["c", "d"], &["b", "d", "a"], &["d", "a"]],
        ];

        for rows in cases {
            let d = ballots(rows);
            let m = Footrule::default().fit(&d).unwrap();
            assert_eq!(m.cost(), brute_force_min(&d), "case {rows:?}");
            // The reported cost is the cost of the reported order.
            assert_eq!(m.cost(), induced_cost(&d, &fitted_positions(&d, &m)));
        }
    }

    #[test]
    fn identical_ballots_reproduce_the_ballot_at_zero_cost() {
        let row: &[&str] = &["a", "b", "c", "d"];
        let d = ballots(&[row, row, row]);
        let m = Footrule::default().fit(&d).unwrap();
        assert_eq!(m.order().collect::<Vec<_>>(), vec!["a", "b", "c", "d"]);
        assert_eq!(m.cost(), 0);
    }

    /// Two reversed ballots: the end items' costs are flat, but the middle
    /// item is strictly cheapest in the middle, so every optimum pins it
    /// there (total cost 2 + 0 + 2 = 4).
    #[test]
    fn reversed_pair_pins_the_middle_item() {
        let d = ballots(&[&["a", "b", "c"], &["c", "b", "a"]]);
        let m = Footrule::default().fit(&d).unwrap();
        let order: Vec<&str> = m.order().collect();
        assert_eq!(order[1], "b");
        assert_eq!(m.cost(), 4);
        assert_eq!(m.cost(), brute_force_min(&d));
    }

    /// Items missing from some ballots still receive positions, and the
    /// induced objective is optimized exactly.
    #[test]
    fn partial_ballots_place_every_item() {
        let d = ballots(&[&["a", "b", "c"], &["b", "d"]]);
        let m = Footrule::default().fit(&d).unwrap();
        let placed: Vec<&str> = m.order().collect();
        assert_eq!(placed.len(), 4);

        for name in ["a", "b", "c", "d"] {
            assert!(placed.contains(&name), "{name} missing from {placed:?}");
        }

        assert_eq!(m.cost(), brute_force_min(&d));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let d = ballots(&[&["a", "b", "c"], &["c", "a", "b"], &["b", "c"]]);
        let m = Footrule::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = FootruleModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
        assert_eq!(loaded.cost(), m.cost());
        assert_eq!(
            loaded.order().collect::<Vec<_>>(),
            m.order().collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let d = RankingsDataset::new();
        assert!(matches!(
            Footrule::default().fit(&d),
            Err(Error::EmptyDataset)
        ));
    }
}
