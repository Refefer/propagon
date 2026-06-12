//! Bootstrap resampling: i.i.d.-with-replacement copies of every dataset
//! shape, over the same entity universe.
//!
//! [`Resample`] is the data-side contract behind
//! [`Bootstrap`](crate::algos::Bootstrap). Each implementation draws as many
//! observations as the source holds, uniformly with replacement from the
//! caller's rng, at the shape's natural exchangeable unit: individual rows
//! (pairwise, rewards, contextual rewards, annotated pairs), edges (graph),
//! whole games, whole ballots, whole matches, and whole **episodes**
//! (trajectories — resampling individual steps would destroy the return
//! structure value estimators read). Every copy clones the source's
//! interner(s), so dense ids and names stay aligned even when an entity's
//! every observation misses the draw; replicate models fitted on copies then
//! score against one shared name universe, which is what lets the bootstrap
//! wrapper aggregate per entity by name.
//!
//! Assumes the chosen unit is exchangeable — interference or drift across
//! rows/games/episodes breaks the bootstrap's premise, not this code.
//!
//! Gotchas: sequential structure that batch rankers ignore is dropped, not
//! resampled — pairwise and games **period boundaries do not survive** a
//! resample (rating-period consumers are `OnlineRanker`s, which the
//! bootstrap wrapper excludes by its `Ranker` bound), and every resampled
//! trajectory episode comes out explicitly closed. An entity can end up with
//! zero observations in a copy; whether a fitter errors, sections it out, or
//! prior-fills it is that fitter's own contract — the wrapper records
//! per-entity coverage instead of legislating here.

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::{
    AnnotatedPairsDataset, ContextualRewardsDataset, GamesDataset, GraphDataset, MatchupsDataset,
    PairwiseDataset, RankingsDataset, RewardsDataset, TrajectoriesDataset,
};

/// A dataset that can produce an i.i.d.-with-replacement copy of itself:
/// same entity universe semantics, same size, rows drawn by the given rng.
pub trait Resample: Sized {
    /// Draws the copy: `len()` units sampled uniformly with replacement,
    /// interner(s) cloned. Deterministic given the rng state; an empty
    /// dataset resamples to an empty copy.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self;
}

impl Resample for PairwiseDataset {
    /// Unit: rows (`winner ≻ loser` with weight, copied verbatim). Period
    /// boundaries are dropped — batch rankers ignore them.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let rows: Vec<(u32, u32, f32)> = self.rows().collect();
        let mut out = self.empty_like();

        for _ in 0..rows.len() {
            let (w, l, x) = rows[rng.random_range(0..rows.len())];
            out.push_row_unchecked(w, l, x);
        }
        out
    }
}

impl Resample for GamesDataset {
    /// Unit: whole games — rosters, outcome, and aggregation weight are
    /// copied verbatim. Period boundaries are dropped — batch rankers
    /// ignore them.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let mut out = self.empty_like();

        for _ in 0..self.len() {
            out.push_game_unchecked(self.game(rng.random_range(0..self.len())));
        }
        out
    }
}

impl Resample for RankingsDataset {
    /// Unit: whole ballots (each drawn ranking keeps its item order).
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let mut out = self.empty_like();

        for _ in 0..self.len() {
            out.push_ranking_unchecked(self.ranking(rng.random_range(0..self.len())));
        }
        out
    }
}

impl Resample for GraphDataset {
    /// Unit: edges (`src → dst` with weight, copied verbatim).
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let edges: Vec<(u32, u32, f32)> = self.view().edges().collect();
        let mut out = self.empty_like();

        for _ in 0..edges.len() {
            let (s, d, x) = edges[rng.random_range(0..edges.len())];
            out.push_edge_unchecked(s, d, x);
        }
        out
    }
}

impl Resample for RewardsDataset {
    /// Unit: rows (`(arm, reward)` events).
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let rows: Vec<(u32, f32)> = self.rows().collect();
        let mut out = self.empty_like();

        for _ in 0..rows.len() {
            let (a, r) = rows[rng.random_range(0..rows.len())];
            out.push_row_unchecked(a, r);
        }
        out
    }
}

impl Resample for MatchupsDataset {
    /// Unit: whole matches — every team's roster and rank copied verbatim.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let mut out = self.empty_like();

        for _ in 0..self.len() {
            out.push_match_unchecked(self.match_teams(rng.random_range(0..self.len())));
        }
        out
    }
}

impl Resample for AnnotatedPairsDataset {
    /// Unit: rows (`(annotator, winner, loser, weight)` votes); both the
    /// entity and annotator interners are cloned.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let rows: Vec<(u32, u32, u32, f32)> = self.rows().collect();
        let mut out = self.empty_like();

        for _ in 0..rows.len() {
            let (a, w, l, x) = rows[rng.random_range(0..rows.len())];
            out.push_ids(a, w, l, x);
        }
        out
    }
}

impl Resample for ContextualRewardsDataset {
    /// Unit: rows (`(arm, reward, context)` events). A non-empty source
    /// yields a copy with the same `dim`; an empty source yields an empty
    /// copy with `dim` unknown.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let rows: Vec<(u32, f32, &[f64])> = self.rows().collect();
        let mut out = self.empty_like();

        for _ in 0..rows.len() {
            let (a, r, x) = rows[rng.random_range(0..rows.len())];
            out.push_row_unchecked(a, r, x);
        }
        out
    }
}

impl Resample for TrajectoriesDataset {
    /// Unit: whole **episodes**, within-episode step order preserved; the
    /// episode count is preserved (the total step count generally is not).
    /// Every drawn episode comes out explicitly closed, including a copy of
    /// a source's trailing open episode.
    fn resample(&self, rng: &mut Xoshiro256PlusPlus) -> Self {
        let n = self.n_episodes();
        let mut out = self.empty_like();

        for _ in 0..n {
            let (states, rewards) = self.episode(rng.random_range(0..n));
            out.push_episode_unchecked(states, rewards);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::dataset::GameOutcome;

    fn rng() -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(26)
    }

    #[test]
    fn pairwise_preserves_len_and_universe_and_drops_periods() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.new_period();
        d.push("b", "c", 2.0);
        d.push("a", "c", 3.0);

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_entities(), 3);
        assert_eq!(r.interner().get("c"), d.interner().get("c"));
        assert_eq!(r.n_periods(), 1, "boundaries are dropped");

        let originals: Vec<_> = d.rows().collect();
        assert!(r.rows().all(|row| originals.contains(&row)));

        // bit-stable at a fixed seed
        let r2 = d.resample(&mut rng());
        assert_eq!(r.rows().collect::<Vec<_>>(), r2.rows().collect::<Vec<_>>());

        // empty in, empty out — no panic on the empty range
        let empty = PairwiseDataset::new().resample(&mut rng());
        assert!(empty.is_empty());
    }

    #[test]
    fn games_preserve_whole_games_and_drop_periods() {
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c", "d"], GameOutcome::Side1Win(3.0), 1.0)
            .unwrap();
        d.new_period();
        d.push_game(&["c"], &["a"], GameOutcome::Tie, 2.0).unwrap();
        d.push_game(&["d"], &["b"], GameOutcome::Side2Win(1.0), 1.5)
            .unwrap();

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_entities(), d.n_entities());
        assert_eq!(r.n_periods(), 1, "boundaries are dropped");

        let originals: Vec<_> = d.games().collect();
        assert!(r.games().all(|g| originals.contains(&g)));
    }

    #[test]
    fn rankings_preserve_whole_ballots() {
        let mut d = RankingsDataset::new();
        d.push_ranking(["a", "b", "c"]).unwrap();
        d.push_ranking(["c", "a"]).unwrap();
        d.push_ranking(["b", "a", "c"]).unwrap();

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_entities(), d.n_entities());

        let originals: Vec<&[u32]> = d.rankings().collect();
        assert!(r.rankings().all(|b| originals.contains(&b)));
    }

    #[test]
    fn graph_preserves_edges_and_node_universe() {
        let mut d = GraphDataset::new();
        d.push("x", "y", 1.5);
        d.push("y", "z", 1.0);
        d.push("z", "x", 0.5);

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_nodes(), d.n_nodes());

        let originals: Vec<_> = d.view().edges().collect();
        assert!(r.view().edges().all(|e| originals.contains(&e)));
    }

    #[test]
    fn rewards_preserve_rows_and_arm_universe() {
        let mut d = RewardsDataset::new();
        d.push("A", 1.0);
        d.push("B", 0.0);
        d.push("A", 0.5);

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_arms(), d.n_arms());

        let originals: Vec<_> = d.rows().collect();
        assert!(r.rows().all(|row| originals.contains(&row)));
    }

    #[test]
    fn matchups_preserve_whole_matches() {
        let mut d = MatchupsDataset::new();
        d.push_ordered(&[&["alice", "bob"], &["carol"]]).unwrap();
        d.push_match(&[&["alice"], &["carol"], &["bob"]], &[1, 1, 3])
            .unwrap();

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_entities(), d.n_entities());

        let collect = |m: &MatchupsDataset, i: usize| -> Vec<(u32, Vec<u32>)> {
            m.match_teams(i).map(|(rk, p)| (rk, p.to_vec())).collect()
        };
        let originals: Vec<_> = (0..d.len()).map(|i| collect(&d, i)).collect();
        assert!((0..r.len()).all(|i| originals.contains(&collect(&r, i))));
    }

    #[test]
    fn annotated_preserves_rows_and_both_universes() {
        let mut d = AnnotatedPairsDataset::new();
        d.push("judge1", "a", "b", 1.0);
        d.push("judge2", "b", "a", 2.0);
        d.push("judge1", "a", "c", 1.0);

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_entities(), d.n_entities());
        assert_eq!(r.n_annotators(), d.n_annotators());

        let originals: Vec<_> = d.rows().collect();
        assert!(r.rows().all(|row| originals.contains(&row)));
    }

    #[test]
    fn contextual_preserves_rows_and_dim() {
        let mut d = ContextualRewardsDataset::new();
        d.push("A", 1.0, &[0.5, 1.5]).unwrap();
        d.push("B", 0.0, &[1.0, -2.0]).unwrap();
        d.push("A", 0.5, &[0.0, 0.25]).unwrap();

        let r = d.resample(&mut rng());
        assert_eq!(r.len(), d.len());
        assert_eq!(r.n_arms(), d.n_arms());
        assert_eq!(r.dim(), Some(2));

        let originals: Vec<(u32, f32, Vec<f64>)> =
            d.rows().map(|(a, x, f)| (a, x, f.to_vec())).collect();
        assert!(
            r.rows()
                .all(|(a, x, f)| originals.contains(&(a, x, f.to_vec())))
        );

        // empty source: dim stays unknown
        let empty = ContextualRewardsDataset::new().resample(&mut rng());
        assert_eq!(empty.dim(), None);
        assert!(empty.is_empty());
    }

    #[test]
    fn trajectories_resample_whole_episodes() {
        let mut d = TrajectoriesDataset::new();
        d.push_step("a", 1.0).unwrap();
        d.push_step("b", 0.5).unwrap();
        d.end_episode();
        d.push_step("b", 2.0).unwrap();
        d.push_step("c", -1.0).unwrap();
        d.push_step("a", 0.0).unwrap();
        d.end_episode();
        d.push_step("c", 3.0).unwrap(); // trailing open episode

        let r = d.resample(&mut rng());
        assert_eq!(r.n_episodes(), 3);
        assert_eq!(r.n_entities(), d.n_entities());

        // every drawn episode is byte-equal to one of the originals, with
        // its within-episode step order intact
        let originals: Vec<(&[u32], &[f32])> = d.episodes().collect();
        assert!(r.episodes().all(|ep| originals.contains(&ep)));
    }
}
