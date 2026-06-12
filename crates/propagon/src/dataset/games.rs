//! Team-vs-team game results with margins — the v2 tournament input.
//!
//! One game is two rosters ("sides"), a [`GameOutcome`] giving the result
//! from side 1's perspective (win margins carried, ties explicit), and an
//! aggregation weight counting how often the game was observed. The CLI
//! parses the `side1 <TAB> side2 <TAB> threshold [<TAB> count]` text rows;
//! this type is the in-memory form, and consumers lower from it:
//! [`GamesDataset::to_pairwise`] for the win/loss family,
//! [`GamesDataset::margin_pairs`] for margin consumers (Massey, Keener,
//! HodgeRank), and [`GamesDataset::to_matchups`] for Weng-Lin.
//!
//! Storage is two-level CSR like [`MatchupsDataset`]: `players` holds every
//! roster concatenated and `side_offsets` cuts it into per-game sides.
//! **Periods** partition games in insertion order with exactly
//! [`PairwiseDataset`]'s semantics (empty periods collapse).
//!
//! Invariants (validated at [`GamesDataset::push_game`]): both sides
//! non-empty, no player twice in one game (within or across sides), weight
//! finite and positive, margins finite and positive. Every stored id was
//! produced by the owning interner, so `Interner::resolve` cannot miss.
//!
//! Gotcha: [`GamesDataset::from_pairwise`] deliberately bypasses push
//! validation — `PairwiseDataset::push` accepts self-pairs and non-positive
//! weights, and a lossless embedding must keep those rows. Lowerings
//! re-validate whatever they depend on.

use std::collections::HashSet;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::interner::Interner;

use super::{MatchupsDataset, PairwiseDataset};

/// Outcome of one game, from side 1's perspective. Margin payloads are
/// validated > 0 and finite at push time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameOutcome {
    /// Side 1 won by this margin.
    Side1Win(f32),
    /// Side 2 won by this margin.
    Side2Win(f32),
    Tie,
}

/// One game borrowed out of a [`GamesDataset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameView<'a> {
    pub side1: &'a [u32],
    pub side2: &'a [u32],
    pub outcome: GameOutcome,
    /// Aggregation weight (repeat count); 1.0 = one observed game.
    pub weight: f32,
}

/// How [`GamesDataset::to_pairwise`] lowers tied games, since win/loss
/// consumers have no native tie representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TiePolicy {
    #[default]
    Error,
    Discard,
    /// Each tie becomes two half-weight rows, one per direction.
    HalfWin,
}

/// How [`GamesDataset::margin_pairs`] lowers tied games, since margin
/// consumers read the weight column as a point differential.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MarginTies {
    #[default]
    Error,
    Discard,
    /// A tie is a zero-margin observation.
    Zero,
}

/// Columnar team-vs-team game dataset.
#[derive(Clone, Debug)]
pub struct GamesDataset {
    interner: Interner,
    players: Vec<u32>,
    /// `side_offsets[2g]`, `side_offsets[2g+1]`, `side_offsets[2g+2]` cut
    /// `players` into game `g`'s side-1 and side-2 rosters.
    side_offsets: Vec<usize>,
    outcomes: Vec<GameOutcome>,
    /// Aggregation weight (repeat count); 1.0 = one observed game.
    weights: Vec<f32>,
    /// Game indices where periods after the first begin; strictly increasing.
    period_starts: Vec<usize>,
}

impl Default for GamesDataset {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            players: Vec::new(),
            side_offsets: vec![0],
            outcomes: Vec::new(),
            weights: Vec::new(),
            period_starts: Vec::new(),
        }
    }
}

impl GamesDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one game: `side1` faced `side2` with `outcome` (side 1's
    /// perspective), observed `weight` times.
    ///
    /// Validates shape before mutating anything: both sides non-empty, no
    /// player twice in one game (within or across sides), `weight` finite
    /// and positive, win margins finite and positive. Interns names.
    pub fn push_game(
        &mut self,
        side1: &[&str],
        side2: &[&str],
        outcome: GameOutcome,
        weight: f32,
    ) -> Result<()> {
        if side1.is_empty() || side2.is_empty() {
            return Err(Error::InvalidInput("a side cannot be empty".into()));
        }

        let mut seen = HashSet::new();
        for player in side1.iter().chain(side2) {
            if !seen.insert(*player) {
                return Err(Error::InvalidInput(format!(
                    "player {player:?} appears twice in one game"
                )));
            }
        }

        if !(weight.is_finite() && weight > 0.0) {
            return Err(Error::InvalidInput(format!(
                "game weight must be finite and positive, got {weight}"
            )));
        }

        if let GameOutcome::Side1Win(m) | GameOutcome::Side2Win(m) = outcome
            && !(m.is_finite() && m > 0.0)
        {
            return Err(Error::InvalidInput(format!(
                "win margin must be finite and positive, got {m}"
            )));
        }

        for player in side1 {
            let id = self.interner.intern(player);
            self.players.push(id);
        }

        self.side_offsets.push(self.players.len());

        for player in side2 {
            let id = self.interner.intern(player);
            self.players.push(id);
        }

        self.side_offsets.push(self.players.len());
        self.outcomes.push(outcome);
        self.weights.push(weight);
        Ok(())
    }

    /// Convenience 1v1 embedding of the v1 row shape: `winner` beat `loser`
    /// (margin 1) with aggregation weight `weight`.
    ///
    /// Fallible because `winner == loser` is a duplicate-player violation
    /// and `weight` is validated like any other game's.
    pub fn push_pair(&mut self, winner: &str, loser: &str, weight: f32) -> Result<()> {
        self.push_game(&[winner], &[loser], GameOutcome::Side1Win(1.0), weight)
    }

    /// Lossless embedding of a pairwise dataset: every `winner ≻ loser` row
    /// becomes a 1v1 [`GameOutcome::Side1Win`] with margin 1 and the row's
    /// weight; the interner and period boundaries carry over exactly.
    ///
    /// Bypasses push validation deliberately — `PairwiseDataset::push`
    /// accepts self-pairs and non-positive weights, and those rows must
    /// survive the embedding unchanged.
    pub fn from_pairwise(p: &PairwiseDataset) -> Self {
        let mut out = Self {
            interner: p.interner().clone(),
            ..Self::default()
        };

        let mut boundaries = p.period_starts_for_io().into_iter().peekable();
        for (i, (w, l, x)) in p.rows().enumerate() {
            if boundaries.peek() == Some(&i) {
                boundaries.next();
                out.new_period();
            }
            out.push_1v1_unchecked(w, l, x);
        }

        // A boundary at the very end (period with no rows yet) re-injects.
        if boundaries.next().is_some() {
            out.new_period();
        }
        out
    }

    /// Same interner (so the same entity universe) with no games and no
    /// period boundaries — the seed for resampled copies.
    pub(crate) fn empty_like(&self) -> Self {
        Self {
            interner: self.interner.clone(),
            ..Self::default()
        }
    }

    /// Appends one game copied verbatim from a dataset sharing this
    /// interner (the resample path); the source already validated it, so
    /// push validation is skipped like [`GamesDataset::push_1v1_unchecked`].
    pub(crate) fn push_game_unchecked(&mut self, view: GameView<'_>) {
        self.players.extend_from_slice(view.side1);
        self.side_offsets.push(self.players.len());
        self.players.extend_from_slice(view.side2);
        self.side_offsets.push(self.players.len());
        self.outcomes.push(view.outcome);
        self.weights.push(view.weight);
    }

    /// Appends one 1v1 row without validation; only the lossless
    /// [`GamesDataset::from_pairwise`] embedding may use it.
    fn push_1v1_unchecked(&mut self, winner: u32, loser: u32, weight: f32) {
        self.players.push(winner);
        self.side_offsets.push(self.players.len());
        self.players.push(loser);
        self.side_offsets.push(self.players.len());
        self.outcomes.push(GameOutcome::Side1Win(1.0));
        self.weights.push(weight);
    }

    /// Starts a new period at the current end of the dataset. Calling this
    /// with no games since the previous boundary is a no-op (empty periods
    /// collapse, matching v1's blank-line semantics).
    pub fn new_period(&mut self) {
        let here = self.outcomes.len();
        if here == 0 {
            return;
        }
        if self.period_starts.last() != Some(&here) {
            self.period_starts.push(here);
        }
    }

    /// Number of games.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Number of distinct entities seen by the interner.
    pub fn n_entities(&self) -> usize {
        self.interner.len()
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Game `g` as borrowed rosters plus its outcome and weight.
    pub fn game(&self, g: usize) -> GameView<'_> {
        GameView {
            side1: &self.players[self.side_offsets[2 * g]..self.side_offsets[2 * g + 1]],
            side2: &self.players[self.side_offsets[2 * g + 1]..self.side_offsets[2 * g + 2]],
            outcome: self.outcomes[g],
            weight: self.weights[g],
        }
    }

    /// All games in insertion order.
    pub fn games(&self) -> impl Iterator<Item = GameView<'_>> {
        (0..self.len()).map(|g| self.game(g))
    }

    /// Number of non-empty periods (0 for an empty dataset, 1 when no
    /// boundaries are set). A boundary at the very end of the data marks the
    /// start of a period that has no games yet and is not counted.
    pub fn n_periods(&self) -> usize {
        self.periods().count()
    }

    /// Period boundaries as game ranges, in order.
    pub fn periods(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        let len = self.len();
        let starts = std::iter::once(0).chain(self.period_starts.iter().copied());
        let ends = self
            .period_starts
            .iter()
            .copied()
            .chain(std::iter::once(len));

        starts.zip(ends).filter(|(s, e)| e > s).map(|(s, e)| s..e)
    }

    /// Games of one period (as produced by [`GamesDataset::periods`]).
    pub fn period_games(&self, range: Range<usize>) -> impl Iterator<Item = GameView<'_>> {
        range.map(|g| self.game(g))
    }

    /// Iteratively removes games any of whose players appear in fewer than
    /// `min_count` surviving games (v1's `--min-count` cascade; counts are
    /// per game, not weight-scaled). Unlike the pairwise filter, the
    /// interner is rebuilt in first-seen order over surviving games, so
    /// dropped-out players vanish from the vocabulary. Period boundaries
    /// stay between the same surviving games.
    pub fn filter_min_count(&self, min_count: usize) -> Self {
        if min_count <= 1 {
            return self.clone();
        }

        let n = self.n_entities();
        let mut keep: Vec<bool> = vec![true; self.len()];
        loop {
            let mut degree = vec![0usize; n];
            for (g, view) in self.games().enumerate() {
                if keep[g] {
                    for &p in view.side1.iter().chain(view.side2) {
                        degree[p as usize] += 1;
                    }
                }
            }

            let mut changed = false;
            for (g, view) in self.games().enumerate() {
                if keep[g]
                    && view
                        .side1
                        .iter()
                        .chain(view.side2)
                        .any(|&p| degree[p as usize] < min_count)
                {
                    keep[g] = false;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let mut out = Self::new();
        let mut boundaries = self.period_starts.iter().copied().peekable();
        for (g, view) in self.games().enumerate() {
            if boundaries.peek() == Some(&g) {
                boundaries.next();
                out.new_period();
            }

            if !keep[g] {
                continue;
            }

            for &p in view.side1 {
                let id = out.interner.intern(self.interner.resolve(p));
                out.players.push(id);
            }

            out.side_offsets.push(out.players.len());

            for &p in view.side2 {
                let id = out.interner.intern(self.interner.resolve(p));
                out.players.push(id);
            }

            out.side_offsets.push(out.players.len());
            out.outcomes.push(view.outcome);
            out.weights.push(view.weight);
        }

        if boundaries.next().is_some() {
            out.new_period();
        }
        out
    }

    /// Lowers to win/loss pairs for the Bradley-Terry family: the weight
    /// column carries the repeat count, margins are dropped.
    ///
    /// Only 1v1 games lower (multi-player sides are a typed error pointing
    /// at the team-aware consumers); ties follow `ties`. Period boundaries
    /// land between the same games even when a policy drops games, so the
    /// surviving period structure is preserved. The produced interner is
    /// first-seen order over emitted rows.
    pub fn to_pairwise(&self, ties: TiePolicy) -> Result<PairwiseDataset> {
        let mut out = PairwiseDataset::new();
        let mut boundaries = self.period_starts.iter().copied().peekable();

        for (g, view) in self.games().enumerate() {
            if boundaries.peek() == Some(&g) {
                boundaries.next();
                out.new_period();
            }

            let (a, b) = self.one_v_one(g, view, "win/loss")?;
            match (view.outcome, ties) {
                (GameOutcome::Side1Win(_), _) => out.push(a, b, view.weight),
                (GameOutcome::Side2Win(_), _) => out.push(b, a, view.weight),
                (GameOutcome::Tie, TiePolicy::Error) => {
                    return Err(Error::InvalidInput(format!(
                        "game {g} is a tie; pass --ties discard|half-win or use the \
                         tie-native generalized-bradley-terry"
                    )));
                }
                (GameOutcome::Tie, TiePolicy::Discard) => {}
                (GameOutcome::Tie, TiePolicy::HalfWin) => {
                    out.push(a, b, view.weight / 2.0);
                    out.push(b, a, view.weight / 2.0);
                }
            }
        }

        if boundaries.next().is_some() {
            out.new_period();
        }
        Ok(out)
    }

    /// Lowers to margin rows for Massey/Keener/HodgeRank mean-margin: each
    /// game emits its win margin in the weight column, repeated once per
    /// observed game (so repeat counts must be whole numbers — margins
    /// cannot be weight-scaled). Tie games follow `ties`; with
    /// [`MarginTies::Zero`] a tie emits `(side1, side2, 0.0)` rows (fixed
    /// orientation). 1v1 only; period boundaries are preserved exactly like
    /// [`GamesDataset::to_pairwise`].
    pub fn margin_pairs(&self, ties: MarginTies) -> Result<PairwiseDataset> {
        let mut out = PairwiseDataset::new();
        let mut boundaries = self.period_starts.iter().copied().peekable();

        for (g, view) in self.games().enumerate() {
            if boundaries.peek() == Some(&g) {
                boundaries.next();
                out.new_period();
            }

            let (a, b) = self.one_v_one(g, view, "margin")?;
            let reps = Self::whole_repeats(g, view.weight, "margin consumers")?;
            match (view.outcome, ties) {
                (GameOutcome::Side1Win(m), _) => {
                    for _ in 0..reps {
                        out.push(a, b, m);
                    }
                }
                (GameOutcome::Side2Win(m), _) => {
                    for _ in 0..reps {
                        out.push(b, a, m);
                    }
                }
                (GameOutcome::Tie, MarginTies::Error) => {
                    return Err(Error::InvalidInput(format!(
                        "game {g} is a tie; pass --margin-ties discard|zero to lower \
                         ties for margin consumers"
                    )));
                }
                (GameOutcome::Tie, MarginTies::Discard) => {}
                (GameOutcome::Tie, MarginTies::Zero) => {
                    for _ in 0..reps {
                        out.push(a, b, 0.0);
                    }
                }
            }
        }

        if boundaries.next().is_some() {
            out.new_period();
        }
        Ok(out)
    }

    /// Lowers to two-team matches for Weng-Lin/OpenSkill: ranks `[1, 2]`,
    /// `[2, 1]`, `[1, 1]` for side-1 win / side-2 win / tie. Margins are
    /// dropped and repeat counts unroll — each game is pushed `weight`
    /// times, so weights must be whole numbers. Matchups carry no period
    /// structure, so boundaries are dropped.
    pub fn to_matchups(&self) -> Result<MatchupsDataset> {
        let mut out = MatchupsDataset::new();

        for (g, view) in self.games().enumerate() {
            let reps = Self::whole_repeats(g, view.weight, "matchups")?;
            let side1: Vec<&str> = view
                .side1
                .iter()
                .map(|&p| self.interner.resolve(p))
                .collect();

            let side2: Vec<&str> = view
                .side2
                .iter()
                .map(|&p| self.interner.resolve(p))
                .collect();

            let ranks: [u32; 2] = match view.outcome {
                GameOutcome::Side1Win(_) => [1, 2],
                GameOutcome::Side2Win(_) => [2, 1],
                GameOutcome::Tie => [1, 1],
            };

            for _ in 0..reps {
                out.push_match(&[side1.as_slice(), side2.as_slice()], &ranks)?;
            }
        }
        Ok(out)
    }

    /// Resolves game `g` to its two player names for the 1v1-only pairwise
    /// lowerings, erroring when either side has more than one player.
    fn one_v_one(&self, g: usize, view: GameView<'_>, what: &str) -> Result<(&str, &str)> {
        match (view.side1, view.side2) {
            (&[a], &[b]) => Ok((self.interner.resolve(a), self.interner.resolve(b))),
            _ => Err(Error::InvalidInput(format!(
                "game {g} has a multi-player side; {what} lowering needs 1v1 games — \
                 use team-bradley-terry or weng-lin on a matchups dataset for teams"
            ))),
        }
    }

    /// Checks game `g`'s aggregation weight is usable as a whole repeat
    /// count — consumers that replicate rows or matches cannot honor
    /// fractional weights.
    fn whole_repeats(g: usize, weight: f32, what: &str) -> Result<usize> {
        if weight.is_finite() && weight > 0.0 && weight.fract() == 0.0 {
            Ok(weight as usize)
        } else {
            Err(Error::InvalidInput(format!(
                "game {g} has repeat count {weight}; repeat counts must be whole \
                 numbers for {what}"
            )))
        }
    }

    /// Used by dataset io; ids must come from this dataset's interner.
    /// Re-checks the structural invariants (non-empty sides, ids in range,
    /// no duplicate player) so corrupted chunks cannot enter.
    pub(crate) fn push_game_ids(
        &mut self,
        side1: &[u32],
        side2: &[u32],
        outcome: GameOutcome,
        weight: f32,
    ) -> Result<()> {
        if side1.is_empty() || side2.is_empty() {
            return Err(Error::State("games chunk has an empty side".into()));
        }

        let n = self.interner.len() as u32;
        let mut seen = HashSet::new();
        for &p in side1.iter().chain(side2) {
            if p >= n {
                return Err(Error::State("games chunk id out of range".into()));
            }
            if !seen.insert(p) {
                return Err(Error::State(
                    "games chunk repeats a player in one game".into(),
                ));
            }
        }

        self.players.extend_from_slice(side1);
        self.side_offsets.push(self.players.len());
        self.players.extend_from_slice(side2);
        self.side_offsets.push(self.players.len());
        self.outcomes.push(outcome);
        self.weights.push(weight);
        Ok(())
    }

    /// Raw period boundaries for serialization.
    pub(crate) fn period_starts_for_io(&self) -> Vec<usize> {
        self.period_starts.clone()
    }

    pub(crate) fn set_interner(&mut self, interner: Interner) {
        self.interner = interner;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three periods with a tie in the middle one:
    /// P0: a≻b ×1 · P1: c~d ×1, a≻c ×2 · P2: d≻b ×1.
    fn tie_fixture() -> GamesDataset {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(2.0), 1.0)
            .unwrap();
        d.new_period();
        d.push_game(&["c"], &["d"], GameOutcome::Tie, 1.0).unwrap();
        d.push_game(&["a"], &["c"], GameOutcome::Side1Win(1.0), 2.0)
            .unwrap();
        d.new_period();
        d.push_game(&["b"], &["d"], GameOutcome::Side2Win(3.0), 1.0)
            .unwrap();
        d
    }

    #[test]
    fn push_validation_table() {
        let mut d = GamesDataset::new();
        // empty sides
        assert!(d.push_game(&[], &["b"], GameOutcome::Tie, 1.0).is_err());
        assert!(d.push_game(&["a"], &[], GameOutcome::Tie, 1.0).is_err());
        // duplicate within a side
        assert!(
            d.push_game(&["a", "a"], &["b"], GameOutcome::Tie, 1.0)
                .is_err()
        );
        // duplicate across sides
        assert!(
            d.push_game(&["a", "b"], &["b"], GameOutcome::Tie, 1.0)
                .is_err()
        );
        // winner == loser via push_pair
        assert!(d.push_pair("a", "a", 1.0).is_err());
        // weight must be finite and positive
        assert!(d.push_game(&["a"], &["b"], GameOutcome::Tie, 0.0).is_err());
        assert!(d.push_game(&["a"], &["b"], GameOutcome::Tie, -1.0).is_err());
        assert!(
            d.push_game(&["a"], &["b"], GameOutcome::Tie, f32::NAN)
                .is_err()
        );
        assert!(
            d.push_game(&["a"], &["b"], GameOutcome::Tie, f32::INFINITY)
                .is_err()
        );
        // margins must be finite and positive
        assert!(
            d.push_game(&["a"], &["b"], GameOutcome::Side1Win(0.0), 1.0)
                .is_err()
        );
        assert!(
            d.push_game(&["a"], &["b"], GameOutcome::Side1Win(-2.0), 1.0)
                .is_err()
        );
        assert!(
            d.push_game(&["a"], &["b"], GameOutcome::Side2Win(f32::NAN), 1.0)
                .is_err()
        );

        // failed pushes leave the dataset (and interner) untouched
        assert!(d.is_empty());
        assert_eq!(d.n_entities(), 0);

        d.push_game(&["a", "b"], &["c", "d"], GameOutcome::Side1Win(7.0), 2.0)
            .unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d.n_entities(), 4);
        let g = d.game(0);
        assert_eq!(g.side1, &[0, 1]);
        assert_eq!(g.side2, &[2, 3]);
        assert_eq!(g.outcome, GameOutcome::Side1Win(7.0));
        assert_eq!(g.weight, 2.0);
    }

    #[test]
    fn period_semantics_match_pairwise() {
        let mut d = GamesDataset::new();
        d.new_period(); // before any game: no-op
        d.push_pair("a", "b", 1.0).unwrap();
        d.push_pair("a", "c", 2.0).unwrap();
        d.new_period();
        d.push_pair("b", "c", 1.0).unwrap();

        assert_eq!(d.n_periods(), 2);
        assert_eq!(d.periods().collect::<Vec<_>>(), vec![0..2, 2..3]);
        let last: Vec<f32> = d.period_games(2..3).map(|g| g.weight).collect();
        assert_eq!(last, vec![1.0]);

        // empty period boundaries collapse
        d.new_period();
        d.new_period();
        assert_eq!(d.n_periods(), 2);
    }

    #[test]
    fn from_pairwise_round_trips() {
        let mut p = PairwiseDataset::new();
        p.push("a", "b", 1.0);
        p.push("a", "a", 0.0); // pairwise permits self-pairs and zero weights
        p.new_period();
        p.push("b", "c", 2.5);

        let g = GamesDataset::from_pairwise(&p);
        assert_eq!(g.len(), p.rows().count());
        assert_eq!(g.n_entities(), p.n_entities());
        assert_eq!(
            g.periods().collect::<Vec<_>>(),
            p.periods().collect::<Vec<_>>()
        );
        assert!(g.games().all(|v| v.outcome == GameOutcome::Side1Win(1.0)));

        let back = g.to_pairwise(TiePolicy::Error).unwrap();
        assert_eq!(
            back.rows().collect::<Vec<_>>(),
            p.rows().collect::<Vec<_>>()
        );
        assert_eq!(
            back.periods().collect::<Vec<_>>(),
            p.periods().collect::<Vec<_>>()
        );
    }

    #[test]
    fn to_pairwise_error_names_the_game() {
        let err = tie_fixture().to_pairwise(TiePolicy::Error).unwrap_err();
        assert!(err.to_string().contains("game 1"), "{err}");

        let mut teams = GamesDataset::new();
        teams
            .push_game(&["a", "b"], &["c"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        let err = teams.to_pairwise(TiePolicy::Error).unwrap_err();
        assert!(err.to_string().contains("multi-player"), "{err}");
    }

    #[test]
    fn to_pairwise_half_win_splits_weight() {
        let p = tie_fixture().to_pairwise(TiePolicy::HalfWin).unwrap();
        // first-seen interner over emitted rows: a=0, b=1, c=2, d=3
        let rows: Vec<_> = p.rows().collect();
        assert_eq!(
            rows,
            vec![
                (0, 1, 1.0),
                (2, 3, 0.5),
                (3, 2, 0.5),
                (0, 2, 2.0),
                (3, 1, 1.0),
            ]
        );
        assert_eq!(p.periods().collect::<Vec<_>>(), vec![0..1, 1..4, 4..5]);
        assert_eq!(p.interner().get("a"), Some(0));
        assert_eq!(p.interner().get("d"), Some(3));
    }

    #[test]
    fn to_pairwise_discard_keeps_boundaries_straight() {
        let p = tie_fixture().to_pairwise(TiePolicy::Discard).unwrap();
        let rows: Vec<_> = p.rows().collect();
        assert_eq!(rows, vec![(0, 1, 1.0), (0, 2, 2.0), (3, 1, 1.0)]);
        assert_eq!(p.periods().collect::<Vec<_>>(), vec![0..1, 1..2, 2..3]);

        // a period that is entirely ties collapses away, exactly like an
        // empty blank-line batch
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        d.new_period();
        d.push_game(&["c"], &["d"], GameOutcome::Tie, 1.0).unwrap();
        d.new_period();
        d.push_pair("b", "a", 1.0).unwrap();
        let p = d.to_pairwise(TiePolicy::Discard).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p.periods().collect::<Vec<_>>(), vec![0..1, 1..2]);
    }

    #[test]
    fn margin_pairs_put_margins_in_the_weight_column() {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(3.5), 2.0)
            .unwrap();
        d.push_game(&["c"], &["d"], GameOutcome::Side2Win(2.0), 1.0)
            .unwrap();
        let p = d.margin_pairs(MarginTies::Error).unwrap();
        // weight 2 unrolls into two copies; Side2Win flips orientation, so
        // the winner d interns first in the lowered dataset (first-seen on
        // emission: a=0, b=1, d=2, c=3).
        let rows: Vec<_> = p.rows().collect();
        assert_eq!(rows, vec![(0, 1, 3.5), (0, 1, 3.5), (2, 3, 2.0)]);

        // repeat counts must be whole numbers
        let mut frac = GamesDataset::new();
        frac.push_pair("a", "b", 1.5).unwrap();
        let err = frac.margin_pairs(MarginTies::Error).unwrap_err();
        assert!(err.to_string().contains("whole numbers"), "{err}");
    }

    #[test]
    fn margin_pairs_tie_policies() {
        let err = tie_fixture().margin_pairs(MarginTies::Error).unwrap_err();
        assert!(err.to_string().contains("game 1"), "{err}");

        let p = tie_fixture().margin_pairs(MarginTies::Discard).unwrap();
        let rows: Vec<_> = p.rows().collect();
        assert_eq!(
            rows,
            vec![(0, 1, 2.0), (0, 2, 1.0), (0, 2, 1.0), (3, 1, 3.0)]
        );
        assert_eq!(p.periods().collect::<Vec<_>>(), vec![0..1, 1..3, 3..4]);

        // Zero: side1-first orientation, one zero row per repeat
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 2.0).unwrap();
        let p = d.margin_pairs(MarginTies::Zero).unwrap();
        assert_eq!(p.rows().collect::<Vec<_>>(), vec![(0, 1, 0.0), (0, 1, 0.0)]);
    }

    #[test]
    fn to_matchups_ranks_and_repeats() {
        let m = tie_fixture().to_matchups().unwrap();
        // g2 has weight 2 and unrolls into two matches
        assert_eq!(m.len(), 5);
        let ranks = |i: usize| m.match_teams(i).map(|(r, _)| r).collect::<Vec<_>>();
        assert_eq!(ranks(0), vec![1, 2]); // side-1 win
        assert_eq!(ranks(1), vec![1, 1]); // tie
        assert_eq!(ranks(2), vec![1, 2]);
        assert_eq!(ranks(3), vec![1, 2]);
        assert_eq!(ranks(4), vec![2, 1]); // side-2 win

        // multi-player rosters carry over
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c"], GameOutcome::Side2Win(1.0), 1.0)
            .unwrap();
        let m = d.to_matchups().unwrap();
        let teams: Vec<(u32, Vec<u32>)> = m.match_teams(0).map(|(r, p)| (r, p.to_vec())).collect();
        assert_eq!(teams, vec![(2, vec![0, 1]), (1, vec![2])]);

        // repeat counts must be whole numbers
        let mut frac = GamesDataset::new();
        frac.push_pair("a", "b", 0.5).unwrap();
        assert!(frac.to_matchups().is_err());
    }

    #[test]
    fn min_count_filter_matches_pairwise_cascade() {
        // stable triangle (every player keeps 2 games) + one pendant game
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        d.push_pair("b", "c", 1.0).unwrap();
        d.new_period();
        d.push_pair("c", "a", 1.0).unwrap();
        d.push_pair("x", "a", 1.0).unwrap();

        let f = d.filter_min_count(2);
        assert_eq!(f.len(), 3);
        // unlike the pairwise filter, the interner is rebuilt: x vanishes
        assert_eq!(f.n_entities(), 3);
        assert_eq!(f.interner().get("x"), None);
        assert_eq!(f.periods().collect::<Vec<_>>(), vec![0..2, 2..3]);

        // parity with PairwiseDataset::filter_min_count on the same rows
        let p = f.to_pairwise(TiePolicy::Error).unwrap();
        let reference = d.to_pairwise(TiePolicy::Error).unwrap().filter_min_count(2);
        assert_eq!(
            p.rows().collect::<Vec<_>>(),
            reference.rows().collect::<Vec<_>>()
        );

        // removal cascades: a 2-chain fully erodes under min_count=2
        let mut chain = GamesDataset::new();
        chain.push_pair("p", "q", 1.0).unwrap();
        chain.push_pair("q", "r", 1.0).unwrap();
        assert!(chain.filter_min_count(2).is_empty());

        // counts are per game played, not weight-scaled
        let mut weighted = GamesDataset::new();
        weighted.push_pair("p", "q", 10.0).unwrap();
        assert!(weighted.filter_min_count(2).is_empty());
    }
}
