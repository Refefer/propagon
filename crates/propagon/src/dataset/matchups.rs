//! Multi-team, multiplayer match results: each match is a list of teams
//! with **competition-style ranks** (1 = best; tied teams share a rank and
//! the next rank skips, e.g. 1, 2, 2, 4), each team a list of players.
//!
//! The input shape for Weng-Lin/OpenSkill ratings. Two-level CSR storage:
//! `players` holds every roster concatenated; `team_offsets` cuts it into
//! teams; `match_offsets` cuts the team list into matches.
//!
//! Invariants (validated at `push_match`): ≥ 2 teams per match, no empty
//! team, no player on two teams in one match. Ranks only need to order
//! teams within their match — any monotone labels work; `push_ordered`
//! assigns 1..k for pre-sorted input.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Ragged matches → teams → players, with per-team ranks.
#[derive(Clone, Debug)]
pub struct MatchupsDataset {
    interner: Interner,
    players: Vec<u32>,
    team_offsets: Vec<usize>,
    team_ranks: Vec<u32>,
    match_offsets: Vec<usize>,
}

impl Default for MatchupsDataset {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            players: Vec::new(),
            team_offsets: vec![0],
            team_ranks: Vec::new(),
            match_offsets: vec![0],
        }
    }
}

impl MatchupsDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one match: `teams[i]` (a non-empty roster) finished with
    /// `ranks[i]` (smaller = better; equal = tied).
    pub fn push_match(&mut self, teams: &[&[&str]], ranks: &[u32]) -> Result<()> {
        if teams.len() < 2 {
            return Err(Error::InvalidInput(
                "a match needs at least two teams".into(),
            ));
        }
        if teams.len() != ranks.len() {
            return Err(Error::InvalidInput(format!(
                "{} teams but {} ranks",
                teams.len(),
                ranks.len()
            )));
        }
        if teams.iter().any(|t| t.is_empty()) {
            return Err(Error::InvalidInput("a team cannot be empty".into()));
        }

        let mut seen = std::collections::HashSet::new();
        for team in teams {
            for player in *team {
                if !seen.insert(*player) {
                    return Err(Error::InvalidInput(format!(
                        "player {player:?} appears twice in one match"
                    )));
                }
            }
        }

        for (team, &rank) in teams.iter().zip(ranks) {
            for player in *team {
                self.players.push(self.interner.intern(player));
            }
            self.team_offsets.push(self.players.len());
            self.team_ranks.push(rank);
        }
        self.match_offsets.push(self.team_ranks.len());
        Ok(())
    }

    /// Appends a match whose teams are already listed best-first
    /// (ranks 1..k, no ties).
    pub fn push_ordered(&mut self, teams: &[&[&str]]) -> Result<()> {
        let ranks: Vec<u32> = (1..=teams.len() as u32).collect();
        self.push_match(teams, &ranks)
    }

    /// Number of matches.
    pub fn len(&self) -> usize {
        self.match_offsets.len() - 1
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

    /// Match `m` as `(rank, roster)` pairs in stored order.
    pub fn match_teams(&self, m: usize) -> impl Iterator<Item = (u32, &[u32])> {
        let teams = self.match_offsets[m]..self.match_offsets[m + 1];
        teams.map(|t| {
            (
                self.team_ranks[t],
                &self.players[self.team_offsets[t]..self.team_offsets[t + 1]],
            )
        })
    }

    /// All matches in insertion order.
    pub fn matches(&self) -> impl Iterator<Item = impl Iterator<Item = (u32, &[u32])>> {
        (0..self.len()).map(|m| self.match_teams(m))
    }

    /// Used by dataset io; ids must come from this dataset's interner.
    pub(crate) fn push_match_ids(&mut self, teams: &[Vec<u32>], ranks: &[u32]) -> Result<()> {
        if teams.len() < 2 || teams.len() != ranks.len() || teams.iter().any(Vec::is_empty) {
            return Err(Error::State("malformed matchup chunk".into()));
        }

        for (team, &rank) in teams.iter().zip(ranks) {
            for &player in team {
                if player as usize >= self.interner.len() {
                    return Err(Error::State("matchup chunk id out of range".into()));
                }
                self.players.push(player);
            }
            self.team_offsets.push(self.players.len());
            self.team_ranks.push(rank);
        }
        self.match_offsets.push(self.team_ranks.len());
        Ok(())
    }

    pub(crate) fn set_interner(&mut self, interner: Interner) {
        self.interner = interner;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_iterate() {
        let mut d = MatchupsDataset::new();
        d.push_ordered(&[&["alice", "bob"], &["carol"]]).unwrap();
        d.push_match(&[&["alice"], &["carol"], &["bob"]], &[1, 1, 3])
            .unwrap();

        assert_eq!(d.len(), 2);
        assert_eq!(d.n_entities(), 3);

        let m0: Vec<(u32, Vec<u32>)> = d.match_teams(0).map(|(r, p)| (r, p.to_vec())).collect();
        assert_eq!(m0, vec![(1, vec![0, 1]), (2, vec![2])]);

        let m1_ranks: Vec<u32> = d.match_teams(1).map(|(r, _)| r).collect();
        assert_eq!(m1_ranks, vec![1, 1, 3]);
    }

    #[test]
    fn validation() {
        let mut d = MatchupsDataset::new();
        assert!(d.push_ordered(&[&["a"]]).is_err());
        assert!(d.push_match(&[&["a"], &[]], &[1, 2]).is_err());
        assert!(d.push_match(&[&["a"], &["b"]], &[1]).is_err());
        assert!(d.push_match(&[&["a", "b"], &["b"]], &[1, 2]).is_err());
        assert_eq!(d.len(), 0);
    }
}
