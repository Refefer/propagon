//! Team Bradley-Terry (`docs/algorithms.md` §1.2; Huang, Weng & Lin,
//! JMLR 7 2006).
//!
//! Rates individual *players* from team-vs-team games: a team's strength is
//! an aggregate of its members' strengths — additive `s_T = Σ_{i∈T} π_i` or
//! product `s_T = Π_{i∈T} π_i` — and `P(T₁ ≻ T₂) = s_T₁ / (s_T₁ + s_T₂)`.
//! Fitting iterates the stationarity fixed point of the log-likelihood
//! (HWL's updates), Jacobi-style for determinism, normalizing `Σπ = 1` each
//! sweep. On singleton teams both aggregates reduce exactly to Hunter's
//! plain Bradley-Terry MM.
//!
//! Assumes the team analogue of Ford's condition: every player must appear
//! at least once on a non-winning side and once on a non-losing side, else
//! their MLE diverges to 0/∞ — checked up front, offenders surface as
//! [`Error::Numeric`]. Half-win ties count both ways for this check.
//!
//! Gotcha: ties follow [`TiePolicy`] — `Error` (default) rejects, `Discard`
//! drops, `HalfWin` splits the game into two half-weight wins. Players seen
//! *only* in discarded tie games have no information and fail the Ford-style
//! guard rather than silently rating at the initialization value.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{GameOutcome, GamesDataset, TiePolicy};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// How a team's strength aggregates from its members'.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeamAggregate {
    /// `s_T = Σ_{i∈T} π_i` (HWL's sum model).
    #[default]
    Additive,
    /// `s_T = Π_{i∈T} π_i` (HWL's log-linear model).
    Product,
}

/// Team Bradley-Terry parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TeamBradleyTerry {
    /// Member-to-team strength aggregation.
    pub aggregate: TeamAggregate,
    /// Maximum fixed-point sweeps.
    pub iterations: usize,
    /// Early exit when the mean absolute strength change drops below this.
    pub tolerance: f64,
    /// What to do with tied games (`HalfWin` = half a win each way).
    pub ties: TiePolicy,
}

impl Default for TeamBradleyTerry {
    fn default() -> Self {
        Self {
            aggregate: TeamAggregate::default(),
            iterations: 10_000,
            tolerance: 1e-9,
            ties: TiePolicy::default(),
        }
    }
}

/// One decided game: winning roster, losing roster, weight.
struct Decided {
    win: Vec<u32>,
    lose: Vec<u32>,
    weight: f64,
}

impl TeamBradleyTerry {
    /// Lowers games to (winner side, loser side, weight) records under the
    /// tie policy: ties error, vanish, or split into two half-weight wins.
    fn decide(&self, data: &GamesDataset) -> Result<Vec<Decided>> {
        let mut out = Vec::with_capacity(data.len());

        for (g, view) in data.games().enumerate() {
            let weight = f64::from(view.weight);
            match (view.outcome, self.ties) {
                (GameOutcome::Side1Win(_), _) => out.push(Decided {
                    win: view.side1.to_vec(),
                    lose: view.side2.to_vec(),
                    weight,
                }),
                (GameOutcome::Side2Win(_), _) => out.push(Decided {
                    win: view.side2.to_vec(),
                    lose: view.side1.to_vec(),
                    weight,
                }),
                (GameOutcome::Tie, TiePolicy::Error) => {
                    return Err(Error::InvalidInput(format!(
                        "game {g} is a tie; set ties to discard or half-win, or use \
                         the tie-native generalized-bradley-terry for 1v1 games"
                    )));
                }
                (GameOutcome::Tie, TiePolicy::Discard) => {}
                (GameOutcome::Tie, TiePolicy::HalfWin) => {
                    out.push(Decided {
                        win: view.side1.to_vec(),
                        lose: view.side2.to_vec(),
                        weight: weight / 2.0,
                    });
                    out.push(Decided {
                        win: view.side2.to_vec(),
                        lose: view.side1.to_vec(),
                        weight: weight / 2.0,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Ford-style pre-check: a player only ever on winning sides has an MLE
    /// at +∞, only ever on losing sides at 0. Half-win ties produce a record
    /// in each direction, so tied players pass both ways.
    fn check_ford(games: &[Decided], names: &Interner) -> Result<()> {
        let n = names.len();
        let mut on_winning = vec![false; n];
        let mut on_losing = vec![false; n];

        for g in games {
            for &p in &g.win {
                on_winning[p as usize] = true;
            }
            for &p in &g.lose {
                on_losing[p as usize] = true;
            }
        }

        let offenders: Vec<&str> = (0..n)
            .filter(|&i| !on_winning[i] || !on_losing[i])
            .filter_map(|i| names.name(i as u32))
            .take(5)
            .collect();

        if offenders.is_empty() {
            Ok(())
        } else {
            Err(Error::Numeric(format!(
                "team BT MLE diverges: {} appear only on winning sides or only on \
                 losing sides (half-win ties count both ways)",
                offenders.join(", ")
            )))
        }
    }

    /// Side strength under the configured aggregate.
    fn side_strength(&self, side: &[u32], pi: &[f64]) -> f64 {
        let members = side.iter().map(|&p| pi[p as usize]);
        match self.aggregate {
            TeamAggregate::Additive => members.sum(),
            TeamAggregate::Product => members.product(),
        }
    }
}

impl Ranker for TeamBradleyTerry {
    type Data = GamesDataset;
    type Model = TeamBtModel;

    fn fit_opts(&self, data: &GamesDataset, opts: &FitOptions<'_>) -> Result<TeamBtModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let games = self.decide(data)?;
        Self::check_ford(&games, data.interner())?;

        let n = data.n_entities();
        let progress = opts.progress;
        progress.start("team-bt sweeps", Some(self.iterations as u64));

        let mut pi = vec![1.0 / n as f64; n];
        for iter in 0..self.iterations {
            // Stationarity fixed point, all entities from the previous sweep:
            //   additive: π_i ← π_i · (Σ_{i wins} w/s_win) / (Σ_{i plays} w/(s₁+s₂))
            //   product:  π_i ← (Σ_{i wins} w) / (Σ_{i plays} w·(s_T/π_i)/(s₁+s₂))
            // where s_T/π_i is computed as the product over teammates.
            let mut num = vec![0.0f64; n];
            let mut den = vec![0.0f64; n];

            for g in &games {
                let s_win = self.side_strength(&g.win, &pi);
                let s_lose = self.side_strength(&g.lose, &pi);
                let total = s_win + s_lose;

                match self.aggregate {
                    TeamAggregate::Additive => {
                        for &p in &g.win {
                            num[p as usize] += g.weight / s_win;
                        }
                        for &p in g.win.iter().chain(&g.lose) {
                            den[p as usize] += g.weight / total;
                        }
                    }
                    TeamAggregate::Product => {
                        for &p in &g.win {
                            num[p as usize] += g.weight;
                        }
                        for side in [&g.win, &g.lose] {
                            for &p in side.iter() {
                                let teammates: f64 = side
                                    .iter()
                                    .filter(|&&q| q != p)
                                    .map(|&q| pi[q as usize])
                                    .product();
                                den[p as usize] += g.weight * teammates / total;
                            }
                        }
                    }
                }
            }

            let mut next: Vec<f64> = (0..n)
                .map(|i| match self.aggregate {
                    TeamAggregate::Additive => pi[i] * num[i] / den[i],
                    TeamAggregate::Product => num[i] / den[i],
                })
                .collect();
            let total: f64 = next.iter().sum();
            next.iter_mut().for_each(|v| *v /= total);

            let err = pi
                .iter()
                .zip(&next)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n as f64;
            pi = next;
            progress.update(iter as u64 + 1);

            if iter % 10 == 0 {
                progress.message(&format!("error {err:0.3e}"));
            }
            if err < self.tolerance {
                break;
            }
        }
        progress.finish();

        if pi.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numeric(
                "team BT produced non-finite strengths; the team comparison \
                 graph is likely too sparse or disconnected"
                    .into(),
            ));
        }

        Ok(TeamBtModel {
            params: *self,
            names: data.interner().clone(),
            scores: pi,
        })
    }
}

/// Fitted player strengths (normalized to sum to 1; higher is better).
#[derive(Debug, Clone)]
pub struct TeamBtModel {
    params: TeamBradleyTerry,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(TeamBtModel, "team-bradley-terry");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;
    use crate::algos::BradleyTerryMM;
    use crate::dataset::PairwiseDataset;

    fn scores_of(m: &TeamBtModel) -> std::collections::HashMap<String, f64> {
        m.scores().map(|(n, s)| (n.to_string(), s)).collect()
    }

    /// Bidirectional 1v1 rows shared with the plain-BT comparison.
    const SINGLETON_ROWS: &[(&str, &str, f32)] = &[
        ("a", "b", 5.0),
        ("b", "a", 2.0),
        ("a", "c", 4.0),
        ("c", "a", 3.0),
        ("b", "c", 6.0),
        ("c", "b", 2.0),
        ("c", "d", 3.0),
        ("d", "c", 1.0),
        ("d", "a", 1.0),
        ("a", "d", 2.0),
        ("b", "d", 2.0),
        ("d", "b", 1.0),
    ];

    #[test]
    fn singleton_teams_reduce_to_plain_bt() {
        let mut games = GamesDataset::new();
        let mut pairs = PairwiseDataset::new();
        for &(w, l, x) in SINGLETON_ROWS {
            games.push_pair(w, l, x).unwrap();
            pairs.push(w, l, x);
        }

        let reference = BradleyTerryMM {
            iterations: 50_000,
            tolerance: 1e-13,
            ..BradleyTerryMM::default()
        }
        .fit(&pairs)
        .unwrap();
        let ref_s: std::collections::HashMap<_, _> = reference.scores().collect();

        for aggregate in [TeamAggregate::Additive, TeamAggregate::Product] {
            let m = TeamBradleyTerry {
                aggregate,
                iterations: 50_000,
                tolerance: 1e-13,
                ..TeamBradleyTerry::default()
            }
            .fit(&games)
            .unwrap();
            let s = scores_of(&m);
            // Both normalize Σπ = 1, so per-pair strength ratios must agree.
            for (x, y) in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")] {
                let got = s[x] / s[y];
                let want = ref_s[x] / ref_s[y];
                assert!(
                    (got - want).abs() < 1e-6,
                    "{aggregate:?} {x}/{y}: {got} vs plain BT {want}"
                );
            }
        }
    }

    /// Strengths a=4, b=2, c=1, d=1 under the additive model: the three 2v2
    /// partitions of {a,b,c,d} pushed at the exact model win rates pin the
    /// MLE to the generating strengths (the partition-sum equations are
    /// linear with the unique solution (4,2,1,1)/8).
    #[test]
    fn additive_recovery_from_exact_rates() {
        let mut d = GamesDataset::new();
        // {a,b}=6 vs {c,d}=2; {a,c}=5 vs {b,d}=3; {a,d}=5 vs {b,c}=3.
        for (s1, s2, w1, w2) in [
            (["a", "b"], ["c", "d"], 6.0, 2.0),
            (["a", "c"], ["b", "d"], 5.0, 3.0),
            (["a", "d"], ["b", "c"], 5.0, 3.0),
        ] {
            d.push_game(&s1, &s2, GameOutcome::Side1Win(1.0), w1)
                .unwrap();
            d.push_game(&s1, &s2, GameOutcome::Side2Win(1.0), w2)
                .unwrap();
        }

        let m = TeamBradleyTerry {
            iterations: 100_000,
            tolerance: 1e-13,
            ..TeamBradleyTerry::default()
        }
        .fit(&d)
        .unwrap();
        let s = scores_of(&m);

        assert!(s["a"] > s["b"] && s["b"] > s["c"], "{s:?}");
        assert!(
            (s["c"] - s["d"]).abs() < 1e-9,
            "c and d are symmetric: {s:?}"
        );
        for (name, want) in [("a", 0.5), ("b", 0.25), ("c", 0.125), ("d", 0.125)] {
            assert!(
                (s[name] - want).abs() < 1e-4,
                "{name}: {} vs exact {want}",
                s[name]
            );
        }
    }

    /// Adding a tie (half-win) between a strong and a weak team pulls their
    /// strengths together compared to adding another strong win.
    #[test]
    fn half_win_ties_pull_teams_together() {
        let base = |extra: GameOutcome| {
            let mut d = GamesDataset::new();
            d.push_game(
                &["a1", "a2"],
                &["b1", "b2"],
                GameOutcome::Side1Win(1.0),
                3.0,
            )
            .unwrap();
            d.push_game(
                &["a1", "a2"],
                &["b1", "b2"],
                GameOutcome::Side2Win(1.0),
                1.0,
            )
            .unwrap();
            d.push_game(&["a1", "a2"], &["b1", "b2"], extra, 1.0)
                .unwrap();
            d
        };

        let algo = TeamBradleyTerry {
            ties: TiePolicy::HalfWin,
            ..TeamBradleyTerry::default()
        };
        let team_ratio = |m: &TeamBtModel| {
            let s = scores_of(m);
            (s["a1"] + s["a2"]) / (s["b1"] + s["b2"])
        };

        let with_win = algo.fit(&base(GameOutcome::Side1Win(1.0))).unwrap();
        let with_tie = algo.fit(&base(GameOutcome::Tie)).unwrap();
        assert!(
            team_ratio(&with_tie) < team_ratio(&with_win),
            "tie {} should pull teams closer than a win {}",
            team_ratio(&with_tie),
            team_ratio(&with_win)
        );
    }

    #[test]
    fn tie_under_error_policy_is_invalid_input() {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 1.0).unwrap();
        let err = TeamBradleyTerry::default().fit(&d).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(m) if m.contains("tie")));
    }

    #[test]
    fn all_wins_player_is_a_numeric_error() {
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        d.push_pair("b", "a", 1.0).unwrap();
        d.push_game(&["champ", "a"], &["b"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        let err = TeamBradleyTerry::default().fit(&d).unwrap_err();
        assert!(matches!(err, Error::Numeric(m) if m.contains("champ")));
    }

    #[test]
    fn empty_dataset_is_an_error() {
        assert!(matches!(
            TeamBradleyTerry::default().fit(&GamesDataset::new()),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c", "d"], GameOutcome::Side1Win(1.0), 2.0)
            .unwrap();
        d.push_game(&["c", "d"], &["a", "b"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        d.push_game(&["a", "c"], &["b", "d"], GameOutcome::Tie, 1.0)
            .unwrap();

        let m = TeamBradleyTerry {
            ties: TiePolicy::HalfWin,
            ..TeamBradleyTerry::default()
        }
        .fit(&d)
        .unwrap();

        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = TeamBtModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}
