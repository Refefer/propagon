//! Input parsing. Tournament files use the v2 games format
//! (`side1<TAB>side2<TAB>threshold[<TAB>count]`); the other shapes are
//! whitespace-separated rows. Blank lines mark batch boundaries (honored
//! with `--groups-are-separate`).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use propagon::{
    AnnotatedPairsDataset, ContextualRewardsDataset, Error, GameOutcome, GamesDataset,
    GraphDataset, MatchupsDataset, RankingsDataset, Result, RewardsDataset, TrajectoriesDataset,
};

fn rows(path: &Path) -> Result<impl Iterator<Item = std::io::Result<String>>> {
    let f = File::open(path)
        .map_err(|e| Error::InvalidInput(format!("cannot open {}: {e}", path.display())))?;
    Ok(BufReader::new(f).lines())
}

fn parse_row(line: &str, lineno: usize) -> Result<Option<(&str, &str, f32)>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let mut it = line.split_whitespace();
    let a = it.next();
    let b = it.next();
    let (Some(a), Some(b)) = (a, b) else {
        return Err(Error::parse(
            lineno,
            format!("expected at least two fields: {line:?}"),
        ));
    };
    let w = match it.next() {
        None => 1.0,
        Some(t) => t
            .parse::<f32>()
            .map_err(|e| Error::parse(lineno, format!("bad weight {t:?}: {e}")))?,
    };
    Ok(Some((a, b, w)))
}

/// Reads a tournament games file: `side1 <TAB> side2 <TAB> threshold
/// [<TAB> count]` per line — rosters space-separated within a side, signed
/// threshold (`> 0`: side 1 wins by that margin, `< 0`: side 2 wins,
/// `= 0`: tie), optional repeat count (default 1). `periods`: honor
/// blank-line batch boundaries; otherwise everything is one period.
pub fn read_games(path: &Path, periods: bool) -> Result<GamesDataset> {
    let mut ds = GamesDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if periods {
                ds.new_period();
            }
            continue;
        }

        let fields: Vec<&str> = trimmed.split('\t').map(str::trim).collect();
        if !(3..=4).contains(&fields.len()) {
            return Err(Error::parse(
                lineno + 1,
                format!(
                    "expected 'side1<TAB>side2<TAB>threshold[<TAB>count]', \
                     found {} tab-separated fields: {trimmed:?}",
                    fields.len()
                ),
            ));
        }
        let side1: Vec<&str> = fields[0].split_whitespace().collect();
        let side2: Vec<&str> = fields[1].split_whitespace().collect();
        let threshold: f32 = fields[2]
            .parse()
            .map_err(|e| Error::parse(lineno + 1, format!("bad threshold {:?}: {e}", fields[2])))?;
        if !threshold.is_finite() {
            return Err(Error::parse(
                lineno + 1,
                format!("threshold must be finite, got {threshold}"),
            ));
        }
        let count: f32 = match fields.get(3) {
            None => 1.0,
            Some(c) => c
                .parse()
                .map_err(|e| Error::parse(lineno + 1, format!("bad count {c:?}: {e}")))?,
        };

        let outcome = if threshold > 0.0 {
            GameOutcome::Side1Win(threshold)
        } else if threshold < 0.0 {
            GameOutcome::Side2Win(-threshold)
        } else {
            GameOutcome::Tie
        };
        ds.push_game(&side1, &side2, outcome, count)
            .map_err(|e| Error::parse(lineno + 1, e.to_string()))?;
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}

/// Reads a graph edge file. `swap`: store each row `a b` as the edge
/// `b -> a` ("b endorses a") — v1's match-file orientation for `page-rank`.
pub fn read_graph(path: &Path, swap: bool) -> Result<GraphDataset> {
    let mut g = GraphDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        if let Some((a, b, x)) = parse_row(&line, lineno + 1)? {
            if swap {
                g.push(b, a, x);
            } else {
                g.push(a, b, x);
            }
        }
    }
    if g.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(g)
}

/// Reads a ballots file: one ranking per line, items whitespace-separated,
/// best first. Blank lines are skipped.
pub fn read_rankings(path: &Path) -> Result<RankingsDataset> {
    let mut ds = RankingsDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        ds.push_ranking(line.split_whitespace())
            .map_err(|e| Error::parse(lineno + 1, format!("{e}: {line:?}")))?;
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}

/// Reads annotator-tagged votes: `annotator winner loser [weight]` rows.
pub fn read_annotated(path: &Path) -> Result<AnnotatedPairsDataset> {
    let mut ds = AnnotatedPairsDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut it = line.split_whitespace();
        let (Some(annotator), Some(winner), Some(loser)) = (it.next(), it.next(), it.next()) else {
            return Err(Error::parse(
                lineno + 1,
                format!("expected 'annotator winner loser [weight]': {line:?}"),
            ));
        };
        let x = match it.next() {
            None => 1.0,
            Some(t) => t
                .parse::<f32>()
                .map_err(|e| Error::parse(lineno + 1, format!("bad weight {t:?}: {e}")))?,
        };
        ds.push(annotator, winner, loser, x);
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}

/// Reads a matchups file: one match per line, teams separated by `|`
/// (best first), `=` joining teams tied at the same rank, players
/// whitespace-separated within a team. Ranks are derived competition-style.
pub fn read_matchups(path: &Path) -> Result<MatchupsDataset> {
    let mut ds = MatchupsDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut teams: Vec<Vec<&str>> = Vec::new();
        let mut ranks: Vec<u32> = Vec::new();
        let mut next_rank = 1u32;

        for segment in line.split('|') {
            let tied: Vec<&str> = segment.split('=').collect();
            let here = next_rank;

            for team in &tied {
                teams.push(team.split_whitespace().collect());
                ranks.push(here);
            }
            next_rank += tied.len() as u32;
        }

        let refs: Vec<&[&str]> = teams.iter().map(Vec::as_slice).collect();
        ds.push_match(&refs, &ranks)
            .map_err(|e| Error::parse(lineno + 1, format!("{e}: {line:?}")))?;
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}

/// Reads `(arm, reward)` rows for bandits: `arm reward` per line.
pub fn read_rewards(path: &Path) -> Result<RewardsDataset> {
    let mut ds = RewardsDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(arm), Some(reward)) = (it.next(), it.next()) else {
            return Err(Error::parse(
                lineno + 1,
                format!("expected 'arm reward': {line:?}"),
            ));
        };
        let r: f32 = reward
            .parse()
            .map_err(|e| Error::parse(lineno + 1, format!("bad reward {reward:?}: {e}")))?;
        ds.push(arm, r);
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}

/// Reads teleport seeds for personalized PageRank: `name [weight]` per
/// line, weight defaulting to 1.
pub fn read_seeds(path: &Path) -> Result<Vec<(String, f64)>> {
    let mut seeds = Vec::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(name) = it.next() else {
            continue;
        };
        let weight = match it.next() {
            None => 1.0,
            Some(t) => t
                .parse::<f64>()
                .map_err(|e| Error::parse(lineno + 1, format!("bad seed weight {t:?}: {e}")))?,
        };
        seeds.push((name.to_string(), weight));
    }
    if seeds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(seeds)
}

/// Reads an entity feature table for covariate models: `entity x1 x2 ... xd`
/// per line, all rows sharing one dimensionality (validated by the fitter).
pub fn read_features(path: &Path) -> Result<Vec<(String, Vec<f64>)>> {
    let mut features = Vec::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(name) = it.next() else {
            continue;
        };
        let xs: Vec<f64> = it
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|e| Error::parse(lineno + 1, format!("bad feature {v:?}: {e}")))
            })
            .collect::<Result<_>>()?;
        if xs.is_empty() {
            return Err(Error::parse(
                lineno + 1,
                format!("entity {name:?} has no feature values"),
            ));
        }
        features.push((name.to_string(), xs));
    }
    if features.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(features)
}

/// Reads contextual bandit rows: `arm reward x1 x2 ... xd` per line, the
/// dimensionality fixed by the first row.
pub fn read_contextual(path: &Path) -> Result<ContextualRewardsDataset> {
    let mut ds = ContextualRewardsDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(arm), Some(reward)) = (it.next(), it.next()) else {
            return Err(Error::parse(
                lineno + 1,
                format!("expected 'arm reward x1 ... xd': {line:?}"),
            ));
        };
        let r: f32 = reward
            .parse()
            .map_err(|e| Error::parse(lineno + 1, format!("bad reward {reward:?}: {e}")))?;
        let x: Vec<f64> = it
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|e| Error::parse(lineno + 1, format!("bad feature {v:?}: {e}")))
            })
            .collect::<Result<_>>()?;
        ds.push(arm, r, &x)
            .map_err(|e| Error::parse(lineno + 1, e.to_string()))?;
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}

/// Reads reward-bearing episodes: `state reward` per line, a blank line
/// ends the current episode.
pub fn read_trajectories(path: &Path) -> Result<TrajectoriesDataset> {
    let mut ds = TrajectoriesDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            ds.end_episode();
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(state), Some(reward)) = (it.next(), it.next()) else {
            return Err(Error::parse(
                lineno + 1,
                format!("expected 'state reward': {line:?}"),
            ));
        };
        let r: f32 = reward
            .parse()
            .map_err(|e| Error::parse(lineno + 1, format!("bad reward {reward:?}: {e}")))?;
        ds.push_step(state, r)
            .map_err(|e| Error::parse(lineno + 1, e.to_string()))?;
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(ds)
}
