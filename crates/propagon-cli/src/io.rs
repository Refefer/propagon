//! Input parsing with v1 semantics: whitespace-separated `a b [weight]`
//! rows, blank lines as batch boundaries (only honored with
//! `--groups-are-separate`, exactly like v1's reader selection), and
//! iterative `--min-count` filtering.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use propagon::{
    AnnotatedPairsDataset, Error, GraphDataset, MatchupsDataset, PairwiseDataset, RankingsDataset,
    Result, RewardsDataset,
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

/// Reads a pairwise edge file. `periods`: honor blank-line batch boundaries
/// (v1 `--groups-are-separate`); otherwise everything is one period.
pub fn read_pairwise(path: &Path, periods: bool, min_count: usize) -> Result<PairwiseDataset> {
    let mut ds = PairwiseDataset::new();
    for (lineno, line) in rows(path)?.enumerate() {
        let line = line?;
        match parse_row(&line, lineno + 1)? {
            Some((w, l, x)) => ds.push(w, l, x),
            None => {
                if periods {
                    ds.new_period();
                }
            }
        }
    }
    if ds.is_empty() {
        return Err(Error::EmptyDataset);
    }
    Ok(if min_count > 1 {
        ds.filter_min_count(min_count)
    } else {
        ds
    })
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
