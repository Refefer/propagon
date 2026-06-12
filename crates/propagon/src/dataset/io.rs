//! Dataset persistence: the FR-4 header-line JSONL format.
//!
//! ```jsonl
//! {"propagon":1,"kind":"dataset","algorithm":"pairwise","params":{"periods":[2]},"entities":3}
//! {"vocab":["ARI","COL","NYM"]}
//! {"w":[0,0],"l":[1,2],"x":[1.0,1.0]}
//! {"w":[1],"l":[2],"x":[1.0]}
//! ```
//!
//! Vocab and row chunks are capped at [`CHUNK`] entries per line so files
//! stream at any size. Dataset edge columns reference vocab indices (strings
//! are not repeated per row); appending after load is the normal flow.

use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state::{Header, SCHEMA_VERSION};

use super::{
    ContextualRewardsDataset, GameOutcome, GamesDataset, GraphDataset, PairwiseDataset,
    RankingsDataset, RewardsDataset, TrajectoriesDataset,
};

/// Maximum entries per vocab / row-chunk line.
const CHUNK: usize = 65_536;

#[derive(Serialize, Deserialize)]
struct VocabLine {
    vocab: Vec<String>,
}

fn write_header<W: Write>(
    w: &mut W,
    schema: &str,
    params: serde_json::Value,
    entities: usize,
) -> Result<()> {
    let header = Header {
        propagon: SCHEMA_VERSION,
        kind: "dataset".to_string(),
        algorithm: schema.to_string(),
        params,
        entities,
    };
    serde_json::to_writer(&mut *w, &header)?;
    w.write_all(b"\n")?;
    Ok(())
}

fn write_vocab<W: Write>(w: &mut W, interner: &Interner) -> Result<()> {
    let names: Vec<&str> = interner.names().collect();
    for chunk in names.chunks(CHUNK) {
        let line = VocabLine {
            vocab: chunk.iter().map(|s| s.to_string()).collect(),
        };
        serde_json::to_writer(&mut *w, &line)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

struct DatasetReader<R: BufRead> {
    lines: std::io::Lines<R>,
    header: Header,
    interner: Interner,
    /// First non-vocab line, already consumed from the stream.
    pending: Option<String>,
}

fn read_dataset_prefix<R: BufRead>(r: R, schema: &str) -> Result<DatasetReader<R>> {
    let mut lines = r.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| Error::State("empty dataset file".into()))??;
    let header: Header = serde_json::from_str(&header_line)
        .map_err(|e| Error::State(format!("malformed header: {e}")))?;
    if header.propagon > SCHEMA_VERSION {
        return Err(Error::Version {
            found: header.propagon,
            supported: SCHEMA_VERSION,
        });
    }
    if header.kind != "dataset" {
        return Err(Error::State(format!(
            "expected a dataset file, found kind {:?}",
            header.kind
        )));
    }
    if header.algorithm != schema {
        return Err(Error::AlgorithmMismatch {
            expected: schema.to_string(),
            found: header.algorithm,
        });
    }

    let mut interner = Interner::new();
    let mut pending = None;
    for line in lines.by_ref() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<VocabLine>(&line) {
            for name in &v.vocab {
                interner.intern(name);
            }
            if interner.len() >= header.entities {
                break;
            }
        } else {
            pending = Some(line);
            break;
        }
    }
    if interner.len() != header.entities {
        return Err(Error::State(format!(
            "vocab holds {} names but header declares {}",
            interner.len(),
            header.entities
        )));
    }
    Ok(DatasetReader {
        lines,
        header,
        interner,
        pending,
    })
}

impl<R: BufRead> DatasetReader<R> {
    fn rows<L: serde::de::DeserializeOwned>(
        mut self,
        mut apply: impl FnMut(L) -> Result<()>,
    ) -> Result<(Header, Interner)> {
        if let Some(line) = self.pending.take() {
            let chunk: L = serde_json::from_str(&line)
                .map_err(|e| Error::State(format!("bad row chunk: {e}")))?;
            apply(chunk)?;
        }
        for line in self.lines.by_ref() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let chunk: L = serde_json::from_str(&line)
                .map_err(|e| Error::State(format!("bad row chunk: {e}")))?;
            apply(chunk)?;
        }
        Ok((self.header, self.interner))
    }
}

// ---------------------------------------------------------------- pairwise

#[derive(Serialize, Deserialize, Default)]
struct PairwiseMeta {
    periods: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct PairwiseChunk {
    w: Vec<u32>,
    l: Vec<u32>,
    x: Vec<f32>,
}

impl PairwiseDataset {
    /// Serializes the dataset (vocab + columnar row chunks + period marks).
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        let meta = PairwiseMeta {
            periods: self.period_starts_for_io(),
        };
        write_header(
            &mut w,
            "pairwise",
            serde_json::to_value(meta)?,
            self.n_entities(),
        )?;
        write_vocab(&mut w, self.interner())?;
        let rows: Vec<(u32, u32, f32)> = self.rows().collect();
        for chunk in rows.chunks(CHUNK) {
            let line = PairwiseChunk {
                w: chunk.iter().map(|r| r.0).collect(),
                l: chunk.iter().map(|r| r.1).collect(),
                x: chunk.iter().map(|r| r.2).collect(),
            };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Loads a dataset written by [`PairwiseDataset::save_jsonl`]. Appending
    /// more rows afterwards is the normal incremental flow.
    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "pairwise")?;
        let meta: PairwiseMeta = serde_json::from_value(reader.header.params.clone())
            .map_err(|e| Error::State(format!("bad pairwise meta: {e}")))?;

        let mut out = PairwiseDataset::new();
        let interner = reader.interner.clone();
        for name in interner.names() {
            out.intern(name);
        }
        let mut boundaries = meta.periods.into_iter().peekable();
        let mut row_idx = 0usize;
        let (_, _) = reader.rows(|chunk: PairwiseChunk| {
            if chunk.w.len() != chunk.l.len() || chunk.w.len() != chunk.x.len() {
                return Err(Error::State("pairwise chunk column mismatch".into()));
            }
            for i in 0..chunk.w.len() {
                while boundaries.peek() == Some(&row_idx) {
                    boundaries.next();
                    out.new_period();
                }
                out.push_ids(chunk.w[i], chunk.l[i], chunk.x[i])?;
                row_idx += 1;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------- rewards

#[derive(Serialize, Deserialize)]
struct RewardsChunk {
    a: Vec<u32>,
    r: Vec<f32>,
}

impl RewardsDataset {
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        write_header(&mut w, "rewards", serde_json::Value::Null, self.n_arms())?;
        write_vocab(&mut w, self.interner())?;
        let rows: Vec<(u32, f32)> = self.rows().collect();
        for chunk in rows.chunks(CHUNK) {
            let line = RewardsChunk {
                a: chunk.iter().map(|r| r.0).collect(),
                r: chunk.iter().map(|r| r.1).collect(),
            };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "rewards")?;
        let mut out = RewardsDataset::new();
        for name in reader.interner.clone().names() {
            out.intern(name);
        }
        reader.rows(|chunk: RewardsChunk| {
            if chunk.a.len() != chunk.r.len() {
                return Err(Error::State("rewards chunk column mismatch".into()));
            }
            for i in 0..chunk.a.len() {
                out.push_ids(chunk.a[i], chunk.r[i])?;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------- contextual

#[derive(Serialize, Deserialize)]
struct ContextualMeta {
    /// Feature dimensionality; absent (`null`) only for an empty dataset.
    dim: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct ContextualChunk {
    a: Vec<u32>,
    r: Vec<f32>,
    /// Flat features, `a.len() · dim` values.
    x: Vec<f64>,
}

impl ContextualRewardsDataset {
    /// Serializes the dataset (vocab + columnar row chunks with flat,
    /// stride-`dim` feature columns).
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        let meta = ContextualMeta { dim: self.dim() };
        write_header(
            &mut w,
            "contextual-rewards",
            serde_json::to_value(meta)?,
            self.n_arms(),
        )?;
        write_vocab(&mut w, self.interner())?;
        let rows: Vec<(u32, f32, &[f64])> = self.rows().collect();
        for chunk in rows.chunks(CHUNK) {
            let mut line = ContextualChunk {
                a: Vec::with_capacity(chunk.len()),
                r: Vec::with_capacity(chunk.len()),
                x: Vec::new(),
            };
            for (a, r, x) in chunk {
                line.a.push(*a);
                line.r.push(*r);
                line.x.extend_from_slice(x);
            }
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Loads a dataset written by `save_jsonl`. Chunk lengths, arm ids,
    /// dimensionality, and feature finiteness are all re-validated, so
    /// corrupted files surface as typed errors.
    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "contextual-rewards")?;
        let meta: ContextualMeta = serde_json::from_value(reader.header.params.clone())
            .map_err(|e| Error::State(format!("bad contextual meta: {e}")))?;
        if meta.dim == Some(0) {
            return Err(Error::State("contextual meta declares dim 0".into()));
        }

        let mut out = ContextualRewardsDataset::new();
        let interner = reader.interner.clone();
        for name in interner.names() {
            out.intern(name);
        }

        reader.rows(|chunk: ContextualChunk| {
            let dim = match meta.dim {
                Some(d) => d,
                None => {
                    return Err(Error::State(
                        "contextual rows present but meta declares no dim".into(),
                    ));
                }
            };
            if chunk.a.len() != chunk.r.len() || chunk.x.len() != chunk.a.len() * dim {
                return Err(Error::State("contextual chunk column mismatch".into()));
            }
            for (i, x) in chunk.x.chunks(dim).enumerate() {
                out.push_ids(chunk.a[i], chunk.r[i], x)
                    .map_err(|e| Error::State(format!("bad contextual row: {e}")))?;
            }
            Ok(())
        })?;

        // A meta dim with no rows cannot come from `save_jsonl` (dim is fixed
        // by the first pushed row); rejecting it keeps "dim known ⟺ rows
        // exist" true in memory and round trips byte-identical.
        if out.dim() != meta.dim {
            return Err(Error::State(format!(
                "meta declares dim {:?} but the rows imply {:?}",
                meta.dim,
                out.dim()
            )));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- graph

#[derive(Serialize, Deserialize)]
struct GraphChunk {
    s: Vec<u32>,
    d: Vec<u32>,
    x: Vec<f32>,
}

impl GraphDataset {
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        write_header(&mut w, "graph", serde_json::Value::Null, self.n_nodes())?;
        write_vocab(&mut w, self.interner())?;
        let rows: Vec<(u32, u32, f32)> = self.view().edges().collect();
        for chunk in rows.chunks(CHUNK) {
            let line = GraphChunk {
                s: chunk.iter().map(|r| r.0).collect(),
                d: chunk.iter().map(|r| r.1).collect(),
                x: chunk.iter().map(|r| r.2).collect(),
            };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "graph")?;
        let mut out = GraphDataset::new();
        for name in reader.interner.clone().names() {
            out.intern(name);
        }
        reader.rows(|chunk: GraphChunk| {
            if chunk.s.len() != chunk.d.len() || chunk.s.len() != chunk.x.len() {
                return Err(Error::State("graph chunk column mismatch".into()));
            }
            for i in 0..chunk.s.len() {
                out.push_ids(chunk.s[i], chunk.d[i], chunk.x[i])?;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------- rankings

#[derive(Serialize, Deserialize)]
struct RankingsChunk {
    /// Each entry is one complete ranking (vocab indices, best first).
    rk: Vec<Vec<u32>>,
}

impl RankingsDataset {
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        write_header(
            &mut w,
            "rankings",
            serde_json::Value::Null,
            self.n_entities(),
        )?;
        write_vocab(&mut w, self.interner())?;
        let all: Vec<Vec<u32>> = self.rankings().map(|r| r.to_vec()).collect();
        for chunk in all.chunks(1024) {
            let line = RankingsChunk { rk: chunk.to_vec() };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "rankings")?;
        let interner = reader.interner.clone();
        let mut out = RankingsDataset::new();
        reader.rows(|chunk: RankingsChunk| {
            for ranking in &chunk.rk {
                let names: Vec<&str> = ranking
                    .iter()
                    .map(|&id| {
                        interner
                            .name(id)
                            .ok_or_else(|| Error::State(format!("ranking id {id} out of vocab")))
                    })
                    .collect::<Result<_>>()?;
                out.push_ranking(names)?;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

// ---------------------------------------------------------------- trajectories

#[derive(Serialize, Deserialize, Default)]
struct TrajectoriesMeta {
    /// Interior episode boundaries (step indices where episodes after the
    /// first begin; may end with a boundary at the total step count when the
    /// last episode was explicitly ended).
    episodes: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct TrajectoriesChunk {
    s: Vec<u32>,
    r: Vec<f32>,
}

impl TrajectoriesDataset {
    /// Serializes the dataset (vocab + columnar step chunks + episode
    /// boundary marks, mirroring how pairwise persists periods).
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        let meta = TrajectoriesMeta {
            episodes: self.episode_starts_for_io(),
        };
        write_header(
            &mut w,
            "trajectories",
            serde_json::to_value(meta)?,
            self.n_entities(),
        )?;
        write_vocab(&mut w, self.interner())?;
        let rows: Vec<(u32, f32)> = self.steps().collect();
        for chunk in rows.chunks(CHUNK) {
            let line = TrajectoriesChunk {
                s: chunk.iter().map(|r| r.0).collect(),
                r: chunk.iter().map(|r| r.1).collect(),
            };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Loads a dataset written by [`TrajectoriesDataset::save_jsonl`].
    /// Episode boundaries are re-injected between steps; state ids, reward
    /// finiteness, and column lengths are re-validated, so corrupted files
    /// surface as typed errors. A trailing boundary (last episode explicitly
    /// ended) is re-applied after the final step to keep round trips
    /// byte-identical.
    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "trajectories")?;
        let meta: TrajectoriesMeta = serde_json::from_value(reader.header.params.clone())
            .map_err(|e| Error::State(format!("bad trajectories meta: {e}")))?;

        let mut out = TrajectoriesDataset::new();
        let interner = reader.interner.clone();
        for name in interner.names() {
            out.intern(name);
        }

        let mut boundaries = meta.episodes.into_iter().peekable();
        let mut step_idx = 0usize;
        let (_, _) = reader.rows(|chunk: TrajectoriesChunk| {
            if chunk.s.len() != chunk.r.len() {
                return Err(Error::State("trajectories chunk column mismatch".into()));
            }
            for i in 0..chunk.s.len() {
                while boundaries.peek() == Some(&step_idx) {
                    boundaries.next();
                    out.end_episode();
                }
                out.push_step_ids(chunk.s[i], chunk.r[i])
                    .map_err(|e| Error::State(format!("bad trajectory step: {e}")))?;
                step_idx += 1;
            }
            Ok(())
        })?;

        while boundaries.peek() == Some(&step_idx) {
            boundaries.next();
            out.end_episode();
        }

        if let Some(b) = boundaries.next() {
            return Err(Error::State(format!(
                "episode boundary {b} does not match any of the {step_idx} steps present"
            )));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_round_trip_with_periods_and_append() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.new_period();
        d.push("b", "c", 2.5);

        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        let mut d2 = PairwiseDataset::load_jsonl(buf.as_slice()).unwrap();

        assert_eq!(d2.n_periods(), 2);
        assert_eq!(d.rows().collect::<Vec<_>>(), d2.rows().collect::<Vec<_>>());

        // byte-identical re-save
        let mut buf2 = Vec::new();
        d2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);

        // append flow: same ids keep resolving
        d2.new_period();
        d2.push("a", "c", 1.0);
        assert_eq!(d2.n_periods(), 3);
        assert_eq!(d2.interner().get("a"), Some(0));
    }

    #[test]
    fn rewards_graph_rankings_round_trip() {
        let mut r = RewardsDataset::new();
        r.push("A", 1.0);
        r.push("B", 0.25);
        let mut buf = Vec::new();
        r.save_jsonl(&mut buf).unwrap();
        let r2 = RewardsDataset::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(r.rows().collect::<Vec<_>>(), r2.rows().collect::<Vec<_>>());

        let mut g = GraphDataset::new();
        g.push("x", "y", 1.5);
        g.push("y", "z", 1.0);
        let mut buf = Vec::new();
        g.save_jsonl(&mut buf).unwrap();
        let g2 = GraphDataset::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(
            g.view().edges().collect::<Vec<_>>(),
            g2.view().edges().collect::<Vec<_>>()
        );

        let mut k = RankingsDataset::new();
        k.push_ranking(["a", "b", "c"]).unwrap();
        k.push_ranking(["c", "a"]).unwrap();
        let mut buf = Vec::new();
        k.save_jsonl(&mut buf).unwrap();
        let k2 = RankingsDataset::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(k.ranking(1), k2.ranking(1));
    }

    #[test]
    fn contextual_round_trip_and_validation() {
        let mut d = ContextualRewardsDataset::new();
        d.push("A", 1.0, &[0.5, -1.25]).unwrap();
        d.push("B", 0.0, &[2.0, 3.5]).unwrap();
        d.push("A", 0.5, &[0.0, 0.125]).unwrap();

        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        let d2 = ContextualRewardsDataset::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(d2.dim(), Some(2));
        assert_eq!(
            d.rows()
                .map(|(a, r, x)| (a, r, x.to_vec()))
                .collect::<Vec<_>>(),
            d2.rows()
                .map(|(a, r, x)| (a, r, x.to_vec()))
                .collect::<Vec<_>>()
        );

        // byte-identical re-save
        let mut buf2 = Vec::new();
        d2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);

        // empty dataset round trips too (dim stays unknown)
        let empty = ContextualRewardsDataset::new();
        let mut buf = Vec::new();
        empty.save_jsonl(&mut buf).unwrap();
        let e2 = ContextualRewardsDataset::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(e2.dim(), None);
        assert!(e2.is_empty());

        // corrupted chunks are rejected
        let good = {
            let mut buf = Vec::new();
            d.save_jsonl(&mut buf).unwrap();
            String::from_utf8(buf).unwrap()
        };
        for (bad_chunk, why) in [
            (
                r#"{"a":[0],"r":[1.0],"x":[0.5]}"#,
                "flat x shorter than dim",
            ),
            (r#"{"a":[0],"r":[1.0,2.0],"x":[0.5,1.0]}"#, "ragged columns"),
            (
                r#"{"a":[9],"r":[1.0],"x":[0.5,1.0]}"#,
                "arm id out of vocab",
            ),
            (
                r#"{"a":[0],"r":[1.0],"x":[0.5,null]}"#,
                "non-finite feature",
            ),
        ] {
            let mut lines: Vec<&str> = good.lines().collect();
            let last = lines.len() - 1;
            lines[last] = bad_chunk;
            let file = lines.join("\n");
            assert!(
                ContextualRewardsDataset::load_jsonl(file.as_bytes()).is_err(),
                "{why}"
            );
        }
    }

    #[test]
    fn wrong_schema_is_rejected() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        assert!(matches!(
            GraphDataset::load_jsonl(buf.as_slice()),
            Err(Error::AlgorithmMismatch { .. })
        ));
    }

    #[test]
    fn games_round_trip_with_periods_and_append() {
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c", "d"], GameOutcome::Side1Win(3.5), 1.0)
            .unwrap();
        d.new_period();
        d.push_game(&["c"], &["a"], GameOutcome::Tie, 2.0).unwrap();
        d.push_pair("d", "b", 1.0).unwrap();

        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        let mut d2 = GamesDataset::load_jsonl(buf.as_slice()).unwrap();

        assert_eq!(d2.n_periods(), 2);
        assert_eq!(d2.len(), 3);
        for (g1, g2) in d.games().zip(d2.games()) {
            assert_eq!(g1.side1, g2.side1);
            assert_eq!(g1.side2, g2.side2);
            assert_eq!(g1.outcome, g2.outcome);
            assert_eq!(g1.weight, g2.weight);
        }

        // byte-identical re-save
        let mut buf2 = Vec::new();
        d2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);

        // append flow: same ids keep resolving
        d2.new_period();
        d2.push_pair("a", "d", 1.0).unwrap();
        assert_eq!(d2.n_periods(), 3);
        assert_eq!(d2.interner().get("a"), Some(0));
    }

    #[test]
    fn trajectories_round_trip_with_episodes_and_append() {
        let mut d = TrajectoriesDataset::new();
        d.push_step("a", 1.0).unwrap();
        d.push_step("b", -0.5).unwrap();
        d.end_episode();
        d.push_step("b", 2.0).unwrap();
        d.end_episode();

        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        let mut d2 = TrajectoriesDataset::load_jsonl(buf.as_slice()).unwrap();

        assert_eq!(d2.n_episodes(), 2);
        assert_eq!(d2.episode(0), d.episode(0));
        assert_eq!(d2.episode(1), d.episode(1));

        // byte-identical re-save (including the trailing boundary)
        let mut buf2 = Vec::new();
        d2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);

        // append flow: same ids keep resolving
        d2.push_step("a", 0.25).unwrap();
        d2.end_episode();
        assert_eq!(d2.n_episodes(), 3);
        assert_eq!(d2.interner().get("a"), Some(0));

        // an open trailing episode (no final end_episode) also round trips
        let mut open = TrajectoriesDataset::new();
        open.push_step("x", 1.0).unwrap();
        open.end_episode();
        open.push_step("y", 0.0).unwrap();
        let mut buf = Vec::new();
        open.save_jsonl(&mut buf).unwrap();
        let open2 = TrajectoriesDataset::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(open2.n_episodes(), 2);
        let mut buf2 = Vec::new();
        open2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn trajectories_corrupted_files_are_rejected() {
        let mut d = TrajectoriesDataset::new();
        d.push_step("a", 1.0).unwrap();
        d.push_step("b", 0.0).unwrap();
        d.end_episode();
        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        let good = String::from_utf8(buf).unwrap();

        for (bad_chunk, why) in [
            (r#"{"s":[9],"r":[1.0]}"#, "state id out of vocab"),
            (r#"{"s":[0,1],"r":[1.0]}"#, "ragged columns"),
            (r#"{"s":[0],"r":[null]}"#, "non-finite reward"),
        ] {
            let mut lines: Vec<&str> = good.lines().collect();
            let last = lines.len() - 1;
            lines[last] = bad_chunk;
            let file = lines.join("\n");
            assert!(
                matches!(
                    TrajectoriesDataset::load_jsonl(file.as_bytes()),
                    Err(Error::State(_))
                ),
                "{why}"
            );
        }

        // an episode boundary beyond the data is rejected
        let broken = good.replacen(r#""episodes":[2]"#, r#""episodes":[9]"#, 1);
        assert_ne!(broken, good, "fixture must contain the boundary meta");
        assert!(matches!(
            TrajectoriesDataset::load_jsonl(broken.as_bytes()),
            Err(Error::State(_))
        ));
    }

    #[test]
    fn games_corrupted_chunks_are_rejected() {
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        let mut buf = Vec::new();
        d.save_jsonl(&mut buf).unwrap();
        let good = String::from_utf8(buf).unwrap();

        for (bad_chunk, why) in [
            (
                r#"{"s1":[[0]],"s2":[[1]],"o":[2],"m":[1.0],"x":[1.0]}"#,
                "outcome out of range",
            ),
            (
                r#"{"s1":[[0]],"s2":[[1]],"o":[1],"m":[0.0],"x":[1.0]}"#,
                "zero margin on a win",
            ),
            (
                r#"{"s1":[[0]],"s2":[[1]],"o":[0],"m":[1.0],"x":[1.0]}"#,
                "nonzero margin on a tie",
            ),
            (
                r#"{"s1":[[0]],"s2":[[1]],"o":[1],"m":[1.0],"x":[0.0]}"#,
                "zero weight",
            ),
            (
                r#"{"s1":[[0]],"s2":[[1]],"o":[1],"m":[1.0],"x":[1.0,2.0]}"#,
                "ragged columns",
            ),
            (
                r#"{"s1":[[0]],"s2":[[7]],"o":[1],"m":[1.0],"x":[1.0]}"#,
                "id out of vocab",
            ),
            (
                r#"{"s1":[[0]],"s2":[[0]],"o":[1],"m":[1.0],"x":[1.0]}"#,
                "duplicate player",
            ),
            (
                r#"{"s1":[[]],"s2":[[1]],"o":[1],"m":[1.0],"x":[1.0]}"#,
                "empty side",
            ),
        ] {
            let mut lines: Vec<&str> = good.lines().collect();
            let last = lines.len() - 1;
            lines[last] = bad_chunk;
            let file = lines.join("\n");
            assert!(
                matches!(
                    GamesDataset::load_jsonl(file.as_bytes()),
                    Err(Error::State(_))
                ),
                "{why}"
            );
        }
    }
}

// ---------------------------------------------------------------- annotated

#[derive(Serialize, Deserialize)]
struct AnnotatedMeta {
    annotators: usize,
}

#[derive(Serialize, Deserialize)]
struct AnnotatorVocabLine {
    annotators: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct AnnotatedChunk {
    a: Vec<u32>,
    w: Vec<u32>,
    l: Vec<u32>,
    x: Vec<f32>,
}

impl super::AnnotatedPairsDataset {
    /// Serializes votes: entity vocab, then annotator vocab, then chunks.
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        let meta = AnnotatedMeta {
            annotators: self.n_annotators(),
        };
        write_header(
            &mut w,
            "annotated-pairs",
            serde_json::to_value(meta)?,
            self.n_entities(),
        )?;
        write_vocab(&mut w, self.entities())?;

        let names: Vec<&str> = self.annotators().names().collect();
        for chunk in names.chunks(CHUNK) {
            let line = AnnotatorVocabLine {
                annotators: chunk.iter().map(|s| s.to_string()).collect(),
            };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }

        let rows: Vec<(u32, u32, u32, f32)> = self.rows().collect();
        for chunk in rows.chunks(CHUNK) {
            let line = AnnotatedChunk {
                a: chunk.iter().map(|r| r.0).collect(),
                w: chunk.iter().map(|r| r.1).collect(),
                l: chunk.iter().map(|r| r.2).collect(),
                x: chunk.iter().map(|r| r.3).collect(),
            };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Loads a dataset written by `save_jsonl`. The annotator vocab is a
    /// second phase after the entity vocab, sized by the header meta.
    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let mut reader = read_dataset_prefix(r, "annotated-pairs")?;
        let meta: AnnotatedMeta = serde_json::from_value(reader.header.params.clone())
            .map_err(|e| Error::State(format!("bad annotated meta: {e}")))?;

        // Phase 2: annotator vocab (the prefix reader stashed the first
        // non-entity-vocab line as pending).
        let mut annotators = Interner::new();
        while annotators.len() < meta.annotators {
            let line = match reader.pending.take() {
                Some(line) => line,
                None => loop {
                    match reader.lines.next() {
                        Some(line) => {
                            let line = line?;
                            if !line.trim().is_empty() {
                                break line;
                            }
                        }
                        None => {
                            return Err(Error::State(format!(
                                "annotator vocab holds {} names but header declares {}",
                                annotators.len(),
                                meta.annotators
                            )));
                        }
                    }
                },
            };

            let v: AnnotatorVocabLine = serde_json::from_str(&line)
                .map_err(|e| Error::State(format!("bad annotator vocab: {e}")))?;
            for name in &v.annotators {
                annotators.intern(name);
            }
        }

        let n_entities = reader.interner.len() as u32;
        let n_annotators = annotators.len() as u32;
        let mut out = super::AnnotatedPairsDataset::new();

        let (_, entities) = reader.rows(|chunk: AnnotatedChunk| {
            if chunk.a.len() != chunk.w.len()
                || chunk.a.len() != chunk.l.len()
                || chunk.a.len() != chunk.x.len()
            {
                return Err(Error::State("annotated chunk column mismatch".into()));
            }
            for i in 0..chunk.a.len() {
                if chunk.a[i] >= n_annotators
                    || chunk.w[i] >= n_entities
                    || chunk.l[i] >= n_entities
                {
                    return Err(Error::State("annotated chunk id out of range".into()));
                }
                out.push_ids(chunk.a[i], chunk.w[i], chunk.l[i], chunk.x[i]);
            }
            Ok(())
        })?;

        out.set_interners(entities, annotators);
        Ok(out)
    }
}

// ---------------------------------------------------------------- matchups

#[derive(Clone, Serialize, Deserialize)]
struct MatchRec {
    t: Vec<Vec<u32>>,
    r: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct MatchupsChunk {
    mt: Vec<MatchRec>,
}

impl super::MatchupsDataset {
    /// Serializes matches: player vocab, then match chunks (team rosters as
    /// vocab indices plus competition ranks).
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        write_header(
            &mut w,
            "matchups",
            serde_json::Value::Object(Default::default()),
            self.n_entities(),
        )?;
        write_vocab(&mut w, self.interner())?;

        let matches: Vec<MatchRec> = self
            .matches()
            .map(|teams| {
                let mut rec = MatchRec {
                    t: Vec::new(),
                    r: Vec::new(),
                };
                for (rank, roster) in teams {
                    rec.t.push(roster.to_vec());
                    rec.r.push(rank);
                }
                rec
            })
            .collect();

        for chunk in matches.chunks(CHUNK / 16) {
            let line = MatchupsChunk { mt: chunk.to_vec() };
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Loads a dataset written by `save_jsonl`.
    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "matchups")?;
        let mut out = super::MatchupsDataset::new();
        let mut staged: Vec<(Vec<Vec<u32>>, Vec<u32>)> = Vec::new();

        let (_, interner) = reader.rows(|chunk: MatchupsChunk| {
            for rec in chunk.mt {
                staged.push((rec.t, rec.r));
            }
            Ok(())
        })?;

        out.set_interner(interner);
        for (teams, ranks) in staged {
            out.push_match_ids(&teams, &ranks)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- games

#[derive(Serialize, Deserialize, Default)]
struct GamesMeta {
    periods: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct GamesChunk {
    /// Side-1 rosters (vocab indices), one per game.
    s1: Vec<Vec<u32>>,
    /// Side-2 rosters.
    s2: Vec<Vec<u32>>,
    /// Outcome sign per game: 1 = side 1 won, -1 = side 2 won, 0 = tie.
    o: Vec<i8>,
    /// Win margin (> 0 for wins, exactly 0 for ties).
    m: Vec<f32>,
    /// Aggregation weight (repeat count).
    x: Vec<f32>,
}

impl GamesDataset {
    /// Serializes the dataset (vocab + per-game roster/outcome chunks +
    /// period marks).
    pub fn save_jsonl<W: Write>(&self, mut w: W) -> Result<()> {
        let meta = GamesMeta {
            periods: self.period_starts_for_io(),
        };
        write_header(
            &mut w,
            "games",
            serde_json::to_value(meta)?,
            self.n_entities(),
        )?;
        write_vocab(&mut w, self.interner())?;

        let games: Vec<_> = self.games().collect();
        for chunk in games.chunks(CHUNK / 16) {
            let mut line = GamesChunk {
                s1: Vec::with_capacity(chunk.len()),
                s2: Vec::with_capacity(chunk.len()),
                o: Vec::with_capacity(chunk.len()),
                m: Vec::with_capacity(chunk.len()),
                x: Vec::with_capacity(chunk.len()),
            };
            for view in chunk {
                line.s1.push(view.side1.to_vec());
                line.s2.push(view.side2.to_vec());
                let (o, m) = match view.outcome {
                    GameOutcome::Side1Win(m) => (1, m),
                    GameOutcome::Side2Win(m) => (-1, m),
                    GameOutcome::Tie => (0, 0.0),
                };
                line.o.push(o);
                line.m.push(m);
                line.x.push(view.weight);
            }
            serde_json::to_writer(&mut w, &line)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Loads a dataset written by [`GamesDataset::save_jsonl`]. Every chunk
    /// is re-validated (outcome signs, margin/weight constraints, roster
    /// invariants), so corrupted files surface as typed errors rather than
    /// illegal in-memory states.
    pub fn load_jsonl<R: BufRead>(r: R) -> Result<Self> {
        let reader = read_dataset_prefix(r, "games")?;
        let meta: GamesMeta = serde_json::from_value(reader.header.params.clone())
            .map_err(|e| Error::State(format!("bad games meta: {e}")))?;

        let mut out = GamesDataset::new();
        let interner = reader.interner.clone();
        let mut boundaries = meta.periods.into_iter().peekable();
        let mut game_idx = 0usize;
        let mut staged_interner = Some(interner);

        let (_, _) = reader.rows(|chunk: GamesChunk| {
            if let Some(int) = staged_interner.take() {
                out.set_interner(int);
            }
            let len = chunk.s1.len();
            if chunk.s2.len() != len
                || chunk.o.len() != len
                || chunk.m.len() != len
                || chunk.x.len() != len
            {
                return Err(Error::State("games chunk column mismatch".into()));
            }
            for i in 0..len {
                while boundaries.peek() == Some(&game_idx) {
                    boundaries.next();
                    out.new_period();
                }
                let margin_ok = chunk.m[i].is_finite() && chunk.m[i] > 0.0;
                let outcome = match chunk.o[i] {
                    1 if margin_ok => GameOutcome::Side1Win(chunk.m[i]),
                    -1 if margin_ok => GameOutcome::Side2Win(chunk.m[i]),
                    0 if chunk.m[i] == 0.0 => GameOutcome::Tie,
                    o => {
                        return Err(Error::State(format!(
                            "games chunk has outcome {o} with margin {}",
                            chunk.m[i]
                        )));
                    }
                };
                if !(chunk.x[i].is_finite() && chunk.x[i] > 0.0) {
                    return Err(Error::State(format!(
                        "games chunk has weight {}",
                        chunk.x[i]
                    )));
                }
                out.push_game_ids(&chunk.s1[i], &chunk.s2[i], outcome, chunk.x[i])?;
                game_idx += 1;
            }
            Ok(())
        })?;
        // An empty file (header + vocab, no chunks) still carries its vocab.
        if let Some(int) = staged_interner.take() {
            out.set_interner(int);
        }
        Ok(out)
    }
}
