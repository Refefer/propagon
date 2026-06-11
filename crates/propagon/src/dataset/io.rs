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

use super::{GraphDataset, PairwiseDataset, RankingsDataset, RewardsDataset};

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
