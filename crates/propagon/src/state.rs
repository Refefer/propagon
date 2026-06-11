//! Header-line JSONL state files (the FR-4 format).
//!
//! Every state file starts with a self-describing header object:
//!
//! ```jsonl
//! {"propagon":1,"kind":"model","algorithm":"glicko2","params":{...},"entities":30}
//! {"id":"BOS","r":1593.6,"rd":41.2,"sigma":0.06}
//! ```
//!
//! Rules: the `propagon` schema-version integer is mandatory; readers
//! tolerate unknown fields (forward compatibility); model lines carry string
//! ids so files are self-contained and greppable; all model floats are `f64`,
//! and serde_json's shortest-roundtrip formatting makes save → load → save
//! byte-identical.

use std::io::{BufRead, Write};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};

/// The schema version this build writes and the maximum it reads.
pub const SCHEMA_VERSION: u32 = 1;

/// First line of every propagon state file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Header {
    /// Schema version (`SCHEMA_VERSION` on write).
    pub propagon: u32,
    /// `"model"` or `"dataset"`.
    pub kind: String,
    /// Algorithm tag (`"glicko2"`, `"btm-mm"`, …) or dataset schema name.
    pub algorithm: String,
    /// The exact params the state was produced with.
    pub params: serde_json::Value,
    /// Number of entity lines that follow.
    pub entities: usize,
}

/// Writes a complete model file: header plus one JSON line per entity.
pub fn save_model<W, P, L>(
    mut w: W,
    algorithm: &str,
    params: &P,
    lines: &[L],
) -> Result<()>
where
    W: Write,
    P: Serialize,
    L: Serialize,
{
    let header = Header {
        propagon: SCHEMA_VERSION,
        kind: "model".to_string(),
        algorithm: algorithm.to_string(),
        params: serde_json::to_value(params)?,
        entities: lines.len(),
    };
    serde_json::to_writer(&mut w, &header)?;
    w.write_all(b"\n")?;
    for line in lines {
        serde_json::to_writer(&mut w, line)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Reads a complete model file written by [`save_model`], validating the
/// schema version and algorithm tag and re-typing params and entity lines.
pub fn load_model<R, P, L>(r: R, expected_algorithm: &str) -> Result<(P, Vec<L>)>
where
    R: BufRead,
    P: DeserializeOwned,
    L: DeserializeOwned,
{
    let mut lines = r.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| Error::State("empty state file".into()))??;
    let header: Header = serde_json::from_str(&header_line)
        .map_err(|e| Error::State(format!("malformed header: {e}")))?;

    if header.propagon > SCHEMA_VERSION {
        return Err(Error::Version { found: header.propagon, supported: SCHEMA_VERSION });
    }
    if header.kind != "model" {
        return Err(Error::State(format!("expected a model file, found kind {:?}", header.kind)));
    }
    if header.algorithm != expected_algorithm {
        return Err(Error::AlgorithmMismatch {
            expected: expected_algorithm.to_string(),
            found: header.algorithm,
        });
    }

    let params: P = serde_json::from_value(header.params)
        .map_err(|e| Error::State(format!("incompatible params in state file: {e}")))?;

    let mut out = Vec::with_capacity(header.entities);
    for (i, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: L = serde_json::from_str(&line)
            .map_err(|e| Error::parse(i + 2, format!("bad entity line: {e}")))?;
        out.push(parsed);
    }
    if out.len() != header.entities {
        return Err(Error::State(format!(
            "header declares {} entities but file contains {}",
            header.entities,
            out.len()
        )));
    }
    Ok((params, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug, Default)]
    struct P {
        tau: f64,
    }
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct L {
        id: String,
        s: f64,
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let params = P { tau: 0.5 };
        let lines = vec![
            L { id: "a".into(), s: 1.25 },
            L { id: "b".into(), s: -0.3333333333333333 },
        ];
        let mut buf = Vec::new();
        save_model(&mut buf, "test", &params, &lines).unwrap();

        let (p2, l2): (P, Vec<L>) = load_model(buf.as_slice(), "test").unwrap();
        assert_eq!(p2, params);
        assert_eq!(l2, lines);

        let mut buf2 = Vec::new();
        save_model(&mut buf2, "test", &p2, &l2).unwrap();
        assert_eq!(buf, buf2, "save -> load -> save must be byte-identical");
    }

    #[test]
    fn wrong_algorithm_is_rejected() {
        let mut buf = Vec::new();
        save_model(&mut buf, "elo", &P::default(), &Vec::<L>::new()).unwrap();
        let err = load_model::<_, P, L>(buf.as_slice(), "glicko2").unwrap_err();
        assert!(matches!(err, Error::AlgorithmMismatch { .. }), "{err}");
    }

    #[test]
    fn future_version_is_rejected_and_unknown_fields_tolerated() {
        let file = format!(
            "{}\n",
            serde_json::json!({
                "propagon": SCHEMA_VERSION + 1, "kind": "model", "algorithm": "test",
                "params": {"tau": 0.5}, "entities": 0
            })
        );
        let err = load_model::<_, P, L>(file.as_bytes(), "test").unwrap_err();
        assert!(matches!(err, Error::Version { .. }), "{err}");

        // Unknown fields in header and lines are forward-compatible.
        let file = concat!(
            r#"{"propagon":1,"kind":"model","algorithm":"test","params":{"tau":0.5},"entities":1,"future":true}"#,
            "\n",
            r#"{"id":"a","s":1.0,"future_field":[1,2,3]}"#,
            "\n"
        );
        let (p, l): (P, Vec<L>) = load_model(file.as_bytes(), "test").unwrap();
        assert_eq!(p.tau, 0.5);
        assert_eq!(l.len(), 1);
    }
}
