//! String-ID interning: external string identifiers ↔ dense `u32` indices.
//!
//! Every dataset owns an [`Interner`]; algorithms work exclusively on the
//! dense indices, and models resolve names back at output time. Numeric v1
//! edge files keep working unchanged — `"7"` is simply interned as a token.
//!
//! The dense index space is `0..len()`, append-only. The `u32` width caps the
//! entity count at ~4.29B, which exceeds what the most constrained target
//! (wasm32's 4 GB memory ceiling) can hold anyway.

use std::collections::HashMap;

/// Append-only bidirectional map between string names and dense `u32` ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Interner {
    map: HashMap<Box<str>, u32>,
    names: Vec<Box<str>>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the id for `name`, inserting it if unseen.
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.map.get(name) {
            return id;
        }

        let id = match u32::try_from(self.names.len()) {
            Ok(id) => id,
            Err(_) => {
                // 2^32 distinct names is hundreds of GB of strings — beyond
                // any practical dataset. Refuse to grow rather than panic.
                log::error!("interner at u32 capacity; mapping {name:?} to the last id");
                return u32::MAX;
            }
        };

        let boxed: Box<str> = name.into();
        self.names.push(boxed.clone());
        self.map.insert(boxed, id);
        id
    }

    /// Resolves an id that is known to have come from this interner (model
    /// and dataset internals uphold that invariant at construction). The
    /// out-of-range case is unreachable by construction; it degrades to a
    /// placeholder instead of panicking.
    pub(crate) fn resolve(&self, id: u32) -> &str {
        self.names
            .get(id as usize)
            .map(AsRef::as_ref)
            .unwrap_or("<unresolved>")
    }

    /// Looks up an existing id without inserting.
    pub fn get(&self, name: &str) -> Option<u32> {
        self.map.get(name).copied()
    }

    /// Resolves an id back to its name.
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(AsRef::as_ref)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// All names in id order (the id of `names()[i]` is `i`).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(AsRef::as_ref)
    }

    /// Rebuilds an interner from names in id order (state-file loading).
    /// Fails on duplicates, which would corrupt the bidirectional mapping.
    pub fn from_names<I, S>(names: I) -> crate::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut out = Self::new();
        for name in names {
            let name = name.as_ref();
            let before = out.names.len();
            out.intern(name);
            if out.names.len() == before {
                return Err(crate::Error::State(format!(
                    "duplicate name in vocab: {name:?}"
                )));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_round_trip() {
        let mut i = Interner::new();
        let a = i.intern("ARI");
        let b = i.intern("COL");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(i.intern("ARI"), a);
        assert_eq!(i.len(), 2);
        assert_eq!(i.name(a), Some("ARI"));
        assert_eq!(i.get("COL"), Some(b));
        assert_eq!(i.get("NYM"), None);
        assert_eq!(i.name(9), None);
    }

    #[test]
    fn from_names_preserves_order_and_rejects_dupes() {
        let i = Interner::from_names(["x", "y", "z"]).unwrap();
        assert_eq!(i.get("y"), Some(1));
        assert!(Interner::from_names(["x", "x"]).is_err());
    }
}
