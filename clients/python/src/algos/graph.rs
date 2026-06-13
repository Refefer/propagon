//! Algorithms over [`GraphDataset`]: PageRank, HITS, BiRank, degree, harmonic,
//! Katz, k-core, and LeaderRank centralities (all batch).

use pyo3::prelude::*;

use propagon::algos::{
    BiRank as RustBiRank, Degree as RustDegree, Harmonic as RustHarmonic, Hits as RustHits,
    KCore as RustKCore, Katz as RustKatz, LeaderRank as RustLeaderRank, PageRank as RustPageRank,
};

use crate::datasets::GraphDataset;
use crate::enums::{SourceBudget, Teleport, unit_enum};

model_class!(
    PageRankModel,
    "PageRankModel",
    propagon::algos::PageRankModel
);
custom_batch!(
    PageRank,
    "PageRank",
    RustPageRank,
    PageRankModel,
    GraphDataset,
    {
        /// Configure PageRank. `sink` is one of "reverse", "all", "uniform",
        /// "none"; `teleport` is a `Teleport` (uniform or seeded) for
        /// personalization.
        #[new]
        #[pyo3(signature = (*, damping=None, iterations=None, sink=None, teleport=None))]
        fn new(
            damping: Option<f64>,
            iterations: Option<usize>,
            sink: Option<String>,
            teleport: Option<PyRef<'_, Teleport>>,
        ) -> PyResult<Self> {
            let mut p = RustPageRank::default();
            if let Some(v) = damping {
                p.damping = v;
            }
            if let Some(v) = iterations {
                p.iterations = v;
            }
            if let Some(s) = sink {
                p.sink = unit_enum(&s, "sink", "reverse, all, uniform, none")?;
            }
            if let Some(t) = teleport {
                p.teleport = t.inner.clone();
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(HitsModel, "HitsModel", propagon::algos::HitsModel);
scalar_batch!(Hits, "Hits", RustHits, HitsModel, GraphDataset, {
    iterations: usize,
    tolerance: f64,
});

model_class!(BiRankModel, "BiRankModel", propagon::algos::BiRankModel);
scalar_batch!(BiRank, "BiRank", RustBiRank, BiRankModel, GraphDataset, {
    iterations: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
});

model_class!(DegreeModel, "DegreeModel", propagon::algos::DegreeModel);
custom_batch!(Degree, "Degree", RustDegree, DegreeModel, GraphDataset, {
    /// Configure degree centrality. `direction` is "in", "out", or "total".
    #[new]
    #[pyo3(signature = (*, direction=None))]
    fn new(direction: Option<String>) -> PyResult<Self> {
        let mut p = RustDegree::default();
        if let Some(s) = direction {
            p.direction = unit_enum(&s, "direction", "in, out, total")?;
        }
        Ok(Self { inner: p })
    }
});

model_class!(
    HarmonicModel,
    "HarmonicModel",
    propagon::algos::HarmonicModel
);
custom_batch!(
    Harmonic,
    "Harmonic",
    RustHarmonic,
    HarmonicModel,
    GraphDataset,
    {
        /// Configure harmonic centrality. `direction` is "in"/"out"/"total";
        /// `cost` is "unit" or "weight"; `sources` is a `SourceBudget`.
        #[new]
        #[pyo3(signature = (*, direction=None, cost=None, sources=None))]
        fn new(
            direction: Option<String>,
            cost: Option<String>,
            sources: Option<PyRef<'_, SourceBudget>>,
        ) -> PyResult<Self> {
            let mut p = RustHarmonic::default();
            if let Some(s) = direction {
                p.direction = unit_enum(&s, "direction", "in, out, total")?;
            }
            if let Some(s) = cost {
                p.cost = unit_enum(&s, "cost", "unit, weight")?;
            }
            if let Some(b) = sources {
                p.sources = b.inner;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(KatzModel, "KatzModel", propagon::algos::KatzModel);
scalar_batch!(Katz, "Katz", RustKatz, KatzModel, GraphDataset, {
    alpha: f64,
    iterations: usize,
    tolerance: f64,
});

model_class!(KCoreModel, "KCoreModel", propagon::algos::KCoreModel);
nofield_batch!(KCore, "KCore", RustKCore, KCoreModel, GraphDataset);

model_class!(
    LeaderRankModel,
    "LeaderRankModel",
    propagon::algos::LeaderRankModel
);
scalar_batch!(LeaderRank, "LeaderRank", RustLeaderRank, LeaderRankModel, GraphDataset, {
    iterations: usize,
    tolerance: f64,
});
