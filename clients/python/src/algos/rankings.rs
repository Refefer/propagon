//! Algorithms over [`RankingsDataset`]: rank aggregation (batch).

use pyo3::prelude::*;

use propagon::algos::{
    Footrule as RustFootrule, Mallows as RustMallows, Mc4 as RustMc4,
    PlackettLuce as RustPlackettLuce,
};

use crate::datasets::RankingsDataset;
use crate::enums::KemenyPasses;

model_class!(
    PlackettLuceModel,
    "PlackettLuceModel",
    propagon::algos::PlackettLuceModel
);
scalar_batch!(PlackettLuce, "PlackettLuce", RustPlackettLuce, PlackettLuceModel, RankingsDataset, {
    iterations: usize,
    tolerance: f64,
});

model_class!(
    FootruleModel,
    "FootruleModel",
    propagon::algos::FootruleModel
);
nofield_batch!(
    Footrule,
    "Footrule",
    RustFootrule,
    FootruleModel,
    RankingsDataset
);

model_class!(MallowsModel, "MallowsModel", propagon::algos::MallowsModel);
custom_batch!(
    Mallows,
    "Mallows",
    RustMallows,
    MallowsModel,
    RankingsDataset,
    {
        /// Configure Mallows. `passes` is a `KemenyPasses` (auto or fixed).
        #[new]
        #[pyo3(signature = (*, passes=None, seed=None))]
        fn new(passes: Option<PyRef<'_, KemenyPasses>>, seed: Option<u64>) -> Self {
            let mut p = RustMallows::default();
            if let Some(k) = passes {
                p.passes = k.inner;
            }
            if let Some(v) = seed {
                p.seed = v;
            }
            Self { inner: p }
        }
    }
);

model_class!(Mc4Model, "Mc4Model", propagon::algos::Mc4Model);
scalar_batch!(Mc4, "Mc4", RustMc4, Mc4Model, RankingsDataset, {
    damping: f64,
    iterations: usize,
    tolerance: f64,
});
