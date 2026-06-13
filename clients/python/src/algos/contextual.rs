//! Algorithms over [`ContextualRewardsDataset`]: linear contextual bandits.

use pyo3::prelude::*;

use propagon::algos::LinUcb as RustLinUcb;

use crate::datasets::ContextualRewardsDataset;

model_class!(LinUcbModel, "LinUcbModel", propagon::algos::LinUcbModel);
scalar_online!(LinUcb, "LinUcb", RustLinUcb, LinUcbModel, ContextualRewardsDataset, {
    alpha: f64,
    ridge: f64,
});
