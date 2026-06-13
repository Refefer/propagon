//! Algorithms over [`RewardsDataset`]: multi-armed bandits (online).

use pyo3::prelude::*;

use propagon::algos::{Bandit as RustBandit, SlidingWindowUcb as RustSlidingWindowUcb};

use crate::datasets::RewardsDataset;
use crate::enums::BanditPolicy;

model_class!(BanditModel, "BanditModel", propagon::algos::BanditModel);
custom_online!(Bandit, "Bandit", RustBandit, BanditModel, RewardsDataset, {
    /// Configure a bandit. `policy` is a `BanditPolicy`; `seed` drives the
    /// (reproducible) selection stream.
    #[new]
    #[pyo3(signature = (*, policy=None, seed=None))]
    fn new(policy: Option<PyRef<'_, BanditPolicy>>, seed: Option<u64>) -> Self {
        let mut p = RustBandit::default();
        if let Some(pol) = policy {
            p.policy = pol.inner;
        }
        if let Some(v) = seed {
            p.seed = v;
        }
        Self { inner: p }
    }
});

model_class!(
    SwUcbModel,
    "SlidingWindowUcbModel",
    propagon::algos::SwUcbModel
);
scalar_online!(SlidingWindowUcb, "SlidingWindowUcb", RustSlidingWindowUcb, SwUcbModel, RewardsDataset, {
    window: usize,
    exploration: f64,
});
