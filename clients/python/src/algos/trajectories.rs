//! Algorithms over [`TrajectoriesDataset`]: value estimation and behavior
//! cloning (batch), and temporal-difference learning (online).

use pyo3::prelude::*;

use propagon::algos::{
    BehaviorCloning as RustBehaviorCloning, McValue as RustMcValue, TdValue as RustTdValue,
    ValueCompare as RustValueCompare,
};

use crate::datasets::TrajectoriesDataset;
use crate::enums::{Granularity, PairwiseTests, Winsorize, unit_enum};

model_class!(McValueModel, "McValueModel", propagon::algos::McValueModel);
custom_batch!(
    McValue,
    "McValue",
    RustMcValue,
    McValueModel,
    TrajectoriesDataset,
    {
        /// Configure Monte-Carlo value estimation. `visit` is "first" or "every";
        /// `aggregate` is "mean" or "median"; `winsorize` is a `Winsorize`.
        #[new]
        #[pyo3(signature = (*, gamma=None, visit=None, aggregate=None, winsorize=None, min_observations=None))]
        fn new(
            gamma: Option<f64>,
            visit: Option<String>,
            aggregate: Option<String>,
            winsorize: Option<PyRef<'_, Winsorize>>,
            min_observations: Option<usize>,
        ) -> PyResult<Self> {
            let mut p = RustMcValue::default();
            if let Some(v) = gamma {
                p.gamma = v;
            }
            if let Some(s) = visit {
                p.visit = unit_enum(&s, "visit", "first, every")?;
            }
            if let Some(s) = aggregate {
                p.aggregate = unit_enum(&s, "aggregate", "mean, median")?;
            }
            if let Some(w) = winsorize {
                p.winsorize = w.inner;
            }
            if let Some(v) = min_observations {
                p.min_observations = v;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(BcModel, "BehaviorCloningModel", propagon::algos::BcModel);
custom_batch!(
    BehaviorCloning,
    "BehaviorCloning",
    RustBehaviorCloning,
    BcModel,
    TrajectoriesDataset,
    {
        /// Configure behavior cloning. `granularity` is a `Granularity`.
        #[new]
        #[pyo3(signature = (*, granularity=None, smoothing=None))]
        fn new(granularity: Option<PyRef<'_, Granularity>>, smoothing: Option<f64>) -> Self {
            let mut p = RustBehaviorCloning::default();
            if let Some(g) = granularity {
                p.granularity = g.inner;
            }
            if let Some(v) = smoothing {
                p.smoothing = v;
            }
            Self { inner: p }
        }
    }
);

model_class!(
    ValueCompareModel,
    "ValueCompareModel",
    propagon::algos::ValueCompareModel
);
custom_batch!(
    ValueCompare,
    "ValueCompare",
    RustValueCompare,
    ValueCompareModel,
    TrajectoriesDataset,
    {
        /// Configure value comparison. `visit` is "first"/"every"; `method` is
        /// "bootstrap" or "bayesian-bootstrap"; `pairwise` is a `PairwiseTests`.
        #[new]
        #[pyo3(signature = (*, gamma=None, visit=None, replicates=None, method=None, credible=None, pairwise=None, min_observations=None, seed=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            gamma: Option<f64>,
            visit: Option<String>,
            replicates: Option<usize>,
            method: Option<String>,
            credible: Option<f64>,
            pairwise: Option<PyRef<'_, PairwiseTests>>,
            min_observations: Option<usize>,
            seed: Option<u64>,
        ) -> PyResult<Self> {
            let mut p = RustValueCompare::default();
            if let Some(v) = gamma {
                p.gamma = v;
            }
            if let Some(s) = visit {
                p.visit = unit_enum(&s, "visit", "first, every")?;
            }
            if let Some(v) = replicates {
                p.replicates = v;
            }
            if let Some(s) = method {
                p.method = unit_enum(&s, "method", "bootstrap, bayesian-bootstrap")?;
            }
            if let Some(v) = credible {
                p.credible = v;
            }
            if let Some(t) = pairwise {
                p.pairwise = t.inner;
            }
            if let Some(v) = min_observations {
                p.min_observations = v;
            }
            if let Some(v) = seed {
                p.seed = v;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(TdValueModel, "TdValueModel", propagon::algos::TdValueModel);
scalar_online!(TdValue, "TdValue", RustTdValue, TdValueModel, TrajectoriesDataset, {
    alpha: f64,
    gamma: f64,
    passes: usize,
    initial_value: f64,
});
