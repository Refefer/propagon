//! Algorithms over [`AnnotatedPairsDataset`]: crowd-aware Bradley-Terry.

use pyo3::prelude::*;

use propagon::algos::CrowdBt as RustCrowdBt;

use crate::datasets::AnnotatedPairsDataset;

model_class!(CrowdBtModel, "CrowdBtModel", propagon::algos::CrowdBtModel);
// `lambda` is a Python keyword, so it cannot be a scalar-macro field name; the
// constructor exposes it as `lambda_` and maps it onto the Rust `lambda` field.
custom_batch!(
    CrowdBt,
    "CrowdBt",
    RustCrowdBt,
    CrowdBtModel,
    AnnotatedPairsDataset,
    {
        /// Configure crowd Bradley-Terry. `lambda_` is the annotator-reliability
        /// regularizer (named with a trailing underscore: `lambda` is a Python
        /// keyword).
        #[new]
        #[pyo3(signature = (*, lambda_=None, eta_prior_alpha=None, eta_prior_beta=None, iterations=None, tolerance=None, inner_sweeps=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            lambda_: Option<f64>,
            eta_prior_alpha: Option<f64>,
            eta_prior_beta: Option<f64>,
            iterations: Option<usize>,
            tolerance: Option<f64>,
            inner_sweeps: Option<usize>,
        ) -> Self {
            let mut p = RustCrowdBt::default();
            if let Some(v) = lambda_ {
                p.lambda = v;
            }
            if let Some(v) = eta_prior_alpha {
                p.eta_prior_alpha = v;
            }
            if let Some(v) = eta_prior_beta {
                p.eta_prior_beta = v;
            }
            if let Some(v) = iterations {
                p.iterations = v;
            }
            if let Some(v) = tolerance {
                p.tolerance = v;
            }
            if let Some(v) = inner_sweeps {
                p.inner_sweeps = v;
            }
            Self { inner: p }
        }
    }
);
