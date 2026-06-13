//! Algorithms over [`MatchupsDataset`]: Weng-Lin / OpenSkill (online).

use pyo3::prelude::*;

use propagon::algos::{Rating as RustRating, WengLin as RustWengLin};

use crate::datasets::MatchupsDataset;
use crate::enums::unit_enum;

/// One entity's Weng-Lin skill estimate.
#[pyclass(name = "Rating", module = "propagon._propagon", frozen)]
pub struct Rating {
    pub(crate) inner: RustRating,
}

#[pymethods]
impl Rating {
    /// Estimated skill mean.
    #[getter]
    fn mu(&self) -> f64 {
        self.inner.mu
    }

    /// Skill uncertainty (standard deviation).
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    fn __repr__(&self) -> String {
        format!("Rating(mu={}, sigma={})", self.inner.mu, self.inner.sigma)
    }
}

model_class!(WengLinModel, "WengLinModel", propagon::algos::WengLinModel, extras {
    /// Per-entity skill estimates (`weng-lin`-specific): mean and uncertainty.
    fn ratings(&self) -> Vec<(String, Rating)> {
        self.inner
            .ratings()
            .map(|(n, r)| (n.to_string(), Rating { inner: r }))
            .collect()
    }
});
custom_online!(
    WengLin,
    "WengLin",
    RustWengLin,
    WengLinModel,
    MatchupsDataset,
    {
        /// Configure Weng-Lin. `variant` is "bradley-terry-full" or
        /// "thurstone-mosteller-full"; `gamma` is "sigma-over-c" or "one-over-k".
        #[new]
        #[pyo3(signature = (*, variant=None, mu=None, sigma=None, beta=None, kappa=None, epsilon=None, tau=None, gamma=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            variant: Option<String>,
            mu: Option<f64>,
            sigma: Option<f64>,
            beta: Option<f64>,
            kappa: Option<f64>,
            epsilon: Option<f64>,
            tau: Option<f64>,
            gamma: Option<String>,
        ) -> PyResult<Self> {
            let mut p = RustWengLin::default();
            if let Some(s) = variant {
                p.variant = unit_enum(
                    &s,
                    "variant",
                    "bradley-terry-full, thurstone-mosteller-full",
                )?;
            }
            if let Some(v) = mu {
                p.mu = v;
            }
            if let Some(v) = sigma {
                p.sigma = v;
            }
            if let Some(v) = beta {
                p.beta = v;
            }
            if let Some(v) = kappa {
                p.kappa = v;
            }
            if let Some(v) = epsilon {
                p.epsilon = v;
            }
            if let Some(v) = tau {
                p.tau = v;
            }
            if let Some(s) = gamma {
                p.gamma = unit_enum(&s, "gamma", "sigma-over-c, one-over-k")?;
            }
            Ok(Self { inner: p })
        }
    }
);
