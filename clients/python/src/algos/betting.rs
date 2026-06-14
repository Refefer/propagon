//! Betting & portfolio algorithms (§14): odds de-vigging over
//! [`OddsDataset`], opinion pools over [`ForecastDataset`], and the LMSR
//! prediction market (online) over [`MarketDataset`]. Kelly sizing and the
//! calibration diagnostics are free functions (see `functions.rs`).

use pyo3::prelude::*;

use propagon::algos::{
    Lmsr as RustLmsr, OddsDevig as RustOddsDevig, OpinionPool as RustOpinionPool,
};

use crate::datasets::{ForecastDataset, MarketDataset, OddsDataset};
use crate::enums::unit_enum;

// --- de-vigging -------------------------------------------------------------

model_class!(OddsDevigModel, "OddsDevigModel", propagon::algos::OddsDevigModel, extras {
    /// Number of events.
    fn n_events(&self) -> usize {
        self.inner.n_events()
    }

    /// The booksum (overround + 1) of event `e`.
    fn booksum(&self, e: usize) -> Option<f64> {
        self.inner.booksum(e)
    }

    /// The estimated insider share `z` of event `e` (Shin only; 0 otherwise).
    fn insider_share(&self, e: usize) -> Option<f64> {
        self.inner.insider_share(e)
    }
});

custom_batch!(
    OddsDevig,
    "OddsDevig",
    RustOddsDevig,
    OddsDevigModel,
    OddsDataset,
    {
        /// Configure de-vigging. `method` is one of "multiplicative",
        /// "additive", "power", "shin"; `additive_clamp` is "error" or
        /// "clamp-renormalize".
        #[new]
        #[pyo3(signature = (*, method=None, additive_clamp=None, tolerance=None))]
        fn new(
            method: Option<String>,
            additive_clamp: Option<String>,
            tolerance: Option<f64>,
        ) -> PyResult<Self> {
            let mut p = RustOddsDevig::default();
            if let Some(s) = method {
                p.method = unit_enum(&s, "method", "multiplicative, additive, power, shin")?;
            }
            if let Some(s) = additive_clamp {
                p.additive_clamp = unit_enum(&s, "additive_clamp", "error, clamp-renormalize")?;
            }
            if let Some(v) = tolerance {
                p.tolerance = v;
            }
            Ok(Self { inner: p })
        }
    }
);

// --- opinion pools ----------------------------------------------------------

model_class!(
    OpinionPoolModel,
    "OpinionPoolModel",
    propagon::algos::OpinionPoolModel
);

custom_batch!(
    OpinionPool,
    "OpinionPool",
    RustOpinionPool,
    OpinionPoolModel,
    ForecastDataset,
    {
        /// Configure the pool. `kind` is "linear" or "logarithmic"; `missing`
        /// is "error", "skip", or "uniform"; `extremize` >= 1 sharpens the
        /// consensus (1 = none); `eps_floor` optionally clamps source
        /// probabilities before the log pool.
        #[new]
        #[pyo3(signature = (*, kind=None, extremize=None, missing=None, eps_floor=None))]
        fn new(
            kind: Option<String>,
            extremize: Option<f64>,
            missing: Option<String>,
            eps_floor: Option<f64>,
        ) -> PyResult<Self> {
            let mut p = RustOpinionPool::default();
            if let Some(s) = kind {
                p.kind = unit_enum(&s, "kind", "linear, logarithmic")?;
            }
            if let Some(v) = extremize {
                p.extremize = v;
            }
            if let Some(s) = missing {
                p.missing = unit_enum(&s, "missing", "error, skip, uniform")?;
            }
            if let Some(v) = eps_floor {
                p.eps_floor = Some(v);
            }
            Ok(Self { inner: p })
        }
    }
);

// --- LMSR prediction market (online) ----------------------------------------

model_class!(LmsrModel, "LmsrModel", propagon::algos::LmsrModel, extras {
    /// The price of `outcome`, or `None` if it is not in the market.
    fn price(&self, outcome: &str) -> Option<f64> {
        self.inner.price(outcome)
    }

    /// The market maker's current cost `C(q)`.
    fn cost(&self) -> f64 {
        self.inner.cost()
    }

    /// The payment for a prospective trade of `shares` on `outcome`.
    fn trade_cost(&self, outcome: &str, shares: f64) -> Option<f64> {
        self.inner.trade_cost(outcome, shares)
    }
});

scalar_online!(Lmsr, "Lmsr", RustLmsr, LmsrModel, MarketDataset, {
    b: f64,
});
