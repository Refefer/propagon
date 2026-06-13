//! Algorithms over [`GamesDataset`]: Elo, Glicko-2, margin-of-victory Elo,
//! multidimensional Elo (online), and the team/tie-aware batch Bradley-Terry
//! variants.

use pyo3::prelude::*;

use propagon::algos::{
    Elo as RustElo, GeneralizedBt as RustGeneralizedBt, Glicko2 as RustGlicko2, MElo as RustMElo,
    MovElo as RustMovElo, PlayerState as RustPlayerState, TeamBradleyTerry as RustTeamBradleyTerry,
};

use crate::datasets::GamesDataset;
use crate::enums::unit_enum;

/// One entity's Glicko-2 rating state on the display (Elo-like) scale.
#[pyclass(name = "PlayerState", module = "propagon._propagon", frozen)]
pub struct PlayerState {
    pub(crate) inner: RustPlayerState,
}

#[pymethods]
impl PlayerState {
    /// Rating on the display scale.
    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    /// Rating deviation (uncertainty) on the display scale.
    #[getter]
    fn rd(&self) -> f64 {
        self.inner.rd
    }

    /// Volatility.
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    /// 95%-ish interval `(r - 2*rd, r + 2*rd)`.
    fn bounds(&self) -> (f64, f64) {
        self.inner.bounds()
    }

    fn __repr__(&self) -> String {
        format!(
            "PlayerState(r={}, rd={}, sigma={})",
            self.inner.r, self.inner.rd, self.inner.sigma
        )
    }
}

model_class!(EloModel, "EloModel", propagon::algos::EloModel);
scalar_online!(Elo, "Elo", RustElo, EloModel, GamesDataset, {
    k: f64,
    initial_rating: f64,
    scale: f64,
});

// Glicko-2 carries a per-entity extra accessor (`players`) spliced into the
// model's single `#[pymethods]`.
model_class!(Glicko2Model, "Glicko2Model", propagon::algos::Glicko2Model, extras {
    /// Per-entity rating states (`glicko2`-specific).
    fn players(&self) -> Vec<(String, PlayerState)> {
        self.inner
            .players()
            .map(|(n, p)| (n.to_string(), PlayerState { inner: *p }))
            .collect()
    }
});
scalar_online!(Glicko2, "Glicko2", RustGlicko2, Glicko2Model, GamesDataset, {
    tau: f64,
    rating: f64,
    rd: f64,
    sigma: f64,
});

model_class!(MovEloModel, "MovEloModel", propagon::algos::MovEloModel);
scalar_online!(MovElo, "MovElo", RustMovElo, MovEloModel, GamesDataset, {
    k: f64,
    initial_rating: f64,
    scale: f64,
    mov_exponent: f64,
});

model_class!(MEloModel, "MEloModel", propagon::algos::MEloModel);
scalar_online!(MElo, "MElo", RustMElo, MEloModel, GamesDataset, {
    k: usize,
    lr_rating: f64,
    lr_vector: f64,
    initial_rating: f64,
    init_scale: f64,
    seed: u64,
});

model_class!(
    GeneralizedBtModel,
    "GeneralizedBtModel",
    propagon::algos::GeneralizedBtModel
);
custom_batch!(
    GeneralizedBt,
    "GeneralizedBt",
    RustGeneralizedBt,
    GeneralizedBtModel,
    GamesDataset,
    {
        /// Configure tie-aware Bradley-Terry. `ties` is one of "none",
        /// "davidson", "rao-kupper"; `home` is "none" or "estimate".
        #[new]
        #[pyo3(signature = (*, ties=None, home=None, iterations=None, tolerance=None))]
        fn new(
            ties: Option<String>,
            home: Option<String>,
            iterations: Option<usize>,
            tolerance: Option<f64>,
        ) -> PyResult<Self> {
            let mut p = RustGeneralizedBt::default();
            if let Some(s) = ties {
                p.ties = unit_enum(&s, "ties", "none, davidson, rao-kupper")?;
            }
            if let Some(s) = home {
                p.home = unit_enum(&s, "home", "none, estimate")?;
            }
            if let Some(v) = iterations {
                p.iterations = v;
            }
            if let Some(v) = tolerance {
                p.tolerance = v;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(
    TeamBtModel,
    "TeamBradleyTerryModel",
    propagon::algos::TeamBtModel
);
custom_batch!(
    TeamBradleyTerry,
    "TeamBradleyTerry",
    RustTeamBradleyTerry,
    TeamBtModel,
    GamesDataset,
    {
        /// Configure team Bradley-Terry. `aggregate` is "additive" or "product";
        /// `ties` is "error", "discard", or "half-win".
        #[new]
        #[pyo3(signature = (*, aggregate=None, iterations=None, tolerance=None, ties=None))]
        fn new(
            aggregate: Option<String>,
            iterations: Option<usize>,
            tolerance: Option<f64>,
            ties: Option<String>,
        ) -> PyResult<Self> {
            let mut p = RustTeamBradleyTerry::default();
            if let Some(s) = aggregate {
                p.aggregate = unit_enum(&s, "aggregate", "additive, product")?;
            }
            if let Some(v) = iterations {
                p.iterations = v;
            }
            if let Some(v) = tolerance {
                p.tolerance = v;
            }
            if let Some(s) = ties {
                p.ties = unit_enum(&s, "ties", "error, discard, half-win")?;
            }
            Ok(Self { inner: p })
        }
    }
);
