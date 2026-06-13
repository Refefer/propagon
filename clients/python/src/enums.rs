//! Parameter enums that cross the boundary.
//!
//! Two kinds:
//!
//! - **Unit-only enums** (`Sink`, `Direction`, `TieModel`, …) are accepted as
//!   serde kebab-case *strings* at the algorithm constructors (e.g.
//!   `PageRank(sink="uniform")`) and converted with [`unit_enum`]. No wrapper
//!   class is needed; the allowed literals are documented in the type stubs.
//! - **Data-carrying enums** (`BanditPolicy`, `Teleport`, …) get a small
//!   `#[pyclass]` with factory classmethods that carry the payload, e.g.
//!   `BanditPolicy.epsilon_greedy(epsilon=0.1)`.

use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::de::DeserializeOwned;

use propagon::GameOutcome as RustGameOutcome;
use propagon::algos::{
    BanditPolicy as RustBanditPolicy, DuelingPolicy as RustDuelingPolicy,
    Granularity as RustGranularity, KemenyPasses as RustKemenyPasses,
    PairwiseTests as RustPairwiseTests, SourceBudget as RustSourceBudget, Teleport as RustTeleport,
    Winsorize as RustWinsorize,
};

use crate::convert::narrow_f32;
use crate::errors::InvalidInputError;

/// Converts a kebab-case string into a unit-only param enum via serde.
///
/// `allowed` is shown verbatim in the error message so callers see the valid
/// set without consulting the docs.
pub(crate) fn unit_enum<T: DeserializeOwned>(s: &str, field: &str, allowed: &str) -> PyResult<T> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|_| {
        InvalidInputError::new_err(format!("invalid {field} {s:?}; expected one of {allowed}"))
    })
}

/// Outcome of one game, from side 1's perspective.
///
/// Construct with the factory classmethods: `GameOutcome.side1_win(margin=2.0)`,
/// `GameOutcome.side2_win(margin=3.0)`, or `GameOutcome.tie()`. Win margins must
/// be finite and positive.
#[pyclass(name = "GameOutcome", module = "propagon._propagon", frozen)]
pub struct GameOutcome {
    pub(crate) inner: RustGameOutcome,
}

#[pymethods]
impl GameOutcome {
    /// Side 1 won by `margin` (default 1.0).
    #[classmethod]
    #[pyo3(signature = (margin = 1.0))]
    fn side1_win(_cls: &Bound<'_, PyType>, margin: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RustGameOutcome::Side1Win(narrow_f32(margin, "margin")?),
        })
    }

    /// Side 2 won by `margin` (default 1.0).
    #[classmethod]
    #[pyo3(signature = (margin = 1.0))]
    fn side2_win(_cls: &Bound<'_, PyType>, margin: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RustGameOutcome::Side2Win(narrow_f32(margin, "margin")?),
        })
    }

    /// The game was a draw.
    #[classmethod]
    fn tie(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustGameOutcome::Tie,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RustGameOutcome::Side1Win(m) => format!("GameOutcome.side1_win(margin={m})"),
            RustGameOutcome::Side2Win(m) => format!("GameOutcome.side2_win(margin={m})"),
            RustGameOutcome::Tie => "GameOutcome.tie()".to_string(),
        }
    }
}

/// Arm-selection policy for [`crate::algos::rewards`]'s `Bandit`.
#[pyclass(name = "BanditPolicy", module = "propagon._propagon", frozen)]
pub struct BanditPolicy {
    pub(crate) inner: RustBanditPolicy,
}

#[pymethods]
impl BanditPolicy {
    /// Always pull the current best arm.
    #[classmethod]
    fn greedy(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustBanditPolicy::Greedy,
        }
    }

    /// Pull a random arm with probability `epsilon`, else the best.
    #[classmethod]
    fn epsilon_greedy(_cls: &Bound<'_, PyType>, epsilon: f64) -> Self {
        Self {
            inner: RustBanditPolicy::EpsilonGreedy { epsilon },
        }
    }

    /// UCB1 with exploration constant `exploration`.
    #[classmethod]
    fn ucb1(_cls: &Bound<'_, PyType>, exploration: f64) -> Self {
        Self {
            inner: RustBanditPolicy::Ucb1 { exploration },
        }
    }

    /// Thompson sampling with a Beta prior.
    #[classmethod]
    fn thompson_beta(_cls: &Bound<'_, PyType>, prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            inner: RustBanditPolicy::ThompsonBeta {
                prior_alpha,
                prior_beta,
            },
        }
    }

    /// Thompson sampling with a Gaussian prior.
    #[classmethod]
    fn thompson_gaussian(_cls: &Bound<'_, PyType>, prior_mean: f64, prior_weight: f64) -> Self {
        Self {
            inner: RustBanditPolicy::ThompsonGaussian {
                prior_mean,
                prior_weight,
            },
        }
    }

    /// KL-UCB with constant `c`.
    #[classmethod]
    fn kl_ucb(_cls: &Bound<'_, PyType>, c: f64) -> Self {
        Self {
            inner: RustBanditPolicy::KlUcb { c },
        }
    }

    /// EXP3 with exploration `gamma`.
    #[classmethod]
    fn exp3(_cls: &Bound<'_, PyType>, gamma: f64) -> Self {
        Self {
            inner: RustBanditPolicy::Exp3 { gamma },
        }
    }

    fn __repr__(&self) -> String {
        format!("BanditPolicy({:?})", self.inner)
    }
}

/// Comparison policy for `DuelingBandit`.
#[pyclass(name = "DuelingPolicy", module = "propagon._propagon", frozen)]
pub struct DuelingPolicy {
    pub(crate) inner: RustDuelingPolicy,
}

#[pymethods]
impl DuelingPolicy {
    /// Relative UCB with exploration `alpha`.
    #[classmethod]
    fn rucb(_cls: &Bound<'_, PyType>, alpha: f64) -> Self {
        Self {
            inner: RustDuelingPolicy::Rucb { alpha },
        }
    }

    /// Double Thompson sampling with exploration `alpha`.
    #[classmethod]
    fn double_thompson(_cls: &Bound<'_, PyType>, alpha: f64) -> Self {
        Self {
            inner: RustDuelingPolicy::DoubleThompson { alpha },
        }
    }

    fn __repr__(&self) -> String {
        format!("DuelingPolicy({:?})", self.inner)
    }
}

/// Pooling granularity for `BehaviorCloning`.
#[pyclass(name = "Granularity", module = "propagon._propagon", frozen)]
pub struct Granularity {
    pub(crate) inner: RustGranularity,
}

#[pymethods]
impl Granularity {
    /// One global policy over all states. (`global_`: `global` is a Python
    /// keyword.)
    #[classmethod]
    #[pyo3(name = "global_")]
    fn global_(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustGranularity::Global,
        }
    }

    /// Per-state policies keyed by the substring before `separator`.
    #[classmethod]
    fn per_state(_cls: &Bound<'_, PyType>, separator: char) -> Self {
        Self {
            inner: RustGranularity::PerState { separator },
        }
    }

    fn __repr__(&self) -> String {
        format!("Granularity({:?})", self.inner)
    }
}

/// Source-set budget for `Harmonic` centrality.
#[pyclass(name = "SourceBudget", module = "propagon._propagon", frozen)]
pub struct SourceBudget {
    pub(crate) inner: RustSourceBudget,
}

#[pymethods]
impl SourceBudget {
    /// Use every node as a source (exact).
    #[classmethod]
    fn all(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustSourceBudget::All,
        }
    }

    /// Sample `count` source nodes with the given `seed` (approximate).
    #[classmethod]
    fn sample(_cls: &Bound<'_, PyType>, count: usize, seed: u64) -> Self {
        Self {
            inner: RustSourceBudget::Sample { count, seed },
        }
    }

    fn __repr__(&self) -> String {
        format!("SourceBudget({:?})", self.inner)
    }
}

/// Refinement-pass budget for `Kemeny` and `Mallows`.
#[pyclass(name = "KemenyPasses", module = "propagon._propagon", frozen)]
pub struct KemenyPasses {
    pub(crate) inner: RustKemenyPasses,
}

#[pymethods]
impl KemenyPasses {
    /// Choose the pass count automatically.
    #[classmethod]
    fn auto(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustKemenyPasses::Auto,
        }
    }

    /// Use exactly `n` refinement passes.
    #[classmethod]
    fn fixed(_cls: &Bound<'_, PyType>, n: usize) -> Self {
        Self {
            inner: RustKemenyPasses::Fixed(n),
        }
    }

    fn __repr__(&self) -> String {
        format!("KemenyPasses({:?})", self.inner)
    }
}

/// Outlier clipping for `McValue`.
#[pyclass(name = "Winsorize", module = "propagon._propagon", frozen)]
pub struct Winsorize {
    pub(crate) inner: RustWinsorize,
}

#[pymethods]
impl Winsorize {
    /// No clipping.
    #[classmethod]
    fn off(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustWinsorize::Off,
        }
    }

    /// Clip to the given two-sided `percentile` (e.g. 0.05).
    #[classmethod]
    fn percentile(_cls: &Bound<'_, PyType>, percentile: f64) -> Self {
        Self {
            inner: RustWinsorize::Percentile(percentile),
        }
    }

    fn __repr__(&self) -> String {
        format!("Winsorize({:?})", self.inner)
    }
}

/// Restart distribution for `PageRank` (personalization).
#[pyclass(name = "Teleport", module = "propagon._propagon", frozen)]
pub struct Teleport {
    pub(crate) inner: RustTeleport,
}

#[pymethods]
impl Teleport {
    /// Restart uniformly (classic global PageRank).
    #[classmethod]
    fn uniform(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustTeleport::Uniform,
        }
    }

    /// Restart at the named seeds with positive weights (personalized
    /// PageRank / random walk with restart).
    #[classmethod]
    fn seeds(_cls: &Bound<'_, PyType>, seeds: Vec<(String, f64)>) -> Self {
        Self {
            inner: RustTeleport::Seeds(seeds),
        }
    }

    fn __repr__(&self) -> String {
        format!("Teleport({:?})", self.inner)
    }
}

/// Significance-test toggle for `ValueCompare`.
#[pyclass(name = "PairwiseTests", module = "propagon._propagon", frozen)]
pub struct PairwiseTests {
    pub(crate) inner: RustPairwiseTests,
}

#[pymethods]
impl PairwiseTests {
    /// Do not run pairwise significance tests.
    #[classmethod]
    fn off(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustPairwiseTests::Off,
        }
    }

    /// Run permutation tests with `permutations` resamples.
    #[classmethod]
    fn on(_cls: &Bound<'_, PyType>, permutations: usize) -> Self {
        Self {
            inner: RustPairwiseTests::On { permutations },
        }
    }

    fn __repr__(&self) -> String {
        format!("PairwiseTests({:?})", self.inner)
    }
}

/// Registers the enum classes on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameOutcome>()?;
    m.add_class::<BanditPolicy>()?;
    m.add_class::<DuelingPolicy>()?;
    m.add_class::<Granularity>()?;
    m.add_class::<SourceBudget>()?;
    m.add_class::<KemenyPasses>()?;
    m.add_class::<Winsorize>()?;
    m.add_class::<Teleport>()?;
    m.add_class::<PairwiseTests>()?;
    Ok(())
}

/// Names this module contributes to `__all__`.
pub(crate) const EXPORTS: &[&str] = &[
    "GameOutcome",
    "BanditPolicy",
    "DuelingPolicy",
    "Granularity",
    "SourceBudget",
    "KemenyPasses",
    "Winsorize",
    "Teleport",
    "PairwiseTests",
];
