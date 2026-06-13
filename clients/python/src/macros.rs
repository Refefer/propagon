//! Declarative macros that generate the repetitive binding surface.
//!
//! Two families, validated by hand against Glicko-2 (online) and PageRank
//! (batch) before extraction:
//!
//! - [`model_class!`] — a `#[pyclass]` model with the common surface (`scores`,
//!   `sorted_scores`, `score`, `top`, `save_state`, `save_state_bytes`, `load`,
//!   `algorithm`). Optional `extras { … }` splice extra methods into the *same*
//!   `#[pymethods]` impl (PyO3 allows only one such block per type without the
//!   `multiple-pymethods` feature, so extras are pasted in as raw tokens).
//! - [`scalar_online!`] / [`scalar_batch!`] — a params `#[pyclass]` for an
//!   algorithm whose parameters are all scalars (`f64`/`usize`/`u64`/`bool`):
//!   a keyword-only constructor seeded from the Rust `Default`, get/set per
//!   field, `__repr__`, and the fitting methods for its tier.
//!
//! The constructor/getter/setter block is written inline in each tier macro
//! rather than factored into a shared macro: a nested `macro_rules!` call inside
//! `#[pymethods]` is expanded *after* the proc-macro runs, which orphans the
//! generated `#[new]`/`#[getter]` attributes. Trait methods use fully-qualified
//! syntax so the macros do not depend on the using module's imports.

/// Generates a model `#[pyclass]` with the common read/persist surface.
macro_rules! model_class {
    ($name:ident, $py:literal, $rust:ty $(, extras { $($extra:tt)* })?) => {
        #[doc = concat!("A fitted ", $py, ".")]
        #[pyclass(name = $py, module = "propagon._propagon")]
        pub struct $name {
            pub(crate) inner: $rust,
        }

        #[pymethods]
        impl $name {
            /// The stable algorithm tag written into state files.
            #[getter]
            fn algorithm(&self) -> &'static str {
                propagon::RankModel::algorithm(&self.inner)
            }

            /// Primary score per entity as a `dict[str, float]`.
            fn scores<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, ::pyo3::types::PyDict>> {
                let d = ::pyo3::types::PyDict::new(py);
                for (name, s) in propagon::RankModel::scores(&self.inner) {
                    d.set_item(name, s)?;
                }
                Ok(d)
            }

            /// Scores sorted descending, ties broken by name.
            fn sorted_scores(&self) -> Vec<(String, f64)> {
                propagon::RankModel::sorted_scores(&self.inner)
                    .into_iter()
                    .map(|(n, s)| (n.to_string(), s))
                    .collect()
            }

            /// The score for one entity, or `None` if it is unknown.
            fn score(&self, name: &str) -> Option<f64> {
                propagon::RankModel::scores(&self.inner)
                    .find(|(n, _)| *n == name)
                    .map(|(_, s)| s)
            }

            /// The top `k` entities by score.
            fn top(&self, k: usize) -> Vec<(String, f64)> {
                let mut v = self.sorted_scores();
                v.truncate(k);
                v
            }

            /// Serializes the model as header-line JSONL text.
            fn save_state(&self) -> PyResult<String> {
                let mut buf = Vec::new();
                crate::errors::MapPy::map_py(propagon::RankModel::save_jsonl(
                    &self.inner,
                    &mut buf,
                ))?;
                String::from_utf8(buf).map_err(|e| {
                    crate::errors::InvalidInputError::new_err(format!(
                        "state is not valid UTF-8: {e}"
                    ))
                })
            }

            /// Serializes the model as header-line JSONL bytes (avoids text
            /// re-encoding hazards when writing to a file).
            fn save_state_bytes<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, ::pyo3::types::PyBytes>> {
                let mut buf = Vec::new();
                crate::errors::MapPy::map_py(propagon::RankModel::save_jsonl(
                    &self.inner,
                    &mut buf,
                ))?;
                Ok(::pyo3::types::PyBytes::new(py, &buf))
            }

            /// Loads a model previously written by `save_state`. Raises
            /// `AlgorithmMismatchError` on a state file from a different
            /// algorithm.
            #[staticmethod]
            fn load(text: &str) -> PyResult<Self> {
                let inner = crate::errors::MapPy::map_py(
                    <$rust as propagon::RankModel>::load_jsonl(text.as_bytes()),
                )?;
                Ok(Self { inner })
            }

            fn __repr__(&self) -> String {
                let n = propagon::RankModel::scores(&self.inner).count();
                format!(concat!($py, "(entities={})"), n)
            }

            $($($extra)*)?
        }
    };
}

/// An online (incremental) algorithm with scalar params over dataset `$data`,
/// producing model `$model`.
macro_rules! scalar_online {
    (
        $name:ident, $py:literal, $rust:ty, $model:ident, $data:ty,
        { $($field:ident : $fty:ty),* $(,)? }
    ) => {
        #[doc = concat!("The ", $py, " algorithm (incremental).")]
        #[pyclass(name = $py, module = "propagon._propagon")]
        pub struct $name {
            pub(crate) inner: $rust,
        }

        // `paste!` wraps the whole `#[pymethods]` impl so it expands FIRST,
        // resolving `[<set_ $field>]` into real identifiers before the
        // `#[pymethods]` proc-macro runs. A `paste!` (or any function-like
        // macro) *inside* the impl would be expanded too late and orphan the
        // generated `#[setter]` attributes.
        ::paste::paste! {
            #[pymethods]
            impl $name {
                // --- params surface ---
                /// Construct with keyword arguments; any omitted parameter
                /// takes the library default.
                #[new]
                #[pyo3(signature = (*, $($field = None),*))]
                fn new($($field: Option<$fty>),*) -> Self {
                    let mut p = <$rust>::default();
                    $(if let Some(v) = $field { p.$field = v; })*
                    Self { inner: p }
                }
                $(
                    #[getter]
                    fn $field(&self) -> $fty { self.inner.$field }
                    #[setter]
                    fn [<set_ $field>](&mut self, v: $fty) { self.inner.$field = v; }
                )*
                fn __repr__(&self) -> String {
                    let mut parts: Vec<String> = Vec::new();
                    $(parts.push(format!("{}={:?}", stringify!($field), self.inner.$field));)*
                    format!(concat!($py, "({})"), parts.join(", "))
                }

                // --- online fitting surface ---
                /// Fresh state with no observations.
                fn init(&self) -> $model {
                    $model { inner: propagon::OnlineRanker::init(&self.inner) }
                }

                /// Folds one batch into `model`, in place. History is never
                /// replayed, so repeated calls continue the same incremental
                /// state.
                fn update(
                    &self,
                    py: Python<'_>,
                    model: &Bound<'_, $model>,
                    data: &$data,
                ) -> PyResult<()> {
                    let mut guard = model.borrow_mut();
                    let m = &mut guard.inner;
                    let d = &data.inner;
                    let algo = &self.inner;
                    crate::errors::MapPy::map_py(
                        py.detach(|| propagon::OnlineRanker::update(algo, m, d)),
                    )
                }

                /// Convenience: `init()` then a single `update()`. For
                /// incremental resume, use `init`/`update`/`save_state`/`load`.
                fn fit(&self, py: Python<'_>, data: &$data) -> PyResult<$model> {
                    let mut m = propagon::OnlineRanker::init(&self.inner);
                    let d = &data.inner;
                    let algo = &self.inner;
                    crate::errors::MapPy::map_py(
                        py.detach(|| propagon::OnlineRanker::update(algo, &mut m, d)),
                    )?;
                    Ok($model { inner: m })
                }
            }
        }
    };
}

/// A batch algorithm with scalar params over dataset `$data`, producing model
/// `$model`.
macro_rules! scalar_batch {
    (
        $name:ident, $py:literal, $rust:ty, $model:ident, $data:ty,
        { $($field:ident : $fty:ty),* $(,)? }
    ) => {
        #[doc = concat!("The ", $py, " algorithm (batch).")]
        #[pyclass(name = $py, module = "propagon._propagon")]
        pub struct $name {
            pub(crate) inner: $rust,
        }

        // See the note in `scalar_online!` on why `paste!` wraps the impl.
        ::paste::paste! {
            #[pymethods]
            impl $name {
                // --- params surface ---
                /// Construct with keyword arguments; any omitted parameter
                /// takes the library default.
                #[new]
                #[pyo3(signature = (*, $($field = None),*))]
                fn new($($field: Option<$fty>),*) -> Self {
                    let mut p = <$rust>::default();
                    $(if let Some(v) = $field { p.$field = v; })*
                    Self { inner: p }
                }
                $(
                    #[getter]
                    fn $field(&self) -> $fty { self.inner.$field }
                    #[setter]
                    fn [<set_ $field>](&mut self, v: $fty) { self.inner.$field = v; }
                )*
                fn __repr__(&self) -> String {
                    let mut parts: Vec<String> = Vec::new();
                    $(parts.push(format!("{}={:?}", stringify!($field), self.inner.$field));)*
                    format!(concat!($py, "({})"), parts.join(", "))
                }

                // --- batch fitting surface ---
                /// Fits a model from `data`.
                fn fit(&self, py: Python<'_>, data: &$data) -> PyResult<$model> {
                    let algo = &self.inner;
                    let d = &data.inner;
                    let inner = crate::errors::MapPy::map_py(
                        py.detach(|| propagon::Ranker::fit(algo, d)),
                    )?;
                    Ok($model { inner })
                }

                /// Refits from `data` warm-started at `init` (never worse than
                /// `fit`). Iterative algorithms converge faster on appended
                /// data.
                fn fit_warm(
                    &self,
                    py: Python<'_>,
                    data: &$data,
                    init: &$model,
                ) -> PyResult<$model> {
                    let algo = &self.inner;
                    let d = &data.inner;
                    let start = &init.inner;
                    let inner = crate::errors::MapPy::map_py(
                        py.detach(|| propagon::Ranker::fit_warm(algo, d, start)),
                    )?;
                    Ok($model { inner })
                }
            }
        }
    };
}

/// A batch algorithm whose params include enum-valued (or otherwise non-scalar)
/// fields: the caller supplies the `#[new]` constructor (and any getters) as a
/// literal token block, and this macro appends the `fit`/`fit_warm` surface and
/// a serde-backed `__repr__`. No `paste!` is needed because the constructor
/// uses explicit method names.
macro_rules! custom_batch {
    (
        $name:ident, $py:literal, $rust:ty, $model:ident, $data:ty,
        { $($ctor:tt)* }
    ) => {
        #[doc = concat!("The ", $py, " algorithm (batch).")]
        #[pyclass(name = $py, module = "propagon._propagon")]
        pub struct $name {
            pub(crate) inner: $rust,
        }

        #[pymethods]
        impl $name {
            $($ctor)*

            /// Fits a model from `data`.
            fn fit(&self, py: Python<'_>, data: &$data) -> PyResult<$model> {
                let algo = &self.inner;
                let d = &data.inner;
                let inner = crate::errors::MapPy::map_py(
                    py.detach(|| propagon::Ranker::fit(algo, d)),
                )?;
                Ok($model { inner })
            }

            /// Refits from `data` warm-started at `init` (never worse than
            /// `fit`).
            fn fit_warm(
                &self,
                py: Python<'_>,
                data: &$data,
                init: &$model,
            ) -> PyResult<$model> {
                let algo = &self.inner;
                let d = &data.inner;
                let start = &init.inner;
                let inner = crate::errors::MapPy::map_py(
                    py.detach(|| propagon::Ranker::fit_warm(algo, d, start)),
                )?;
                Ok($model { inner })
            }

            fn __repr__(&self) -> String {
                format!(
                    concat!($py, "({})"),
                    serde_json::to_string(&self.inner).unwrap_or_default()
                )
            }
        }
    };
}

/// The online counterpart of [`custom_batch!`].
macro_rules! custom_online {
    (
        $name:ident, $py:literal, $rust:ty, $model:ident, $data:ty,
        { $($ctor:tt)* }
    ) => {
        #[doc = concat!("The ", $py, " algorithm (incremental).")]
        #[pyclass(name = $py, module = "propagon._propagon")]
        pub struct $name {
            pub(crate) inner: $rust,
        }

        #[pymethods]
        impl $name {
            $($ctor)*

            /// Fresh state with no observations.
            fn init(&self) -> $model {
                $model { inner: propagon::OnlineRanker::init(&self.inner) }
            }

            /// Folds one batch into `model`, in place (history is never
            /// replayed).
            fn update(
                &self,
                py: Python<'_>,
                model: &Bound<'_, $model>,
                data: &$data,
            ) -> PyResult<()> {
                let mut guard = model.borrow_mut();
                let m = &mut guard.inner;
                let d = &data.inner;
                let algo = &self.inner;
                crate::errors::MapPy::map_py(
                    py.detach(|| propagon::OnlineRanker::update(algo, m, d)),
                )
            }

            /// Convenience: `init()` then a single `update()`.
            fn fit(&self, py: Python<'_>, data: &$data) -> PyResult<$model> {
                let mut m = propagon::OnlineRanker::init(&self.inner);
                let d = &data.inner;
                let algo = &self.inner;
                crate::errors::MapPy::map_py(
                    py.detach(|| propagon::OnlineRanker::update(algo, &mut m, d)),
                )?;
                Ok($model { inner: m })
            }

            fn __repr__(&self) -> String {
                format!(
                    concat!($py, "({})"),
                    serde_json::to_string(&self.inner).unwrap_or_default()
                )
            }
        }
    };
}

/// A batch algorithm with no parameters: a no-argument constructor plus the
/// `fit`/`fit_warm` surface.
macro_rules! nofield_batch {
    ($name:ident, $py:literal, $rust:ty, $model:ident, $data:ty) => {
        custom_batch!($name, $py, $rust, $model, $data, {
            /// Construct with default parameters (this algorithm has none).
            #[new]
            fn new() -> Self {
                Self {
                    inner: <$rust>::default(),
                }
            }
        });
    };
}
