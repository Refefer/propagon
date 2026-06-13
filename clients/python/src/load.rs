//! `load_state`: reconstruct the right concrete model class from a saved state
//! file by dispatching on its header `algorithm` tag.
//!
//! Each model class also has its own `load` staticmethod (which additionally
//! enforces the tag matches that class); this free function is the
//! "don't-know-the-algorithm-yet" entry point. The arms are written out rather
//! than macro-generated because a `:path` macro fragment cannot form a struct
//! literal (`$model { inner }`).

use pyo3::prelude::*;

use crate::algos::{
    annotated, contextual, games, graph, matchups, pairwise, rankings, rewards, trajectories,
};
use crate::errors::{MapPy, StateError};

/// Loads a state's bytes into model type `R`, wraps it in the Python class `W`,
/// and boxes it as an opaque Python object.
fn load_into<R, W>(py: Python<'_>, text: &str, wrap: impl FnOnce(R) -> W) -> PyResult<Py<PyAny>>
where
    R: propagon::RankModel,
    W: pyo3::PyClass + Into<pyo3::PyClassInitializer<W>>,
{
    let inner = <R as propagon::RankModel>::load_jsonl(text.as_bytes()).map_py()?;
    Ok(Py::new(py, wrap(inner))?.into_any())
}

/// Reconstructs a fitted model from header-line JSONL `text`, returning the
/// concrete model class matching the file's `algorithm` tag.
#[pyfunction]
pub(crate) fn load_state(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    let first = text
        .lines()
        .next()
        .ok_or_else(|| StateError::new_err("empty state file"))?;
    let header: propagon::state::Header = serde_json::from_str(first)
        .map_err(|e| StateError::new_err(format!("malformed state header: {e}")))?;

    match header.algorithm.as_str() {
        // games
        "elo" => load_into(py, text, |inner| games::EloModel { inner }),
        "glicko2" => load_into(py, text, |inner| games::Glicko2Model { inner }),
        "elo-mov" => load_into(py, text, |inner| games::MovEloModel { inner }),
        "melo" => load_into(py, text, |inner| games::MEloModel { inner }),
        "generalized-bt" => load_into(py, text, |inner| games::GeneralizedBtModel { inner }),
        "team-bradley-terry" => load_into(py, text, |inner| games::TeamBtModel { inner }),
        // graph
        "page-rank" => load_into(py, text, |inner| graph::PageRankModel { inner }),
        "hits" => load_into(py, text, |inner| graph::HitsModel { inner }),
        "birank" => load_into(py, text, |inner| graph::BiRankModel { inner }),
        "degree" => load_into(py, text, |inner| graph::DegreeModel { inner }),
        "harmonic" => load_into(py, text, |inner| graph::HarmonicModel { inner }),
        "katz" => load_into(py, text, |inner| graph::KatzModel { inner }),
        "k-core" => load_into(py, text, |inner| graph::KCoreModel { inner }),
        "leader-rank" => load_into(py, text, |inner| graph::LeaderRankModel { inner }),
        // rewards / contextual
        "bandit" => load_into(py, text, |inner| rewards::BanditModel { inner }),
        "sliding-window-ucb" => load_into(py, text, |inner| rewards::SwUcbModel { inner }),
        "lin-ucb" => load_into(py, text, |inner| contextual::LinUcbModel { inner }),
        // annotated
        "crowd-bt" => load_into(py, text, |inner| annotated::CrowdBtModel { inner }),
        // matchups
        "weng-lin" => load_into(py, text, |inner| matchups::WengLinModel { inner }),
        // rankings
        "plackett-luce" => load_into(py, text, |inner| rankings::PlackettLuceModel { inner }),
        "footrule" => load_into(py, text, |inner| rankings::FootruleModel { inner }),
        "mallows" => load_into(py, text, |inner| rankings::MallowsModel { inner }),
        "mc4" => load_into(py, text, |inner| rankings::Mc4Model { inner }),
        // trajectories
        "mc-value" => load_into(py, text, |inner| trajectories::McValueModel { inner }),
        "behavior-cloning" => load_into(py, text, |inner| trajectories::BcModel { inner }),
        "value-compare" => load_into(py, text, |inner| trajectories::ValueCompareModel { inner }),
        "td-value" => load_into(py, text, |inner| trajectories::TdValueModel { inner }),
        // pairwise
        "btm-mm" => load_into(py, text, |inner| pairwise::BtmMmModel { inner }),
        "btm-lr" => load_into(py, text, |inner| pairwise::BtmLrModel { inner }),
        "bayesian-bradley-terry" => load_into(py, text, |inner| pairwise::BayesBtModel { inner }),
        "colley" => load_into(py, text, |inner| pairwise::ColleyModel { inner }),
        "massey" => load_into(py, text, |inner| pairwise::MasseyModel { inner }),
        "keener" => load_into(py, text, |inner| pairwise::KeenerModel { inner }),
        "ilsr" => load_into(py, text, |inner| pairwise::ILsrModel { inner }),
        "nash-averaging" => load_into(py, text, |inner| pairwise::NashAveragingModel { inner }),
        "offense-defense" => load_into(py, text, |inner| pairwise::OffenseDefenseModel { inner }),
        "random-walker" => load_into(py, text, |inner| pairwise::RandomWalkerModel { inner }),
        "rank-centrality" => load_into(py, text, |inner| pairwise::RankCentralityModel { inner }),
        "serial-rank" => load_into(py, text, |inner| pairwise::SerialRankModel { inner }),
        "thurstone-mosteller" => load_into(py, text, |inner| pairwise::ThurstoneModel { inner }),
        "whr" => load_into(py, text, |inner| pairwise::WhrModel { inner }),
        "borda" => load_into(py, text, |inner| pairwise::BordaModel { inner }),
        "copeland" => load_into(py, text, |inner| pairwise::CopelandModel { inner }),
        "blade-chest" => load_into(py, text, |inner| pairwise::BladeChestModel { inner }),
        "es-rum" => load_into(py, text, |inner| pairwise::EsRumModel { inner }),
        "hodge-rank" => load_into(py, text, |inner| pairwise::HodgeModel { inner }),
        "kemeny" => load_into(py, text, |inner| pairwise::KemenyModel { inner }),
        "lsr" => load_into(py, text, |inner| pairwise::LsrModel { inner }),
        "covariate-bt" => load_into(py, text, |inner| pairwise::CovariateBtModel { inner }),
        "rate" => load_into(py, text, |inner| pairwise::WinRateModel { inner }),
        "dueling-bandit" => load_into(py, text, |inner| pairwise::DuelingModel { inner }),
        other => Err(StateError::new_err(format!(
            "unknown algorithm tag {other:?} in state header"
        ))),
    }
}
