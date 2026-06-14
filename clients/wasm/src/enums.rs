//! Conversions between WIT enums/variants and `propagon` enums.
//!
//! Unlike the Python client (which parses kebab-case strings via `unit_enum`),
//! the Component Model carries these as statically-typed `enum`/`variant`s, so
//! conversion is a total `match` with no parse step or "expected one of" error.

use serde::de::DeserializeOwned;

use crate::bindings::exports::propagon::core::graph::{Sink, SourceBudget, Teleport};
use crate::bindings::exports::propagon::core::pairwise::DuelingPolicy;
use crate::bindings::exports::propagon::core::rewards::BanditPolicy;
use crate::bindings::exports::propagon::core::trajectories::{
    Granularity, PairwiseTests, Winsorize,
};
use crate::bindings::exports::propagon::core::types::{Error, GameOutcome, KemenyPasses};
use crate::convert::narrow_f32;

/// Parse a unit-only param enum from a kebab-case string via serde (mirrors the
/// Python client's `unit_enum`). The target type is inferred from the call site.
pub(crate) fn unit_enum<T: DeserializeOwned>(s: &str, field: &str) -> Result<T, Error> {
    serde_json::from_value(serde_json::Value::String(s.to_string()))
        .map_err(|_| Error::InvalidInput(format!("invalid {field}: {s:?}")))
}

/// WIT `game-outcome` -> `propagon::GameOutcome` (margins narrow f64 -> f32).
pub(crate) fn game_outcome(go: GameOutcome) -> Result<propagon::GameOutcome, Error> {
    Ok(match go {
        GameOutcome::Side1Win(m) => propagon::GameOutcome::Side1Win(narrow_f32(m, "margin")?),
        GameOutcome::Side2Win(m) => propagon::GameOutcome::Side2Win(narrow_f32(m, "margin")?),
        GameOutcome::Tie => propagon::GameOutcome::Tie,
    })
}

/// WIT `sink` -> `propagon::algos::Sink`.
pub(crate) fn sink(s: Sink) -> propagon::algos::Sink {
    use propagon::algos::Sink as P;
    match s {
        Sink::Reverse => P::Reverse,
        Sink::All => P::All,
        Sink::Uniform => P::Uniform,
        Sink::None => P::None,
    }
}

/// WIT `teleport` -> `propagon::algos::Teleport`.
pub(crate) fn teleport(t: Teleport) -> propagon::algos::Teleport {
    match t {
        Teleport::Uniform => propagon::algos::Teleport::Uniform,
        Teleport::Seeds(seeds) => propagon::algos::Teleport::Seeds(seeds),
    }
}

/// WIT `source-budget` -> `propagon::algos::SourceBudget`.
pub(crate) fn source_budget(b: SourceBudget) -> propagon::algos::SourceBudget {
    match b {
        SourceBudget::All => propagon::algos::SourceBudget::All,
        SourceBudget::Sample((count, seed)) => propagon::algos::SourceBudget::Sample {
            count: count as usize,
            seed,
        },
    }
}

/// WIT `kemeny-passes` -> `propagon::algos::KemenyPasses`.
pub(crate) fn kemeny_passes(p: KemenyPasses) -> propagon::algos::KemenyPasses {
    match p {
        KemenyPasses::Auto => propagon::algos::KemenyPasses::Auto,
        KemenyPasses::Fixed(n) => propagon::algos::KemenyPasses::Fixed(n as usize),
    }
}

/// WIT `granularity` -> `propagon::algos::Granularity`.
pub(crate) fn granularity(g: Granularity) -> propagon::algos::Granularity {
    match g {
        Granularity::Global => propagon::algos::Granularity::Global,
        Granularity::PerState(separator) => propagon::algos::Granularity::PerState { separator },
    }
}

/// WIT `pairwise-tests` -> `propagon::algos::PairwiseTests`.
pub(crate) fn pairwise_tests(t: PairwiseTests) -> propagon::algos::PairwiseTests {
    match t {
        PairwiseTests::Off => propagon::algos::PairwiseTests::Off,
        PairwiseTests::On(permutations) => propagon::algos::PairwiseTests::On {
            permutations: permutations as usize,
        },
    }
}

/// WIT `winsorize` -> `propagon::algos::Winsorize`.
pub(crate) fn winsorize(w: Winsorize) -> propagon::algos::Winsorize {
    match w {
        Winsorize::Off => propagon::algos::Winsorize::Off,
        Winsorize::Percentile(p) => propagon::algos::Winsorize::Percentile(p),
    }
}

/// WIT `dueling-policy` -> `propagon::algos::DuelingPolicy`.
pub(crate) fn dueling_policy(p: DuelingPolicy) -> propagon::algos::DuelingPolicy {
    match p {
        DuelingPolicy::Rucb(alpha) => propagon::algos::DuelingPolicy::Rucb { alpha },
        DuelingPolicy::DoubleThompson(alpha) => {
            propagon::algos::DuelingPolicy::DoubleThompson { alpha }
        }
    }
}

/// WIT `bandit-policy` -> `propagon::algos::BanditPolicy`.
pub(crate) fn bandit_policy(p: BanditPolicy) -> propagon::algos::BanditPolicy {
    use propagon::algos::BanditPolicy as P;
    match p {
        BanditPolicy::Greedy => P::Greedy,
        BanditPolicy::EpsilonGreedy(epsilon) => P::EpsilonGreedy { epsilon },
        BanditPolicy::Ucb1(exploration) => P::Ucb1 { exploration },
        BanditPolicy::ThompsonBeta((prior_alpha, prior_beta)) => P::ThompsonBeta {
            prior_alpha,
            prior_beta,
        },
        BanditPolicy::ThompsonGaussian((prior_mean, prior_weight)) => P::ThompsonGaussian {
            prior_mean,
            prior_weight,
        },
        BanditPolicy::KlUcb(c) => P::KlUcb { c },
        BanditPolicy::Exp3(gamma) => P::Exp3 { gamma },
    }
}
