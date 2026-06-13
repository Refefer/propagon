//! Algorithms over [`PairwiseDataset`]: the win/loss family — Bradley-Terry
//! variants, least-squares and spectral ratings, rank aggregation reductions,
//! Wilson win-rate (online), and dueling bandits (online).

use pyo3::prelude::*;

use propagon::algos::{
    BayesianBradleyTerry as RustBayesianBradleyTerry, BladeChest as RustBladeChest,
    Borda as RustBorda, BradleyTerryLR as RustBradleyTerryLR, BradleyTerryMM as RustBradleyTerryMM,
    Colley as RustColley, Copeland as RustCopeland, CovariateBt as RustCovariateBt,
    DuelingBandit as RustDuelingBandit, EsRum as RustEsRum, HodgeRank as RustHodgeRank,
    ILsr as RustILsr, Keener as RustKeener, Kemeny as RustKemeny, Lsr as RustLsr,
    Massey as RustMassey, NashAveraging as RustNashAveraging, OffenseDefense as RustOffenseDefense,
    RandomWalker as RustRandomWalker, RankCentrality as RustRankCentrality,
    SerialRank as RustSerialRank, ThurstoneMosteller as RustThurstoneMosteller, Whr as RustWhr,
    WinRate as RustWinRate,
};

use crate::datasets::PairwiseDataset;
use crate::enums::{DuelingPolicy, KemenyPasses, unit_enum};

// --- scalar-param batch algorithms ---

model_class!(
    BtmMmModel,
    "BradleyTerryMMModel",
    propagon::algos::BtmMmModel
);
scalar_batch!(BradleyTerryMM, "BradleyTerryMM", RustBradleyTerryMM, BtmMmModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
    min_graph_size: usize,
    remove_total_losers: bool,
    create_fake_games: f64,
    random_subgraph_links: usize,
    random_subgraph_weight: f64,
    seed: u64,
});

model_class!(
    BtmLrModel,
    "BradleyTerryLRModel",
    propagon::algos::BtmLrModel
);
scalar_batch!(BradleyTerryLR, "BradleyTerryLR", RustBradleyTerryLR, BtmLrModel, PairwiseDataset, {
    passes: usize,
    alpha: f64,
    decay: f64,
    thrifty: bool,
});

model_class!(
    BayesBtModel,
    "BayesianBradleyTerryModel",
    propagon::algos::BayesBtModel
);
scalar_batch!(BayesianBradleyTerry, "BayesianBradleyTerry", RustBayesianBradleyTerry, BayesBtModel, PairwiseDataset, {
    shape: f64,
    rate: f64,
    samples: usize,
    burn_in: usize,
    credible: f64,
    seed: u64,
});

model_class!(ColleyModel, "ColleyModel", propagon::algos::ColleyModel);
scalar_batch!(Colley, "Colley", RustColley, ColleyModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
});

model_class!(MasseyModel, "MasseyModel", propagon::algos::MasseyModel);
scalar_batch!(Massey, "Massey", RustMassey, MasseyModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
});

model_class!(KeenerModel, "KeenerModel", propagon::algos::KeenerModel);
scalar_batch!(Keener, "Keener", RustKeener, KeenerModel, PairwiseDataset, {
    skew: bool,
    normalize_games: bool,
    iterations: usize,
    tolerance: f64,
});

model_class!(ILsrModel, "ILsrModel", propagon::algos::ILsrModel);
scalar_batch!(ILsr, "ILsr", RustILsr, ILsrModel, PairwiseDataset, {
    outer: usize,
    inner_steps: usize,
    tolerance: f64,
});

model_class!(
    NashAveragingModel,
    "NashAveragingModel",
    propagon::algos::NashAveragingModel
);
scalar_batch!(NashAveraging, "NashAveraging", RustNashAveraging, NashAveragingModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
    learning_rate: f64,
    anneal_every: usize,
});

model_class!(
    OffenseDefenseModel,
    "OffenseDefenseModel",
    propagon::algos::OffenseDefenseModel
);
scalar_batch!(OffenseDefense, "OffenseDefense", RustOffenseDefense, OffenseDefenseModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
});

model_class!(
    RandomWalkerModel,
    "RandomWalkerModel",
    propagon::algos::RandomWalkerModel
);
scalar_batch!(RandomWalker, "RandomWalker", RustRandomWalker, RandomWalkerModel, PairwiseDataset, {
    p: f64,
    iterations: usize,
    tolerance: f64,
});

model_class!(
    RankCentralityModel,
    "RankCentralityModel",
    propagon::algos::RankCentralityModel
);
scalar_batch!(RankCentrality, "RankCentrality", RustRankCentrality, RankCentralityModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
});

model_class!(
    SerialRankModel,
    "SerialRankModel",
    propagon::algos::SerialRankModel
);
scalar_batch!(SerialRank, "SerialRank", RustSerialRank, SerialRankModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
    seed: u64,
});

model_class!(
    ThurstoneModel,
    "ThurstoneMostellerModel",
    propagon::algos::ThurstoneModel
);
scalar_batch!(ThurstoneMosteller, "ThurstoneMosteller", RustThurstoneMosteller, ThurstoneModel, PairwiseDataset, {
    iterations: usize,
    tolerance: f64,
    pseudo_count: f64,
});

model_class!(WhrModel, "WhrModel", propagon::algos::WhrModel);
scalar_batch!(Whr, "Whr", RustWhr, WhrModel, PairwiseDataset, {
    w2: f64,
    prior_games: f64,
    iterations: usize,
    tolerance: f64,
});

// --- no-param batch algorithms ---

model_class!(BordaModel, "BordaModel", propagon::algos::BordaModel);
nofield_batch!(Borda, "Borda", RustBorda, BordaModel, PairwiseDataset);

model_class!(
    CopelandModel,
    "CopelandModel",
    propagon::algos::CopelandModel
);
nofield_batch!(
    Copeland,
    "Copeland",
    RustCopeland,
    CopelandModel,
    PairwiseDataset
);

// --- enum-param batch algorithms ---

model_class!(
    BladeChestModel,
    "BladeChestModel",
    propagon::algos::BladeChestModel
);
custom_batch!(
    BladeChest,
    "BladeChest",
    RustBladeChest,
    BladeChestModel,
    PairwiseDataset,
    {
        /// Configure Blade-Chest. `variant` is "inner" or "dist".
        #[new]
        #[pyo3(signature = (*, dims=None, variant=None, lr=None, epochs=None, l2=None, init_scale=None, seed=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            dims: Option<usize>,
            variant: Option<String>,
            lr: Option<f64>,
            epochs: Option<usize>,
            l2: Option<f64>,
            init_scale: Option<f64>,
            seed: Option<u64>,
        ) -> PyResult<Self> {
            let mut p = RustBladeChest::default();
            if let Some(v) = dims {
                p.dims = v;
            }
            if let Some(s) = variant {
                p.variant = unit_enum(&s, "variant", "inner, dist")?;
            }
            if let Some(v) = lr {
                p.lr = v;
            }
            if let Some(v) = epochs {
                p.epochs = v;
            }
            if let Some(v) = l2 {
                p.l2 = v;
            }
            if let Some(v) = init_scale {
                p.init_scale = v;
            }
            if let Some(v) = seed {
                p.seed = v;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(EsRumModel, "EsRumModel", propagon::algos::EsRumModel);
custom_batch!(EsRum, "EsRum", RustEsRum, EsRumModel, PairwiseDataset, {
    /// Configure ES-RUM. `distribution` is "gaussian" or "fixed-normal".
    #[new]
    #[pyo3(signature = (*, distribution=None, passes=None, alpha=None, gamma=None, min_obs=None, prior=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        distribution: Option<String>,
        passes: Option<usize>,
        alpha: Option<f64>,
        gamma: Option<f64>,
        min_obs: Option<usize>,
        prior: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut p = RustEsRum::default();
        if let Some(s) = distribution {
            p.distribution = unit_enum(&s, "distribution", "gaussian, fixed-normal")?;
        }
        if let Some(v) = passes {
            p.passes = v;
        }
        if let Some(v) = alpha {
            p.alpha = v;
        }
        if let Some(v) = gamma {
            p.gamma = v;
        }
        if let Some(v) = min_obs {
            p.min_obs = v;
        }
        if let Some(v) = prior {
            p.prior = v;
        }
        if let Some(v) = seed {
            p.seed = v;
        }
        Ok(Self { inner: p })
    }
});

model_class!(HodgeModel, "HodgeRankModel", propagon::algos::HodgeModel);
custom_batch!(
    HodgeRank,
    "HodgeRank",
    RustHodgeRank,
    HodgeModel,
    PairwiseDataset,
    {
        /// Configure HodgeRank. `flow` is "log-odds", "win-rate-delta", or
        /// "mean-margin".
        #[new]
        #[pyo3(signature = (*, flow=None, iterations=None, tolerance=None))]
        fn new(
            flow: Option<String>,
            iterations: Option<usize>,
            tolerance: Option<f64>,
        ) -> PyResult<Self> {
            let mut p = RustHodgeRank::default();
            if let Some(s) = flow {
                p.flow = unit_enum(&s, "flow", "log-odds, win-rate-delta, mean-margin")?;
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

model_class!(KemenyModel, "KemenyModel", propagon::algos::KemenyModel);
custom_batch!(
    Kemeny,
    "Kemeny",
    RustKemeny,
    KemenyModel,
    PairwiseDataset,
    {
        /// Configure Kemeny. `passes` is a `KemenyPasses`; `algo` is "insertion"
        /// or "diff-evo".
        #[new]
        #[pyo3(signature = (*, passes=None, min_obs=None, algo=None, seed=None))]
        fn new(
            passes: Option<PyRef<'_, KemenyPasses>>,
            min_obs: Option<usize>,
            algo: Option<String>,
            seed: Option<u64>,
        ) -> PyResult<Self> {
            let mut p = RustKemeny::default();
            if let Some(k) = passes {
                p.passes = k.inner;
            }
            if let Some(v) = min_obs {
                p.min_obs = v;
            }
            if let Some(s) = algo {
                p.algo = unit_enum(&s, "algo", "insertion, diff-evo")?;
            }
            if let Some(v) = seed {
                p.seed = v;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(LsrModel, "LsrModel", propagon::algos::LsrModel);
custom_batch!(Lsr, "Lsr", RustLsr, LsrModel, PairwiseDataset, {
    /// Configure Luce Spectral Ranking. `estimator` is "power-method" or
    /// "monte-carlo".
    #[new]
    #[pyo3(signature = (*, steps=None, estimator=None, seed=None))]
    fn new(steps: Option<usize>, estimator: Option<String>, seed: Option<u64>) -> PyResult<Self> {
        let mut p = RustLsr::default();
        if let Some(v) = steps {
            p.steps = v;
        }
        if let Some(s) = estimator {
            p.estimator = unit_enum(&s, "estimator", "power-method, monte-carlo")?;
        }
        if let Some(v) = seed {
            p.seed = v;
        }
        Ok(Self { inner: p })
    }
});

model_class!(
    CovariateBtModel,
    "CovariateBtModel",
    propagon::algos::CovariateBtModel
);
custom_batch!(
    CovariateBt,
    "CovariateBt",
    RustCovariateBt,
    CovariateBtModel,
    PairwiseDataset,
    {
        /// Configure covariate Bradley-Terry. `features` is required: a list of
        /// `(entity, feature_vector)` pairs, all of one dimensionality.
        #[new]
        #[pyo3(signature = (features, *, l2=None, intercepts=None, iterations=None, tolerance=None))]
        fn new(
            features: Vec<(String, Vec<f64>)>,
            l2: Option<f64>,
            intercepts: Option<bool>,
            iterations: Option<usize>,
            tolerance: Option<f64>,
        ) -> Self {
            let mut p = RustCovariateBt::new(features);
            if let Some(v) = l2 {
                p.l2 = v;
            }
            if let Some(v) = intercepts {
                p.intercepts = v;
            }
            if let Some(v) = iterations {
                p.iterations = v;
            }
            if let Some(v) = tolerance {
                p.tolerance = v;
            }
            Self { inner: p }
        }
    }
);

// --- online algorithms ---

model_class!(WinRateModel, "WinRateModel", propagon::algos::WinRateModel);
custom_online!(
    WinRate,
    "WinRate",
    RustWinRate,
    WinRateModel,
    PairwiseDataset,
    {
        /// Configure Wilson win-rate. `confidence` is "P50", "P90", or "P95".
        #[new]
        #[pyo3(signature = (*, confidence=None))]
        fn new(confidence: Option<String>) -> PyResult<Self> {
            let mut p = RustWinRate::default();
            if let Some(s) = confidence {
                p.confidence = unit_enum(&s, "confidence", "P50, P90, P95")?;
            }
            Ok(Self { inner: p })
        }
    }
);

model_class!(
    DuelingModel,
    "DuelingBanditModel",
    propagon::algos::DuelingModel
);
custom_online!(
    DuelingBandit,
    "DuelingBandit",
    RustDuelingBandit,
    DuelingModel,
    PairwiseDataset,
    {
        /// Configure a dueling bandit. `policy` is a `DuelingPolicy`.
        #[new]
        #[pyo3(signature = (*, policy=None, seed=None))]
        fn new(policy: Option<PyRef<'_, DuelingPolicy>>, seed: Option<u64>) -> Self {
            let mut p = RustDuelingBandit::default();
            if let Some(pol) = policy {
                p.policy = pol.inner;
            }
            if let Some(v) = seed {
                p.seed = v;
            }
            Self { inner: p }
        }
    }
);
