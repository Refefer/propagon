//! Algorithms over `pairwise-dataset`: the tournament/ranking family (mostly
//! batch; WinRate and DuelingBandit are online).

use propagon::algos::{
    BayesBtModel as CoreBayesBt, BayesianBradleyTerry, BladeChest,
    BladeChestModel as CoreBladeChest, Borda, BordaModel as CoreBorda, BradleyTerryLR,
    BradleyTerryMM, BtmLrModel as CoreBtmLr, BtmMmModel as CoreBtmMm, Colley,
    ColleyModel as CoreColley, Copeland, CopelandModel as CoreCopeland, CovariateBt,
    CovariateBtModel as CoreCovariateBt, DuelingBandit, DuelingModel as CoreDueling, EsRum,
    EsRumModel as CoreEsRum, HodgeModel as CoreHodge, HodgeRank, ILsr, ILsrModel as CoreILsr,
    Keener, KeenerModel as CoreKeener, Kemeny, KemenyModel as CoreKemeny, Lsr, LsrModel as CoreLsr,
    Massey, MasseyModel as CoreMassey, NashAveraging, NashAveragingModel as CoreNash,
    OffenseDefense, OffenseDefenseModel as CoreOffDef, RandomWalker,
    RandomWalkerModel as CoreRandWalk, RankCentrality, RankCentralityModel as CoreRankCent,
    SerialRank, SerialRankModel as CoreSerial, ThurstoneModel as CoreThurstone, ThurstoneMosteller,
    Whr, WhrModel as CoreWhr, WinRate, WinRateModel as CoreWinRate,
};

use crate::Component;
use crate::datasets::PairwiseData;
use crate::enums::{dueling_policy, kemeny_passes, unit_enum};
use crate::wit::datasets::PairwiseDatasetBorrow;
use crate::wit::pairwise::*;
use crate::wit::types::Error;

// Model wrappers ----------------------------------------------------------------
batch_model!(BtmMmMod, GuestBradleyTerryMmModel, CoreBtmMm);
batch_model!(BordaMod, GuestBordaModel, CoreBorda);
batch_model!(BtmLrMod, GuestBradleyTerryLrModel, CoreBtmLr);
batch_model!(BayesBtMod, GuestBayesianBradleyTerryModel, CoreBayesBt);
batch_model!(ColleyMod, GuestColleyModel, CoreColley);
batch_model!(MasseyMod, GuestMasseyModel, CoreMassey);
batch_model!(KeenerMod, GuestKeenerModel, CoreKeener);
batch_model!(ILsrMod, GuestILsrModel, CoreILsr);
batch_model!(NashMod, GuestNashAveragingModel, CoreNash);
batch_model!(OffDefMod, GuestOffenseDefenseModel, CoreOffDef);
batch_model!(RandWalkMod, GuestRandomWalkerModel, CoreRandWalk);
batch_model!(RankCentMod, GuestRankCentralityModel, CoreRankCent);
batch_model!(SerialMod, GuestSerialRankModel, CoreSerial);
batch_model!(ThurstoneMod, GuestThurstoneMostellerModel, CoreThurstone);
batch_model!(WhrMod, GuestWhrModel, CoreWhr);
batch_model!(CopelandMod, GuestCopelandModel, CoreCopeland);
batch_model!(BladeChestMod, GuestBladeChestModel, CoreBladeChest);
batch_model!(EsRumMod, GuestEsRumModel, CoreEsRum);
batch_model!(HodgeMod, GuestHodgeRankModel, CoreHodge);
batch_model!(KemenyMod, GuestKemenyModel, CoreKemeny);
batch_model!(LsrMod, GuestLsrModel, CoreLsr);
batch_model!(CovariateBtMod, GuestCovariateBtModel, CoreCovariateBt);
online_model!(
    WinRateMod,
    GuestWinRateModel,
    WinRate,
    CoreWinRate,
    PairwiseData,
    PairwiseDatasetBorrow<'_>
);
online_model!(
    DuelingMod,
    GuestDuelingBanditModel,
    DuelingBandit,
    CoreDueling,
    PairwiseData,
    PairwiseDatasetBorrow<'_>
);

// Param builders ----------------------------------------------------------------
fn btm_mm_build(p: BradleyTerryMmParams) -> Result<BradleyTerryMM, Error> {
    Ok(merge_params!(
        p,
        BradleyTerryMM,
        scalar {
            tolerance,
            remove_total_losers,
            create_fake_games,
            random_subgraph_weight,
            seed
        },
        usize {
            iterations,
            min_graph_size,
            random_subgraph_links
        }
    ))
}
fn btm_lr_build(p: BradleyTerryLrParams) -> Result<BradleyTerryLR, Error> {
    Ok(merge_params!(
        p,
        BradleyTerryLR,
        scalar {
            alpha,
            decay,
            thrifty
        },
        usize { passes }
    ))
}
fn bayes_bt_build(p: BayesianBradleyTerryParams) -> Result<BayesianBradleyTerry, Error> {
    Ok(merge_params!(
        p,
        BayesianBradleyTerry,
        scalar {
            shape,
            rate,
            credible,
            seed
        },
        usize { samples, burn_in }
    ))
}
fn colley_build(p: ColleyParams) -> Result<Colley, Error> {
    Ok(merge_params!(
        p,
        Colley,
        scalar { tolerance },
        usize { iterations }
    ))
}
fn massey_build(p: MasseyParams) -> Result<Massey, Error> {
    Ok(merge_params!(
        p,
        Massey,
        scalar { tolerance },
        usize { iterations }
    ))
}
fn keener_build(p: KeenerParams) -> Result<Keener, Error> {
    Ok(merge_params!(
        p,
        Keener,
        scalar {
            skew,
            normalize_games,
            tolerance
        },
        usize { iterations }
    ))
}
fn i_lsr_build(p: ILsrParams) -> Result<ILsr, Error> {
    Ok(merge_params!(
        p,
        ILsr,
        scalar { tolerance },
        usize { outer, inner_steps }
    ))
}
fn nash_build(p: NashAveragingParams) -> Result<NashAveraging, Error> {
    Ok(merge_params!(
        p,
        NashAveraging,
        scalar {
            tolerance,
            learning_rate
        },
        usize {
            iterations,
            anneal_every
        }
    ))
}
fn off_def_build(p: OffenseDefenseParams) -> Result<OffenseDefense, Error> {
    Ok(merge_params!(
        p,
        OffenseDefense,
        scalar { tolerance },
        usize { iterations }
    ))
}
fn rand_walk_build(p: RandomWalkerParams) -> Result<RandomWalker, Error> {
    Ok(merge_params!(
        p,
        RandomWalker,
        scalar { p, tolerance },
        usize { iterations }
    ))
}
fn rank_cent_build(p: RankCentralityParams) -> Result<RankCentrality, Error> {
    Ok(merge_params!(
        p,
        RankCentrality,
        scalar { tolerance },
        usize { iterations }
    ))
}
fn serial_build(p: SerialRankParams) -> Result<SerialRank, Error> {
    Ok(merge_params!(
        p,
        SerialRank,
        scalar { tolerance, seed },
        usize { iterations }
    ))
}
fn thurstone_build(p: ThurstoneMostellerParams) -> Result<ThurstoneMosteller, Error> {
    Ok(merge_params!(
        p,
        ThurstoneMosteller,
        scalar {
            tolerance,
            pseudo_count
        },
        usize { iterations }
    ))
}
fn whr_build(p: WhrParams) -> Result<Whr, Error> {
    Ok(merge_params!(
        p,
        Whr,
        scalar {
            w2,
            prior_games,
            tolerance
        },
        usize { iterations }
    ))
}
fn blade_chest_build(p: BladeChestParams) -> Result<BladeChest, Error> {
    let mut a = merge_params!(
        p,
        BladeChest,
        scalar {
            lr,
            l2,
            init_scale,
            seed
        },
        usize { dims, epochs }
    );
    if let Some(s) = p.bc_variant {
        a.variant = unit_enum(&s, "variant")?;
    }
    Ok(a)
}
fn es_rum_build(p: EsRumParams) -> Result<EsRum, Error> {
    let mut a = merge_params!(
        p,
        EsRum,
        scalar { alpha, gamma, seed },
        usize {
            passes,
            min_obs,
            prior
        }
    );
    if let Some(s) = p.distribution {
        a.distribution = unit_enum(&s, "distribution")?;
    }
    Ok(a)
}
fn hodge_build(p: HodgeRankParams) -> Result<HodgeRank, Error> {
    let mut a = merge_params!(p, HodgeRank, scalar { tolerance }, usize { iterations });
    if let Some(s) = p.flow {
        a.flow = unit_enum(&s, "flow")?;
    }
    Ok(a)
}
fn kemeny_build(p: KemenyParams) -> Result<Kemeny, Error> {
    let mut a = merge_params!(p, Kemeny, scalar { seed }, usize { min_obs });
    if let Some(passes) = p.passes {
        a.passes = kemeny_passes(passes);
    }
    if let Some(s) = p.algo {
        a.algo = unit_enum(&s, "algo")?;
    }
    Ok(a)
}
fn lsr_build(p: LsrParams) -> Result<Lsr, Error> {
    let mut a = merge_params!(p, Lsr, scalar { seed }, usize { steps });
    if let Some(s) = p.estimator {
        a.estimator = unit_enum(&s, "estimator")?;
    }
    Ok(a)
}
fn covariate_bt_build(p: CovariateBtParams) -> Result<CovariateBt, Error> {
    let mut a = CovariateBt::new(p.features);
    if let Some(v) = p.l2 {
        a.l2 = v;
    }
    if let Some(v) = p.intercepts {
        a.intercepts = v;
    }
    if let Some(v) = p.iterations {
        a.iterations = v as usize;
    }
    if let Some(v) = p.tolerance {
        a.tolerance = v;
    }
    Ok(a)
}
fn win_rate_build(p: WinRateParams) -> WinRate {
    let mut a = WinRate::default();
    // A bad confidence level silently keeps the default (init cannot fail).
    if let Some(s) = p.confidence
        && let Ok(c) = unit_enum(&s, "confidence")
    {
        a.confidence = c;
    }
    a
}
fn dueling_build(p: DuelingBanditParams) -> DuelingBandit {
    let mut a = DuelingBandit::default();
    if let Some(policy) = p.policy {
        a.policy = dueling_policy(policy);
    }
    if let Some(seed) = p.seed {
        a.seed = seed;
    }
    a
}

impl Guest for Component {
    batch_algo!(
        BradleyTerryMmModel,
        BtmMmMod,
        CoreBtmMm,
        BradleyTerryMmParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        BradleyTerryMmModel,
        BradleyTerryMmModelBorrow<'_>,
        fit_bradley_terry_mm,
        fit_warm_bradley_terry_mm,
        load_bradley_terry_mm,
        btm_mm_build
    );
    nofield_algo!(
        BordaModel,
        BordaMod,
        CoreBorda,
        Borda,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        BordaModel,
        fit_borda,
        load_borda
    );
    batch_algo!(
        BradleyTerryLrModel,
        BtmLrMod,
        CoreBtmLr,
        BradleyTerryLrParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        BradleyTerryLrModel,
        BradleyTerryLrModelBorrow<'_>,
        fit_bradley_terry_lr,
        fit_warm_bradley_terry_lr,
        load_bradley_terry_lr,
        btm_lr_build
    );
    batch_algo!(
        BayesianBradleyTerryModel,
        BayesBtMod,
        CoreBayesBt,
        BayesianBradleyTerryParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        BayesianBradleyTerryModel,
        BayesianBradleyTerryModelBorrow<'_>,
        fit_bayesian_bradley_terry,
        fit_warm_bayesian_bradley_terry,
        load_bayesian_bradley_terry,
        bayes_bt_build
    );
    batch_algo!(
        ColleyModel,
        ColleyMod,
        CoreColley,
        ColleyParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        ColleyModel,
        ColleyModelBorrow<'_>,
        fit_colley,
        fit_warm_colley,
        load_colley,
        colley_build
    );
    batch_algo!(
        MasseyModel,
        MasseyMod,
        CoreMassey,
        MasseyParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        MasseyModel,
        MasseyModelBorrow<'_>,
        fit_massey,
        fit_warm_massey,
        load_massey,
        massey_build
    );
    batch_algo!(
        KeenerModel,
        KeenerMod,
        CoreKeener,
        KeenerParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        KeenerModel,
        KeenerModelBorrow<'_>,
        fit_keener,
        fit_warm_keener,
        load_keener,
        keener_build
    );
    batch_algo!(
        ILsrModel,
        ILsrMod,
        CoreILsr,
        ILsrParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        ILsrModel,
        ILsrModelBorrow<'_>,
        fit_i_lsr,
        fit_warm_i_lsr,
        load_i_lsr,
        i_lsr_build
    );
    batch_algo!(
        NashAveragingModel,
        NashMod,
        CoreNash,
        NashAveragingParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        NashAveragingModel,
        NashAveragingModelBorrow<'_>,
        fit_nash_averaging,
        fit_warm_nash_averaging,
        load_nash_averaging,
        nash_build
    );
    batch_algo!(
        OffenseDefenseModel,
        OffDefMod,
        CoreOffDef,
        OffenseDefenseParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        OffenseDefenseModel,
        OffenseDefenseModelBorrow<'_>,
        fit_offense_defense,
        fit_warm_offense_defense,
        load_offense_defense,
        off_def_build
    );
    batch_algo!(
        RandomWalkerModel,
        RandWalkMod,
        CoreRandWalk,
        RandomWalkerParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        RandomWalkerModel,
        RandomWalkerModelBorrow<'_>,
        fit_random_walker,
        fit_warm_random_walker,
        load_random_walker,
        rand_walk_build
    );
    batch_algo!(
        RankCentralityModel,
        RankCentMod,
        CoreRankCent,
        RankCentralityParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        RankCentralityModel,
        RankCentralityModelBorrow<'_>,
        fit_rank_centrality,
        fit_warm_rank_centrality,
        load_rank_centrality,
        rank_cent_build
    );
    batch_algo!(
        SerialRankModel,
        SerialMod,
        CoreSerial,
        SerialRankParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        SerialRankModel,
        SerialRankModelBorrow<'_>,
        fit_serial_rank,
        fit_warm_serial_rank,
        load_serial_rank,
        serial_build
    );
    batch_algo!(
        ThurstoneMostellerModel,
        ThurstoneMod,
        CoreThurstone,
        ThurstoneMostellerParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        ThurstoneMostellerModel,
        ThurstoneMostellerModelBorrow<'_>,
        fit_thurstone_mosteller,
        fit_warm_thurstone_mosteller,
        load_thurstone_mosteller,
        thurstone_build
    );
    batch_algo!(
        WhrModel,
        WhrMod,
        CoreWhr,
        WhrParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        WhrModel,
        WhrModelBorrow<'_>,
        fit_whr,
        fit_warm_whr,
        load_whr,
        whr_build
    );
    nofield_algo!(
        CopelandModel,
        CopelandMod,
        CoreCopeland,
        Copeland,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        CopelandModel,
        fit_copeland,
        load_copeland
    );
    batch_algo!(
        BladeChestModel,
        BladeChestMod,
        CoreBladeChest,
        BladeChestParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        BladeChestModel,
        BladeChestModelBorrow<'_>,
        fit_blade_chest,
        fit_warm_blade_chest,
        load_blade_chest,
        blade_chest_build
    );
    batch_algo!(
        EsRumModel,
        EsRumMod,
        CoreEsRum,
        EsRumParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        EsRumModel,
        EsRumModelBorrow<'_>,
        fit_es_rum,
        fit_warm_es_rum,
        load_es_rum,
        es_rum_build
    );
    batch_algo!(
        HodgeRankModel,
        HodgeMod,
        CoreHodge,
        HodgeRankParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        HodgeRankModel,
        HodgeRankModelBorrow<'_>,
        fit_hodge_rank,
        fit_warm_hodge_rank,
        load_hodge_rank,
        hodge_build
    );
    batch_algo!(
        KemenyModel,
        KemenyMod,
        CoreKemeny,
        KemenyParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        KemenyModel,
        KemenyModelBorrow<'_>,
        fit_kemeny,
        fit_warm_kemeny,
        load_kemeny,
        kemeny_build
    );
    batch_algo!(
        LsrModel,
        LsrMod,
        CoreLsr,
        LsrParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        LsrModel,
        LsrModelBorrow<'_>,
        fit_lsr,
        fit_warm_lsr,
        load_lsr,
        lsr_build
    );
    batch_algo!(
        CovariateBtModel,
        CovariateBtMod,
        CoreCovariateBt,
        CovariateBtParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        CovariateBtModel,
        CovariateBtModelBorrow<'_>,
        fit_covariate_bt,
        fit_warm_covariate_bt,
        load_covariate_bt,
        covariate_bt_build
    );
    online_algo!(
        WinRateModel,
        WinRateMod,
        CoreWinRate,
        WinRate,
        WinRateParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        WinRateModel,
        init_win_rate,
        fit_win_rate,
        load_win_rate,
        win_rate_build
    );
    online_algo!(
        DuelingBanditModel,
        DuelingMod,
        CoreDueling,
        DuelingBandit,
        DuelingBanditParams,
        PairwiseData,
        PairwiseDatasetBorrow<'_>,
        DuelingBanditModel,
        init_dueling_bandit,
        fit_dueling_bandit,
        load_dueling_bandit,
        dueling_build
    );
}
