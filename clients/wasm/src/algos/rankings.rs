//! Algorithms over `rankings-dataset`: Plackett-Luce, Footrule, Mallows, MC4.

use propagon::algos::{
    Footrule, FootruleModel as CoreFootrule, Mallows, MallowsModel as CoreMallows, Mc4,
    Mc4Model as CoreMc4, PlackettLuce, PlackettLuceModel as CorePlackettLuce,
};

use crate::Component;
use crate::datasets::RankingsData;
use crate::enums::kemeny_passes;
use crate::wit::datasets::RankingsDatasetBorrow;
use crate::wit::rankings::{
    FootruleModel, Guest, GuestFootruleModel, GuestMallowsModel, GuestMc4Model,
    GuestPlackettLuceModel, MallowsModel, MallowsModelBorrow, MallowsParams, Mc4Model,
    Mc4ModelBorrow, Mc4Params, PlackettLuceModel, PlackettLuceModelBorrow, PlackettLuceParams,
};
use crate::wit::types::Error;

batch_model!(PlackettLuceMod, GuestPlackettLuceModel, CorePlackettLuce);
batch_model!(FootruleMod, GuestFootruleModel, CoreFootrule);
batch_model!(MallowsMod, GuestMallowsModel, CoreMallows);
batch_model!(Mc4Mod, GuestMc4Model, CoreMc4);

fn plackett_luce_build(p: PlackettLuceParams) -> Result<PlackettLuce, Error> {
    Ok(merge_params!(
        p,
        PlackettLuce,
        scalar { tolerance },
        usize { iterations }
    ))
}
fn mallows_build(p: MallowsParams) -> Result<Mallows, Error> {
    let mut m = Mallows::default();
    if let Some(passes) = p.passes {
        m.passes = kemeny_passes(passes);
    }
    if let Some(seed) = p.seed {
        m.seed = seed;
    }
    Ok(m)
}
fn mc4_build(p: Mc4Params) -> Result<Mc4, Error> {
    Ok(merge_params!(
        p,
        Mc4,
        scalar { damping, tolerance },
        usize { iterations }
    ))
}

impl Guest for Component {
    batch_algo!(
        PlackettLuceModel,
        PlackettLuceMod,
        CorePlackettLuce,
        PlackettLuceParams,
        RankingsData,
        RankingsDatasetBorrow<'_>,
        PlackettLuceModel,
        PlackettLuceModelBorrow<'_>,
        fit_plackett_luce,
        fit_warm_plackett_luce,
        load_plackett_luce,
        plackett_luce_build
    );
    nofield_algo!(
        FootruleModel,
        FootruleMod,
        CoreFootrule,
        Footrule,
        RankingsData,
        RankingsDatasetBorrow<'_>,
        FootruleModel,
        fit_footrule,
        load_footrule
    );
    batch_algo!(
        MallowsModel,
        MallowsMod,
        CoreMallows,
        MallowsParams,
        RankingsData,
        RankingsDatasetBorrow<'_>,
        MallowsModel,
        MallowsModelBorrow<'_>,
        fit_mallows,
        fit_warm_mallows,
        load_mallows,
        mallows_build
    );
    batch_algo!(
        Mc4Model,
        Mc4Mod,
        CoreMc4,
        Mc4Params,
        RankingsData,
        RankingsDatasetBorrow<'_>,
        Mc4Model,
        Mc4ModelBorrow<'_>,
        fit_mc4,
        fit_warm_mc4,
        load_mc4,
        mc4_build
    );
}
