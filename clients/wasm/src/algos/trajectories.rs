//! Algorithms over `trajectories-dataset`: Monte-Carlo value, behavior cloning,
//! value comparison (batch), and TD value (online).

use propagon::algos::{
    BcModel as CoreBc, BehaviorCloning, McValue, McValueModel as CoreMcValue, TdValue,
    TdValueModel as CoreTdValue, ValueCompare, ValueCompareModel as CoreValueCompare,
};

use crate::Component;
use crate::datasets::TrajectoriesData;
use crate::enums::{granularity, pairwise_tests, unit_enum, winsorize};
use crate::wit::datasets::TrajectoriesDatasetBorrow;
use crate::wit::trajectories::{
    BehaviorCloningModel, BehaviorCloningModelBorrow, BehaviorCloningParams, Guest,
    GuestBehaviorCloningModel, GuestMcValueModel, GuestTdValueModel, GuestValueCompareModel,
    McValueModel, McValueModelBorrow, McValueParams, TdValueModel, TdValueParams,
    ValueCompareModel, ValueCompareModelBorrow, ValueCompareParams,
};
use crate::wit::types::Error;

batch_model!(McValueMod, GuestMcValueModel, CoreMcValue);
batch_model!(BcMod, GuestBehaviorCloningModel, CoreBc);
batch_model!(ValueCompareMod, GuestValueCompareModel, CoreValueCompare);
online_model!(
    TdValueMod,
    GuestTdValueModel,
    TdValue,
    CoreTdValue,
    TrajectoriesData,
    TrajectoriesDatasetBorrow<'_>
);

fn mc_value_build(p: McValueParams) -> Result<McValue, Error> {
    let mut a = merge_params!(p, McValue, scalar { gamma }, usize { min_observations });
    if let Some(s) = p.visit {
        a.visit = unit_enum(&s, "visit")?;
    }
    if let Some(s) = p.aggregate {
        a.aggregate = unit_enum(&s, "aggregate")?;
    }
    if let Some(w) = p.winsorize {
        a.winsorize = winsorize(w);
    }
    Ok(a)
}
fn bc_build(p: BehaviorCloningParams) -> Result<BehaviorCloning, Error> {
    let mut a = merge_params!(p, BehaviorCloning, scalar { smoothing });
    if let Some(g) = p.granularity {
        a.granularity = granularity(g);
    }
    Ok(a)
}
fn value_compare_build(p: ValueCompareParams) -> Result<ValueCompare, Error> {
    let mut a = merge_params!(
        p,
        ValueCompare,
        scalar {
            gamma,
            credible,
            seed
        },
        usize {
            replicates,
            min_observations
        }
    );
    if let Some(s) = p.visit {
        a.visit = unit_enum(&s, "visit")?;
    }
    if let Some(s) = p.method {
        a.method = unit_enum(&s, "method")?;
    }
    if let Some(t) = p.pairwise {
        a.pairwise = pairwise_tests(t);
    }
    Ok(a)
}
fn td_value_build(p: TdValueParams) -> TdValue {
    merge_params!(
        p,
        TdValue,
        scalar {
            alpha,
            gamma,
            initial_value
        },
        usize { passes }
    )
}

impl Guest for Component {
    batch_algo!(
        McValueModel,
        McValueMod,
        CoreMcValue,
        McValueParams,
        TrajectoriesData,
        TrajectoriesDatasetBorrow<'_>,
        McValueModel,
        McValueModelBorrow<'_>,
        fit_mc_value,
        fit_warm_mc_value,
        load_mc_value,
        mc_value_build
    );
    batch_algo!(
        BehaviorCloningModel,
        BcMod,
        CoreBc,
        BehaviorCloningParams,
        TrajectoriesData,
        TrajectoriesDatasetBorrow<'_>,
        BehaviorCloningModel,
        BehaviorCloningModelBorrow<'_>,
        fit_behavior_cloning,
        fit_warm_behavior_cloning,
        load_behavior_cloning,
        bc_build
    );
    batch_algo!(
        ValueCompareModel,
        ValueCompareMod,
        CoreValueCompare,
        ValueCompareParams,
        TrajectoriesData,
        TrajectoriesDatasetBorrow<'_>,
        ValueCompareModel,
        ValueCompareModelBorrow<'_>,
        fit_value_compare,
        fit_warm_value_compare,
        load_value_compare,
        value_compare_build
    );
    online_algo!(
        TdValueModel,
        TdValueMod,
        CoreTdValue,
        TdValue,
        TdValueParams,
        TrajectoriesData,
        TrajectoriesDatasetBorrow<'_>,
        TdValueModel,
        init_td_value,
        fit_td_value,
        load_td_value,
        td_value_build
    );
}
