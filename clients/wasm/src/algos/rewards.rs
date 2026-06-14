//! Algorithms over reward logs: the multi-armed Bandit and sliding-window UCB
//! (over `rewards-dataset`) and LinUCB (over `contextual-rewards-dataset`). All
//! online.

use propagon::algos::{
    Bandit, BanditModel as CoreBandit, LinUcb, LinUcbModel as CoreLinUcb, SlidingWindowUcb,
    SwUcbModel as CoreSwUcb,
};

use crate::Component;
use crate::datasets::{ContextualRewardsData, RewardsData};
use crate::enums::bandit_policy;
use crate::wit::datasets::{ContextualRewardsDatasetBorrow, RewardsDatasetBorrow};
use crate::wit::rewards::{
    BanditModel, BanditParams, Guest, GuestBanditModel, GuestLinUcbModel,
    GuestSlidingWindowUcbModel, LinUcbModel, LinUcbParams, SlidingWindowUcbModel,
    SlidingWindowUcbParams,
};

online_model!(
    BanditMod,
    GuestBanditModel,
    Bandit,
    CoreBandit,
    RewardsData,
    RewardsDatasetBorrow<'_>
);
online_model!(
    SwUcbMod,
    GuestSlidingWindowUcbModel,
    SlidingWindowUcb,
    CoreSwUcb,
    RewardsData,
    RewardsDatasetBorrow<'_>
);
online_model!(
    LinUcbMod,
    GuestLinUcbModel,
    LinUcb,
    CoreLinUcb,
    ContextualRewardsData,
    ContextualRewardsDatasetBorrow<'_>
);

fn bandit_build(p: BanditParams) -> Bandit {
    let mut b = Bandit::default();
    if let Some(policy) = p.policy {
        b.policy = bandit_policy(policy);
    }
    if let Some(seed) = p.seed {
        b.seed = seed;
    }
    b
}
fn sw_ucb_build(p: SlidingWindowUcbParams) -> SlidingWindowUcb {
    merge_params!(
        p,
        SlidingWindowUcb,
        scalar { exploration },
        usize { window }
    )
}
fn lin_ucb_build(p: LinUcbParams) -> LinUcb {
    merge_params!(p, LinUcb, scalar { alpha, ridge })
}

impl Guest for Component {
    online_algo!(
        BanditModel,
        BanditMod,
        CoreBandit,
        Bandit,
        BanditParams,
        RewardsData,
        RewardsDatasetBorrow<'_>,
        BanditModel,
        init_bandit,
        fit_bandit,
        load_bandit,
        bandit_build
    );
    online_algo!(
        SlidingWindowUcbModel,
        SwUcbMod,
        CoreSwUcb,
        SlidingWindowUcb,
        SlidingWindowUcbParams,
        RewardsData,
        RewardsDatasetBorrow<'_>,
        SlidingWindowUcbModel,
        init_sliding_window_ucb,
        fit_sliding_window_ucb,
        load_sliding_window_ucb,
        sw_ucb_build
    );
    online_algo!(
        LinUcbModel,
        LinUcbMod,
        CoreLinUcb,
        LinUcb,
        LinUcbParams,
        ContextualRewardsData,
        ContextualRewardsDatasetBorrow<'_>,
        LinUcbModel,
        init_lin_ucb,
        fit_lin_ucb,
        load_lin_ucb,
        lin_ucb_build
    );
}
