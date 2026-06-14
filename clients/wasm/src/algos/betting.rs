//! Betting & portfolio bindings (§14): de-vigging over `odds-dataset`, opinion
//! pools over `forecast-dataset`, the LMSR market (online) over
//! `market-dataset`, plus Kelly staking and calibration diagnostics as free
//! functions on the `betting` interface.

use propagon::algos::{
    Lmsr, LmsrModel as CoreLmsr, OddsDevig, OddsDevigModel as CoreDevig, OpinionPool,
    OpinionPoolModel as CorePool, Opportunity,
};

use crate::Component;
use crate::datasets::{ForecastData, MarketData, OddsData};
use crate::enums::unit_enum;
use crate::errors::MapWit;
use crate::wit::betting::*;
use crate::wit::datasets::{ForecastDatasetBorrow, MarketDatasetBorrow, OddsDatasetBorrow};

batch_model!(DevigMod, GuestOddsDevigModel, CoreDevig, extras {
    fn n_events(&self) -> u32 {
        self.0.n_events() as u32
    }
    fn booksum(&self, event: u32) -> Option<f64> {
        self.0.booksum(event as usize)
    }
    fn insider_share(&self, event: u32) -> Option<f64> {
        self.0.insider_share(event as usize)
    }
});

batch_model!(PoolMod, GuestOpinionPoolModel, CorePool);

online_model!(
    LmsrMod, GuestLmsrModel, Lmsr, CoreLmsr, MarketData, MarketDatasetBorrow<'_>,
    extras {
        fn price(&self, outcome: String) -> Option<f64> {
            self.model.borrow().price(&outcome)
        }
        fn cost(&self) -> f64 {
            self.model.borrow().cost()
        }
        fn trade_cost(&self, outcome: String, shares: f64) -> Option<f64> {
            self.model.borrow().trade_cost(&outcome, shares)
        }
    }
);

fn devig_build(p: OddsDevigParams) -> Result<OddsDevig, crate::wit::types::Error> {
    let mut a = merge_params!(p, OddsDevig, scalar { tolerance });
    if let Some(s) = p.method {
        a.method = unit_enum(&s, "method")?;
    }
    if let Some(s) = p.additive_clamp {
        a.additive_clamp = unit_enum(&s, "additive-clamp")?;
    }
    Ok(a)
}

fn pool_build(p: OpinionPoolParams) -> Result<OpinionPool, crate::wit::types::Error> {
    let mut a = merge_params!(p, OpinionPool, scalar { extremize });
    if let Some(s) = p.kind {
        a.kind = unit_enum(&s, "kind")?;
    }
    if let Some(s) = p.missing {
        a.missing = unit_enum(&s, "missing")?;
    }
    if let Some(e) = p.eps_floor {
        a.eps_floor = Some(e);
    }
    Ok(a)
}

fn lmsr_build(p: LmsrParams) -> Lmsr {
    merge_params!(p, Lmsr, scalar { b })
}

impl Guest for Component {
    batch_algo!(
        OddsDevigModel,
        DevigMod,
        CoreDevig,
        OddsDevigParams,
        OddsData,
        OddsDatasetBorrow<'_>,
        OddsDevigModel,
        OddsDevigModelBorrow<'_>,
        fit_odds_devig,
        fit_warm_odds_devig,
        load_odds_devig,
        devig_build
    );

    batch_algo!(
        OpinionPoolModel,
        PoolMod,
        CorePool,
        OpinionPoolParams,
        ForecastData,
        ForecastDatasetBorrow<'_>,
        OpinionPoolModel,
        OpinionPoolModelBorrow<'_>,
        fit_opinion_pool,
        fit_warm_opinion_pool,
        load_opinion_pool,
        pool_build
    );

    online_algo!(
        LmsrModel,
        LmsrMod,
        CoreLmsr,
        Lmsr,
        LmsrParams,
        MarketData,
        MarketDatasetBorrow<'_>,
        LmsrModel,
        init_lmsr,
        fit_lmsr,
        load_lmsr,
        lmsr_build
    );

    // --- §14.4 Kelly (free functions) ---
    fn kelly_fraction(p: f64, b: f64) -> Result<f64, crate::wit::types::Error> {
        propagon::algos::kelly_fraction(p, b).map_wit()
    }
    fn fractional_kelly(p: f64, b: f64, lambda: f64) -> Result<f64, crate::wit::types::Error> {
        propagon::algos::fractional_kelly(p, b, lambda).map_wit()
    }
    fn portfolio_kelly(
        opportunities: Vec<(f64, f64)>,
    ) -> Result<Vec<f64>, crate::wit::types::Error> {
        let opps: Vec<Opportunity> = opportunities
            .into_iter()
            .map(|(p, b)| Opportunity { p, b })
            .collect();
        propagon::algos::portfolio_kelly(&opps).map_wit()
    }

    // --- §14.5 diagnostics (free functions) ---
    fn brier_score(
        forecasts: Vec<f64>,
        outcomes: Vec<bool>,
    ) -> Result<f64, crate::wit::types::Error> {
        propagon::algos::brier_score(&forecasts, &outcomes).map_wit()
    }
    fn log_loss(forecasts: Vec<f64>, outcomes: Vec<bool>) -> Result<f64, crate::wit::types::Error> {
        propagon::algos::log_loss(&forecasts, &outcomes).map_wit()
    }
    fn closing_line_value(taken: f64, closing: f64) -> Result<f64, crate::wit::types::Error> {
        propagon::algos::closing_line_value(taken, closing).map_wit()
    }
    fn calibration_table(
        implied: Vec<f64>,
        outcomes: Vec<bool>,
        n_buckets: u32,
    ) -> Result<Vec<CalibrationBin>, crate::wit::types::Error> {
        let table = propagon::algos::calibration_table(&implied, &outcomes, n_buckets as usize)
            .map_wit()?;
        Ok(table
            .into_iter()
            .map(|b| CalibrationBin {
                lo: b.lo,
                hi: b.hi,
                mean_pred: b.mean_pred,
                realized_freq: b.realized_freq,
                count: b.count as u32,
            })
            .collect())
    }
}
