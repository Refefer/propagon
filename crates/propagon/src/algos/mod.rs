//! Algorithm implementations.
//!
//! Each module pairs an algorithm struct (its public fields are the params,
//! `Default` gives sensible values) with an owned, serializable model type.
//! See `docs/algorithms.md` for the survey entry behind each method and
//! `docs/PRD.md` FR-5 for which algorithms support incremental update vs
//! warm-start refitting.

mod bandits;
mod bayes_bt;
mod behavior_cloning;
mod birank;
mod blade_chest;
mod bootstrap;
mod borda;
mod bt_lr;
mod bt_mm;
mod colley;
mod common;
mod components;
mod copeland;
mod covariate_bt;
mod crowd_bt;
mod de;
mod degree;
mod devig;
mod diagnostics;
mod dueling;
mod elo;
mod elo_mov;
mod esrum;
mod footrule;
mod generalized_bt;
mod glicko2;
mod harmonic;
mod hits;
mod hodge;
mod ilsr;
mod katz;
mod kcore;
mod keener;
mod kelly;
mod kemeny;
mod leader_rank;
mod lin_ucb;
mod lmsr;
mod lsr;
mod mallows;
mod massey;
mod mc4;
mod mc_value;
mod melo;
mod nash;
mod offense_defense;
mod opinion_pool;
mod pagerank;
mod plackett_luce;
mod random_walker;
mod rank_centrality;
mod rate;
mod serial_rank;
mod sw_ucb;
mod td_value;
mod team_bt;
mod thurstone;
mod value_compare;
mod weng_lin;
mod whr;

pub use bandits::{Bandit, BanditModel, BanditPolicy};
pub use bayes_bt::{BayesBtModel, BayesianBradleyTerry};
pub use behavior_cloning::{BcModel, BehaviorCloning, Granularity};
pub use birank::{BiRank, BiRankModel};
pub use blade_chest::{BladeChest, BladeChestModel, BladeChestVariant};
pub use bootstrap::{Bootstrap, BootstrapModel};
pub use borda::{Borda, BordaModel};
pub use bt_lr::{BradleyTerryLR, BtmLrModel};
pub use bt_mm::{BradleyTerryMM, BtmMmModel, Section, SectionKind};
pub use colley::{Colley, ColleyModel};
pub use components::extract_components;
pub use copeland::{Copeland, CopelandModel};
pub use covariate_bt::{CovariateBt, CovariateBtModel};
pub use crowd_bt::{CrowdBt, CrowdBtModel};
pub use de::{DifferentialEvolution, Fitness};
pub use degree::{Degree, DegreeModel, Direction};
pub use devig::{AdditiveClamp, DevigMethod, OddsDevig, OddsDevigModel};
pub use diagnostics::{
    CalibrationBin, brier_score, calibration_table, closing_line_value, log_loss, log_loss_eps,
};
pub use dueling::{DuelingBandit, DuelingModel, DuelingPolicy};
pub use elo::{Elo, EloModel};
pub use elo_mov::{MovElo, MovEloModel};
pub use esrum::{EsRum, EsRumModel, RumDistribution};
pub use footrule::{Footrule, FootruleModel};
pub use generalized_bt::{GeneralizedBt, GeneralizedBtModel, HomeAdvantage, TieModel};
pub use glicko2::{Glicko2, Glicko2Model, PlayerState};
pub use harmonic::{EdgeCost, Harmonic, HarmonicModel, SourceBudget};
pub use hits::{Hits, HitsModel};
pub use hodge::{HodgeFlow, HodgeModel, HodgeRank};
pub use ilsr::{ILsr, ILsrModel};
pub use katz::{Katz, KatzModel};
pub use kcore::{KCore, KCoreModel};
pub use keener::{Keener, KeenerModel};
pub use kelly::{
    MAX_PORTFOLIO_BETS, Opportunity, fractional_kelly, kelly_fraction, portfolio_kelly,
};
pub use kemeny::{Kemeny, KemenyAlgo, KemenyModel, KemenyPasses};
pub use leader_rank::{LeaderRank, LeaderRankModel};
pub use lin_ucb::{LinUcb, LinUcbModel};
pub use lmsr::{Lmsr, LmsrModel};
pub use lsr::{Estimator, Lsr, LsrModel};
pub use mallows::{Mallows, MallowsModel};
pub use massey::{Massey, MasseyModel};
pub use mc_value::{Aggregate, McValue, McValueModel, Visit, Winsorize};
pub use mc4::{Mc4, Mc4Model};
pub use melo::{MElo, MEloModel};
pub use nash::{NashAveraging, NashAveragingModel};
pub use offense_defense::{OffenseDefense, OffenseDefenseModel};
pub use opinion_pool::{Missing, OpinionPool, OpinionPoolModel, PoolKind};
pub use pagerank::{PageRank, PageRankModel, Sink, Teleport};
pub use plackett_luce::{PlackettLuce, PlackettLuceModel};
pub use random_walker::{RandomWalker, RandomWalkerModel};
pub use rank_centrality::{RankCentrality, RankCentralityModel};
pub use rate::{Confidence, WinRate, WinRateModel, wilson_interval};
pub use serial_rank::{SerialRank, SerialRankModel};
pub use sw_ucb::{SlidingWindowUcb, SwUcbModel};
pub use td_value::{TdValue, TdValueModel};
pub use team_bt::{TeamAggregate, TeamBradleyTerry, TeamBtModel};
pub use thurstone::{ThurstoneModel, ThurstoneMosteller};
pub use value_compare::{PairwiseTests, ResampleScheme, ValueCompare, ValueCompareModel};
pub use weng_lin::{GammaPolicy, Rating, WengLin, WengLinModel, WengLinVariant};
pub use whr::{Whr, WhrModel};
