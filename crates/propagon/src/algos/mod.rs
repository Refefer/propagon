//! Algorithm implementations.
//!
//! Each module pairs an algorithm struct (its public fields are the params,
//! `Default` gives sensible values) with an owned, serializable model type.
//! See `docs/algorithms.md` for the survey entry behind each method and
//! `docs/PRD.md` FR-5 for which algorithms support incremental update vs
//! warm-start refitting.

mod bandits;
mod bayes_bt;
mod birank;
mod borda;
mod bt_lr;
mod bt_mm;
mod colley;
mod common;
mod components;
mod copeland;
mod de;
mod elo;
mod esrum;
mod glicko2;
mod hodge;
mod keener;
mod kemeny;
mod lsr;
mod massey;
mod mc4;
mod pagerank;
mod plackett_luce;
mod rank_centrality;
mod rate;

pub use bandits::{Bandit, BanditModel, BanditPolicy};
pub use bayes_bt::{BayesBtModel, BayesianBradleyTerry};
pub use birank::{BiRank, BiRankModel};
pub use borda::{Borda, BordaModel};
pub use bt_lr::{BradleyTerryLR, BtmLrModel};
pub use bt_mm::{BradleyTerryMM, BtmMmModel, Section, SectionKind};
pub use colley::{Colley, ColleyModel};
pub use components::extract_components;
pub use copeland::{Copeland, CopelandModel};
pub use de::{DifferentialEvolution, Fitness};
pub use elo::{Elo, EloModel};
pub use esrum::{EsRum, EsRumModel, RumDistribution};
pub use glicko2::{Glicko2, Glicko2Model, PlayerState};
pub use hodge::{HodgeFlow, HodgeModel, HodgeRank};
pub use keener::{Keener, KeenerModel};
pub use kemeny::{Kemeny, KemenyAlgo, KemenyModel, KemenyPasses};
pub use lsr::{Estimator, Lsr, LsrModel};
pub use massey::{Massey, MasseyModel};
pub use mc4::{Mc4, Mc4Model};
pub use pagerank::{PageRank, PageRankModel, Sink};
pub use plackett_luce::{PlackettLuce, PlackettLuceModel};
pub use rank_centrality::{RankCentrality, RankCentralityModel};
pub use rate::{Confidence, WinRate, WinRateModel, wilson_interval};
