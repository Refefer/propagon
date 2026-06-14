//! Algorithms over `games-dataset`: Elo, Glicko-2, margin-of-victory Elo and
//! multidimensional Elo (online), plus the tie/team-aware batch Bradley-Terry
//! variants.

use propagon::algos::{
    Elo, EloModel as CoreElo, GeneralizedBt, GeneralizedBtModel as CoreGenBt, Glicko2,
    Glicko2Model as CoreGlicko2, MElo, MEloModel as CoreMElo, MovElo, MovEloModel as CoreMovElo,
    TeamBradleyTerry, TeamBtModel as CoreTeamBt,
};

use crate::Component;
use crate::datasets::GamesData;
use crate::enums::unit_enum;
use crate::wit::datasets::GamesDatasetBorrow;
use crate::wit::games::{
    EloModel, EloParams, GeneralizedBtModel, GeneralizedBtModelBorrow, GeneralizedBtParams,
    Glicko2Model, Glicko2Params, Guest, GuestEloModel, GuestGeneralizedBtModel, GuestGlicko2Model,
    GuestMEloModel, GuestMovEloModel, GuestTeamBradleyTerryModel, MEloModel, MEloParams,
    MovEloModel, MovEloParams, PlayerState, TeamBradleyTerryModel, TeamBradleyTerryModelBorrow,
    TeamBradleyTerryParams,
};
use crate::wit::types::Error;

online_model!(
    EloMod,
    GuestEloModel,
    Elo,
    CoreElo,
    GamesData,
    GamesDatasetBorrow<'_>
);
online_model!(
    Glicko2Mod, GuestGlicko2Model, Glicko2, CoreGlicko2, GamesData, GamesDatasetBorrow<'_>,
    extras {
        fn players(&self) -> Vec<(String, PlayerState)> {
            self.model
                .borrow()
                .players()
                .map(|(n, p)| (n.to_string(), PlayerState { r: p.r, rd: p.rd, sigma: p.sigma }))
                .collect()
        }
    }
);
online_model!(
    MovEloMod,
    GuestMovEloModel,
    MovElo,
    CoreMovElo,
    GamesData,
    GamesDatasetBorrow<'_>
);
online_model!(
    MEloMod,
    GuestMEloModel,
    MElo,
    CoreMElo,
    GamesData,
    GamesDatasetBorrow<'_>
);
batch_model!(GenBtMod, GuestGeneralizedBtModel, CoreGenBt);
batch_model!(TeamBtMod, GuestTeamBradleyTerryModel, CoreTeamBt);

fn elo_build(p: EloParams) -> Elo {
    merge_params!(
        p,
        Elo,
        scalar {
            k,
            initial_rating,
            scale
        }
    )
}
fn glicko2_build(p: Glicko2Params) -> Glicko2 {
    merge_params!(
        p,
        Glicko2,
        scalar {
            tau,
            rating,
            rd,
            sigma
        }
    )
}
fn mov_elo_build(p: MovEloParams) -> MovElo {
    merge_params!(
        p,
        MovElo,
        scalar {
            k,
            initial_rating,
            scale,
            mov_exponent
        }
    )
}
fn m_elo_build(p: MEloParams) -> MElo {
    merge_params!(
        p,
        MElo,
        scalar {
            lr_rating,
            lr_vector,
            initial_rating,
            init_scale,
            seed
        },
        usize { k }
    )
}
fn gen_bt_build(p: GeneralizedBtParams) -> Result<GeneralizedBt, Error> {
    let mut a = merge_params!(p, GeneralizedBt, scalar { tolerance }, usize { iterations });
    if let Some(s) = p.ties {
        a.ties = unit_enum(&s, "ties")?;
    }
    if let Some(s) = p.home {
        a.home = unit_enum(&s, "home")?;
    }
    Ok(a)
}
fn team_bt_build(p: TeamBradleyTerryParams) -> Result<TeamBradleyTerry, Error> {
    let mut a = merge_params!(
        p,
        TeamBradleyTerry,
        scalar { tolerance },
        usize { iterations }
    );
    if let Some(s) = p.aggregate {
        a.aggregate = unit_enum(&s, "aggregate")?;
    }
    if let Some(s) = p.ties {
        a.ties = unit_enum(&s, "ties")?;
    }
    Ok(a)
}

impl Guest for Component {
    online_algo!(
        EloModel,
        EloMod,
        CoreElo,
        Elo,
        EloParams,
        GamesData,
        GamesDatasetBorrow<'_>,
        EloModel,
        init_elo,
        fit_elo,
        load_elo,
        elo_build
    );
    online_algo!(
        Glicko2Model,
        Glicko2Mod,
        CoreGlicko2,
        Glicko2,
        Glicko2Params,
        GamesData,
        GamesDatasetBorrow<'_>,
        Glicko2Model,
        init_glicko2,
        fit_glicko2,
        load_glicko2,
        glicko2_build
    );
    online_algo!(
        MovEloModel,
        MovEloMod,
        CoreMovElo,
        MovElo,
        MovEloParams,
        GamesData,
        GamesDatasetBorrow<'_>,
        MovEloModel,
        init_mov_elo,
        fit_mov_elo,
        load_mov_elo,
        mov_elo_build
    );
    online_algo!(
        MEloModel,
        MEloMod,
        CoreMElo,
        MElo,
        MEloParams,
        GamesData,
        GamesDatasetBorrow<'_>,
        MEloModel,
        init_m_elo,
        fit_m_elo,
        load_m_elo,
        m_elo_build
    );
    batch_algo!(
        GeneralizedBtModel,
        GenBtMod,
        CoreGenBt,
        GeneralizedBtParams,
        GamesData,
        GamesDatasetBorrow<'_>,
        GeneralizedBtModel,
        GeneralizedBtModelBorrow<'_>,
        fit_generalized_bt,
        fit_warm_generalized_bt,
        load_generalized_bt,
        gen_bt_build
    );
    batch_algo!(
        TeamBradleyTerryModel,
        TeamBtMod,
        CoreTeamBt,
        TeamBradleyTerryParams,
        GamesData,
        GamesDatasetBorrow<'_>,
        TeamBradleyTerryModel,
        TeamBradleyTerryModelBorrow<'_>,
        fit_team_bradley_terry,
        fit_warm_team_bradley_terry,
        load_team_bradley_terry,
        team_bt_build
    );
}
