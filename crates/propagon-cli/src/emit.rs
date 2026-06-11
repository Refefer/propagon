//! Output formatting. `tsv` reproduces v1's `"{id}: {value}"` shape
//! (including the per-algorithm special formats); `jsonl` streams the
//! model's state-file form to stdout.

use std::io::Write;

use propagon::algos::{
    BayesBtModel, BiRankModel, BtmMmModel, CrowdBtModel, Glicko2Model, PlackettLuceModel,
    SectionKind, WengLinModel,
};
use propagon::{RankModel, Result};

/// `"{id}: {score}"` per line, sorted descending (ties by name).
pub fn scores<M: RankModel>(out: &mut impl Write, model: &M) -> Result<()> {
    for (name, score) in model.sorted_scores() {
        writeln!(out, "{name}: {score}")?;
    }
    Ok(())
}

/// v1 glicko2 format: `mu, rd, lower, upper` at 4 decimals (or mu alone),
/// sorted by mu descending (v2 sorts numerically; v1 sorted formatted
/// strings, which misordered negatives).
pub fn glicko2(out: &mut impl Write, model: &Glicko2Model, use_mu: bool) -> Result<()> {
    let mut players: Vec<_> = model.players().collect();
    players.sort_by(|a, b| b.1.r.total_cmp(&a.1.r).then_with(|| a.0.cmp(b.0)));
    for (name, p) in players {
        if use_mu {
            writeln!(out, "{name}: {}", model.mu(p))?;
        } else {
            let (lo, hi) = p.bounds();
            writeln!(
                out,
                "{name}: {:.4}\t{:.4}\t{:.4}\t{:.4}",
                model.mu(p),
                p.rd,
                lo,
                hi
            )?;
        }
    }
    Ok(())
}

/// v1 es-rum format: `mu sigma` at 6 decimals, sorted by mu descending
/// (v1 emitted in hash order; v2 sorts for determinism).
pub fn esrum(out: &mut impl Write, model: &propagon::algos::EsRumModel) -> Result<()> {
    let mut rows: Vec<_> = model.distributions().collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (name, mu, sigma) in rows {
        writeln!(out, "{name}: {mu:.6} {sigma:.6}")?;
    }
    Ok(())
}

/// v1 btm-mm format: ranked sections in order, then undefeated, then
/// winless, separated by blank lines (trailing sections always printed,
/// even when empty — exactly v1's print sequence).
pub fn btm_mm(out: &mut impl Write, model: &BtmMmModel) -> Result<()> {
    let mut first_section = true;
    for section in model.sections() {
        let ranked = section.kind == SectionKind::Ranked;
        if !first_section {
            writeln!(out)?;
        }
        first_section = false;
        for &(id, score) in &section.entries {
            let name = model.name(id).unwrap_or("<unresolved>");
            let _ = ranked;
            writeln!(out, "{name}: {score}")?;
        }
    }
    Ok(())
}

/// Bayesian BT: `mean lo hi` per entity at 6 decimals, sorted by posterior
/// mean descending.
pub fn bayes_bt(out: &mut impl Write, model: &BayesBtModel) -> Result<()> {
    let mut rows: Vec<_> = model.posteriors().collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (name, mean, lo, hi) in rows {
        writeln!(out, "{name}: {mean:.6} {lo:.6} {hi:.6}")?;
    }
    Ok(())
}

/// Weng-Lin: `mu sigma ordinal` per player at 4 decimals (ordinal =
/// mu - 3 sigma, the conservative openskill convention), sorted by mu.
pub fn weng_lin(out: &mut impl Write, model: &WengLinModel) -> Result<()> {
    let mut rows: Vec<_> = model.ratings().collect();
    rows.sort_by(|a, b| b.1.mu.total_cmp(&a.1.mu).then_with(|| a.0.cmp(b.0)));
    for (name, r) in rows {
        writeln!(
            out,
            "{name}: {:.4}\t{:.4}\t{:.4}",
            r.mu,
            r.sigma,
            r.ordinal(3.0)
        )?;
    }
    Ok(())
}

/// Crowd-BT: entity log-strengths, blank line, then annotator
/// reliabilities, each sorted descending.
pub fn crowd_bt(out: &mut impl Write, model: &CrowdBtModel) -> Result<()> {
    let mut entities: Vec<_> = model.scores().collect();
    entities.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (name, s) in entities {
        writeln!(out, "{name}: {s}")?;
    }

    writeln!(out)?;
    let mut annotators: Vec<_> = model.annotators().collect();
    annotators.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (name, eta) in annotators {
        writeln!(out, "{name}: {eta:.4}")?;
    }
    Ok(())
}

/// Plackett-Luce sections in btm style: ranked components first, then
/// undefeated, then winless, blank-line separated.
pub fn plackett_luce(out: &mut impl Write, model: &PlackettLuceModel) -> Result<()> {
    let mut first = true;
    for section in model.sections() {
        if !first {
            writeln!(out)?;
        }
        first = false;

        for &(id, score) in &section.entries {
            let name = model.name(id).unwrap_or("<unresolved>");
            writeln!(out, "{name}: {score}")?;
        }
    }
    Ok(())
}

/// v1 birank format: src-side scores, blank line, dst-side scores, each
/// sorted descending.
pub fn birank(out: &mut impl Write, model: &BiRankModel) -> Result<()> {
    let emit_side = |out: &mut dyn Write, rows: Vec<(&str, f64)>| -> Result<()> {
        let mut rows = rows;
        rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        for (name, score) in rows {
            writeln!(out, "{name}: {score}")?;
        }
        Ok(())
    };
    emit_side(out, model.src_scores().collect())?;
    writeln!(out)?;
    emit_side(out, model.dst_scores().collect())
}

/// The model's own JSONL state representation.
pub fn jsonl<M: RankModel>(out: &mut impl Write, model: &M) -> Result<()> {
    model.save_jsonl(out)
}
