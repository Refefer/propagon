//! End-to-end numerical regression against captured v1 outputs
//! (`tests/golden/`). The CLI surface is v2's grouped form
//! (`tournament`/`graph`/`bandit`); only the *numbers* are held to v1.
//!
//! Tiers (see `scripts/capture_golden.sh`):
//! - **T (tolerance)**: numeric agreement per entity — rate, glicko2,
//!   btm-mm, btm-lr, kemeny (insertion), lsr (power), page-rank.
//! - **S (sanity)**: rank correlation ≥ 0.95 — es-rum, birank (their RNG
//!   streams legitimately differ from v1's retired `random`/xorshift crates).
//!
//! Note on `rate-090.out`: v1 declared `--confidence-interval 0.9` but its
//! handler matched `"0.90"`, so the capture silently produced **P50** values.
//! v2 fixes the mapping; the golden is compared against `0.5`.
//!
//! Per AGENTS.md rule 7, integration tests are not exempt from the
//! unwrap/expect denies: every test propagates errors via `Result`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn repo(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

const EDGES: &str = "examples/tournament/baseball.2018.edges";

/// Runs `propagon <args...> <example-edges-path>` and returns stdout.
fn run(args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
        .args(args)
        .arg(repo(EDGES))
        .output()?;
    assert!(
        out.status.success(),
        "propagon {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(String::from_utf8(out.stdout)?)
}

/// Parses `id: v [v ...]` lines into id -> columns; ignores blank lines.
fn parse(text: &str) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();

    for line in text.lines() {
        let Some((id, rest)) = line.split_once(": ") else {
            continue;
        };
        let cols: Vec<f64> = rest
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;

        if !cols.is_empty() {
            out.insert(id.to_string(), cols);
        }
    }

    Ok(out)
}

fn golden(name: &str) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(repo(&format!("crates/propagon-cli/tests/golden/{name}")))?;
    parse(&text)
}

fn assert_tier_t(args: &[&str], golden_file: &str, tol: f64) -> TestResult {
    let got = parse(&run(args)?)?;
    let want = golden(golden_file)?;
    assert_eq!(got.len(), want.len(), "{golden_file}: entity count");

    for (id, want_cols) in &want {
        let got_cols = got
            .get(id)
            .ok_or_else(|| format!("{golden_file}: missing {id}"))?;
        assert_eq!(
            got_cols.len(),
            want_cols.len(),
            "{golden_file} {id}: column count"
        );

        for (g, w) in got_cols.iter().zip(want_cols) {
            assert!((g - w).abs() < tol, "{golden_file} {id}: v2 {g} vs v1 {w}");
        }
    }

    Ok(())
}

/// Spearman rank correlation over shared keys (first column).
fn spearman(a: &HashMap<String, Vec<f64>>, b: &HashMap<String, Vec<f64>>) -> f64 {
    let mut keys: Vec<&String> = a.keys().collect();
    keys.retain(|k| b.contains_key(*k));

    let rank = |m: &HashMap<String, Vec<f64>>| -> HashMap<String, f64> {
        let mut sorted: Vec<&String> = keys.clone();
        sorted.sort_by(|x, y| m[*y][0].total_cmp(&m[*x][0]));
        sorted
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), i as f64))
            .collect()
    };
    let ra = rank(a);
    let rb = rank(b);
    let n = keys.len() as f64;
    let d2: f64 = keys.iter().map(|k| (ra[*k] - rb[*k]).powi(2)).sum();
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

#[test]
fn rate_matches_golden() -> TestResult {
    assert_tier_t(
        &["tournament", "win-rate", "--confidence-interval", "0.5"],
        "rate-090.out",
        1e-6,
    )?;
    assert_tier_t(&["tournament", "win-rate"], "rate-095.out", 1e-5)
}

#[test]
fn glicko2_matches_golden() -> TestResult {
    assert_tier_t(&["tournament", "glicko2"], "glicko2.out", 5e-3)?;
    assert_tier_t(
        &["tournament", "glicko2", "--use-mu"],
        "glicko2-mu.out",
        2e-3,
    )
}

#[test]
fn btm_mm_matches_golden() -> TestResult {
    assert_tier_t(&["tournament", "bradley-terry-model"], "btm-mm.out", 1e-4)
}

#[test]
fn btm_lr_matches_golden() -> TestResult {
    assert_tier_t(
        &["tournament", "bradley-terry-model", "--estimator", "sgd"],
        "btm-lr.out",
        1e-3,
    )
}

#[test]
fn kemeny_matches_golden() -> TestResult {
    assert_tier_t(
        &["tournament", "kemeny", "--passes", "5"],
        "kemeny.out",
        0.5,
    )
}

#[test]
fn lsr_matches_golden() -> TestResult {
    assert_tier_t(
        &["tournament", "luce-spectral-ranking", "--steps", "20"],
        "lsr.out",
        2e-3,
    )
}

#[test]
fn page_rank_matches_golden() -> TestResult {
    // --matches reproduces v1's orientation: 'winner loser' rows become
    // loser -> winner endorsements.
    assert_tier_t(&["graph", "page-rank", "--matches"], "page-rank.out", 1e-5)
}

#[test]
fn es_rum_rank_correlates_with_golden() -> TestResult {
    let got = parse(&run(&[
        "tournament",
        "random-utility-model",
        "--passes",
        "100",
    ])?)?;
    let want = golden("es-rum.out")?;
    let rho = spearman(&got, &want);
    assert!(rho >= 0.95, "es-rum spearman {rho}");
    Ok(())
}

#[test]
fn birank_rank_correlates_with_golden() -> TestResult {
    // Both outputs hold two 30-line sections (src side then dst side) whose
    // ids overlap; correlate section-wise.
    type Sides = (HashMap<String, Vec<f64>>, HashMap<String, Vec<f64>>);
    let split = |text: &str| -> Result<Sides, Box<dyn std::error::Error>> {
        let lines: Vec<&str> = text.lines().filter(|l| l.contains(": ")).collect();
        let mid = lines.len() / 2;
        Ok((
            parse(&lines[..mid].join("\n"))?,
            parse(&lines[mid..].join("\n"))?,
        ))
    };

    let (got_u, got_p) = split(&run(&["graph", "birank"])?)?;
    let golden_text = std::fs::read_to_string(repo("crates/propagon-cli/tests/golden/birank.out"))?;
    let (want_u, want_p) = split(&golden_text)?;

    let rho_u = spearman(&got_u, &want_u);
    let rho_p = spearman(&got_p, &want_p);
    assert!(rho_u >= 0.95, "birank u-side spearman {rho_u}");
    assert!(rho_p >= 0.95, "birank p-side spearman {rho_p}");
    Ok(())
}

/// FR-5 acceptance at the CLI: glicko2 two-batch run via --save/--load-state
/// equals a single run over both periods.
#[test]
fn glicko2_save_load_state_flow() -> TestResult {
    let dir = std::env::temp_dir().join(format!("propagon-golden-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let p1 = dir.join("p1.edges");
    let p2 = dir.join("p2.edges");
    let both = dir.join("both.edges");
    std::fs::write(&p1, "a b\nc b\n")?;
    std::fs::write(&p2, "b a\na c\n")?;
    std::fs::write(&both, "a b\nc b\n\nb a\na c\n")?;
    let state = dir.join("state.jsonl");
    let state_arg = state.to_str().ok_or("non-utf8 temp path")?;

    let run_at = |extra: &[&str], path: &PathBuf| -> Result<String, Box<dyn std::error::Error>> {
        let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
            .args(extra)
            .arg(path)
            .output()?;
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        Ok(String::from_utf8(out.stdout)?)
    };

    let _ = run_at(&["tournament", "glicko2", "--save-state", state_arg], &p1)?;
    let resumed = run_at(&["tournament", "glicko2", "--load-state", state_arg], &p2)?;
    let continuous = run_at(&["tournament", "glicko2", "--groups-are-separate"], &both)?;
    assert_eq!(
        resumed, continuous,
        "resume must equal continuous two-period run"
    );

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

/// New-in-v2 tournament algorithms smoke-run on the example data, and the
/// short visible aliases resolve to the same commands.
#[test]
fn new_subcommands_run() -> TestResult {
    for algo in ["elo", "borda-count", "copeland", "rank-centrality"] {
        let out = parse(&run(&["tournament", algo])?)?;
        assert_eq!(out.len(), 30, "{algo} ranks all 30 teams");
    }

    for alias in ["rate", "btm", "lsr", "rum", "borda"] {
        let extra: &[&str] = if alias == "lsr" {
            &["--steps", "5"]
        } else {
            &[]
        };
        let mut args = vec!["tournament", alias];
        args.extend_from_slice(extra);
        let out = parse(&run(&args)?)?;
        assert_eq!(out.len(), 30, "alias {alias} works");
    }

    Ok(())
}

/// Bandit group: per-policy subcommands, scores and seeded selection.
#[test]
fn bandit_subcommand_runs() -> TestResult {
    let dir = std::env::temp_dir().join(format!("propagon-bandit-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let rewards = dir.join("rewards");
    std::fs::write(&rewards, "A 1\nA 1\nB 0\nB 1\nC 0\n")?;

    let run_b = |extra: &[&str]| -> Result<String, Box<dyn std::error::Error>> {
        let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
            .args(extra)
            .arg(&rewards)
            .output()?;
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        Ok(String::from_utf8(out.stdout)?)
    };

    let scores = parse(&run_b(&["bandit", "greedy"])?)?;
    assert_eq!(scores.get("A").ok_or("missing arm A")?[0], 1.0);
    assert_eq!(scores.get("B").ok_or("missing arm B")?[0], 0.5);

    let pick1 = run_b(&["bandit", "thompson-beta", "--seed", "9", "--select", "1"])?;
    let pick2 = run_b(&["bandit", "ts-beta", "--seed", "9", "--select", "1"])?; // alias
    assert_eq!(pick1, pick2, "seeded selection is deterministic");
    assert!(["A", "B", "C"].contains(&pick1.trim()));

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
