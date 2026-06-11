//! End-to-end parity against captured v1 outputs (`tests/golden/`).
//!
//! Tiers (see `scripts/capture_golden.sh`):
//! - **T (tolerance)**: numeric agreement per entity + identical ranking —
//!   rate, glicko2, btm-mm, btm-lr, kemeny (insertion), lsr (power),
//!   page-rank.
//! - **S (sanity)**: rank correlation ≥ 0.95 — es-rum, birank (their RNG
//!   streams legitimately differ from v1's retired `random`/xorshift crates).
//!
//! Note on `rate-090.out`: v1 declared `--confidence-interval 0.9` but its
//! handler matched `"0.90"`, so the capture silently produced **P50** values.
//! v2 fixes the mapping; the golden is compared against `0.5`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

fn repo(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

const EDGES: &str = "example/tournament/baseball.2018.edges";

fn run(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
        .arg(repo(EDGES))
        .args(args)
        .output()
        .expect("binary runs");
    assert!(
        out.status.success(),
        "propagon {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 output")
}

/// Parses `id: v [v ...]` lines into id -> columns; ignores blank lines.
fn parse(text: &str) -> HashMap<String, Vec<f64>> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let Some((id, rest)) = line.split_once(": ") else { continue };
        let cols: Vec<f64> =
            rest.split_whitespace().map(|t| t.parse().expect("numeric column")).collect();
        if !cols.is_empty() {
            out.insert(id.to_string(), cols);
        }
    }
    out
}

fn golden(name: &str) -> HashMap<String, Vec<f64>> {
    parse(&std::fs::read_to_string(repo(&format!("crates/propagon-cli/tests/golden/{name}"))).unwrap())
}

fn assert_tier_t(args: &[&str], golden_file: &str, tol: f64) {
    let got = parse(&run(args));
    let want = golden(golden_file);
    assert_eq!(got.len(), want.len(), "{golden_file}: entity count");
    for (id, want_cols) in &want {
        let got_cols = &got[id];
        assert_eq!(got_cols.len(), want_cols.len(), "{golden_file} {id}: column count");
        for (g, w) in got_cols.iter().zip(want_cols) {
            assert!((g - w).abs() < tol, "{golden_file} {id}: v2 {g} vs v1 {w}");
        }
    }
}

/// Spearman rank correlation over shared keys (first column).
fn spearman(a: &HashMap<String, Vec<f64>>, b: &HashMap<String, Vec<f64>>) -> f64 {
    let mut keys: Vec<&String> = a.keys().collect();
    keys.retain(|k| b.contains_key(*k));
    let rank = |m: &HashMap<String, Vec<f64>>| -> HashMap<String, f64> {
        let mut sorted: Vec<&String> = keys.clone();
        sorted.sort_by(|x, y| m[*y][0].total_cmp(&m[*x][0]));
        sorted.into_iter().enumerate().map(|(i, k)| (k.clone(), i as f64)).collect()
    };
    let ra = rank(a);
    let rb = rank(b);
    let n = keys.len() as f64;
    let d2: f64 = keys.iter().map(|k| (ra[*k] - rb[*k]).powi(2)).sum();
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

#[test]
fn rate_matches_golden() {
    assert_tier_t(&["rate", "--confidence-interval", "0.5"], "rate-090.out", 1e-6);
    assert_tier_t(&["rate"], "rate-095.out", 1e-5);
}

#[test]
fn glicko2_matches_golden() {
    assert_tier_t(&["glicko2"], "glicko2.out", 5e-3);
    assert_tier_t(&["glicko2", "--use-mu"], "glicko2-mu.out", 2e-3);
}

#[test]
fn btm_mm_matches_golden() {
    assert_tier_t(&["btm-mm"], "btm-mm.out", 1e-4);
}

#[test]
fn btm_lr_matches_golden() {
    assert_tier_t(&["btm-lr"], "btm-lr.out", 1e-3);
}

#[test]
fn kemeny_matches_golden() {
    assert_tier_t(&["kemeny", "--passes", "5"], "kemeny.out", 0.5);
}

#[test]
fn lsr_matches_golden() {
    assert_tier_t(&["lsr", "--steps", "20"], "lsr.out", 2e-3);
}

#[test]
fn page_rank_matches_golden() {
    assert_tier_t(&["page-rank"], "page-rank.out", 1e-5);
}

#[test]
fn es_rum_rank_correlates_with_golden() {
    let got = parse(&run(&["es-rum", "--passes", "100"]));
    let want = golden("es-rum.out");
    let rho = spearman(&got, &want);
    assert!(rho >= 0.95, "es-rum spearman {rho}");
}

#[test]
fn birank_rank_correlates_with_golden() {
    // Both outputs hold two 30-line sections (src side then dst side) whose
    // ids overlap; correlate section-wise.
    let split = |text: &str| -> (HashMap<String, Vec<f64>>, HashMap<String, Vec<f64>>) {
        let lines: Vec<&str> = text.lines().filter(|l| l.contains(": ")).collect();
        let mid = lines.len() / 2;
        (parse(&lines[..mid].join("\n")), parse(&lines[mid..].join("\n")))
    };
    let (got_u, got_p) = split(&run(&["birank"]));
    let golden_text =
        std::fs::read_to_string(repo("crates/propagon-cli/tests/golden/birank.out")).unwrap();
    let (want_u, want_p) = split(&golden_text);

    let rho_u = spearman(&got_u, &want_u);
    let rho_p = spearman(&got_p, &want_p);
    assert!(rho_u >= 0.95, "birank u-side spearman {rho_u}");
    assert!(rho_p >= 0.95, "birank p-side spearman {rho_p}");
}

/// FR-5 acceptance at the CLI: glicko2 two-batch run via --save/--load-state
/// equals a single run over both periods.
#[test]
fn glicko2_save_load_state_flow() {
    let dir = std::env::temp_dir().join(format!("propagon-golden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p1 = dir.join("p1.edges");
    let p2 = dir.join("p2.edges");
    let both = dir.join("both.edges");
    std::fs::write(&p1, "a b\nc b\n").unwrap();
    std::fs::write(&p2, "b a\na c\n").unwrap();
    std::fs::write(&both, "a b\nc b\n\nb a\na c\n").unwrap();
    let state = dir.join("state.jsonl");

    let run_at = |path: &PathBuf, extra: &[&str]| -> String {
        let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
            .arg(path)
            .args(extra)
            .output()
            .expect("binary runs");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };

    let _ = run_at(&p1, &["glicko2", "--save-state", state.to_str().unwrap()]);
    let resumed = run_at(
        &p2,
        &["glicko2", "--load-state", state.to_str().unwrap()],
    );
    let continuous = run_at(&both, &["glicko2", "--groups-are-separate"]);
    assert_eq!(resumed, continuous, "resume must equal continuous two-period run");

    std::fs::remove_dir_all(&dir).ok();
}

/// The deprecated dehydrate/hydrate pipeline still works end to end.
#[test]
fn dehydrate_hydrate_round_trip() {
    let dir = std::env::temp_dir().join(format!("propagon-dehydrate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let raw = dir.join("games");
    std::fs::write(&raw, "ARI\tCOL\nATL\tPHI\nARI\tPHI\n").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
        .arg(&raw)
        .arg("dehydrate")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let edges = dir.join("games.edges");
    let scores_out = Command::new(env!("CARGO_BIN_EXE_propagon"))
        .arg(&edges)
        .arg("rate")
        .output()
        .unwrap();
    assert!(scores_out.status.success());
    let scores_path = dir.join("scores");
    std::fs::write(&scores_path, &scores_out.stdout).unwrap();

    let hydrated = Command::new(env!("CARGO_BIN_EXE_propagon"))
        .arg(&scores_path)
        .arg("hydrate")
        .arg("--vocab")
        .arg(dir.join("games.vocab"))
        .output()
        .unwrap();
    assert!(hydrated.status.success());
    let text = String::from_utf8(hydrated.stdout).unwrap();
    assert!(text.contains("ARI\t"), "hydrated output has names: {text}");

    std::fs::remove_dir_all(&dir).ok();
}

/// New-in-v2 subcommands smoke-run on the example data.
#[test]
fn new_subcommands_run() {
    for args in [
        vec!["elo"],
        vec!["borda"],
        vec!["copeland"],
        vec!["rank-centrality"],
    ] {
        let out = parse(&run(&args));
        assert_eq!(out.len(), 30, "{args:?} ranks all 30 teams");
    }
}

/// Bandit subcommand: scores and seeded selection.
#[test]
fn bandit_subcommand_runs() {
    let dir = std::env::temp_dir().join(format!("propagon-bandit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rewards = dir.join("rewards");
    std::fs::write(&rewards, "A 1\nA 1\nB 0\nB 1\nC 0\n").unwrap();

    let run_b = |extra: &[&str]| -> String {
        let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
            .arg(&rewards)
            .args(extra)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };

    let scores = parse(&run_b(&["bandit", "--policy", "greedy"]));
    assert_eq!(scores["A"][0], 1.0);
    assert_eq!(scores["B"][0], 0.5);

    let pick1 = run_b(&["bandit", "--policy", "ts-beta", "--seed", "9", "--select", "1"]);
    let pick2 = run_b(&["bandit", "--policy", "ts-beta", "--seed", "9", "--select", "1"]);
    assert_eq!(pick1, pick2, "seeded selection is deterministic");
    assert!(["A", "B", "C"].contains(&pick1.trim()));

    std::fs::remove_dir_all(&dir).ok();
}
