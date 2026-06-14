//! End-to-end tests for the `odds` group (§14). Self-contained — each test
//! writes a tiny fixture to a temp file, runs the built binary, and checks the
//! output. New algorithms have no v1 baseline, so these live outside the v1
//! `golden.rs` regression set.
//!
//! Per AGENTS.md rule 7, integration tests still propagate errors via `Result`.

use std::io::Write;
use std::process::Command;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Writes `contents` to a uniquely-named temp file and returns its path.
fn fixture(name: &str, contents: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let mut path = std::env::temp_dir();
    path.push(format!("propagon-odds-{}-{name}", std::process::id()));
    let mut f = std::fs::File::create(&path)?;
    f.write_all(contents.as_bytes())?;
    Ok(path)
}

fn run(args: &[&str], path: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let out = Command::new(env!("CARGO_BIN_EXE_propagon"))
        .args(args)
        .arg(path)
        .output()?;
    assert!(
        out.status.success(),
        "exit: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(String::from_utf8(out.stdout)?)
}

/// `name: value` lines into a map.
fn parse(out: &str) -> std::collections::HashMap<String, f64> {
    out.lines()
        .filter_map(|l| l.split_once(": "))
        .filter_map(|(k, v)| v.trim().parse().ok().map(|x| (k.to_string(), x)))
        .collect()
}

#[test]
fn devig_fair_probs_sum_to_one() -> TestResult {
    let f = fixture("devig", "home 4.20\ndraw 3.70\naway 1.95\n")?;
    for method in ["multiplicative", "power", "shin"] {
        let scores = parse(&run(&["odds", "devig", "--method", method], &f)?);
        let sum: f64 = scores.values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "{method}: Σ = {sum}");
    }
    Ok(())
}

#[test]
fn opinion_pool_consolidates() -> TestResult {
    let f = fixture(
        "pool",
        "s1 home 0.8\ns1 away 0.2\ns2 home 0.6\ns2 away 0.4\n",
    )?;
    let linear = parse(&run(&["odds", "opinion-pool", "--kind", "linear"], &f)?);
    // Linear pool is the arithmetic mean: (0.8 + 0.6)/2 = 0.7.
    assert!((linear["home"] - 0.7).abs() < 1e-9);
    Ok(())
}

#[test]
fn lmsr_prices_sum_to_one() -> TestResult {
    let f = fixture("lmsr", "yes 100\nno 20\nmaybe 5\n")?;
    let prices = parse(&run(&["odds", "lmsr", "--liquidity", "50"], &f)?);
    let sum: f64 = prices.values().sum();
    assert!((sum - 1.0).abs() < 1e-9, "Σ = {sum}");
    assert!(prices["yes"] > prices["no"], "more shares ⇒ higher price");
    Ok(())
}

#[test]
fn kelly_matches_closed_form() -> TestResult {
    let f = fixture("kelly", "pick 0.6 2.0\nnoedge 0.4 2.0\n")?;
    // Even money (b=1), p=0.6 ⇒ f* = 0.2; half-Kelly ⇒ 0.1. No edge ⇒ 0.
    let half = parse(&run(&["odds", "kelly", "--fraction", "0.5"], &f)?);
    assert!((half["pick"] - 0.1).abs() < 1e-9, "got {}", half["pick"]);
    assert_eq!(half["noedge"], 0.0);
    Ok(())
}

#[test]
fn clv_sign_convention() -> TestResult {
    let f = fixture("clv", "beat 2.10 2.00\nlost 1.90 2.00\n")?;
    let clv = parse(&run(&["odds", "clv"], &f)?);
    assert!((clv["beat"] - 0.05).abs() < 1e-9);
    assert!((clv["lost"] + 0.05).abs() < 1e-9);
    Ok(())
}
