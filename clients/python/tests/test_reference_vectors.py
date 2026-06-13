"""Correctness against published reference vectors.

Each expected number below is read from a primary or authoritative source,
mirroring the Rust crate's ``tests/reference.rs`` (cited inline). These are the
strongest correctness checks: they pin the bindings to externally-known answers,
not just to the Rust core's current output.
"""

import math

import propagon
import pytest
from conftest import ACC_2005, TENNESSEE, tennessee_pairwise


def test_bradley_terry_matches_agresti_baseball():
    """Agresti (1990) 1987 AL East Bradley-Terry abilities (log scale)."""
    teams = ["Milwaukee", "Detroit", "Toronto", "New York", "Boston", "Cleveland", "Baltimore"]
    wins = [
        [0, 7, 9, 7, 7, 9, 11],
        [6, 0, 7, 5, 11, 9, 9],
        [4, 6, 0, 7, 7, 8, 12],
        [6, 8, 6, 0, 6, 7, 10],
        [6, 2, 6, 7, 0, 7, 12],
        [4, 4, 5, 6, 6, 0, 6],
        [2, 4, 1, 3, 1, 7, 0],
    ]
    expected = [1.5814, 1.4364, 1.2945, 1.2476, 1.1077, 0.6839]

    d = propagon.PairwiseDataset()
    total = 0
    for i, row in enumerate(wins):
        for j, w in enumerate(row):
            if w > 0:
                d.push(teams[i], teams[j], float(w))
                total += w
    assert total == 273

    scores = propagon.BradleyTerryMM(tolerance=1e-12).fit(d).scores()
    baltimore = scores["Baltimore"]
    for team, exp in zip(teams, expected):
        ability = math.log(scores[team] / baltimore)
        assert abs(ability - exp) < 1e-3, f"{team}: {ability:.4f} vs {exp}"


def test_wilson_intervals_match_newcombe_table():
    """Newcombe (1998) Table I, method 3 (95%, z=1.959964)."""
    z = 1.959964
    for successes, n, lo, hi in [
        (81.0, 263.0, 0.2553, 0.3662),
        (15.0, 148.0, 0.0624, 0.1605),
        (0.0, 20.0, 0.0000, 0.1611),
        (1.0, 29.0, 0.0061, 0.1718),
    ]:
        got_lo, got_hi = propagon.wilson_interval(successes, n - successes, z)
        assert abs(got_lo - lo) < 5e-5, f"{successes}/{n} lower {got_lo:.4f} vs {lo}"
        assert abs(got_hi - hi) < 5e-5, f"{successes}/{n} upper {got_hi:.4f} vs {hi}"


def test_borda_matches_tennessee_example():
    """Published Borda totals (3/2/1/0): Memphis 126, Nashville 194, etc.

    Borda over the ballots-as-ordered-pairs equals the classic Borda count:
    an entity's score is its total weighted pairwise wins.
    """
    scores = propagon.Borda().fit(tennessee_pairwise()).scores()
    assert scores["Memphis"] == pytest.approx(126.0)
    assert scores["Nashville"] == pytest.approx(194.0)
    assert scores["Chattanooga"] == pytest.approx(173.0)
    assert scores["Knoxville"] == pytest.approx(107.0)


def test_copeland_finds_tennessee_condorcet_winner():
    """Copeland order: Nashville 3 > Chattanooga 2 > Knoxville 1 > Memphis 0."""
    scores = propagon.Copeland().fit(tennessee_pairwise()).scores()
    assert scores["Nashville"] == pytest.approx(3.0)
    assert scores["Chattanooga"] == pytest.approx(2.0)
    assert scores["Knoxville"] == pytest.approx(1.0)
    assert scores["Memphis"] == pytest.approx(0.0)


def test_pagerank_matches_langville_meyer_example():
    """Langville & Meyer (2005) 6-node example, alpha=0.9, uniform dangling."""
    g = propagon.GraphDataset()
    for s, t in [
        ("1", "2"), ("1", "3"), ("3", "1"), ("3", "2"), ("3", "5"),
        ("4", "5"), ("4", "6"), ("5", "4"), ("5", "6"), ("6", "4"),
    ]:
        g.push(s, t)
    scores = propagon.PageRank(damping=0.9, iterations=200, sink="uniform").fit(g).scores()
    expected = {"1": 0.03721, "2": 0.05396, "3": 0.04151, "4": 0.3751, "5": 0.206, "6": 0.2862}
    for node, exp in expected.items():
        assert abs(scores[node] - exp) < 1e-4, f"node {node}: {scores[node]:.5f} vs {exp}"


def test_massey_matches_whos_number_one():
    """Massey ratings (Who's #1?, ch. 2): Duke -24.8, Miami 18.2, etc."""
    d = propagon.PairwiseDataset()
    for w, l, margin in ACC_2005:
        d.push(w, l, margin)
    scores = propagon.Massey().fit(d).scores()
    for team, exp in [("Duke", -24.8), ("Miami", 18.2), ("UNC", -8.0), ("UVA", -3.4), ("VT", 18.0)]:
        assert abs(scores[team] - exp) < 0.05, f"{team}: {scores[team]:.2f} vs {exp}"


def test_colley_matches_whos_number_one():
    """Colley ratings (Who's #1?, ch. 3): 0.2143, 0.7857, 0.5, 0.3571, 0.6429."""
    d = propagon.PairwiseDataset()
    for w, l, _margin in ACC_2005:
        d.push(w, l, 1.0)  # Colley reads weights as game counts, not margins
    scores = propagon.Colley().fit(d).scores()
    for team, exp in [
        ("Duke", 0.2143), ("Miami", 0.7857), ("UNC", 0.5), ("UVA", 0.3571), ("VT", 0.6429)
    ]:
        assert abs(scores[team] - exp) < 1e-3, f"{team}: {scores[team]:.4f} vs {exp}"


def test_thompson_beta_posterior_mean_is_analytic():
    """Beta(1,1) prior + 7 successes / 3 failures -> posterior mean 8/12."""
    data = propagon.RewardsDataset()
    for _ in range(7):
        data.push("arm", 1.0)
    for _ in range(3):
        data.push("arm", 0.0)
    bandit = propagon.Bandit(policy=propagon.BanditPolicy.thompson_beta(1.0, 1.0), seed=1)
    model = bandit.fit(data)
    estimate = model.score("arm")
    assert abs(estimate - 8.0 / 12.0) < 1e-12, f"posterior mean {estimate}"
