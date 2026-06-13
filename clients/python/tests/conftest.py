"""Shared fixtures and dataset builders for the propagon test suite."""

import propagon
import pytest


@pytest.fixture
def round_robin_pairwise():
    """A small balanced pairwise dataset where every entity wins and loses."""
    d = propagon.PairwiseDataset()
    for w, l in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A"), ("C", "B")]:
        d.push(w, l)
    return d


@pytest.fixture
def round_robin_games():
    d = propagon.GamesDataset()
    for w, l in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A"), ("C", "B")]:
        d.push_pair(w, l)
    return d


@pytest.fixture
def small_graph():
    d = propagon.GraphDataset()
    for s, t in [("a", "b"), ("b", "c"), ("c", "a"), ("a", "c"), ("b", "a")]:
        d.push(s, t)
    return d


# The Tennessee capital election (Wikipedia, "Borda count" / "Condorcet method").
TENNESSEE = [
    (42, ["Memphis", "Nashville", "Chattanooga", "Knoxville"]),
    (26, ["Nashville", "Chattanooga", "Knoxville", "Memphis"]),
    (15, ["Chattanooga", "Knoxville", "Nashville", "Memphis"]),
    (17, ["Knoxville", "Chattanooga", "Nashville", "Memphis"]),
]


def tennessee_pairwise():
    """Tennessee ballots lowered to ordered pairs (one win per ranked pair)."""
    d = propagon.PairwiseDataset()
    for count, order in TENNESSEE:
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                d.push(order[i], order[j], count)
    return d


# 2005 ACC season (Langville & Meyer, "Who's #1?", ch. 2): (winner, loser, margin).
ACC_2005 = [
    ("Miami", "Duke", 45.0),
    ("UNC", "Duke", 3.0),
    ("UVA", "Duke", 31.0),
    ("VT", "Duke", 45.0),
    ("Miami", "UNC", 18.0),
    ("Miami", "UVA", 8.0),
    ("Miami", "VT", 20.0),
    ("UNC", "UVA", 2.0),
    ("VT", "UNC", 27.0),
    ("VT", "UVA", 38.0),
]
