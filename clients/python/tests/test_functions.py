"""Module-level free functions: extract_components and load_state edges."""

import propagon
import pytest


def test_extract_components_splits_disjoint_graphs():
    g = propagon.GraphDataset()
    for a, b in [("a", "b"), ("b", "c"), ("c", "a"), ("x", "y"), ("y", "z"), ("z", "x")]:
        g.push(a, b)
    comps = propagon.extract_components(g, min_size=2)
    assert len(comps) == 2
    assert all(isinstance(c, propagon.GraphDataset) for c in comps)
    assert sorted(c.n_nodes() for c in comps) == [3, 3]


def test_extract_components_min_size_filters():
    g = propagon.GraphDataset()
    g.push("a", "b")            # 2-node component
    g.push("x", "y")
    g.push("y", "z")           # 3-node component
    big = propagon.extract_components(g, min_size=3)
    assert len(big) == 1
    assert big[0].n_nodes() == 3


def test_load_state_round_trips_every_shape():
    """A model from each dataset shape reloads to the right class via load_state."""
    g = propagon.GamesDataset()
    g.push_pair("A", "B")
    g.push_pair("B", "C")
    state = propagon.Elo().fit(g).save_state()
    reloaded = propagon.load_state(state)
    assert reloaded.algorithm == "elo"
    assert reloaded.save_state() == state


def test_load_state_empty_raises():
    with pytest.raises(propagon.StateError):
        propagon.load_state("")
