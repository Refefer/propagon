"""State persistence: byte-identical round-trips, dispatch, and the FR-5
'split update equals continuous update' incremental guarantee."""

import propagon
import pytest


def test_save_load_save_is_byte_identical(round_robin_games, small_graph):
    """One model per fitting tier round-trips byte-for-byte."""
    online = propagon.Glicko2().fit(round_robin_games)
    s1 = online.save_state()
    assert propagon.Glicko2Model.load(s1).save_state() == s1

    batch = propagon.PageRank().fit(small_graph)
    s2 = batch.save_state()
    assert propagon.PageRankModel.load(s2).save_state() == s2


def test_save_state_bytes_matches_text(round_robin_games):
    model = propagon.Glicko2().fit(round_robin_games)
    assert model.save_state_bytes() == model.save_state().encode("utf-8")


def test_load_state_dispatches_on_tag(round_robin_games, small_graph):
    """The top-level load_state picks the right concrete class from the tag."""
    g = propagon.Glicko2().fit(round_robin_games).save_state()
    assert type(propagon.load_state(g)).__name__ == "Glicko2Model"
    p = propagon.PageRank().fit(small_graph).save_state()
    assert type(propagon.load_state(p)).__name__ == "PageRankModel"


def test_wrong_algorithm_load_raises(round_robin_games):
    """Loading a glicko2 state as PageRank raises AlgorithmMismatchError."""
    g = propagon.Glicko2().fit(round_robin_games).save_state()
    with pytest.raises(propagon.AlgorithmMismatchError):
        propagon.PageRankModel.load(g)


def test_online_split_update_equals_continuous():
    """FR-5: update -> save -> load -> update equals one continuous run."""
    algo = propagon.Glicko2(tau=0.5)

    both = propagon.GamesDataset()
    both.push_pair("a", "b")
    both.push_pair("c", "b")
    both.new_period()
    both.push_pair("b", "a")
    both.push_pair("a", "c")
    continuous = algo.init()
    algo.update(continuous, both)

    p1 = propagon.GamesDataset()
    p1.push_pair("a", "b")
    p1.push_pair("c", "b")
    p2 = propagon.GamesDataset()
    p2.push_pair("b", "a")
    p2.push_pair("a", "c")

    resumed = algo.init()
    algo.update(resumed, p1)
    resumed = propagon.Glicko2Model.load(resumed.save_state())  # persist between batches
    algo.update(resumed, p2)

    assert resumed.save_state() == continuous.save_state()
