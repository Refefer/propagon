"""Dataset construction, introspection, periods, and input validation."""

import propagon
import pytest


def test_pairwise_len_and_entities():
    d = propagon.PairwiseDataset()
    d.push("A", "B")
    d.push("A", "C", 2.0)
    assert len(d) == 2
    assert d.n_entities() == 3
    assert not d.is_empty()


def test_games_periods():
    d = propagon.GamesDataset()
    d.push_pair("a", "b")
    d.push_pair("a", "c")
    d.new_period()
    d.push_pair("b", "c")
    assert d.n_periods() == 2
    assert len(d) == 3


def test_games_push_game_with_teams():
    d = propagon.GamesDataset()
    d.push_game(["A", "B"], ["C", "D"], propagon.GameOutcome.side1_win(7.0), 2.0)
    assert len(d) == 1
    assert d.n_entities() == 4


def test_graph_nodes():
    d = propagon.GraphDataset()
    d.push("a", "b")
    d.push("b", "c", 2.0)
    assert d.n_nodes() == 3
    assert len(d) == 2


def test_trajectories_episodes():
    d = propagon.TrajectoriesDataset()
    d.push_step("s0", 0.0)
    d.push_step("s1", 1.0)
    d.end_episode()
    d.push_step("s0", 0.5)
    d.end_episode()
    assert d.n_episodes() == 2


def test_rankings_and_matchups_and_annotated_and_rewards():
    r = propagon.RankingsDataset()
    r.push_ranking(["A", "B", "C"])
    assert len(r) == 1 and r.n_entities() == 3

    m = propagon.MatchupsDataset()
    m.push_match([["A"], ["B"]], [1, 2])
    assert len(m) == 1 and m.n_entities() == 2

    a = propagon.AnnotatedPairsDataset()
    a.push("u1", "A", "B")
    assert a.n_annotators() == 1 and a.n_entities() == 2

    w = propagon.RewardsDataset()
    w.push("arm", 1.0)
    assert w.n_arms() == 1


def test_overflow_weight_is_rejected():
    """A finite float that overflows f32 range is rejected at the boundary."""
    d = propagon.PairwiseDataset()
    with pytest.raises(propagon.InvalidInputError):
        d.push("A", "B", 1e40)


def test_fractional_weight_is_accepted():
    """Single-precision rounding of fractional weights is fine (not rejected)."""
    d = propagon.PairwiseDataset()
    d.push("A", "B", 0.1)
    assert len(d) == 1


def test_repr_is_informative():
    d = propagon.GraphDataset()
    d.push("a", "b")
    assert repr(d) == "GraphDataset(edges=1)"
