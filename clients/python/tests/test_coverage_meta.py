"""Coverage meta-test: every algorithm fits a valid fixture and round-trips
its state. This guarantees the comprehensive surface stays wired — if an
algorithm is added to the registry but mis-mapped, this fails."""

import propagon
import pytest


def _pairwise():
    d = propagon.PairwiseDataset()
    for w, l in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A"), ("C", "B")]:
        d.push(w, l)
    return d


def _games():
    d = propagon.GamesDataset()
    for w, l in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A"), ("C", "B")]:
        d.push_pair(w, l)
    return d


def _graph():
    d = propagon.GraphDataset()
    for a, b in [("a", "b"), ("b", "c"), ("c", "a"), ("a", "c"), ("b", "a")]:
        d.push(a, b)
    return d


def _rewards():
    d = propagon.RewardsDataset()
    for arm, r in [("a", 1.0), ("a", 1.0), ("b", 0.0), ("b", 1.0), ("c", 0.0), ("c", 1.0)]:
        d.push(arm, r)
    return d


def _contextual():
    d = propagon.ContextualRewardsDataset()
    for arm, r, x in [("a", 1.0, [1.0, 0.0]), ("b", 0.0, [0.0, 1.0]),
                      ("a", 1.0, [1.0, 0.1]), ("b", 1.0, [0.2, 1.0])]:
        d.push(arm, r, x)
    return d


def _matchups():
    d = propagon.MatchupsDataset()
    for teams, ranks in [([["A"], ["B"]], [1, 2]), ([["B"], ["C"]], [1, 2]),
                         ([["C"], ["A"]], [1, 2]), ([["A"], ["C"]], [1, 2])]:
        d.push_match(teams, ranks)
    return d


def _annotated():
    d = propagon.AnnotatedPairsDataset()
    for ann, w, l in [("u1", "A", "B"), ("u1", "B", "C"), ("u2", "C", "A"),
                      ("u2", "A", "C"), ("u1", "A", "C")]:
        d.push(ann, w, l)
    return d


def _rankings():
    d = propagon.RankingsDataset()
    for b in [["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["A", "B", "C"]]:
        d.push_ranking(b)
    return d


def _trajectories():
    d = propagon.TrajectoriesDataset()
    for ep in [[("s0", 0.0), ("s1", 1.0)], [("s0", 0.0), ("s2", 0.0)], [("s1", 1.0), ("s0", 0.5)]]:
        for s, r in ep:
            d.push_step(s, r)
        d.end_episode()
    return d


def _odds():
    d = propagon.OddsDataset()
    d.push_event([("home", 2.1), ("draw", 3.4), ("away", 3.9)])
    d.push_event([("p", 1.8), ("q", 2.2)])
    return d


def _forecasts():
    d = propagon.ForecastDataset()
    d.push_source("s1", [("a", 0.5), ("b", 0.3), ("c", 0.2)])
    d.push_source("s2", [("a", 0.4), ("b", 0.4), ("c", 0.2)])
    return d


def _market():
    d = propagon.MarketDataset()
    for o, s in [("yes", 30.0), ("no", 5.0), ("yes", 10.0)]:
        d.push_trade(o, s)
    return d


# (factory, fixture builder) for every fittable algorithm.
ALGORITHMS = [
    (lambda: propagon.Elo(), _games),
    (lambda: propagon.Glicko2(), _games),
    (lambda: propagon.MovElo(), _games),
    (lambda: propagon.MElo(), _games),
    (lambda: propagon.GeneralizedBt(), _games),
    (lambda: propagon.TeamBradleyTerry(), _games),
    (lambda: propagon.PageRank(), _graph),
    (lambda: propagon.Hits(), _graph),
    (lambda: propagon.BiRank(), _graph),
    (lambda: propagon.Degree(), _graph),
    (lambda: propagon.Harmonic(), _graph),
    (lambda: propagon.Katz(), _graph),
    (lambda: propagon.KCore(), _graph),
    (lambda: propagon.LeaderRank(), _graph),
    (lambda: propagon.Bandit(), _rewards),
    (lambda: propagon.SlidingWindowUcb(), _rewards),
    (lambda: propagon.LinUcb(), _contextual),
    (lambda: propagon.CrowdBt(), _annotated),
    (lambda: propagon.WengLin(), _matchups),
    (lambda: propagon.PlackettLuce(), _rankings),
    (lambda: propagon.Footrule(), _rankings),
    (lambda: propagon.Mallows(), _rankings),
    (lambda: propagon.Mc4(), _rankings),
    (lambda: propagon.McValue(), _trajectories),
    (lambda: propagon.BehaviorCloning(), _trajectories),
    (lambda: propagon.ValueCompare(), _trajectories),
    (lambda: propagon.TdValue(), _trajectories),
    (lambda: propagon.BradleyTerryMM(), _pairwise),
    (lambda: propagon.BradleyTerryLR(), _pairwise),
    (lambda: propagon.BayesianBradleyTerry(), _pairwise),
    (lambda: propagon.Colley(), _pairwise),
    (lambda: propagon.Massey(), _pairwise),
    (lambda: propagon.Keener(), _pairwise),
    (lambda: propagon.ILsr(), _pairwise),
    (lambda: propagon.NashAveraging(), _pairwise),
    (lambda: propagon.OffenseDefense(), _pairwise),
    (lambda: propagon.RandomWalker(), _pairwise),
    (lambda: propagon.RankCentrality(), _pairwise),
    (lambda: propagon.SerialRank(), _pairwise),
    (lambda: propagon.ThurstoneMosteller(), _pairwise),
    (lambda: propagon.Whr(), _pairwise),
    (lambda: propagon.Borda(), _pairwise),
    (lambda: propagon.Copeland(), _pairwise),
    (lambda: propagon.BladeChest(), _pairwise),
    (lambda: propagon.EsRum(), _pairwise),
    (lambda: propagon.HodgeRank(), _pairwise),
    (lambda: propagon.Kemeny(), _pairwise),
    (lambda: propagon.Lsr(), _pairwise),
    (lambda: propagon.WinRate(), _pairwise),
    (lambda: propagon.DuelingBandit(), _pairwise),
    (lambda: propagon.CovariateBt(features=[("A", [1.0]), ("B", [0.0]), ("C", [-1.0])]), _pairwise),
    (lambda: propagon.OddsDevig(), _odds),
    (lambda: propagon.OpinionPool(), _forecasts),
    (lambda: propagon.Lmsr(), _market),
]


@pytest.mark.parametrize("factory,fixture", ALGORITHMS)
def test_every_algorithm_fits_and_round_trips(factory, fixture):
    model = factory().fit(fixture())
    scores = model.sorted_scores()
    assert isinstance(scores, list)
    state = model.save_state()
    # the concrete class round-trips, and top-level load_state finds the right class
    assert type(model).load(state).save_state() == state
    assert propagon.load_state(state).save_state() == state


def test_algorithm_count_is_comprehensive():
    """Sanity: we cover the full catalog (54 fittable algorithms)."""
    assert len(ALGORITHMS) == 54
