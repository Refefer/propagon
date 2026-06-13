"""Parameter enums: factory constructors and unit-enum string parsing."""

import propagon
import pytest


def test_game_outcome_factories():
    assert "side1_win" in repr(propagon.GameOutcome.side1_win(2.0))
    assert "side2_win" in repr(propagon.GameOutcome.side2_win(3.0))
    assert repr(propagon.GameOutcome.tie()) == "GameOutcome.tie()"


def test_bandit_policy_factories_roundtrip_through_repr():
    p = propagon.Bandit(policy=propagon.BanditPolicy.ucb1(2.0), seed=1)
    assert "Ucb1" in repr(p) or "ucb1" in repr(p)
    # every factory constructs without error
    propagon.BanditPolicy.greedy()
    propagon.BanditPolicy.epsilon_greedy(0.1)
    propagon.BanditPolicy.thompson_beta(1.0, 1.0)
    propagon.BanditPolicy.thompson_gaussian(0.0, 1.0)
    propagon.BanditPolicy.kl_ucb(1.0)
    propagon.BanditPolicy.exp3(0.1)


def test_teleport_seeds():
    t = propagon.Teleport.seeds([("a", 1.0), ("b", 2.0)])
    assert "Seeds" in repr(t)
    propagon.Teleport.uniform()


def test_data_carrying_enums_construct():
    propagon.DuelingPolicy.rucb(0.5)
    propagon.DuelingPolicy.double_thompson(0.5)
    propagon.Granularity.global_()  # `global` is a Python keyword
    propagon.Granularity.per_state(separator="/")
    propagon.SourceBudget.all()
    propagon.SourceBudget.sample(10, 1)
    propagon.KemenyPasses.auto()
    propagon.KemenyPasses.fixed(5)
    propagon.Winsorize.off()
    propagon.Winsorize.percentile(0.05)
    propagon.PairwiseTests.off()
    propagon.PairwiseTests.on(1000)


def test_unit_enum_strings_are_accepted():
    propagon.PageRank(sink="reverse")
    propagon.PageRank(sink="uniform")
    propagon.Degree(direction="in")
    propagon.Degree(direction="out")
    propagon.WinRate(confidence="P90")
    propagon.Lsr(estimator="monte-carlo")


def test_bad_unit_enum_string_raises():
    with pytest.raises(propagon.InvalidInputError):
        propagon.Degree(direction="sideways")
    with pytest.raises(propagon.InvalidInputError):
        propagon.WinRate(confidence="P99")
