"""Seeded stochastic algorithms reproduce their output exactly."""

import propagon


def _games():
    d = propagon.GamesDataset()
    for w, l in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A")]:
        d.push_pair(w, l)
    return d


def test_melo_is_reproducible_at_fixed_seed():
    a = propagon.MElo(seed=7).fit(_games()).sorted_scores()
    b = propagon.MElo(seed=7).fit(_games()).sorted_scores()
    assert a == b


def test_bandit_is_reproducible_at_fixed_seed():
    def rewards():
        d = propagon.RewardsDataset()
        for arm, r in [("a", 1.0), ("b", 0.0), ("a", 1.0), ("c", 0.0), ("b", 1.0)]:
            d.push(arm, r)
        return d

    pol = propagon.BanditPolicy.thompson_beta(1.0, 1.0)
    a = propagon.Bandit(policy=pol, seed=3).fit(rewards()).sorted_scores()
    b = propagon.Bandit(policy=pol, seed=3).fit(rewards()).sorted_scores()
    assert a == b


def test_sorted_scores_breaks_ties_by_name():
    """Determinism of the public ordering: equal scores sort by name ascending."""
    # Borda over a fully symmetric pair gives equal scores; order must be stable.
    d = propagon.PairwiseDataset()
    d.push("B", "A")
    d.push("A", "B")
    names = [n for n, _ in propagon.Borda().fit(d).sorted_scores()]
    assert names == sorted(names)
