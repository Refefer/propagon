"""Per-model extra accessors and the common model surface."""

import propagon


def _games():
    d = propagon.GamesDataset()
    for w, l in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C")]:
        d.push_pair(w, l)
    return d


def test_glicko2_players_extra():
    model = propagon.Glicko2().fit(_games())
    players = model.players()
    assert all(isinstance(name, str) for name, _ in players)
    name, state = players[0]
    # PlayerState exposes r / rd / sigma and bounds()
    assert state.r == model.score(name)
    lo, hi = state.bounds()
    assert lo < state.r < hi


def test_weng_lin_ratings_extra():
    m = propagon.MatchupsDataset()
    for teams, ranks in [([["A"], ["B"]], [1, 2]), ([["B"], ["C"]], [1, 2]), ([["C"], ["A"]], [1, 2])]:
        m.push_match(teams, ranks)
    model = propagon.WengLin().fit(m)
    ratings = dict(model.ratings())
    assert set(ratings) == {"A", "B", "C"}
    r = ratings["A"]
    assert hasattr(r, "mu") and hasattr(r, "sigma")
    assert r.sigma > 0


def test_common_model_surface():
    model = propagon.Glicko2().fit(_games())
    scores = model.scores()
    assert isinstance(scores, dict)
    ss = model.sorted_scores()
    assert ss == sorted(ss, key=lambda kv: (-kv[1], kv[0]))  # desc, name-tiebreak
    assert model.top(2) == ss[:2]
    assert model.score("A") == scores["A"]
    assert model.score("nonexistent") is None
    assert model.algorithm == "glicko2"
