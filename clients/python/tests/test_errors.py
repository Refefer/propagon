"""Error paths: every propagon::Error variant surfaces as the right Python
exception subclass."""

import propagon
import pytest


def test_empty_dataset_raises_empty_dataset_error():
    with pytest.raises(propagon.EmptyDatasetError):
        propagon.PageRank().fit(propagon.GraphDataset())


def test_duplicate_player_raises_invalid_input():
    d = propagon.GamesDataset()
    with pytest.raises(propagon.InvalidInputError):
        d.push_pair("x", "x")  # winner == loser


def test_multi_player_side_on_1v1_algo_raises_invalid_input():
    """A multi-player roster fed to Glicko-2 (a 1v1 method) is a typed error."""
    d = propagon.GamesDataset()
    d.push_game(["A", "B"], ["C"], propagon.GameOutcome.side1_win())
    d.push_pair("A", "C")
    with pytest.raises(propagon.InvalidInputError):
        propagon.Glicko2().fit(d)


def test_param_mismatch_on_online_update_raises():
    """Updating a model with a differently-configured updater is rejected."""
    g = propagon.GamesDataset()
    g.push_pair("a", "b")
    model = propagon.Glicko2(tau=0.5).init()
    with pytest.raises(propagon.ParamMismatchError):
        propagon.Glicko2(tau=1.0).update(model, g)


def test_unknown_enum_string_raises_invalid_input_with_allowed_set():
    with pytest.raises(propagon.InvalidInputError) as exc:
        propagon.PageRank(sink="bogus")
    assert "reverse" in str(exc.value)  # the allowed set is shown


def test_malformed_state_raises_state_error():
    with pytest.raises(propagon.StateError):
        propagon.load_state("not json at all")


def test_future_schema_version_raises_state_error():
    future = '{"propagon":999,"kind":"model","algorithm":"glicko2","params":{},"entities":0}\n'
    with pytest.raises(propagon.StateError):
        propagon.Glicko2Model.load(future)


def test_exception_hierarchy():
    assert issubclass(propagon.AlgorithmMismatchError, propagon.StateError)
    assert issubclass(propagon.StateError, propagon.PropagonError)
    for name in [
        "InvalidInputError", "EmptyDatasetError", "NumericError",
        "ParamMismatchError", "IoError",
    ]:
        assert issubclass(getattr(propagon, name), propagon.PropagonError)
    # everything is catchable as the base class
    assert issubclass(propagon.PropagonError, Exception)
