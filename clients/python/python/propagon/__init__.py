"""Python bindings for propagon — ranking from revealed preferences.

propagon turns match outcomes, pairwise choices, rankings, interaction graphs,
reward events, and trajectories into rankings, via a broad catalog of algorithms
(Bradley-Terry, Elo, Glicko-2, Luce spectral ranking, rank aggregation,
centrality, multi-armed bandits, value estimation).

The shape of the API mirrors the Rust crate:

1. Build one of the dataset types (string ids are interned for you).
2. Configure an algorithm (all parameters have defaults).
3. ``fit`` (batch) or ``init`` + ``update`` (incremental).
4. Read ``sorted_scores``/``top``, or persist with ``save_state`` and resume
   later with :func:`load_state`.

>>> import propagon
>>> games = propagon.GamesDataset()
>>> _ = games.push_pair("ARI", "COL")
>>> _ = games.push_pair("ARI", "NYM")
>>> _ = games.push_pair("COL", "NYM")
>>> model = propagon.Glicko2().fit(games)
>>> model.top(1)[0][0]
'ARI'
"""

from ._propagon import *  # noqa: F401,F403
from ._propagon import __all__ as _all  # re-exported symbol list from Rust

# Surface the compiled module's curated export list as the package's.
__all__ = list(_all)

try:  # version is injected by the compiled module
    from ._propagon import __version__  # noqa: F401
except ImportError:  # pragma: no cover - always present in a built wheel
    __version__ = "0+unknown"
