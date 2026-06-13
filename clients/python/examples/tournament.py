"""Tournament ranking: rate teams from head-to-head games.

Run after `maturin develop`:  python examples/tournament.py
"""

import propagon


def main() -> None:
    games = propagon.GamesDataset()
    # winner, loser (each game is one observation)
    for winner, loser in [
        ("Sharks", "Bears"),
        ("Sharks", "Wolves"),
        ("Bears", "Wolves"),
        ("Wolves", "Bears"),
        ("Sharks", "Bears"),
    ]:
        games.push_pair(winner, loser)

    # Incremental rating systems
    for algo in (propagon.Elo(k=24.0), propagon.Glicko2()):
        model = algo.fit(games)
        print(f"\n{type(algo).__name__}:")
        for name, score in model.sorted_scores():
            print(f"  {name:8s} {score:8.2f}")

    # Glicko-2 also exposes rating deviation and volatility per team.
    print("\nGlicko-2 detail (rating ± 2·RD):")
    model = propagon.Glicko2().fit(games)
    for name, state in model.players():
        lo, hi = state.bounds()
        print(f"  {name:8s} {state.r:7.1f}  [{lo:.0f}, {hi:.0f}]  sigma={state.sigma:.4f}")


if __name__ == "__main__":
    main()
