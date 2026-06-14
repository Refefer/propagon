"""Rank funnel states by expected value from session trajectories.

Each episode is a sequence of (state, reward) steps. Monte-Carlo value
estimates the discounted return reachable from each state — contrast with
behavior cloning, which ranks states by how often they're *visited*
(popularity), not how valuable they are.

Run after `maturin develop`:  python examples/trajectories.py
"""

import propagon

# A tiny checkout funnel. Most sessions browse and leave; a few convert.
EPISODES = [
    [("landing", 0.0), ("browse", 0.0), ("cart", 0.0), ("checkout", 1.0)],
    [("landing", 0.0), ("browse", 0.0), ("cart", 0.0)],
    [("landing", 0.0), ("browse", 0.0)],
    [("landing", 0.0), ("browse", 0.0), ("cart", 0.0), ("checkout", 1.0)],
    [("landing", 0.0), ("search", 0.0), ("cart", 0.0), ("checkout", 1.0)],
    [("landing", 0.0), ("search", 0.0)],
    [("landing", 0.0), ("browse", 0.0), ("search", 0.0), ("cart", 0.0)],
    [("landing", 0.0), ("browse", 0.0), ("cart", 0.0), ("checkout", 1.0)],
]


def main() -> None:
    data = propagon.TrajectoriesDataset()
    for episode in EPISODES:
        for state, reward in episode:
            data.push_step(state, reward)
        data.end_episode()

    print("Monte-Carlo state values (expected discounted return):")
    for state, value in propagon.McValue(gamma=0.9).fit(data).sorted_scores():
        print(f"  {state:10s} {value:.4f}")

    print("\nBehavior cloning (visit frequency = popularity, not value):")
    for state, freq in propagon.BehaviorCloning().fit(data).sorted_scores():
        print(f"  {state:10s} {freq:.4f}")


if __name__ == "__main__":
    main()
