"""Rank aggregation: combine best-first ballots into one ranking.

Run after `maturin develop`:  python examples/rankings.py
"""

import propagon


def main() -> None:
    ballots = propagon.RankingsDataset()
    for ballot in [
        ["espresso", "latte", "drip", "tea"],
        ["espresso", "drip", "latte", "tea"],
        ["latte", "espresso", "tea", "drip"],
        ["drip", "espresso", "latte", "tea"],
    ]:
        ballots.push_ranking(ballot)

    print("Plackett-Luce strengths:")
    for name, score in propagon.PlackettLuce().fit(ballots).sorted_scores():
        print(f"  {name:9s} {score:.4f}")

    # Kemeny / Borda work on pairwise data; lower ballots to ordered pairs.
    pairs = propagon.PairwiseDataset()
    for ballot in [
        ["espresso", "latte", "drip", "tea"],
        ["espresso", "drip", "latte", "tea"],
        ["latte", "espresso", "tea", "drip"],
        ["drip", "espresso", "latte", "tea"],
    ]:
        for i in range(len(ballot)):
            for j in range(i + 1, len(ballot)):
                pairs.push(ballot[i], ballot[j])

    print("\nBorda count:")
    for name, score in propagon.Borda().fit(pairs).sorted_scores():
        print(f"  {name:9s} {score:.0f}")


if __name__ == "__main__":
    main()
