"""Weng-Lin (OpenSkill) ratings from team matches with rotating partners.

Each match ranks teams best-first; players within a team share the result,
and Weng-Lin splits every team update across its members so individuals
emerge even when partners change between matches.

Run after `maturin develop`:  python examples/matchups.py
"""

import propagon


def main() -> None:
    data = propagon.MatchupsDataset()
    # teams = list of rosters (best-first); ranks = finishing place per team
    # (lower is better, equal ranks tie). Partners rotate match to match.
    for teams, ranks in [
        ([["maxpax", "spirit"], ["trigger", "showtime"]], [1, 2]),
        ([["maxpax", "trigger"], ["spirit", "showtime"]], [1, 2]),
        ([["spirit", "trigger"], ["maxpax", "showtime"]], [2, 1]),
        ([["maxpax", "showtime"], ["spirit", "trigger"]], [1, 2]),
        ([["spirit", "showtime"], ["maxpax", "trigger"]], [2, 1]),
    ]:
        data.push_match(teams, ranks)

    model = propagon.WengLin().fit(data)

    print("Player ratings (ordinal = mu - 3*sigma, the conservative display):")
    for name, rating in model.ratings():
        ordinal = rating.mu - 3.0 * rating.sigma
        print(f"  {name:10s} mu={rating.mu:6.3f}  sigma={rating.sigma:5.3f}  ordinal={ordinal:6.3f}")

    print("\nRanked by ordinal score:")
    for name, score in model.sorted_scores():
        print(f"  {name:10s} {score:.4f}")


if __name__ == "__main__":
    main()
