"""Crowdsourced pairwise votes with unreliable annotators (Crowd-BT).

Three annotators judge the same items with very different reliability:
alice is diligent, bob is a coin-flipping spammer, mallory votes against
the truth. Crowd-BT weights each annotator's votes by an estimated
reliability, so mallory's inverted votes *add* signal instead of poisoning
the fit. Contrast with plain Bradley-Terry that ignores who voted.

Run after `maturin develop`:  python examples/crowd.py
"""

import random

import propagon

TRUE_ORDER = ["sonnet", "haiku", "limerick", "ballad", "ode", "epigram"]


def votes():
    """Deterministic synthetic votes: (annotator, winner, loser)."""
    rng = random.Random(2024)
    annotators = [("alice", 0.92), ("bob", 0.50), ("mallory", 0.12)]
    for _ in range(200):
        for name, fidelity in annotators:
            i, j = sorted(rng.sample(range(len(TRUE_ORDER)), 2))
            better, worse = TRUE_ORDER[i], TRUE_ORDER[j]  # i<j => better truth
            if rng.random() < fidelity:
                yield name, better, worse
            else:
                yield name, worse, better


def main() -> None:
    crowd = propagon.AnnotatedPairsDataset()
    naive = propagon.PairwiseDataset()
    for annotator, winner, loser in votes():
        crowd.push(annotator, winner, loser)
        naive.push(winner, loser)  # same votes, annotator column dropped

    print(f"True order: {' > '.join(TRUE_ORDER)}\n")

    print("Crowd-BT (reliability-aware):")
    for name, score in propagon.CrowdBt().fit(crowd).sorted_scores():
        print(f"  {name:10s} {score:7.4f}")

    print("\nPlain Bradley-Terry (ignores who voted):")
    for name, score in propagon.BradleyTerryMM().fit(naive).sorted_scores():
        print(f"  {name:10s} {score:7.4f}")


if __name__ == "__main__":
    main()
