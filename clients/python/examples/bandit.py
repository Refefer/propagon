"""Multi-armed bandits: estimate arm values from a reward log, with
incremental state you can persist and resume.

Run after `maturin develop`:  python examples/bandit.py
"""

import propagon


def main() -> None:
    day1 = propagon.RewardsDataset()
    for arm, reward in [("A", 1.0), ("B", 0.0), ("A", 1.0), ("C", 0.0), ("B", 1.0)]:
        day1.push(arm, reward)

    bandit = propagon.Bandit(policy=propagon.BanditPolicy.ucb1(exploration=2.0), seed=1)
    model = bandit.init()
    bandit.update(model, day1)
    print("After day 1:", model.sorted_scores())

    # Persist the selection state, then resume the next day without replay.
    state = model.save_state()
    model = propagon.BanditModel.load(state)

    day2 = propagon.RewardsDataset()
    for arm, reward in [("C", 1.0), ("C", 1.0), ("A", 0.0)]:
        day2.push(arm, reward)

    bandit.update(model, day2)
    print("After day 2:", model.sorted_scores())


if __name__ == "__main__":
    main()
