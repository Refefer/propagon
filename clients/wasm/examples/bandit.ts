// Multi-armed bandit with state persistence: fit day 1, save, resume day 2
// without replaying day 1.
//   node examples/bandit.ts
import { datasets, rewards, loadState } from "../index.js";

const day1 = new datasets.RewardsDataset();
for (const [arm, reward] of [
  ["A", 1],
  ["B", 0],
  ["A", 1],
  ["C", 1],
  ["B", 0],
] as [string, number][]) {
  day1.push(arm, reward);
}

const model = rewards.initBandit({ policy: { tag: "ucb1", val: 2.0 }, seed: 1n });
model.update(day1);
console.log("After day 1:", model.sortedScores());

const state = model.saveState();
const resumed = loadState(state) as ReturnType<typeof rewards.initBandit>;

const day2 = new datasets.RewardsDataset();
for (const [arm, reward] of [["C", 1], ["C", 1], ["A", 0]] as [string, number][]) {
  day2.push(arm, reward);
}
resumed.update(day2);
console.log("After day 2:", resumed.sortedScores());
