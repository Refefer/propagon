// Rank funnel states by value (Monte-Carlo) vs popularity (behavior cloning).
//   node examples/trajectories.ts
import { datasets, trajectories } from "../index.js";

const episodes: [string, number][][] = [
  [["landing", 0], ["browse", 0], ["cart", 0], ["checkout", 1]],
  [["landing", 0], ["browse", 0], ["cart", 0]],
  [["landing", 0], ["browse", 0]],
  [["landing", 0], ["search", 0], ["cart", 0], ["checkout", 1]],
  [["landing", 0], ["browse", 0], ["cart", 0], ["checkout", 1]],
  [["landing", 0], ["search", 0]],
];

const d = new datasets.TrajectoriesDataset();
for (const ep of episodes) {
  for (const [state, reward] of ep) d.pushStep(state, reward);
  d.endEpisode();
}

console.log("Monte-Carlo state values (expected discounted return):");
for (const [state, v] of trajectories.fitMcValue({ gamma: 0.9 }, d).sortedScores()) {
  console.log(`  ${state.padEnd(10)} ${v.toFixed(4)}`);
}
console.log("\nBehavior cloning (visit frequency = popularity):");
for (const [state, f] of trajectories.fitBehaviorCloning({}, d).sortedScores()) {
  console.log(`  ${state.padEnd(10)} ${f.toFixed(4)}`);
}
