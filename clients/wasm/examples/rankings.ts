// Rank aggregation: Plackett-Luce over ballots, Borda over pairwise data.
//   node examples/rankings.ts
import { datasets, rankings, pairwise } from "../index.js";

const ballots = [
  ["espresso", "latte", "drip", "cold-brew"],
  ["espresso", "drip", "latte", "cold-brew"],
  ["latte", "espresso", "cold-brew", "drip"],
  ["espresso", "latte", "cold-brew", "drip"],
];

const r = new datasets.RankingsDataset();
for (const ballot of ballots) r.pushRanking(ballot);

console.log("Plackett-Luce:");
for (const [name, score] of rankings.fitPlackettLuce({}, r).sortedScores()) {
  console.log(`  ${name.padEnd(10)} ${score.toFixed(4)}`);
}

// Borda over pairwise preferences extracted from the same ballots.
const p = new datasets.PairwiseDataset();
for (const ballot of ballots) {
  for (let i = 0; i < ballot.length; i++) {
    for (let j = i + 1; j < ballot.length; j++) p.push(ballot[i], ballot[j], 1);
  }
}
console.log("\nBorda (from pairwise):");
for (const [name, score] of pairwise.fitBorda(p).sortedScores()) {
  console.log(`  ${name.padEnd(10)} ${score.toFixed(4)}`);
}
