// Tournament ratings from head-to-head games: Elo and Glicko-2.
//
// Build first (npm run build), then run with Node 24+ type stripping:
//   node examples/tournament.ts
import { datasets, games } from "../index.js";

const d = new datasets.GamesDataset();
for (const [winner, loser] of [
  ["sharks", "bears"],
  ["sharks", "wolves"],
  ["bears", "wolves"],
  ["wolves", "bears"],
  ["sharks", "bears"],
]) {
  d.pushPair(winner, loser, 1);
}

console.log("Elo:");
for (const [name, score] of games.fitElo({ k: 24 }, d).sortedScores()) {
  console.log(`  ${name.padEnd(8)} ${score.toFixed(1)}`);
}

console.log("\nGlicko-2 (rating ± 2·RD):");
const g = games.fitGlicko2({}, d);
for (const [name, state] of g.players()) {
  console.log(
    `  ${name.padEnd(8)} ${state.r.toFixed(0)} ± ${(2 * state.rd).toFixed(0)}`,
  );
}
