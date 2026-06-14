// Weng-Lin (OpenSkill) ratings from team matches with rotating partners.
//   node examples/matchups.ts
import { datasets, matchups } from "../index.js";

const d = new datasets.MatchupsDataset();
// teams (best-first rosters), ranks (1 = winner). Partners rotate match to match.
const fixtures: [string[][], number[]][] = [
  [[["maxpax", "spirit"], ["trigger", "showtime"]], [1, 2]],
  [[["maxpax", "trigger"], ["spirit", "showtime"]], [1, 2]],
  [[["spirit", "trigger"], ["maxpax", "showtime"]], [2, 1]],
  [[["maxpax", "showtime"], ["spirit", "trigger"]], [1, 2]],
];
for (const [teams, ranks] of fixtures) {
  d.pushMatch(teams, new Uint32Array(ranks));
}

const model = matchups.fitWengLin({}, d);
console.log("Player ratings (ordinal = mu - 3·sigma):");
for (const [name, r] of model.ratings()) {
  const ordinal = r.mu - 3 * r.sigma;
  console.log(
    `  ${name.padEnd(10)} mu=${r.mu.toFixed(3)}  sigma=${r.sigma.toFixed(3)}  ordinal=${ordinal.toFixed(3)}`,
  );
}
