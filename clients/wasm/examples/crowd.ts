// Crowdsourced votes with unreliable annotators: Crowd-BT vs plain Bradley-Terry.
//   node examples/crowd.ts
import { datasets, annotated, pairwise } from "../index.js";

const TRUE_ORDER = ["sonnet", "haiku", "limerick", "ballad", "ode", "epigram"];

// Deterministic synthetic votes from a diligent, a spammer, and an adversary.
function mulberry32(a: number): () => number {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function* votes(): Generator<[string, string, string]> {
  const rand = mulberry32(2024);
  // A diligent annotator and an adversary who votes against the truth. Plain
  // Bradley-Terry sees the two cancel out; Crowd-BT detects that mallory is
  // inverted and turns her votes back into signal.
  const annotators: [string, number][] = [["alice", 0.9], ["mallory", 0.1]];
  for (let n = 0; n < 400; n++) {
    for (const [name, fidelity] of annotators) {
      let i = Math.floor(rand() * TRUE_ORDER.length);
      let j = Math.floor(rand() * TRUE_ORDER.length);
      if (i === j) continue;
      if (i > j) [i, j] = [j, i];
      const [better, worse] = [TRUE_ORDER[i], TRUE_ORDER[j]];
      yield rand() < fidelity ? [name, better, worse] : [name, worse, better];
    }
  }
}

const crowd = new datasets.AnnotatedPairsDataset();
const naive = new datasets.PairwiseDataset();
for (const [annotator, winner, loser] of votes()) {
  crowd.push(annotator, winner, loser, 1);
  naive.push(winner, loser, 1);
}

console.log(`True order: ${TRUE_ORDER.join(" > ")}\n`);
console.log("Crowd-BT (reliability-aware):");
for (const [name, score] of annotated.fitCrowdBt({}, crowd).sortedScores()) {
  console.log(`  ${name.padEnd(10)} ${score.toFixed(4)}`);
}
console.log("\nPlain Bradley-Terry (ignores who voted):");
for (const [name, score] of pairwise.fitBradleyTerryMm({}, naive).sortedScores()) {
  console.log(`  ${name.padEnd(10)} ${score.toFixed(4)}`);
}
