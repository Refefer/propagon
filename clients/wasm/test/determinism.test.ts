import { describe, it, expect } from "vitest";
import { datasets, games, graph, rewards, functions } from "../index.js";

function games3() {
  const d = new datasets.GamesDataset();
  for (const [w, l] of [["a", "b"], ["a", "c"], ["b", "c"], ["c", "a"]]) {
    d.pushPair(w, l, 1);
  }
  return d;
}

describe("determinism", () => {
  it("seeded MElo is bit-stable across runs", () => {
    const a = games.fitMElo({ seed: 7n }, games3()).scoresBulk();
    const b = games.fitMElo({ seed: 7n }, games3()).scoresBulk();
    expect(Array.from(a.scores)).toEqual(Array.from(b.scores));
    expect(a.ids).toEqual(b.ids);
  });

  it("seeded Bandit replay is bit-stable across runs", () => {
    const log = () => {
      const r = new datasets.RewardsDataset();
      for (const [arm, v] of [["A", 1], ["B", 0], ["A", 1], ["C", 1]] as [string, number][]) {
        r.push(arm, v);
      }
      return r;
    };
    const a = rewards
      .fitBandit({ policy: { tag: "epsilon-greedy", val: 0.1 }, seed: 3n }, log())
      .scoresBulk();
    const b = rewards
      .fitBandit({ policy: { tag: "epsilon-greedy", val: 0.1 }, seed: 3n }, log())
      .scoresBulk();
    expect(Array.from(a.scores)).toEqual(Array.from(b.scores));
  });

  it("PageRank is deterministic", () => {
    const g = () => {
      const x = new datasets.GraphDataset();
      x.push("a", "b", 1);
      x.push("b", "c", 1);
      x.push("c", "a", 1);
      return x;
    };
    expect(graph.fitPageRank({}, g()).sortedScores()).toEqual(
      graph.fitPageRank({}, g()).sortedScores(),
    );
  });
});

describe("reference vectors", () => {
  it("Wilson interval matches the Newcombe table (8/10, z=1.96)", () => {
    const [lo, hi] = functions.wilsonInterval(8, 2, 1.96);
    expect(lo).toBeCloseTo(0.4901, 3);
    expect(hi).toBeCloseTo(0.9433, 3);
  });
});
