import { describe, it, expect } from "vitest";
import {
  datasets,
  games,
  graph,
  pairwise,
  rewards,
  rankings,
  trajectories,
  loadState,
} from "../index.js";

// A near-transitive round robin a > b > c > d where the favorite wins most but
// every entity has at least one win and one loss (so MM/LSR/offense-defense and
// Thurstone don't hit their undefeated/winless/perfect-separation guards).
function roundRobin() {
  const d = new datasets.PairwiseDataset();
  const order = ["a", "b", "c", "d"];
  for (let i = 0; i < order.length; i++) {
    for (let j = i + 1; j < order.length; j++) {
      d.push(order[i], order[j], 3); // higher-ranked usually wins
      d.push(order[j], order[i], 1); // but the underdog wins sometimes
    }
  }
  return d;
}

function smallGraph() {
  const g = new datasets.GraphDataset();
  for (const [s, t] of [
    ["a", "b"],
    ["b", "c"],
    ["c", "a"],
    ["a", "c"],
    ["d", "a"],
  ]) {
    g.push(s, t, 1);
  }
  return g;
}

describe("phase-3 pairwise coverage", () => {
  const fits: Array<[string, (d: any) => any]> = [
    ["bradley-terry-mm", (d) => pairwise.fitBradleyTerryMm({}, d)],
    ["bradley-terry-lr", (d) => pairwise.fitBradleyTerryLr({}, d)],
    ["bayesian-bradley-terry", (d) => pairwise.fitBayesianBradleyTerry({}, d)],
    ["colley", (d) => pairwise.fitColley({}, d)],
    ["massey", (d) => pairwise.fitMassey({}, d)],
    ["keener", (d) => pairwise.fitKeener({}, d)],
    ["i-lsr", (d) => pairwise.fitILsr({}, d)],
    ["nash-averaging", (d) => pairwise.fitNashAveraging({}, d)],
    ["offense-defense", (d) => pairwise.fitOffenseDefense({}, d)],
    ["random-walker", (d) => pairwise.fitRandomWalker({}, d)],
    ["rank-centrality", (d) => pairwise.fitRankCentrality({}, d)],
    ["serial-rank", (d) => pairwise.fitSerialRank({}, d)],
    ["thurstone-mosteller", (d) => pairwise.fitThurstoneMosteller({}, d)],
    ["whr", (d) => pairwise.fitWhr({}, d)],
    ["borda", (d) => pairwise.fitBorda(d)],
    ["copeland", (d) => pairwise.fitCopeland(d)],
    ["hodge-rank", (d) => pairwise.fitHodgeRank({}, d)],
    ["kemeny", (d) => pairwise.fitKemeny({}, d)],
    ["lsr", (d) => pairwise.fitLsr({}, d)],
    ["blade-chest", (d) => pairwise.fitBladeChest({}, d)],
    ["es-rum", (d) => pairwise.fitEsRum({}, d)],
  ];

  for (const [name, fit] of fits) {
    it(`fits ${name} and round-trips its state`, () => {
      const model = fit(roundRobin());
      expect(model.sortedScores().length).toBe(4);
      const state = model.saveState();
      const reloaded = loadState(state) as { saveState(): string };
      expect(reloaded.saveState()).toBe(state);
    });
  }

  it("CovariateBt fits with feature vectors", () => {
    const params = {
      features: [
        ["a", [3.0]],
        ["b", [2.0]],
        ["c", [1.0]],
        ["d", [0.0]],
      ] as [string, number[]][],
    };
    const m = pairwise.fitCovariateBt(params, roundRobin());
    expect(m.sortedScores().length).toBe(4);
  });

  it("WinRate (online) and DuelingBandit (online) run over pairwise data", () => {
    const wr = pairwise.initWinRate({});
    wr.update(roundRobin());
    expect(wr.sortedScores().length).toBe(4);
    const db = pairwise.initDuelingBandit({ seed: 1n });
    db.update(roundRobin());
    expect(db.sortedScores().length).toBeGreaterThan(0);
  });
});

describe("phase-3 graph coverage", () => {
  const fits: Array<[string, (g: any) => any]> = [
    ["bi-rank", (g) => graph.fitBiRank({}, g)],
    ["degree", (g) => graph.fitDegree({ direction: "total" }, g)],
    ["harmonic", (g) => graph.fitHarmonic({}, g)],
    ["katz", (g) => graph.fitKatz({}, g)],
    ["k-core", (g) => graph.fitKCore(g)],
    ["leader-rank", (g) => graph.fitLeaderRank({}, g)],
  ];
  for (const [name, fit] of fits) {
    it(`fits ${name}`, () => {
      const m = fit(smallGraph());
      expect(m.sortedScores().length).toBeGreaterThan(0);
    });
  }
});

describe("phase-3 other families", () => {
  it("games: MovElo, MElo, GeneralizedBt, TeamBradleyTerry", () => {
    const d = new datasets.GamesDataset();
    // Balanced: every team has wins and losses (no undefeated/winless team).
    for (const [w, l, n] of [
      ["a", "b", 3],
      ["b", "a", 1],
      ["b", "c", 3],
      ["c", "b", 1],
      ["c", "a", 2],
      ["a", "c", 2],
    ] as [string, string, number][]) {
      for (let i = 0; i < n; i++) d.pushPair(w, l, 1);
    }
    expect(games.fitMovElo({}, d).sortedScores().length).toBe(3);
    expect(games.fitMElo({ seed: 1n }, d).sortedScores().length).toBe(3);
    expect(games.fitGeneralizedBt({ ties: "davidson" }, d).sortedScores().length).toBe(3);
    expect(games.fitTeamBradleyTerry({}, d).sortedScores().length).toBe(3);
  });

  it("rankings: Footrule, Mallows, Mc4", () => {
    const d = new datasets.RankingsDataset();
    d.pushRanking(["a", "b", "c"]);
    d.pushRanking(["a", "c", "b"]);
    d.pushRanking(["b", "a", "c"]);
    expect(rankings.fitFootrule(d).sortedScores().length).toBe(3);
    expect(rankings.fitMallows({}, d).sortedScores().length).toBe(3);
    expect(rankings.fitMc4({}, d).sortedScores().length).toBe(3);
  });

  it("rewards: SlidingWindowUcb + LinUcb (contextual)", () => {
    const r = new datasets.RewardsDataset();
    for (const [arm, v] of [["A", 1], ["A", 1], ["B", 0]] as [string, number][]) r.push(arm, v);
    expect(rewards.fitSlidingWindowUcb({ window: 10 }, r).sortedScores().length).toBe(2);

    const c = new datasets.ContextualRewardsDataset();
    c.push("A", 1, new Float64Array([1, 0]));
    c.push("B", 0, new Float64Array([0, 1]));
    expect(rewards.fitLinUcb({}, c).sortedScores().length).toBe(2);
  });

  it("trajectories: BehaviorCloning, ValueCompare, TdValue", () => {
    const d = new datasets.TrajectoriesDataset();
    for (let i = 0; i < 4; i++) {
      d.pushStep("s0", 0);
      d.pushStep("s1", 1);
      d.endEpisode();
    }
    expect(trajectories.fitBehaviorCloning({}, d).sortedScores().length).toBe(2);
    expect(trajectories.fitValueCompare({ replicates: 50 }, d).sortedScores().length).toBe(2);
    const td = trajectories.initTdValue({});
    td.update(d);
    expect(td.sortedScores().length).toBe(2);
  });

  it("loadState dispatches across families", () => {
    const g = games.fitGlicko2({}, (() => {
      const d = new datasets.GamesDataset();
      d.pushPair("a", "b", 1);
      return d;
    })());
    const state = g.saveState();
    const back = loadState(state) as { algorithm(): string };
    expect(back.algorithm()).toContain("glicko");
    expect(() => loadState('{"algorithm":"nope"}')).toThrow();
  });
});
