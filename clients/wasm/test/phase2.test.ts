import { describe, it, expect } from "vitest";
import {
  datasets,
  games,
  graph,
  pairwise,
  rewards,
  matchups,
  annotated,
  rankings,
  trajectories,
} from "../dist/propagon.js";

describe("phase-2 algorithms", () => {
  it("Glicko-2 fits and exposes per-player states", () => {
    const d = new datasets.GamesDataset();
    d.pushPair("a", "b", 1);
    d.pushPair("a", "c", 1);
    d.pushPair("b", "c", 1);
    const m = games.fitGlicko2({ tau: 0.5 }, d);
    expect(m.algorithm()).toContain("glicko");
    const players = m.players();
    expect(players.length).toBe(3);
    const [, state] = players[0];
    expect(state.r).toBeGreaterThan(0);
    expect(state.rd).toBeGreaterThan(0);
  });

  it("HITS fits over a graph", () => {
    const g = new datasets.GraphDataset();
    g.push("a", "b", 1);
    g.push("b", "c", 1);
    g.push("a", "c", 1);
    const m = graph.fitHits({}, g);
    expect(m.sortedScores().length).toBe(3);
  });

  it("Bradley-Terry (MM) and Borda rank a round robin", () => {
    const d = new datasets.PairwiseDataset();
    d.push("a", "b", 1);
    d.push("a", "c", 1);
    d.push("b", "c", 1);
    const bt = pairwise.fitBradleyTerryMm({}, d);
    expect(bt.sortedScores()[0][0]).toBe("a");
    const borda = pairwise.fitBorda(d);
    expect(borda.sortedScores()[0][0]).toBe("a");
  });

  it("Bandit replays a reward log under a UCB1 policy with state resume", () => {
    const d = new datasets.RewardsDataset();
    for (const [arm, r] of [
      ["A", 1],
      ["A", 1],
      ["B", 0],
      ["A", 1],
      ["B", 0],
    ] as [string, number][]) {
      d.push(arm, r);
    }
    const m = rewards.initBandit({ policy: { tag: "ucb1", val: 2.0 }, seed: 1n });
    m.update(d);
    expect(m.score("A")!).toBeGreaterThan(m.score("B")!);
    // resume from saved state
    const state = m.saveState();
    const resumed = rewards.loadBandit(state);
    expect(resumed.saveState()).toBe(state);
  });

  it("Weng-Lin rates team matches and exposes (mu, sigma)", () => {
    const d = new datasets.MatchupsDataset();
    d.pushMatch([["maxpax", "spirit"], ["trigger", "showtime"]], new Uint32Array([1, 2]));
    d.pushMatch([["maxpax", "trigger"], ["spirit", "showtime"]], new Uint32Array([1, 2]));
    const m = matchups.fitWengLin({}, d);
    const ratings = m.ratings();
    expect(ratings.length).toBe(4);
    const [, r] = ratings[0];
    expect(r.mu).toBeTypeOf("number");
    expect(r.sigma).toBeGreaterThan(0);
  });

  it("Crowd-BT recovers order from annotator-tagged votes", () => {
    const d = new datasets.AnnotatedPairsDataset();
    for (let i = 0; i < 20; i++) {
      d.push("alice", "good", "bad", 1);
      d.push("alice", "good", "meh", 1);
      d.push("alice", "meh", "bad", 1);
    }
    const m = annotated.fitCrowdBt({}, d);
    const order = m.sortedScores().map(([n]) => n);
    expect(order.indexOf("good")).toBeLessThan(order.indexOf("bad"));
  });

  it("Plackett-Luce aggregates ballots", () => {
    const d = new datasets.RankingsDataset();
    d.pushRanking(["a", "b", "c"]);
    d.pushRanking(["a", "c", "b"]);
    d.pushRanking(["b", "a", "c"]);
    const m = rankings.fitPlackettLuce({}, d);
    expect(m.sortedScores()[0][0]).toBe("a");
  });

  it("Monte-Carlo value ranks funnel states", () => {
    const d = new datasets.TrajectoriesDataset();
    for (const ep of [
      [["landing", 0], ["cart", 0], ["checkout", 1]],
      [["landing", 0], ["cart", 0]],
      [["landing", 0], ["cart", 0], ["checkout", 1]],
    ] as [string, number][][]) {
      for (const [s, r] of ep) d.pushStep(s, r);
      d.endEpisode();
    }
    const m = trajectories.fitMcValue({ gamma: 0.9 }, d);
    expect(m.score("checkout")!).toBeGreaterThan(m.score("landing")!);
  });
});
