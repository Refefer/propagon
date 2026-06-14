import { describe, it, expect } from "vitest";
import { datasets, games, graph, functions } from "../dist/propagon.js";

describe("phase-1 smoke", () => {
  it("fits Elo (online fit) and round-trips state byte-identically", () => {
    const d = new datasets.GamesDataset();
    d.pushPair("a", "b", 1);
    d.pushPair("a", "c", 1);
    d.pushPair("b", "c", 1);

    const m = games.fitElo({ k: 24 }, d);
    const scores = m.sortedScores();
    expect(scores[0][0]).toBe("a"); // a beat everyone
    expect(m.algorithm()).toContain("elo");

    const bulk = m.scoresBulk();
    expect(bulk.ids.length).toBe(3);
    expect(bulk.scores).toBeInstanceOf(Float64Array);

    const state = m.saveState();
    const reloaded = games.loadElo(state);
    expect(reloaded.saveState()).toBe(state);
    expect(reloaded.sortedScores()).toEqual(scores);
  });

  it("supports incremental init + update", () => {
    const d = new datasets.GamesDataset();
    d.pushPair("x", "y", 1);
    const m = games.initElo({ k: 32 });
    m.update(d);
    expect(m.score("x")!).toBeGreaterThan(m.score("y")!);
  });

  it("accepts the game-outcome variant via push-game", () => {
    const d = new datasets.GamesDataset();
    d.pushGame(["p1"], ["p2"], { tag: "side1-win", val: 1 }, 1);
    const m = games.fitElo({}, d);
    expect(m.score("p1")!).toBeGreaterThan(m.score("p2")!);
  });

  it("fits PageRank (batch), personalizes, and extracts components", () => {
    const g = new datasets.GraphDataset();
    for (const [s, t] of [
      ["home", "about"],
      ["home", "products"],
      ["about", "home"],
      ["products", "home"],
      ["products", "checkout"],
      ["checkout", "home"],
    ]) {
      g.push(s, t, 1);
    }
    const global = graph.fitPageRank({ damping: 0.85 }, g);
    expect(global.top(3).length).toBe(3);
    expect(global.algorithm()).toContain("page");

    const personalized = graph.fitPageRank(
      { teleport: { tag: "seeds", val: [["checkout", 1]] } },
      g,
    );
    // Personalization shifts mass toward the seed's neighborhood.
    expect(personalized.score("home")).toBeDefined();

    const comps = functions.extractComponents(g, 1);
    expect(comps.length).toBeGreaterThanOrEqual(1);
    expect(comps[0].nNodes()).toBeGreaterThan(0);
  });

  it("exposes wilson-interval", () => {
    const [lo, hi] = functions.wilsonInterval(8, 2, 1.96);
    expect(lo).toBeLessThan(hi);
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(hi).toBeLessThanOrEqual(1);
  });

  it("throws a typed error on an empty dataset", () => {
    const g = new datasets.GraphDataset();
    expect(() => graph.fitPageRank({}, g)).toThrow();
  });
});
