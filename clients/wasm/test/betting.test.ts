import { describe, it, expect } from "vitest";
import { datasets, betting, loadState } from "../index.js";

describe("betting & portfolio (§14)", () => {
  it("de-vigs odds into fair probabilities that sum to 1", () => {
    const d = new datasets.OddsDataset();
    d.pushEvent([["home", 4.2], ["draw", 3.7], ["away", 1.95]]);

    for (const method of ["multiplicative", "power", "shin"]) {
      const m = betting.fitOddsDevig({ method }, d);
      const sum = m.sortedScores().reduce((a, [, s]) => a + s, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-9);
    }

    const shin = betting.fitOddsDevig({ method: "shin" }, d);
    expect(shin.algorithm()).toBe("odds-devig");
    expect(shin.insiderShare(0)).toBeGreaterThan(0);

    // round-trips via the global loader.
    const state = shin.saveState();
    expect(loadState(state).saveState()).toBe(state);
  });

  it("consolidates forecasts via the linear pool", () => {
    const d = new datasets.ForecastDataset();
    d.pushSource("s1", 1, [["a", 0.8], ["b", 0.2]]);
    d.pushSource("s2", 1, [["a", 0.6], ["b", 0.4]]);
    const m = betting.fitOpinionPool({ kind: "linear" }, d);
    const s = new Map(m.sortedScores());
    expect(Math.abs(s.get("a")! - 0.7)).toBeLessThan(1e-9);
  });

  it("runs an LMSR market with prices summing to 1", () => {
    const d = new datasets.MarketDataset();
    d.pushTrade("yes", 100);
    d.pushTrade("no", 20);
    const m = betting.fitLmsr({ b: 50 }, d);
    const sum = m.sortedScores().reduce((a, [, s]) => a + s, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-9);
    expect(m.price("yes")!).toBeGreaterThan(m.price("no")!);
    expect(m.cost()).toBeGreaterThan(0);
  });

  it("sizes Kelly stakes and scores diagnostics", () => {
    expect(Math.abs(betting.kellyFraction(0.6, 1.0) - 0.2)).toBeLessThan(1e-12);
    expect(betting.kellyFraction(0.4, 1.0)).toBe(0);
    expect(Math.abs(betting.fractionalKelly(0.6, 1.0, 0.5) - 0.1)).toBeLessThan(1e-12);
    expect(betting.portfolioKelly([[0.6, 1.0]])[0]).toBeGreaterThan(0);

    expect(Math.abs(betting.brierScore([0.9, 0.2, 0.7], [true, false, true]) - 0.14 / 3)).toBeLessThan(1e-12);
    expect(Math.abs(betting.closingLineValue(2.1, 2.0) - 0.05)).toBeLessThan(1e-12);

    const table = betting.calibrationTable([0.1, 0.9, 0.9], [false, true, true], 10);
    expect(table.length).toBe(10);
    expect(table[9].count).toBe(2);
  });

  it("throws a typed error on invalid Kelly inputs", () => {
    expect(() => betting.kellyFraction(1.5, 1.0)).toThrow();
  });
});
