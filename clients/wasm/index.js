// Ergonomic entry point: re-exports the jco-generated interface namespaces and
// adds `loadState`, which reconstructs the right model from a saved state's
// header `algorithm` tag (mirrors the Python client's `load_state`). Each model
// also has its own per-algorithm `load*` function in its interface namespace.
export * from "./dist/propagon.js";

import {
  games,
  graph,
  pairwise,
  rewards,
  matchups,
  annotated,
  rankings,
  trajectories,
} from "./dist/propagon.js";

// algorithm tag (state header) -> loader. Tags match propagon's RankModel::algorithm().
const LOADERS = {
  // games
  "elo": (s) => games.loadElo(s),
  "glicko2": (s) => games.loadGlicko2(s),
  "elo-mov": (s) => games.loadMovElo(s),
  "melo": (s) => games.loadMElo(s),
  "generalized-bt": (s) => games.loadGeneralizedBt(s),
  "team-bradley-terry": (s) => games.loadTeamBradleyTerry(s),
  // graph
  "page-rank": (s) => graph.loadPageRank(s),
  "hits": (s) => graph.loadHits(s),
  "birank": (s) => graph.loadBiRank(s),
  "degree": (s) => graph.loadDegree(s),
  "harmonic": (s) => graph.loadHarmonic(s),
  "katz": (s) => graph.loadKatz(s),
  "k-core": (s) => graph.loadKCore(s),
  "leader-rank": (s) => graph.loadLeaderRank(s),
  // rewards / contextual
  "bandit": (s) => rewards.loadBandit(s),
  "sliding-window-ucb": (s) => rewards.loadSlidingWindowUcb(s),
  "lin-ucb": (s) => rewards.loadLinUcb(s),
  // annotated
  "crowd-bt": (s) => annotated.loadCrowdBt(s),
  // matchups
  "weng-lin": (s) => matchups.loadWengLin(s),
  // rankings
  "plackett-luce": (s) => rankings.loadPlackettLuce(s),
  "footrule": (s) => rankings.loadFootrule(s),
  "mallows": (s) => rankings.loadMallows(s),
  "mc4": (s) => rankings.loadMc4(s),
  // trajectories
  "mc-value": (s) => trajectories.loadMcValue(s),
  "behavior-cloning": (s) => trajectories.loadBehaviorCloning(s),
  "value-compare": (s) => trajectories.loadValueCompare(s),
  "td-value": (s) => trajectories.loadTdValue(s),
  // pairwise
  "btm-mm": (s) => pairwise.loadBradleyTerryMm(s),
  "btm-lr": (s) => pairwise.loadBradleyTerryLr(s),
  "bayesian-bradley-terry": (s) => pairwise.loadBayesianBradleyTerry(s),
  "colley": (s) => pairwise.loadColley(s),
  "massey": (s) => pairwise.loadMassey(s),
  "keener": (s) => pairwise.loadKeener(s),
  "ilsr": (s) => pairwise.loadILsr(s),
  "nash-averaging": (s) => pairwise.loadNashAveraging(s),
  "offense-defense": (s) => pairwise.loadOffenseDefense(s),
  "random-walker": (s) => pairwise.loadRandomWalker(s),
  "rank-centrality": (s) => pairwise.loadRankCentrality(s),
  "serial-rank": (s) => pairwise.loadSerialRank(s),
  "thurstone-mosteller": (s) => pairwise.loadThurstoneMosteller(s),
  "whr": (s) => pairwise.loadWhr(s),
  "borda": (s) => pairwise.loadBorda(s),
  "copeland": (s) => pairwise.loadCopeland(s),
  "blade-chest": (s) => pairwise.loadBladeChest(s),
  "es-rum": (s) => pairwise.loadEsRum(s),
  "hodge-rank": (s) => pairwise.loadHodgeRank(s),
  "kemeny": (s) => pairwise.loadKemeny(s),
  "lsr": (s) => pairwise.loadLsr(s),
  "covariate-bt": (s) => pairwise.loadCovariateBt(s),
  "rate": (s) => pairwise.loadWinRate(s),
  "dueling-bandit": (s) => pairwise.loadDuelingBandit(s),
};

/**
 * Reconstruct a fitted model from header-line JSONL, dispatching on the state's
 * `algorithm` tag. Returns the concrete model resource for that algorithm.
 */
export function loadState(state) {
  const first = state.split("\n", 1)[0];
  let header;
  try {
    header = JSON.parse(first);
  } catch (e) {
    throw new Error(`malformed state header: ${e.message}`);
  }
  const loader = LOADERS[header.algorithm];
  if (!loader) {
    throw new Error(
      `unknown algorithm tag ${JSON.stringify(header.algorithm)} in state header`,
    );
  }
  return loader(state);
}
