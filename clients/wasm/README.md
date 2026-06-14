# propagon (WebAssembly Component)

WebAssembly **Component Model** bindings for [propagon](../../): ranking, rating,
multi-armed bandit, graph-centrality, and trajectory-value algorithms compiled
to a single portable `.wasm` component.

One artifact, many hosts:

- **TypeScript / JavaScript** consume it via [jco](https://github.com/bytecodealliance/jco)
  (this package ships the transpiled JS + `.d.ts`).
- **Other languages** (Rust, Python, Go, …) can run the *same* component via
  `wasmtime` + their own component bindgen — no reimplementation.

The surface mirrors the [Python client](../python): nine dataset shapes and
~50 algorithms across every family.

## Build

Requires the Rust `wasm32-wasip2` target, [`cargo-component`](https://github.com/bytecodealliance/cargo-component),
and Node 20+.

```sh
cargo install cargo-component --locked
npm install
npm run build      # cargo component build --release && jco transpile -> dist/
npm test           # vitest host suite
```

`npm run build` produces `dist/propagon.js` + `dist/propagon.d.ts` (plus the
`.wasm` chunks). The package's entry point (`index.js`) re-exports the generated
interface namespaces and adds `loadState`.

## Usage

```ts
import { datasets, games, graph, loadState } from "propagon";

// Rate teams from head-to-head games (Elo).
const d = new datasets.GamesDataset();
d.pushPair("sharks", "bears", 1);
d.pushPair("sharks", "wolves", 1);
const elo = games.fitElo({ k: 24 }, d);
console.log(elo.sortedScores());          // [["sharks", ...], ...]

// Bulk export crosses the boundary as a Float64Array + id list.
const { ids, scores } = elo.scoresBulk();

// Persist and restore (dispatches on the state's algorithm tag).
const state = elo.saveState();
const restored = loadState(state);

// Personalized PageRank over a graph.
const g = new datasets.GraphDataset();
g.push("home", "checkout", 1);
const pr = graph.fitPageRank(
  { teleport: { tag: "seeds", val: [["checkout", 1.0]] } },
  g,
);
```

Algorithms are grouped by dataset shape into interface namespaces: `datasets`,
`games`, `graph`, `pairwise`, `rewards`, `matchups`, `annotated`, `rankings`,
`trajectories`, and `functions`. Each fitted model exposes `algorithm`,
`sortedScores`, `score`, `top`, `scoresBulk`, and `saveState`; online models add
`update`. Errors are thrown (idiomatic `try`/`catch`).

Parameters are plain objects; omit a field to take the core default
(`fitElo({ k: 24 }, d)`). Unit-enum params are kebab-case strings
(`{ direction: "total" }`); data-carrying params are tagged variants
(`{ tag: "ucb1", val: 2.0 }`). `u64` seeds are JavaScript `BigInt`s
(`{ seed: 1n }`).

## Examples

See [`examples/`](examples) (runnable on Node 24+ via type stripping, e.g.
`node examples/tournament.ts` after `npm run build`).

## Threading

The component is **single-threaded**. See [THREADING.md](THREADING.md) for why,
and the status of a parallel build.
