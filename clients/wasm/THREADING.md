# Threading

The propagon component is built **single-threaded** and that is currently the
only supported mode.

## Why single-threaded

- The core is compiled **parallel-OFF** (`propagon` with `default-features =
  false`, i.e. no `rayon`). Every algorithm runs sequentially with identical
  results — the `parallel`-off build is a first-class, tested configuration of
  the core.
- WebAssembly threading is **not** a runtime flag. A multi-threaded wasm module
  needs a different memory model (shared memory + atomics) chosen at compile
  time, plus a Web-Worker pool to drive it.
- The usual browser path, `wasm-bindgen-rayon`, is **wasm-bindgen-specific**: it
  patches wasm-bindgen's JS glue. It has no Component-Model equivalent, and jco
  does not transpile threaded components into a working Web-Worker pool. The
  Component-Model threading proposals (shared-everything threads / `wasi-threads`)
  are not yet stable.

So a single npm package that flips between single- and multi-threaded at runtime
is **not achievable on the Component-Model track today**. The two are different
build pipelines.

## Keeping the UI responsive

Single-threaded does not mean blocking. Run fits inside a **Web Worker** so a
large fit never blocks the main thread, and post results back. This is the
recommended pattern regardless of threading.

## Deferred: a parallel build

A parallel variant remains possible as the ecosystem matures, via either of:

1. **wasm-bindgen + wasm-bindgen-rayon** — a separate, JS-only package
   (`propagon/parallel` subpath) requiring cross-origin isolation
   (`Cross-Origin-Opener-Policy: same-origin`,
   `Cross-Origin-Embedder-Policy: require-corp`). This abandons the
   cross-language Component-Model reach that motivated this client.
2. **Component-Model shared-everything threads** — once stable and supported by
   jco/wasmtime.

Both are out of scope for the current client. The core is already parallel-ready
(`propagon/parallel` feature), so only the binding/build layer would change.

## Determinism

Because the core never calls a clock or an OS RNG (all randomness is explicitly
seeded — see `seed` params), the component is fully deterministic given the same
inputs and seeds, in any host. The `wasm32-wasip1` build does carry dead
`wasi:random`/`wasi:clocks`/`wasi:filesystem` imports pulled in transitively by
`std`; they are never reached on any algorithm path. Eliminating them (e.g. a
custom `getrandom` backend, or a `wasm32-unknown-unknown` build) is a
size/packaging refinement, not a correctness one.
