// Public type surface: the jco-generated interface namespaces plus `loadState`.
export * from "./dist/propagon.js";

/**
 * Reconstruct a fitted model from header-line JSONL, dispatching on the state's
 * `algorithm` tag. The concrete return type depends on the tag; narrow via the
 * model's `algorithm()` or use the per-algorithm `load*` function for a typed
 * result.
 */
export function loadState(state: string): unknown;
