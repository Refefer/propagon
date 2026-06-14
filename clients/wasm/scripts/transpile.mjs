// Transpile the built component to JS + TypeScript types with jco.
//
// Resolves the Cargo target directory via `cargo metadata` so this works both
// locally (where a machine-local .cargo/config may redirect target-dir to
// another filesystem) and on CI (default in-tree target). Avoids hardcoding a
// path and the cross-device rename that `--target-dir` triggers here.
import { execSync } from "node:child_process";

const meta = JSON.parse(
  execSync("cargo metadata --format-version=1 --no-deps", { encoding: "utf8" }),
);
const wasm = `${meta.target_directory}/wasm32-wasip1/release/propagon_wasm.wasm`;

execSync(`npx jco transpile ${wasm} -o dist --name propagon`, {
  stdio: "inherit",
});
