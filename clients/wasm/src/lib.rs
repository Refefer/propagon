//! Propagon WebAssembly Component-Model bindings.
//!
//! One component, many hosts: TypeScript consumes it via jco; other languages
//! via wasmtime + their bindgen. The surface mirrors the Python client
//! (clients/python). The core is built parallel-OFF (single-threaded) and never
//! touches the filesystem, so the component has no ambient capabilities.

// Generated guest bindings (cargo-component / wit-bindgen). Exempt from the
// crate's lints — including the no-panic/no-unwrap restriction gate — since the
// codegen is not ours to change.
#[allow(warnings)]
#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::restriction,
    clippy::cargo
)]
mod bindings;

mod algos;
mod convert;
mod datasets;
mod enums;
mod errors;
mod functions;

/// The component entry point. Implements every exported interface's `Guest`
/// trait across the submodules (datasets, games, graph, functions).
struct Component;

bindings::export!(Component with_types_in bindings);
