# gamma — coding rules

These rules are non-negotiable. They override anything that conflicts in
the global config, AGENTS.md, or default Claude behavior. They apply to
every file in this workspace.

## 1. Explicit over implicit

**Never rely on implicit defaults.** A setting may live in exactly one
place. If a value can be configured, the configuration file must list
it; if it can come from code, the code must spell it out.

This means:

- No `#[serde(default)]` / `#[serde(default = "...")]` on user-facing
  config. Either the field is required, or the type makes the choice
  visible (see rule 3).
- No `impl Default` for config types that the user is supposed to
  fill in. `Default` is reserved for internal types whose "zero"
  state is genuinely meaningful.
- No two sources of truth for the same value. If TOML and code both
  provide a default, that's two — pick one.
- When you load a file, render it absent (`Option`-from-IO, see
  rule 3) — don't silently fabricate a default config.

Why: defaults that live in code are invisible to the user. They
discover them by hitting a bug. When a setting matters enough to
exist, it matters enough to write down.

## 2. Make illegal state unrepresentable

We are in a language with sum types, exhaustive matching, and zero-cost
enums. Use them. The compiler is the cheapest reviewer we have.

This means:

- If two fields are mutually exclusive, they go in an enum, not two
  `Option`s with a comment that says "exactly one of these".
- If a sequence of states is required (idle → streaming → done),
  encode it as a state enum that owns only the data each state
  needs. Don't carry "this `Vec` is only meaningful when `streaming`
  is true" as a runtime invariant.
- Branching is a smell. Every `if let Some(...) = ...` over a field
  that "should always be Some when X" is a missing variant. Hoist
  the guarantee into the type.
- Prefer `match` over `if`. Exhaustiveness is free correctness.
- New runtime check before access? Stop. Refactor the type so the
  check is impossible to forget.

Why: structural complexity becomes branching complexity becomes
bugs. Cheaper to spend the type-design effort up front than to debug
"impossible" states.

## 3. `Option` means *missing*, never *default*

`Option<T>` represents data that legitimately may be absent: a row
that wasn't returned, a header the server didn't send, a file that
doesn't exist, a CLI flag the user didn't pass. That is its only
job in this codebase.

**Never use `Option<T>` to mean "use the default":**

- ❌ `temperature: Option<f32>` (None = "let provider decide")
- ✅ `temperature: Temperature` where
    `enum Temperature { Default, Fixed(f32) }`

- ❌ `timeout: Option<Duration>` (None = unlimited)
- ✅ `enum Timeout { Unlimited, After(Duration) }`

- ❌ `max_retries: Option<u32>` (None = no retry)
- ✅ `enum RetryPolicy { Never, UpTo(u32) }`

- ❌ Using `0` or `-1` as a sentinel for "unset" or "default"
- ✅ A named enum variant.

If you find yourself documenting that "`None` means X" or
"`-1` means use the provider's default", you are wrong. Stop and
write the enum.

The check: read each `Option<T>` aloud as "this might be missing
because the source didn't have it." If that doesn't fit, the type
is wrong.

## 4. Comments earn their place by carrying the *why*

Three tiers, used deliberately. None of them is decoration.

### Syntax

Use standard Rust doc comments. The line-comment forms are the default
because they match rustdoc examples, rustfmt behavior, editor wrapping,
and the standard library's style.

- `//!` — inner docs for module-level documentation. Use this for every
  module doc, whether it is one line or many.
- `///` — outer docs for items: functions, structs, enums, traits,
  trait methods, modules declared with `mod`, and re-exports.
- `/*! ... */` and `/** ... */` — block doc comments. Avoid these in
  ordinary code; reserve them only for generated code or cases where a
  tool specifically requires block syntax.

Don't mix line and block doc styles on the same item. Prefer wrapping
or splitting `//!` / `///` lines over converting a doc comment to block
form just because it grew.

### Module-level docs (`//!`) — required, every file

Every `.rs` file opens with a module-level doc comment that answers
three questions. Skip any of them and the comment is incomplete:

1. **What this file does.** Not "utility helpers" — name the
   responsibility. "Owns the conversation, dispatches LLM turns and
   tool calls, and manages two-tier observational memory compaction."
2. **What it assumes.** What invariants must hold for this code to
   work? What does the caller have to set up? "Assumes the provider
   client returns at least one terminal `Done` chunk per request."
   "Assumes `cwd` is canonical — symlinks are not resolved here."
3. **The gotchas.** What surprised you (or would surprise the next
   reader) while writing it? "Stale `LlmChunk` messages can arrive
   after `Cancel`; the actor drops them silently rather than asserting,
   because cancel-races are racy by definition." If there are no
   gotchas, omit the section — don't pad.

### Function docs (`///`) — required on every non-trivial fn

For every public function, and every private function whose body is
more than a few lines, write a doc comment that covers:

- **What it solves.** The user-visible problem, in one sentence.
  "Drains observed messages from the working set and appends the
  observer's summary to the log." Not "applies the observation."
- **How it solves it.** The mechanism, briefly. "Drains
  `[observed_up_to .. len - preserve_recent)` from `messages` so the
  recent tail stays in context, then advances `observed_up_to` past
  the new tail boundary." Enough that a reader doesn't have to
  reverse-engineer the body to verify the doc.

Trivial getters, one-line constructors, and `Debug`/`Default`/`From`
impls don't need docs — the signature already tells the whole story.
Public *trait methods* always do, even if short.

### Inline `//` comments — only for gotchas

Inline comments are for things the code *cannot* tell you. Every one
should answer "why" — never "what". If you find yourself writing a
comment that paraphrases the next line, delete it.

Good inline comments:

- "// SAFETY: the caller has already verified the index is in bounds."
- "// Anthropic returns `stop_reason = "refusal"` for newer models;
   older models would have used `end_turn` here."
- "// Stale chunk from a cancelled stream — silently drop."
- "// Reflector may shrink the log to empty; that's a valid state."

Bad inline comments (delete on sight):

- "// loop over messages" before `for m in messages`
- "// increment counter" before `n += 1`
- "// TODO: clean this up" with no ticket and no condition for cleanup

Function-internal step comments ("// 1. validate", "// 2. dispatch")
are usually a sign the function should be split. If you can't split
it, the steps belong in the function-level doc — not as a running
narrative inside the body.

### What to avoid

- Doc comments that restate the type signature ("`/// Returns the
  number of messages.`" on `fn message_count() -> usize`).
- TODOs without a clear trigger condition or owner.
- "Removed X" / "Used to do Y" historical commentary — that's what
  git history is for.
- Comments that document obsolete behavior. When you change the code,
  update or delete the comment in the same edit.

## 5. Don't override clippy — fix the underlying problem

Clippy's complexity and style lints exist to catch real smells:
functions that are too long, branches that are too tangled, casts that
silently lose data, names that confuse the reader. When a lint fires,
the answer is to **fix the code**, not silence the lint.

**Forbidden** (per-item bypass of an active lint):

```rust
#[allow(clippy::too_many_arguments, clippy::cognitive_complexity, clippy::too_many_lines)]
fn handle_event(
    event_type: &Option<String>,
    data: &str,
    pending: &mut Vec<PendingToolUse>,
    last_stop_reason: &mut Option<String>,
    emitted_text: &mut bool,
    emitted_tool_calls: &mut bool,
    tx: &mpsc::UnboundedSender<LlmStreamChunk>,
) { ... }
```

That's three lints triggered by one function. The function is the
problem, not the lints. Split it. Bundle the bool flags into a struct.
Move the per-event branch into separate `handle_text_delta`,
`handle_tool_use`, etc.

When a lint fires, three responses are acceptable:

1. **Refactor the code** so the lint stops firing. This is the default.
2. **Relax the lint at the workspace level** if you've decided as a
   project that this lint doesn't apply here (e.g. the workspace's
   existing `cast_possible_truncation = "allow"`). Project-wide policy
   lives in the workspace `Cargo.toml`, gets discussed once, and
   applies uniformly.
3. **Disable the lint for an entire test module via `cfg_attr(test, ...)`**.
   Tests legitimately use `unwrap`/`expect`/`panic` to fail loud, and
   that's category-different from production code smells.

**Not acceptable**:

- `#[allow(clippy::too_many_lines)]` on a single function. The
  function has design issues — fix them.
- `#[allow(clippy::cognitive_complexity)]` on a match arm. Same.
- `#[allow(clippy::cast_precision_loss)]` to silence a number-conversion
  warning. Either widen the receiver to `f64`, use a checked
  conversion, or accept the loss at the workspace level — but don't
  hide it on one line.
- `#[allow(clippy::match_same_arms)]` to dodge merging arms. Merge
  them with `|`.
- `#[allow(clippy::print_stdout)]` / `print_stderr` because you're
  doing legitimate CLI output. Configure the workspace to permit it
  in CLI binaries, or use `std::io::stdout()` / `stderr()` directly.

If a per-item allow is *truly* unavoidable (e.g. a derive macro
generates code that trips a lint), the allow must be accompanied by an
inline comment explaining why the workspace-level fix doesn't apply.
Bare `#[allow(...)]` with no comment is rejected on review.

## 6. Keep type definitions and their impls together

Every `struct`, `enum`, or `trait` definition is immediately followed
by **all** of its inherent `impl` blocks, in source order. Don't move
methods to a different file, and don't put another type's definition
between a type and its impls. The reader scans top-to-bottom and
expects "here is `Foo`, here are its methods" without having to scroll
or grep.

This means:

- For `struct Foo { ... }`, every `impl Foo { ... }` block lives in
  the same module, *immediately* below the definition (and any
  derived-trait `impl Trait for Foo` blocks).
- For `enum Bar { ... }`, same rule. If `Bar` has no methods, no impl
  block is needed — but the *type* definition stays unbroken.
- Don't interleave: `struct A`, `struct B`, `impl A`, `impl B` is wrong.
  Do `struct A`, `impl A`, `struct B`, `impl B`.
- A free function that operates on `Foo` and is closely tied to it
  should live next to `Foo`'s definition too — either as an inherent
  method (preferred) or as a free fn placed right after the impls.
- Trait `impl`s for foreign traits (e.g. `Serialize`, `Deserialize`)
  also stay with the type. Don't park them at the bottom of the file.

A common smell to watch for: a small "helpers" function block at the
bottom of a file that's actually doing work for a type defined at the
top. That's two pieces of one feature pulled apart by file order. Move
the helpers up.

If a type's impl set is genuinely too large to read top-to-bottom, the
type is probably doing too much — split it into smaller types, each
with their own colocated impl, before reaching for `impl`-by-feature
file organization.

## 7. Never `unwrap()` or `expect()` in production code

Production code **handles** `Option` and `Result` — it does not assert
them away. Both `.unwrap()` and `.expect("...")` panic on the wrong
shape; that's a runtime crash on data we should have reasoned about
at compile time.

Forbidden in any non-test file:

- `result.unwrap()` — propagate with `?`, match on the variants, or
  convert to a typed error.
- `option.unwrap()` — match, use `if let Some`, or convert to a typed
  error if `None` is genuinely unexpected at this point in the flow
  (and if it *is* unexpected, that's an "illegal state" — see rule 2).
- `.expect("this can't be None")` — if you're sure it can't be `None`,
  the type is wrong; refactor (rule 2). If you're not sure, you're
  guessing; handle it.

Allowed (these are not the panicking variants):

- `.unwrap_or(default)`, `.unwrap_or_else(|| ...)`,
  `.unwrap_or_default()` — these *provide* a value for the `None`/`Err`
  case; they don't panic.
- `.map_or(...)`, `.map_or_else(...)`, `.ok_or(...)`,
  `.ok_or_else(...)` — same category.

Tests are exempt by construction: every crate's `lib.rs` carries
`#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]`.
Test code is *supposed* to fail loud on a wrong shape — that's the
test's job. Do not extend this exemption to integration tests or
binaries; only `cfg(test)` modules within the lib.

Enforcement: workspace `Cargo.toml` denies `unwrap_used`,
`expect_used`, and `panic` at the clippy level. Combined with
`-D warnings` in CI, any production violation is a hard build
failure. If a lint fires, fix the call site — don't silence the lint
(that's rule 5).

## 8. Blank line after a multi-line statement

When two statements sit at the same indent level inside a function body
and either is multi-line, **separate them with a blank line.** The
blank line is the visual seam between two distinct phases of the
function; without it, a dense body reads as one undifferentiated wall.

In particular: a blank line is required before any `if`, `if let`,
`for`, `while`, `match`, `loop`, or any other block-introducing
statement that follows a multi-line statement.

❌ Wrong (no separation):

```rust
let mut body = serde_json::json!({
    "model": request.model,
    "max_tokens": request.max_tokens,
    "messages": messages,
    "stream": true,
});
if let Temperature::Fixed(t) = request.temperature {
    body["temperature"] = serde_json::json!(t);
}
if !request.tools.is_empty() {
    body["tools"] = serde_json::json!(
        request.tools.iter().map(convert_tool_schema).collect::<Vec<_>>()
    );
}
body
```

✅ Right:

```rust
let mut body = serde_json::json!({
    "model": request.model,
    "max_tokens": request.max_tokens,
    "messages": messages,
    "stream": true,
});

if let Temperature::Fixed(t) = request.temperature {
    body["temperature"] = serde_json::json!(t);
}

if !request.tools.is_empty() {
    body["tools"] = serde_json::json!(
        request.tools.iter().map(convert_tool_schema).collect::<Vec<_>>()
    );
}

body
```

Two consecutive single-line statements (`x = 1; y = 2;` style) do not
need a blank line. The rule kicks in as soon as one of the adjacent
statements spans multiple lines or ends with `}`/`);` from a
multi-line construct.

This is enforced by review, not by rustfmt — rustfmt has no option
that operates on statement-level whitespace inside function bodies.

## How to use these rules

When you encounter existing code that violates them: flag it, then
fix it. Don't add new violations even if a neighbour has them.
When you write new code: design the types first, then the logic.
The types should make the logic obvious.

When in doubt, ask. These rules cost some up-front design time —
that is the point.
