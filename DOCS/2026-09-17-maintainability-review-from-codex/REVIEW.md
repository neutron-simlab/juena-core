# Maintainability review: `juena-core`

Date: 2026-09-17  
Review type: documentation-only, current implementation compared with plans 00–03,
`juena-chatbot`, Vitess AI Agent v1, and the current Vitess AI Agent v2 tree.

## Decision summary

`juena-core` is not generally over-engineered. Its apparent size is mainly essential
complexity: authenticated identity, PostgreSQL persistence, resumable streams,
interrupts, server-authored execution evidence, artifact ownership, optional MCP and
sandbox boundaries, and a reusable UI/client layer. The package has no internal import
cycle, its optional extras stay out of the base import, and both real applications use
broad parts of the public surface. Splitting large files or adding another plugin or
dependency-injection framework would increase indirection without removing a source of
truth.

The useful simplifications are narrower:

1. centralise the duplicated graph-run-id lookup;
2. remove four values that core does not consume from `CoreSettings`;
3. make the one helper used by an application subclass public instead of importing a
   private symbol; and
4. replace source-code references to absent checkpoint documents with local invariant
   explanations.

No recommendation changes identity semantics, persistence, fail-closed behaviour,
streaming, artifacts, middleware ordering, or the optional sandbox boundary.

## Snapshot and review controls

### Start snapshot

Captured at `2026-09-17T22:36:11+02:00` before analysis or writes.

| Repository | Branch | Commit | Working tree |
|---|---|---|---|
| `juena-core` | `juena-core-cutover-runtime` | `41371869d67e54a6849e4059b14e21e264d13053` | dirty only in the pre-existing `.gitignore` change that ignores `DOCS/` |
| `juena-chatbot` | `juena-core-cutover-runtime` | `27e5016d20db153edb107bb0e3f1b6a6fe4aa75c` | clean; read-only evidence |
| Vitess v1 | `feature/migrate-to-juena` | `2715f1e7b0ed0ac78090d4f00e00d4cc85c9731f` | clean; read-only evidence |
| Vitess v2 | `feature/vitess-rag-tools` | `688b2b141b2e0a7be8f27804725402bd194f038e` | clean; read-only consumer evidence |

The core tracked-worktree diff SHA-256 was
`23d32c4cf38ca80d73ff7ca5e02e885fc4bb9f3acac259329cd769c05a15d4d9`.
The staged-diff SHA-256 was the empty-input digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The tracked change was not touched. This review file is ignored by the existing rule,
as requested.

### End snapshot

Filled after the completed review and verification:

| Repository | Branch | Commit | Working tree |
|---|---|---|---|
| `juena-core` | `juena-core-cutover-runtime` | `41371869d67e54a6849e4059b14e21e264d13053` | tracked tree clean; `DOCS/` is now visible as untracked and includes pre-existing reviews plus this review |
| `juena-chatbot` | `juena-core-cutover-runtime` | `27e5016d20db153edb107bb0e3f1b6a6fe4aa75c` | clean |
| Vitess v1 | `feature/migrate-to-juena` | `420ccb0e5f339dfbf96f48713300fe18b3d150d1` | clean |
| Vitess v2 | `feature/vitess-rag-tools` | `dabd14839324a55bb275903806bf0b70fe98ede3` | tracked tree clean; this requested review directory is untracked |

End core tracked-worktree diff SHA-256:
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.  
End v2 tracked-worktree diff SHA-256:
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Only this requested review directory was created in core. No runtime code, tests,
prompts, workflows, or comparison repositories were modified.

The snapshot guard detected concurrent changes during the review. V2 advanced from
`688b2b1` to `dabd148` (actionable RAG failure text, its tests, and a versioned design
note); v1 advanced from `2715f1e` to `420ccb0` (plan-03 documentation only); and the
uncommitted core `.gitignore` change disappeared without a core commit change. The v2
and v1 suites were rerun on the new commits, the changed retrieval code was reinspected,
and the static findings were refreshed before finalising this document.

## Verification actually performed

| Check | Result | Scope and limitation |
|---|---|---|
| Core non-PostgreSQL suite | **473 passed, 5 skipped**, 2 warnings, 6.45 s | Excluded `test_identity_postgres.py`, `test_server_routes_postgres.py`, `test_sandbox_jobs_postgres.py`, and `cp0b/test_checkpointing.py` |
| Core import-direction gate | **passed**: `import direction ok` | `bash scripts/check-imports.sh` |
| PostgreSQL-dependent core tests | **not run** | `docker compose -f tests/compose.postgres.yml ps` showed no service; this remains an environmental gate |
| Vitess v1 reference suite | **186 passed**, 2 warnings, 1.52 s | Final clean v1 checkout at `420ccb0` |
| Vitess v2 complete suite | **438 passed**, 4 warnings, 24.75 s | Final v2 code snapshot at `dabd148`, including retrieval work |

The core warnings were the LangChain MCP beta warning and Starlette's deprecated
`anyio_backend_name` warning. No browser flow, live model call, successful retrieval
query, real SAML ACS/session round trip, or PostgreSQL integration suite was exercised.
Those are not implied by the green unit suites.

## Plans 00–03 reconciled with the code

- **Plan 00 — boundary:** its central boundary still holds. Core owns generic agent,
  server, persistence, streaming, UI, evidence, artifact, MCP, and optional sandbox
  mechanics. Authentication policy, retrieval corpus, domain prompts, simulator
  schemas, and deployment policy remain application-owned.
- **Plan 01 — extraction:** the extracted modules are real and used by both consumers.
  The plan is historical evidence, not a current API reference. Core source currently
  contains 28 lines matching historical checkpoint references even though those plan
  files are not part of this repository; `CORE-002` removes that maintenance trap.
- **Plan 02 — chatbot cutover:** code/API verification landed, but the plan itself
  still records browser rendering and a real IFFLogin → `/auth/acs` session round trip
  as open. This review does not relabel them complete.
- **Plan 03 — Vitess v2:** the current v2 consumer is later than the plan's status
  table. It includes the post-CP6 retrieval work and passes 438 tests. Conversely, that
  does not close plan 02's human/browser gates. The current checkout, not a “landed”
  paragraph, is the API evidence used below.

## The API the two applications actually use

An AST import inventory over application source (not tests) found 31 distinct
`juena_core` modules and 90 imported names in `juena-chatbot`, and 27 modules and 47
names in v2. Counts include aliases and multi-name imports; they are a breadth signal,
not a stability promise.

| Core boundary | `juena-chatbot` | Vitess v2 | Assessment |
|---|---|---|---|
| Root configuration | `CoreSettings`, `configure`, `settings` | same | Keep the explicit application-to-core hand-off; narrow it with `CROSS-001` |
| Model/provider selection | provider/model enums, model builders and availability | same | Keep; runtime selection is shared infrastructure |
| Agent assembly | ask-user, supervisor backend, specialist runtime/outcome; findings utilities | ask-user, supervisor backend, delegation, specialist runtime/outcome | Keep shared mechanics and application-owned prompts |
| Agent registry/runtime context | factory registration and `RuntimeModelContext` | same | Keep process lifecycle; simplify duplicated run-id lookup with `CORE-001` |
| Persistence | checkpointer, store, DB sessions/models, owned-chat lookup | checkpointer, store, sessions, owned-chat lookup | Keep; identity and ownership checks are essential complexity |
| Server composition | `create_app`, `StreamPolicy`, `ThreadWorkspace`, application routers | `create_app`, fixed local principal, upload router | Keep the composition root rather than creating a framework of hooks |
| Client and UI | `BaseAgentClient`, chat storage, streaming, shared components | same smaller subset | Keep; publish the error formatter used by the chatbot subclass via `CROSS-002` |
| Evidence and artifacts | middleware-mediated use | direct artifact registration plus evidence middleware | Keep server-owned evidence and delivery verification |
| MCP | optional discovery for JüNA | required discovery for VITESS | Keep separate required/optional entry points; failure semantics differ deliberately |
| Sandbox | approvals, runtime, config and middleware | unused | Keep as an optional leaf; do not make v2 install or configure it |

The only external import of a private core function found in either application is
`juena-chatbot` importing `_http_error_message`. Its use of the protected `_client` and
`_headers` members is the intended subclass extension seam, not an accidental second
transport implementation.

## Structural evidence

The source contains 71 Python modules and 156 internal import edges. An AST dependency
scan found **zero cycles**. Importing only `juena_core` did not load `streamlit`,
`podman`, or `fastmcp`, confirming that the `[ui]`, `[sandbox]`, and `[mcp]` extras do
not leak into the base import.

Source size by responsibility is led by `server` (3,864 lines), `sandbox` (2,569),
`agents` (1,663), and `ui` (1,637). These are distinct subsystems rather than one
generic layer consuming every optional dependency.

The largest functions were:

| Symbol | Approx. span / branches | Decision |
|---|---:|---|
| `server.api.endpoints.build_api_router` | 461 / 31 | **Keep.** Large but cohesive: one router closure over the same auth, store, agent, stream, and artifact dependencies. Splitting it produces router plumbing without removing state or policy. |
| `sandbox.backend.PodmanSandboxBackend.execute` | 131 / 14 | **Keep.** One execution lifecycle; branches are cleanup, timeout, result, and containment cases. |
| `server.service.create_app` | 108 / 4 | **Keep.** A readable composition root; replacing its explicit arguments with a container/config object hides the API used by both apps. |
| `agents.specialist_runtime.build_specialist_middleware` | 99 / 9 | **Keep.** The ordering of middleware and tool exposure is part of the contract. |
| `clients.base._parse_sse_data` | 92 / 11 | **Keep.** Branches describe the wire vocabulary and intentionally pass unknown application frames through. |

File size alone therefore produces no accepted split.

## Responsibility review

| Area | Evidence | Decision |
|---|---|---|
| Configuration | Core does not read `.env`; apps construct one frozen settings object | **Keep**, but remove four fields core never reads (`CROSS-001`) |
| Registries | Agent factories, interrupts, store/checkpointer, and settings are process-scoped and have explicit startup/shutdown/reset behaviour | **Keep.** They reflect the ASGI process lifecycle; request-scoped dependency injection would be more machinery |
| Middleware composition | Evidence, artifact, model, delegation, and sandbox policies remain separately named and ordered | **Keep.** These encode trust boundaries, not convenience wrappers |
| Persistence | PostgreSQL checkpointer/store and owned-chat lookup enforce durable identity | **Keep.** In-memory substitution would recreate v1's cross-request/global-state weakness |
| Streaming | SSE parsing, resume, interrupt and unknown-frame pass-through share one protocol surface | **Keep** |
| UI | Optional Streamlit helpers contain no application navigation or domain prompt | **Keep** as optional extra |
| Artifacts | Ownership, registration, audit, dropped-delivery reasons and message attachment are one delivery contract | **Keep.** Do not split metadata from verification |
| MCP | Discovery is thin and fail-closed where required | **Keep** |
| Sandbox | Large, but isolated behind an extra and used only by the chatbot | **Keep in core for now.** Extraction would create a third package and coordinated release without simplifying either consumer |
| Runtime context | Two copies of graph-run-id precedence plus a private exported value helper | **Simplify** with `CORE-001` |
| Historical comments | Checkpoint paths point outside this repository | **Simplify** with `CORE-002` |

## Accepted recommendations

### CORE-001 — one runtime-context lookup for graph run identity

- **Priority / phase:** P1, low-risk internal simplification; no dependency.
- **Evidence and symbols:**
  `agents.specialist_outcome._graph_run_id` and
  `sandbox.middleware._graph_run_id` implement the same precedence:
  `runtime.context.run_id`, then `runtime.execution_info.run_id`, then
  `runtime.config["run_id"]`. Both import the exported-private
  `runtime_context._context_value`.
- **Why accidental:** this is defect-sensitive identity precedence duplicated across
  the writer and reader of execution evidence. Their only intended difference is that
  sandbox execution requires a value while outcome rendering may return no value.
- **Selected target interface:**

  ```python
  # juena_core.runtime_context
  def context_value(context: object, name: str) -> str | None: ...
  def graph_run_id(runtime: object) -> str | None: ...
  ```

  Export both through `runtime_context.__all__`. There is no `_context_value` alias.
- **Move/consolidate/delete:** rename `_context_value`; move the shared precedence
  implementation into `graph_run_id`; delete both private `_graph_run_id` functions.
- **Consumer migration:** both core modules import the public helpers. Sandbox keeps
  its present `RuntimeError("Sandbox execution requires a graph run_id")` when the
  optional result is absent. No application code changes.
- **Protected invariants:** context wins across supervisor/subagent boundaries;
  execution-info and config remain direct-invocation fallbacks; missing identity never
  becomes an invented ID.
- **Regression tests:** direct mapping/object tests for `context_value`; precedence and
  empty-value tests for `graph_run_id`; existing specialist-outcome and sandbox
  middleware suites; full non-PostgreSQL core suite and import-direction gate.

### CORE-002 — make source explanations repository-local

- **Priority / phase:** P3, low-risk cleanup; after functional refactors so diffs stay
  reviewable.
- **Evidence and symbols:** 28 matching lines under `src/` and `examples/` refer to
  `00-BOUNDARY`, `01-EXTRACT`, `03/CP*`, or equivalent checkpoint names. Those files
  live in the v1 repository, not in the installed package or this checkout.
- **Why accidental:** a future maintainer cannot follow the reference from core, and
  chronological labels obscure the invariant the comment is meant to protect.
- **Selected target design:** every runtime-source docstring stands alone. Replace each
  plan label with the local symbol, state channel, route, or invariant it explains.
  Keep one README-level link to historical design records if provenance is wanted.
- **Move/consolidate/delete:** edit comments/docstrings only; delete checkpoint labels
  and obsolete “what checkpoint added this” prose. Do not remove rationale about
  identity, evidence, optional imports, or fail-closed behaviour.
- **Consumer migration:** none.
- **Protected invariants:** no executable line, public docstring contract, prompt, or
  workflow changes.
- **Regression tests:** import-direction gate, compile/import smoke test, and the normal
  core suite. Review the diff to ensure only comments/docstrings changed.

### CROSS-001 — reduce `CoreSettings` to values core consumes

- **Priority / phase:** P2, coordinated core/API change; do after `CORE-001` and in one
  commit spanning core, chatbot, and v2.
- **Evidence and symbols:** `CoreSettings.OPENAI_DEFAULT_MODEL`,
  `BLABLADOR_DEFAULT_MODEL`, `SESSION_COOKIE_SECURE`, and `LOG_DIR` are declared in
  core but have no read in core source. Both applications pass them through
  `to_core_settings()` solely to satisfy the dataclass constructor. Chatbot genuinely
  reads its own provider defaults and `Config.SESSION_COOKIE_SECURE`; v2 does not read
  its provider-specific defaults or `LOG_DIR`, and no Python source in either app reads
  `LOG_DIR`.
- **Why accidental:** this duplicates application configuration in the shared contract
  and falsely implies that core owns cookie and log-directory policy.
- **Selected target interface:** delete exactly those four dataclass fields. Keep
  `DEFAULT_MODEL`, provider availability strings, `LOG_LEVEL`, persistence, artifact,
  timeout, bind, and publication fields because core reads them.
- **Move/consolidate/delete:** remove the four declarations and their four arguments at
  `juena-chatbot.Config.to_core_settings()` and `vitess_ai.Config.to_core_settings()`;
  update core examples and fixtures. Retain chatbot's provider-specific defaults and
  `SESSION_COOKIE_SECURE`, which it consumes. Delete v2's now-unreferenced
  `OPENAI_DEFAULT_MODEL`, `BLABLADOR_DEFAULT_MODEL`, and `LOG_DIR`; delete chatbot's
  unreferenced `LOG_DIR` plus its Compose/example forwarding after confirming the final
  repository-wide search is empty.
- **Consumer migration:** core, chatbot, and v2 land together. No deprecated
  properties, `**kwargs`, compatibility dataclass, or dual old/new constructor.
- **Protected invariants:** model defaults remain identical; production cookie checks
  remain in chatbot; v2 still has a fixed local principal; stdout/container logging is
  unchanged; `configure()` remains immutable and fail-fast.
- **Regression tests:** core config and model-provider tests; both applications'
  `to_core_settings` tests and full suites; import-direction checks.

### CROSS-002 — publish the HTTP error formatter used by client subclasses

- **Priority / phase:** P2, coordinated core/chatbot API change; independent of
  `CROSS-001`.
- **Evidence and symbols:** `juena.clients.client.AgentClient` imports
  `juena_core.clients.base._http_error_message`, the only private core function imported
  by either application.
- **Why accidental:** the base client explicitly supports application subclasses, but
  its shared error translation helper is private. The underscore makes the actual
  extension contract dishonest.
- **Selected target interface:**

  ```python
  def http_error_message(exc: httpx.HTTPError) -> str: ...
  ```

  Export it from `juena_core.clients.base.__all__`.
- **Move/consolidate/delete:** rename the existing function and update
  `_raise_http_error`; do not retain a private alias.
- **Consumer migration:** change the chatbot import and its four call sites in the same
  commit. V2 has no call site.
- **Protected invariants:** HTTP 401 remains distinguishable; FastAPI `detail` remains
  preferred; transport errors retain their text; the client does not turn failures
  into successful empty results.
- **Regression tests:** core client-contract tests, chatbot `test_agent_client.py`, and
  both full suites.

## Ordered roadmap

### 1. Correctness prerequisites

No new core correctness prerequisite was found in the exercised non-PostgreSQL paths.
Before any release, however, run the excluded PostgreSQL files against the required
service. Browser rendering and real SAML acceptance remain release evidence gaps for
the chatbot, not refactoring prerequisites to disguise as unit coverage.

### 2. Low-risk internal simplifications

1. `CORE-001` — one graph-run-id resolver.
2. `CORE-002` — localise historical source comments after functional diffs settle.

### 3. Coordinated core/API changes

1. `CROSS-001` — delete four unused settings and migrate both applications atomically.
2. `CROSS-002` — public client error formatter and atomic chatbot migration.

Run core, chatbot, and v2 suites from the same sibling checkout after each coordinated
change. A core-only green suite is insufficient for these two items.

### 4. Rejected or deferred ideas

| Idea | Decision and reason |
|---|---|
| Split `build_api_router` by route count | **Rejected.** Shared closure dependencies and streaming/ownership policy would move into more constructors without reducing branches |
| Replace process registries with a generic service container | **Rejected.** More indirection for an ASGI process lifecycle already explicit in startup/shutdown |
| Add plugin discovery for agents or middleware | **Rejected.** Both applications have a small explicit set; import-time registration is visible at their composition roots |
| Move application `Config` or retrieval into core | **Rejected.** Authentication, corpus, upload, and deployment policy differ materially |
| Merge MCP required and optional discovery | **Rejected.** “Unavailable” is a valid optional result for JüNA but a fail-closed boundary for VITESS execution |
| Extract sandbox into another distribution now | **Deferred.** It is already an optional leaf with no base-import leak; a package split adds coordinated release cost without current duplication |
| Replace server-owned evidence with prompt instructions | **Rejected.** Evidence and artifact delivery are trust boundaries, not presentation logic |
| Add compatibility shims for `CROSS-001` or `CROSS-002` | **Rejected.** Migrate both checked-out consumers together and keep one API |

## Completion criteria for the later refactor

The roadmap is complete only when all accepted IDs have their named regression tests,
both consumers have moved atomically for cross-package changes, the import-direction
gate remains green, optional imports remain lazy, and PostgreSQL/browser/SAML evidence
is reported at its actually exercised level. No file-count or line-count reduction is
itself an acceptance criterion.
