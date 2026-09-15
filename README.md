# juena-core

`juena-core` is reusable infrastructure for building JüNA applications. It
provides a Postgres-checkpointed LangGraph server, authenticated chat and
streaming routes, agent middleware, a Python client, and optional Streamlit
MCP, and isolated-code-execution integrations.

The package deliberately does **not** contain an application's prompts,
domain tools, authentication provider, retrieval system, or deployment
policy. An application selects the modules it needs and supplies those parts.
The dependency always points inward: `juena_core` never imports an application.

> `juena-core` is currently `0.1.x`. Keep a committed dependency lock and run
> the application test suite when updating it. The LangChain MCP API is beta
> and is isolated in `juena_core.mcp` for that reason.

## Try the chatbot demo

[`examples/simple_chat`](examples/simple_chat) is a runnable, local-only
chatbot: one tool-free agent, the core FastAPI service, Postgres history, the
core HTTP client, and a small Streamlit page.

From this repository:

```bash
uv sync --extra ui --group dev
docker compose -f examples/simple_chat/compose.postgres.yml up -d --wait
cp examples/simple_chat/.env.example examples/simple_chat/.env
```

Add valid provider credentials to `.env`, source it in two terminals, then
start the API and UI:

```bash
set -a
source examples/simple_chat/.env
set +a
uv run uvicorn examples.simple_chat.service:app --host 127.0.0.1 --port 8080
```

```bash
set -a
source examples/simple_chat/.env
set +a
uv run --extra ui streamlit run examples/simple_chat/streamlit_app.py
```

See the [demo guide](examples/simple_chat/README.md) for the design and extension
points. The fixed demo identity is intentionally restricted to loopback; it is
not a production login mechanism.

## Install in an application

Python 3.11 or newer is required. The current distribution model is a sibling
checkout. Declare the base package for the agent, server, schemas, and HTTP
client, then select only the extras the application uses:

```toml
[project]
dependencies = [
    "juena-core",          # base only
    # "juena-core[ui]",    # add the Streamlit shell
    # "juena-core[mcp]",   # add the LangChain MCP client
    # "juena-core[sandbox]", # add rootless-Podman execution
    # "juena-core[ui,mcp,sandbox]", # add all optional features
]

[tool.uv.sources]
juena-core = { path = "../juena-core" }
```

Keep exactly one of those dependency lines, then run `uv lock` and commit the
application's `uv.lock`. A path source does not
pin the contents of `../juena-core`; for a reproducible team or CI build, use
an immutable Git commit/tag or a published package. For a personal Docker
build, at minimum build from a clean core tree and record its commit SHA and
the resulting image digest.

## Choose the pieces you need

| Need | Import | Application supplies |
| --- | --- | --- |
| Settings and model providers | `juena_core.config`, `juena_core.llms_providers` | Environment parsing, secrets, provider policy |
| Agent building blocks | `juena_core.agents` | Prompts, domain tools, specialist definitions |
| FastAPI service | `juena_core.server` | Agent factories, identity, app routes and lifespans |
| HTTP client | `juena_core.clients` | Thin methods for app-specific routes |
| Streamlit shell | `juena_core.ui` | Pages, navigation, branding and session policy |
| MCP discovery | `juena_core.mcp` | Server configuration and tool security facade |
| Isolated execution | `juena_core.sandbox` | Opt-in settings, worker deployment and sandbox image |
| Shared contracts | `juena_core.schema`, `juena_core.artifacts` | Domain schemas and artifact producers |

Applications can use the client or schemas without running the server, and
can use the server without installing the UI or MCP extras.

## Application wiring

The base application runtime never reads environment variables or a `.env`
file. The application owns that policy, constructs one immutable
`CoreSettings`, and calls `configure()` once. The optional standalone sandbox
worker is the deliberate exception: `SandboxWorkerSettings.from_env()` reads
its host-process settings once at startup. The demo's
[`config.py`](examples/simple_chat/config.py) contains a complete base example.

An application then registers one async factory per stable agent ID:

```python
from juena_core.server.agent.registry import register_agent_factory


async def build_agent(provider: str, model: str):
    # Build the model, tools, middleware and checkpointed LangGraph here.
    return application_resources, compiled_graph


register_agent_factory("assistant", build_agent, set_as_default=True)
```

Core builds each registered agent once per process and closes resources that
implement `close()` or `aclose()` during shutdown. Chats retain their
`agent_id`, so another graph cannot accidentally resume the thread.

Finally, choose an identity and create the service:

```python
import os
from uuid import UUID

from juena_core.server.identity import local_principal
from juena_core.server.service import create_app

app = create_app(
    principal=local_principal(user_id=UUID(os.environ["LOCAL_USER_ID"])),
    title="My assistant",
    version="0.1.0",
)
```

The UUID must stay stable across restarts because it owns the local user's
chats and memory. `local_principal` refuses a published or non-loopback API.
For multiple users, replace it with an application-owned SAML/OIDC/session
dependency; the core route ownership checks remain unchanged.

`create_app()` provides health, chat CRUD, streaming/resume,
pending-interrupt, artifact, and thread-deletion routes. Use `extra_routers`
for authentication or domain endpoints and `extra_lifespans` for
application-owned resources. Use `ThreadWorkspace` when an external process
must see staged files, and `StreamPolicy` to allow application events or
silence noisy tools.

## Client and UI

`BaseAgentClient` is synchronous and owns only routes served by core:

```python
from uuid import uuid4

from juena_core.clients.base import BaseAgentClient

thread_id = str(uuid4())
with BaseAgentClient(
    "http://127.0.0.1:8080",
    agent="assistant",
    timeout=60,
) as client:
    client.create_chat(thread_id, agent_id="assistant", title="First chat")
    for event in client.stream("Hello", thread_id=thread_id):
        print(event)
```

Subclass it to add methods for application-owned routes such as `/auth/me` or
`/research`. The `juena_core.ui` modules provide client setup, chat storage,
streaming, shared components, and math rendering without deciding the page
layout or branding.

## MCP tools

Required integrations fail clearly during discovery; optional integrations
can degrade without removing the rest of the agent:

```python
from juena_core.mcp import discover_optional_tools, discover_tools

simulation_tools = await discover_tools(simulation_mcp_url, label="Simulation")
documentation_tools = await discover_optional_tools(
    {"mcpServers": {"docs": {"url": documentation_mcp_url}}},
    label="Documentation",
)
```

Discovered tools reconnect for each call, so no adapter must remain open. Do
not bind raw tools containing ownership identifiers to a model. Wrap them in
application facade tools that obtain the principal, `thread_id`, and run
identifiers from trusted runtime state.

## Optional sandbox execution

`juena_core.sandbox` provides a durable Postgres job queue, tenant-scoped
workspaces, a rootless-Podman worker, Deep Agents backend, human approval,
stream events, execution evidence, and output-artifact collection. It is
strictly opt-in: an application that does not install `[sandbox]`, import this
package, or call `configure_sandbox()` loads no Podman client and creates no
`sandbox_jobs` table. VITESS can therefore continue to use core without
acquiring or running the sandbox.

The application owns environment parsing. Configure both core and the sandbox
before constructing its FastAPI app:

```python
from pathlib import Path

from juena_core.sandbox.config import SandboxRuntimeSettings, configure_sandbox

configure_sandbox(SandboxRuntimeSettings(
    enabled=True,
    identity_secret="load-at-least-32-random-characters-from-a-secret-store",
    workspace_root=Path("/var/lib/my-agent-sandbox/workspaces"),
))
```

Wire the server-side parts explicitly:

```python
from juena_core.sandbox.approvals import (
    SandboxResumeInput,
    register_sandbox_interrupt,
)
from juena_core.sandbox.runtime import (
    delete_runtime_workspace,
    sandbox_lifespan,
    stage_runtime_inputs,
)
from juena_core.server.service import StreamPolicy, ThreadWorkspace, create_app

register_sandbox_interrupt()
app = create_app(
    principal=principal,
    resume_input=SandboxResumeInput,
    workspace=ThreadWorkspace(
        stage=stage_runtime_inputs,
        delete=delete_runtime_workspace,
    ),
    stream_policy=StreamPolicy(
        custom_event_types=frozenset({"sandbox_status"}),
        silent_tools=frozenset({"execute"}),
    ),
    extra_lifespans=(sandbox_lifespan,),
)
```

For each execution-capable specialist, build its filesystem backend with
`build_sandbox_backend(...)`, then pass
`execution_middleware=(SandboxExecutionMiddleware(),)` and
`interrupt_on=sandbox_interrupt_on()` to `build_specialist_middleware(...)`.
Those two arguments intentionally travel together: an agent must not execute a
generated command without the matching approval boundary.

Run the worker as a separate host-side process:

```bash
uv run --extra sandbox python -m juena_core.sandbox.worker
```

The API process and worker share Postgres and `SANDBOX_WORKSPACE_ROOT`; only the
worker receives access to the rootless Podman socket. Keep the socket out of the
API container. Worker limits such as concurrency, CPU, memory, timeout, output
size, workspace quota, and TTL must match the `SandboxRuntimeSettings` values
the API uses for admission and approval display. The sandbox image, installed
scientific software, service unit, Compose wiring, and production hardening are
application/deployment assets, not library policy.

## Scale beyond one laptop

For a personal machine, the smallest deployment is Postgres plus one
application process. Keep `BIND_HOST` on loopback and `API_PUBLISHED=false`
when using `local_principal`; do not publish that API port.

When the deployment grows:

- Replace `local_principal` with a real application-owned identity dependency.
- Put Postgres on durable storage shared by all API replicas. Checkpoints and
  chat ownership then survive process and container restarts.
- Remember that the agent registry is process-local. Every worker constructs
  its own graph and resources, so size pools and model clients per worker.
- Put `ARTIFACT_ROOT` and any `ThreadWorkspace` on shared storage when requests
  may reach different replicas. Postgres does not make local files portable.
- Keep Postgres and MCP/execution services on a private network. Terminate TLS
  and enforce public access at the application or reverse proxy.
- Treat automatic table creation as bootstrap only. Use reviewed migrations
  and backups before changing a populated production database.
- Move long unattended workflows into an application-owned background-job
  system instead of keeping an HTTP stream alive indefinitely.

These changes replace deployment-specific seams; they do not require a rewrite
of the core server, client, or conversation contract.

## Development

```bash
uv sync --extra ui --extra sandbox --group dev
docker compose -f tests/compose.postgres.yml up -d --wait
uv run --frozen --extra ui --extra sandbox --group dev pytest -q
./scripts/check-imports.sh
docker compose -f tests/compose.postgres.yml down --volumes
```

The import-boundary check rejects dependencies on `juena-chatbot`,
`vitess-ai-agent`, `juena-rag`, or `vitess-rag` from core.
