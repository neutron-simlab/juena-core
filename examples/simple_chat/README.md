# Simple chat example

This is a minimal application built from `juena-core`: one tool-free
LangChain agent, the core FastAPI service, Postgres conversation history, the
core HTTP client, and a small Streamlit page.

The example deliberately does not enable the optional sandbox. It demonstrates
the smallest base server; add `[sandbox]` and the explicit wiring from the main
README only when an agent needs isolated command execution.

It is intentionally local-only. The fixed demo identity is safe only because
the API binds to loopback and cannot be configured as published. A real
multi-user application must replace `local_principal` with its own authenticated
principal dependency.

The Compose file therefore runs **Postgres only**. The API and UI run on the
host, where the API can stay on `127.0.0.1`. A container normally binds its
service to `0.0.0.0`; core correctly refuses that combination with a fixed
identity. Containerize the API only after adding real authentication.

## Run it

From the `juena-core` repository root:

```bash
uv sync --extra ui --group dev
docker compose -f examples/simple_chat/compose.postgres.yml up -d --wait
cp examples/simple_chat/.env.example examples/simple_chat/.env
```

Edit `.env` with valid credentials, then load it in each terminal:

```bash
set -a
source examples/simple_chat/.env
set +a
```

Start the API in the first terminal:

```bash
uv run uvicorn examples.simple_chat.service:app --host 127.0.0.1 --port 8080
```

Start the UI in the second:

```bash
uv run --extra ui streamlit run examples/simple_chat/streamlit_app.py
```

Open the URL printed by Streamlit. Postgres history persists across service
restarts. The UI deliberately starts a fresh thread for a fresh browser
session to keep the example small.

Stop Postgres without deleting its conversation volume:

```bash
docker compose -f examples/simple_chat/compose.postgres.yml down
```

Add `--volumes` only when you also want to delete the demo history.

## Where to extend it

- `config.py` translates application environment into immutable core settings.
- `agent.py` owns the prompt, tools, middleware, and compiled graph.
- `service.py` registers agents and chooses the identity and HTTP surface.
- `streamlit_app.py` owns the product page while reusing core rendering and
  streaming contracts.

Add tools or middleware in `agent.py`; add another stable agent ID and factory
for another mode; add application routers and lifespans in `service.py`. The
main repository README covers the deployment changes needed beyond one laptop.
