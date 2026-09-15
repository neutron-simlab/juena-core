# juena-core

Shared agent, server and UI infrastructure extracted from `juena-chatbot`:
Postgres-checkpointed agent kit, streaming server, specialist delegation with
verified reports, and (behind extras) a client and a Streamlit UI shell.

Consumed by `juena-chatbot` and `vitess-ai-agent` as a sibling-directory uv path
source. Never imports either application — see `scripts/check-imports.sh`.
