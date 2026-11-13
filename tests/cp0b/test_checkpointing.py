"""A two-node graph compiles and round-trips through ``AsyncPostgresSaver``
against a real Postgres.

Requires a throwaway Postgres:
``docker compose -f tests/compose.postgres.yml up -d --wait``
It does not skip when Postgres is unreachable, because a skip would hide
whether checkpointing really works.
"""

from typing import TypedDict

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

DSN = "postgresql://juena_core_test:juena_core_test@127.0.0.1:55432/juena_core_test"


class _State(TypedDict):
    count: int


def _node_a(state: _State) -> dict:
    return {"count": state["count"] + 1}


def _node_b(state: _State) -> dict:
    return {"count": state["count"] + 10}


def _build_graph():
    g = StateGraph(_State)
    g.add_node("a", _node_a)
    g.add_node("b", _node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


@pytest.mark.asyncio
async def test_graph_round_trips_through_real_postgres():
    cfg = {"configurable": {"thread_id": "cp0b-checkpoint-test"}}

    async with AsyncPostgresSaver.from_conn_string(DSN) as saver:
        await saver.setup()
        graph = _build_graph().compile(checkpointer=saver)
        result = await graph.ainvoke({"count": 0}, cfg)
        assert result["count"] == 11

    # A second saver instance against the same DSN: proves the state was
    # written to Postgres, not held in an in-process cache.
    async with AsyncPostgresSaver.from_conn_string(DSN) as saver2:
        graph2 = _build_graph().compile(checkpointer=saver2)
        state = await graph2.aget_state(cfg)
        assert state.values["count"] == 11
