"""Importing ``langchain.mcp`` raises ``LangChainBetaWarning`` once per process.

juena_core lets the warning surface rather than suppressing it or turning it
into an error. If a release takes the namespace out of beta, the warning
disappears and this test fails, so that choice can be revisited.
"""

import importlib
import sys

import pytest


def test_langchain_mcp_raises_beta_warning():
    sys.modules.pop("langchain.mcp", None)
    sys.modules.pop("langchain.mcp.adapter", None)

    with pytest.warns(Warning) as record:
        importlib.import_module("langchain.mcp")

    assert any(w.category.__name__ == "LangChainBetaWarning" for w in record)
