"""CP0b, row 8: importing langchain.mcp raises LangChainBetaWarning once per
process, and the plan's decision is to let it surface rather than suppress it
globally or turn it into an error. This test only asserts the warning exists
and is the expected category — juena_core does not filter it, and this guards
against a future release silently promoting the namespace out of beta (which
would make the warning disappear) without anyone noticing the plan's
"accepted deliberately" language is now moot.
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
