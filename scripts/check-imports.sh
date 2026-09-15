#!/usr/bin/env bash
# The dependency runs one way: juena_core must never import an application.
# `juena\b` does not match `juena_core` -- `_` is a word character, so
# `juena_core` has no boundary after `juena`. Exact; needs no exclusion list.
# `tests/` is included deliberately: a test that pulls an application in is
# how this rule actually gets broken, and it would otherwise go unnoticed
# until someone installed core on its own.
set -euo pipefail
if grep -rnE '^\s*(from|import) (juena|vitess_ai|juena_rag)\b' src/ tests/; then
    echo "juena_core imported an application. The dependency runs one way." >&2
    exit 1
fi
echo "import direction ok"
