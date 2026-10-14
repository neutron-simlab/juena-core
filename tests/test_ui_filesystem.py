"""Python bridge contracts; browser navigation is exercised by Playwright."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from juena_core.ui import filesystem


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    (tmp_path / "filesystem.js").write_text("export default () => {}")
    monkeypatch.setattr(filesystem, "_ASSETS", tmp_path)
    state = State(thread_id="one")
    component = Mock()
    component.side_effect = lambda **kwargs: SimpleNamespace(ask=state.get(kwargs["key"], {}).get("ask"))
    st = SimpleNamespace(session_state=state, rerun=Mock(), error=Mock())
    monkeypatch.setattr(filesystem, "st", st)
    monkeypatch.setattr(filesystem, "_component", lambda: component)

    def render(thread="one", scope="opaque"):
        state.thread_id = thread
        filesystem.render_filesystem.__wrapped__(
            api_url="https://public.example/api/", thread_id=thread, session_scope=scope,
            refresh_generation=3,
        )
        return component.call_args.kwargs
    return state, st, render


def test_bridge_uses_only_public_configuration_and_stable_identity(bridge):
    state, st, render = bridge
    first = render()
    assert first["data"] == {
        "apiUrl": "https://public.example/api", "threadId": "one",
        "sessionScope": "opaque", "refreshGeneration": 3, "initialView": None,
    }
    assert render()["key"] == first["key"]
    assert render("two")["key"] != first["key"]
    st.rerun.assert_not_called()


def test_drafts_are_partitioned_by_conversation_and_reset_on_session_change(bridge):
    state, st, render = bridge
    first = render()
    draft = {"cwd": "/workspace", "draft": {"text": "unsaved"}}
    state[first["key"]] = {"view": draft}
    first["on_view_change"]()
    assert render("two")["data"]["initialView"] is None
    assert render("one")["data"]["initialView"] == draft
    assert render(scope="new session")["data"]["initialView"] is None


def test_composer_event_is_consumed_once_and_old_conversation_events_are_ignored(bridge):
    state, st, render = bridge
    args = render()
    event = {"id": "unique", "thread_id": "one", "path": "/workspace/code.py"}
    state[args["key"]] = {"ask": event}
    args["on_ask_change"]()
    st.rerun.assert_not_called()  # Streamlit disallows reruns inside callbacks.
    render()
    assert state.pop("filesystem_mention") == event
    st.rerun.assert_called_once_with(scope="app")
    args["on_ask_change"]()
    render()
    st.rerun.assert_called_once()
    state.thread_id = "two"
    state[args["key"]]["ask"] = {**event, "id": "late"}
    args["on_ask_change"]()
    assert "filesystem_mention" not in state


def test_auth_loss_clears_preserved_views_and_queued_paths(bridge):
    state, st, render = bridge
    args = render()
    state.filesystem_views["one"] = {"draft": "private"}
    state.filesystem_mention = {"path": "private"}
    args["on_auth_lost_change"]()
    assert not state.filesystem_views
    assert "filesystem_mention" not in state
