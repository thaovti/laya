"""`/mcp` on laya-serve: Streamable HTTP MCP sharing the server's Router."""
import json
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")
from fastapi.testclient import TestClient  # noqa: E402

from laya import Router  # noqa: E402
from laya.mcp import server as mcp_server  # noqa: E402
from laya.serve import create_app  # noqa: E402

ROUTE_ARGS = {
    "state": {"text": "hello world"},
    "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "first", "b": "second"}}},
}


@pytest.fixture(autouse=True)
def _restore_mcp_binding():
    saved = (mcp_server._ROUTER, mcp_server._RUN)
    yield
    mcp_server._ROUTER, mcp_server._RUN = saved


def test_bind_router_replaces_router_and_runs_tools_through_hook():
    router = Router()
    seen = []

    def run(fn):
        seen.append(threading.current_thread().name)
        return fn()

    mcp_server.bind_router(router, run=run)
    assert mcp_server._ensure_router() is router
    out = json.loads(mcp_server.laya_route_tool(**ROUTE_ARGS))
    assert out["model"]
    assert len(seen) == 1


def test_unbound_wrap_calls_directly():
    mcp_server.bind_router(Router())
    out = json.loads(mcp_server.laya_route_tool(**ROUTE_ARGS))
    assert out["model"]
