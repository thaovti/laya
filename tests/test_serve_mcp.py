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


HDR = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _sse_json(resp):
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:])
    return resp.json()


class McpSession:
    """Minimal MCP client over a TestClient: initialize, then call methods."""

    def __init__(self, client, headers=None, prefix=""):
        self.c, self.n, self.url = client, 0, prefix + "/mcp"
        self.h = dict(HDR, **(headers or {}))
        r = self._post("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "t", "version": "0"}})
        self.h["mcp-session-id"] = r.headers["mcp-session-id"]
        self.c.post(self.url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=self.h)

    def _post(self, method, params):
        self.n += 1
        return self.c.post(self.url, json={"jsonrpc": "2.0", "id": self.n, "method": method, "params": params},
                           headers=self.h)

    def call(self, method, params=None):
        return _sse_json(self._post(method, params or {}))


def test_mcp_lists_tools_and_shares_router(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    router = Router()
    with TestClient(create_app(router)) as client:
        s = McpSession(client)
        names = {t["name"] for t in s.call("tools/list")["result"]["tools"]}
        assert {"laya_status", "laya_route", "laya_predict", "laya_decide"} <= names
        assert mcp_server._ROUTER is router
        res = s.call("tools/call", {"name": "laya_route", "arguments": ROUTE_ARGS})["result"]
        assert not res.get("isError")
        assert json.loads(res["content"][0]["text"])["model"]


def test_mcp_tool_error_keeps_json_payload(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    with TestClient(create_app(Router())) as client:
        s = McpSession(client)
        res = s.call("tools/call", {"name": "laya_route",
                                    "arguments": {"state": {"text": "x"}, "questions": {}}})["result"]
        assert res["isError"] is True
        # the SDK prefixes "Error executing tool laya_route: " to the preserved JSON payload
        assert "error" in json.loads(res["content"][0]["text"].split(": ", 1)[1])


def test_mcp_requires_bearer_when_api_key_set(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "s3cret")
    with TestClient(create_app(Router())) as client:
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        assert client.post("/mcp", json=body, headers=HDR).status_code == 401
        assert client.post("/mcp", json=body, headers=dict(HDR, Authorization="Bearer nope")).status_code == 401
        # latin-1 header bytes are legal on the wire; the comparison must answer 401, not raise
        bad = dict(HDR, Authorization=b"Bearer s\xe9cret")
        assert client.post("/mcp", json=body, headers=bad).status_code == 401
        s = McpSession(client, headers={"Authorization": "Bearer s3cret"})
        assert s.call("tools/list")["result"]["tools"]


def test_mcp_disabled_by_env(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_MCP", "0")
    with TestClient(create_app(Router())) as client:
        assert client.post("/mcp", json={}, headers=HDR).status_code == 404
        assert client.get("/health").status_code == 200


def test_mcp_skipped_when_package_missing(monkeypatch, caplog):
    import builtins

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.delenv("LAYA_MCP", raising=False)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "mcp" and "server" in (fromlist or ()):
            raise ImportError("No module named 'mcp'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level("INFO"):
        app = create_app(Router())
    monkeypatch.setattr(builtins, "__import__", real_import)
    with TestClient(app) as client:
        assert client.post("/mcp", json={}, headers=HDR).status_code == 404
    assert any("/mcp skipped" in r.getMessage() for r in caplog.records)


def test_two_apps_in_one_process_both_serve_mcp(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    for _ in range(2):
        with TestClient(create_app(Router())) as client:
            assert McpSession(client).call("tools/list")["result"]["tools"]


def test_mcp_answers_under_root_path(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_ROOT_PATH", "/laya")
    with TestClient(create_app(Router()), root_path="/laya") as client:
        assert McpSession(client).call("tools/list")["result"]["tools"]
