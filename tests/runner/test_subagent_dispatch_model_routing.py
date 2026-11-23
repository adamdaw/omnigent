"""``sys_session_send`` tags an explicit model and reports a routed one."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from omnigent.runner import tool_dispatch
from omnigent.runner.subagent_routing import EXPLICIT_MODEL_LABEL_KEY, ROUTING_DECISION_LABEL_KEY

PARENT_ID = "conv_parent"
CHILD_ID = "conv_child"


def _server(
    created: list[dict[str, Any]],
    child_snapshot: dict[str, Any],
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/sessions":
            created.append(json.loads(request.content))
            return httpx.Response(201, json={"id": CHILD_ID})
        if request.method == "POST" and path.endswith("/events"):
            return httpx.Response(202, json={})
        if request.method == "GET" and path == f"/v1/sessions/{CHILD_ID}":
            return httpx.Response(200, json=child_snapshot)
        if request.method == "GET" and path == f"/v1/sessions/{PARENT_ID}":
            return httpx.Response(200, json={"agent_id": "agent_parent"})
        if request.method == "GET" and "child_sessions" in path:
            return httpx.Response(200, json={"data": [], "has_more": False})
        return httpx.Response(200, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://server")


async def _dispatch(client: httpx.AsyncClient, **dispatch_args: Any) -> dict[str, Any]:
    spec = SimpleNamespace(
        sub_agents=[
            SimpleNamespace(
                name="codex",
                id="agent_codex",
                executor=SimpleNamespace(type="codex", config={}, model=None),
            )
        ]
    )
    raw = await tool_dispatch._execute_subagent_tool(
        {"agent": "codex", "args": {"input": "do it", **dispatch_args}},
        server_client=client,
        conversation_id=PARENT_ID,
        agent_spec=spec,  # type: ignore[arg-type]
        session_inbox=asyncio.Queue(),
    )
    return json.loads(raw)


@pytest.mark.asyncio
async def test_explicit_model_is_tagged_and_routed_model_is_reported() -> None:
    created: list[dict[str, Any]] = []
    snapshot = {
        "model_override": "gpt-5-6-luna",
        "labels": {ROUTING_DECISION_LABEL_KEY: "dec_1"},
    }
    async with _server(created, snapshot) as client:
        handle = await _dispatch(client, model="gpt-6-astra")

    assert created[0]["model_override"] == "gpt-6-astra"
    assert created[0]["labels"][EXPLICIT_MODEL_LABEL_KEY] == "1"
    assert handle["model_routing"] == {
        "requested_model": "gpt-6-astra",
        "model": "gpt-5-6-luna",
        "routed": True,
    }


@pytest.mark.asyncio
async def test_unrouted_dispatch_has_no_routing_field_and_no_tag() -> None:
    created: list[dict[str, Any]] = []
    async with _server(created, {"model_override": None, "labels": {}}) as client:
        handle = await _dispatch(client)

    assert EXPLICIT_MODEL_LABEL_KEY not in created[0]["labels"]
    assert "model_routing" not in handle


@pytest.mark.asyncio
async def test_routing_with_no_named_model_reports_a_null_request() -> None:
    created: list[dict[str, Any]] = []
    snapshot = {"model_override": "glm-5-2", "labels": {ROUTING_DECISION_LABEL_KEY: "dec_2"}}
    async with _server(created, snapshot) as client:
        handle = await _dispatch(client)

    assert handle["model_routing"] == {
        "requested_model": None,
        "model": "glm-5-2",
        "routed": True,
    }
