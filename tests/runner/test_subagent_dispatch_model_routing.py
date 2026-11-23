"""``sys_session_send`` tags an explicit model and reports a routed one."""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from omnigent.runner import tool_dispatch
from omnigent.runner.subagent_routing import EXPLICIT_MODEL_LABEL_KEY

PARENT_ID = "conv_parent"
CHILD_ID = "conv_child"


def _decision(picked: str, *, applied: bool = True, model: str | None = None) -> dict[str, Any]:
    """A ``model_routing`` object as the message POST response carries it."""
    return {"picked_model": picked, "applied": applied, "model": model or picked}


def _server(
    created: list[dict[str, Any]],
    event_response: dict[str, Any] | bytes,
    *,
    child_id: str = CHILD_ID,
    child_title: str = "codex:worker",
    metadata_status: int = 200,
) -> httpx.AsyncClient:
    """Mock server: the message POST answers *event_response*."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/sessions":
            created.append(json.loads(request.content))
            return httpx.Response(201, json={"id": child_id})
        if request.method == "POST" and path.endswith("/events"):
            if isinstance(event_response, bytes):
                return httpx.Response(202, content=event_response)
            return httpx.Response(202, json=event_response)
        if request.method == "GET" and path == f"/v1/sessions/{child_id}":
            return httpx.Response(
                metadata_status,
                json={
                    "parent_session_id": PARENT_ID,
                    "title": child_title,
                    "model_override": None,
                    "labels": {},
                },
            )
        if request.method == "GET" and path == f"/v1/sessions/{PARENT_ID}":
            return httpx.Response(200, json={"agent_id": "agent_parent"})
        if request.method == "GET" and "child_sessions" in path:
            return httpx.Response(
                200, json={"data": [{"id": child_id, "title": child_title}], "has_more": False}
            )
        return httpx.Response(200, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://server")


async def _dispatch(
    client: httpx.AsyncClient, *, title: str | None = None, **dispatch_args: Any
) -> dict[str, Any]:
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
        {
            "agent": "codex",
            **({"title": title} if title is not None else {}),
            "args": {"input": "do it", **dispatch_args},
        },
        server_client=client,
        conversation_id=PARENT_ID,
        agent_spec=spec,  # type: ignore[arg-type]
        session_inbox=asyncio.Queue(),
    )
    return json.loads(raw)


@pytest.mark.asyncio
async def test_explicit_model_is_tagged_and_routed_model_is_reported() -> None:
    created: list[dict[str, Any]] = []
    response = {"queued": True, "model_routing": _decision("gpt-5-6-luna")}
    async with _server(created, response) as client:
        handle = await _dispatch(client, model="gpt-6-astra")

    assert created[0]["model_override"] == "gpt-6-astra"
    assert created[0]["labels"][EXPLICIT_MODEL_LABEL_KEY] == "1"
    assert handle["model_routing"] == {
        "requested_model": "gpt-6-astra",
        "model": "gpt-5-6-luna",
        "routed": True,
        "picked_model": "gpt-5-6-luna",
    }


@pytest.mark.asyncio
async def test_unrouted_dispatch_has_no_routing_field_and_no_tag() -> None:
    created: list[dict[str, Any]] = []
    async with _server(created, {"queued": True}) as client:
        handle = await _dispatch(client)

    assert EXPLICIT_MODEL_LABEL_KEY not in created[0]["labels"]
    assert "model_routing" not in handle


@pytest.mark.asyncio
async def test_routing_with_no_named_model_reports_a_null_request() -> None:
    created: list[dict[str, Any]] = []
    async with _server(created, {"model_routing": _decision("glm-5-2")}) as client:
        handle = await _dispatch(client)

    assert handle["model_routing"] == {
        "requested_model": None,
        "model": "glm-5-2",
        "routed": True,
        "picked_model": "glm-5-2",
    }


@pytest.mark.asyncio
async def test_unapplied_pick_is_reported_as_not_routed() -> None:
    created: list[dict[str, Any]] = []
    response = {"model_routing": _decision("glm-5-2", applied=False, model="gpt-5-6-luna")}
    async with _server(created, response) as client:
        handle = await _dispatch(client)

    # The child still runs its inherited model; the pick is shown, not claimed.
    assert handle["model_routing"] == {
        "requested_model": None,
        "model": "gpt-5-6-luna",
        "routed": False,
        "picked_model": "glm-5-2",
    }


@pytest.mark.asyncio
async def test_unapplied_pick_on_an_unset_model_does_not_claim_a_model() -> None:
    created: list[dict[str, Any]] = []
    response = {"model_routing": {"picked_model": "glm-5-2", "applied": False, "model": None}}
    async with _server(created, response) as client:
        handle = await _dispatch(client)

    assert handle["model_routing"]["routed"] is False
    assert handle["model_routing"]["model"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_response",
    [b"not json", b"[]", b'{"model_routing": "oops"}', b'{"model_routing": {"applied": true}}'],
)
async def test_malformed_response_does_not_fail_an_accepted_dispatch(
    event_response: bytes,
) -> None:
    async with _server([], event_response) as client:
        handle = await _dispatch(client)

    assert handle["status"] == "launching"
    assert "model_routing" not in handle


@pytest.mark.asyncio
async def test_metadata_read_failures_cannot_hide_or_fake_a_report() -> None:
    # Reports come from the POST response, so a failing child read changes nothing.
    child_id = f"conv_unreadable_{uuid.uuid4().hex[:8]}"
    response = {"model_routing": _decision("glm-5-2")}
    async with _server([], response, child_id=child_id, metadata_status=500) as client:
        handle = await _dispatch(client, title="worker")

    assert handle["model_routing"]["picked_model"] == "glm-5-2"


@pytest.mark.asyncio
async def test_routing_on_a_continuation_send_is_reported() -> None:
    child_id = f"conv_continued_{uuid.uuid4().hex[:8]}"
    response = {"model_routing": _decision("glm-5-2")}
    async with _server([], response, child_id=child_id) as client:
        handle = await _dispatch(client, title="worker")

    assert handle["model_routing"]["picked_model"] == "glm-5-2"
    assert handle["model_routing"]["routed"] is True


@pytest.mark.asyncio
async def test_a_continuation_the_server_did_not_route_reports_nothing() -> None:
    child_id = f"conv_unrouted_{uuid.uuid4().hex[:8]}"
    async with _server([], {"queued": True}, child_id=child_id) as client:
        handle = await _dispatch(client, title="worker")

    assert "model_routing" not in handle


@pytest.mark.asyncio
async def test_send_by_session_id_reports_routing() -> None:
    child_id = f"conv_by_id_{uuid.uuid4().hex[:8]}"
    response = {"model_routing": _decision("glm-5-2")}
    async with _server([], response, child_id=child_id) as client:
        raw = await tool_dispatch._send_to_existing_session(
            child_id, "continue", server_client=client, conversation_id=PARENT_ID
        )

    assert json.loads(raw)["model_routing"] == {
        "requested_model": None,
        "model": "glm-5-2",
        "routed": True,
        "picked_model": "glm-5-2",
    }


@pytest.mark.asyncio
async def test_send_by_session_id_without_a_decision_reports_nothing() -> None:
    child_id = f"conv_by_id_none_{uuid.uuid4().hex[:8]}"
    async with _server([], {"queued": True}, child_id=child_id) as client:
        raw = await tool_dispatch._send_to_existing_session(
            child_id, "continue", server_client=client, conversation_id=PARENT_ID
        )

    assert "model_routing" not in json.loads(raw)


@pytest.mark.asyncio
async def test_interleaved_sends_each_report_their_own_decision() -> None:
    # Both POSTs are in flight together while every child read shows a decision
    # label from elsewhere. Each handle reports its own response's decision only.
    suffix = uuid.uuid4().hex[:8]
    child_ids = {"a": f"conv_race_a_{suffix}", "b": f"conv_race_b_{suffix}"}
    both_posted = asyncio.Event()
    posted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posted
        path = request.url.path
        if request.method == "POST" and path.endswith("/events"):
            posted += 1
            if posted == 2:
                both_posted.set()
            await both_posted.wait()
            text = json.loads(request.content)["data"]["content"][0]["text"]
            return httpx.Response(202, json={"model_routing": _decision(f"pick-for-{text}")})
        if request.method == "GET" and path.startswith("/v1/sessions/conv_race_"):
            return httpx.Response(
                200,
                json={
                    "parent_session_id": PARENT_ID,
                    "title": "codex:worker",
                    "model_override": "someone-elses-pick",
                    "labels": {"omnigent.routing.decision_id": "dec_other"},
                },
            )
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://server"
    ) as client:
        first, second = await asyncio.gather(
            *(
                tool_dispatch._send_to_existing_session(
                    child_id, key, server_client=client, conversation_id=PARENT_ID
                )
                for key, child_id in child_ids.items()
            )
        )

    assert json.loads(first)["model_routing"]["picked_model"] == "pick-for-a"
    assert json.loads(second)["model_routing"]["picked_model"] == "pick-for-b"
