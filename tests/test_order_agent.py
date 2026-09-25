from __future__ import annotations

from typing import Any

import pytest

from student_agent.agents.order_agent import OrderAgent


class FakeGateway:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((tool_name, kwargs))
        if tool_name in self.responses:
            return self.responses[tool_name]
        return {"data": {}, "evidence_ref": f"ev_{tool_name}_12345678901234567890"}


class FakeTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


@pytest.mark.anyio
async def test_order_agent_canceled() -> None:
    gateway = FakeGateway({
        "get_order": {
            "evidence_ref": "ev_order_12345678901234567890",
            "data": {
                "order_id": "order-1",
                "order_status": "canceled",
                "order_purchase_timestamp": "2018-01-01T10:00:00Z",
            },
        },
        "get_order_items": {
            "evidence_ref": "ev_items_12345678901234567890",
            "data": [
                {"order_item_id": 1, "shipping_limit_date": "2018-01-05T10:00:00Z"},
                {"order_item_id": 2, "shipping_limit_date": "2017-01-01T10:00:00Z"},
            ],
        },
    })
    trace = FakeTrace()
    agent = OrderAgent(gateway, trace)  # type: ignore[arg-type]

    case = {
        "case_id": "L3A_CASE_001",
        "opened_at": "2018-01-10T10:00:00Z",
        "customer_request": {
            "claimed_order_id": "order-1",
            "claims": [{"claim_id": "c1", "topic": "canceled_order_paid"}],
        },
    }
    state: dict[str, Any] = {}
    await agent.run(case, state)

    assert state["order_data"]["order_status"] == "canceled"
    assert state["primary_issue_candidate"] == "canceled_order_paid"
    assert len(state["items_in_window"]) == 1
    assert state["items_in_window"][0]["order_item_id"] == 1
    assert len(state["claim_assessments"]) == 1
    assert state["claim_assessments"][0]["verdict"] == "supported"
    assert state["claim_assessments"][0]["confidence"] == 0.9

    event_types = [e["event_type"] for e in trace.events]
    assert "tool_result_consumed" in event_types
    assert "handoff" in event_types
