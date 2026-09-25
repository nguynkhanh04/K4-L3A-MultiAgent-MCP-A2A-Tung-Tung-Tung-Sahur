from __future__ import annotations

from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter
from .evidence_rules import WINDOW_SLACK, parse_ts


class OrderAgent:
    """Order & Claims Specialist Agent (Member 2).

    Investigates order status, items in window, and assesses claims.
    """

    ACTOR_NAME: str = "order-agent"

    def __init__(self, gateway: EvidenceGateway, trace: TraceWriter) -> None:
        self.gateway = gateway
        self.trace = trace

    async def _fetch(
        self, tool_name: str, case_id: str, **arguments: Any
    ) -> dict[str, Any] | None:
        try:
            res = await self.gateway.call(tool_name, case_id=case_id, **arguments)
            ev_ref = res.get("evidence_ref")
            self.trace.emit(
                case_id=case_id,
                event_type="tool_result_consumed",
                actor=self.ACTOR_NAME,
                tool_name=tool_name,
                evidence_refs=[ev_ref] if ev_ref else None,
            )
            return res
        except Exception:
            return None

    async def run(self, case: dict[str, Any], state: dict[str, Any]) -> None:
        case_id = case["case_id"]
        customer_request = case.get("customer_request") or {}
        order_id = customer_request.get("claimed_order_id")
        if not order_id:
            return

        order_ev = await self._fetch("get_order", case_id, order_id=order_id)
        items_ev = await self._fetch("get_order_items", case_id, order_id=order_id)
        order = (order_ev or {}).get("data") or {}
        state["order_data"] = order

        start = parse_ts(order.get("order_purchase_timestamp"))
        end = parse_ts(case.get("opened_at"))
        items = [
            i
            for i in ((items_ev or {}).get("data") or [])
            if isinstance(i, dict)
            and (
                start is None
                or end is None
                or (
                    start - WINDOW_SLACK
                    <= (parse_ts(i.get("shipping_limit_date")) or end + WINDOW_SLACK)
                    <= end
                )
            )
        ]
        state["items_in_window"] = items

        state.setdefault("affected_entities", {})
        state["affected_entities"].setdefault("order_ids", set()).add(str(order_id))
        item_ids = state["affected_entities"].setdefault("item_ids", set())
        for i in items:
            if i.get("order_item_id"):
                item_ids.add(str(i["order_item_id"]))

        status = order.get("order_status")
        topics = {
            c["claim_id"]: c["topic"]
            for c in customer_request.get("claims", [])
            if isinstance(c, dict) and "claim_id" in c and "topic" in c
        }
        for claim_id, topic in topics.items():
            if topic not in ("canceled_order_paid", "unavailable_order_paid"):
                continue
            expected = "canceled" if topic == "canceled_order_paid" else "unavailable"
            ev_refs = (
                [order_ev["evidence_ref"]]
                if order_ev and order_ev.get("evidence_ref")
                else []
            )
            state.setdefault("claim_assessments", []).append(
                {
                    "claim_id": claim_id,
                    "verdict": "supported" if status == expected else "unsupported",
                    "confidence": 0.9 if order_ev else 0.3,
                    "evidence_refs": ev_refs,
                }
            )

        decision_code = f"ORDER_{(status or 'unknown').upper()}"
        self.trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=self.ACTOR_NAME,
            target="coordinator",
            decision_code=decision_code,
        )
