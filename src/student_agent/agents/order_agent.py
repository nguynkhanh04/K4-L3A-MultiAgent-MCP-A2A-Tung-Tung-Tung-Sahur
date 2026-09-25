from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter

try:
    from .evidence_rules import WINDOW_SLACK, parse_ts
except ImportError:
    WINDOW_SLACK = timedelta(days=1)

    def parse_ts(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, Exception):
            return None


class OrderAgent:
    """Order & Claims Specialist Agent.

    Owner: Thành viên 2 (Khánh)
    """

    def __init__(self, gateway: EvidenceGateway, trace: TraceWriter) -> None:
        self.gateway = gateway
        self.trace = trace

    async def _fetch(
        self, tool_name: str, case_id: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        try:
            ev = await self.gateway.call(tool_name, case_id=case_id, **kwargs)
            ref = ev.get("evidence_ref") if isinstance(ev, dict) else None
            if ref:
                self.trace.emit(
                    case_id=case_id,
                    event_type="tool_result_consumed",
                    actor="order-agent",
                    tool_name=tool_name,
                    evidence_refs=[ref],
                )
            return ev
        except Exception:
            return None

    async def run(self, case: dict[str, Any], state: dict[str, Any]) -> None:
        case_id = case["case_id"]
        order_id = case.get("customer_request", {}).get("claimed_order_id")
        if not order_id:
            return

        state.setdefault("affected_entities", {})
        state["affected_entities"].setdefault("order_ids", set()).add(order_id)
        state["affected_entities"].setdefault("item_ids", set())
        state.setdefault("evidence_refs", [])

        order_ev = await self._fetch("get_order", case_id, order_id=order_id)
        items_ev = await self._fetch("get_order_items", case_id, order_id=order_id)

        if order_ev and order_ev.get("evidence_ref"):
            state["evidence_refs"].append(order_ev["evidence_ref"])
        if items_ev and items_ev.get("evidence_ref"):
            state["evidence_refs"].append(items_ev["evidence_ref"])

        order = (order_ev or {}).get("data") or {}
        state["order_data"] = order

        start = parse_ts(order.get("order_purchase_timestamp"))
        end = parse_ts(case.get("opened_at"))
        items = [
            i
            for i in ((items_ev or {}).get("data") or [])
            if start
            and end
            and start - WINDOW_SLACK
            <= (parse_ts(i.get("shipping_limit_date")) or end + WINDOW_SLACK)
            <= end
        ]
        state["items_in_window"] = items
        for item in items:
            if "order_item_id" in item:
                state["affected_entities"]["item_ids"].add(str(item["order_item_id"]))

        status = order.get("order_status")
        if status in ("canceled", "unavailable"):
            state["primary_issue_candidate"] = (
                "canceled_order_paid" if status == "canceled" else "unavailable_order_paid"
            )

        topics = {
            c.get("claim_id"): c.get("topic")
            for c in case.get("customer_request", {}).get("claims", [])
        }
        for claim_id, topic in topics.items():
            if not claim_id or topic not in ("canceled_order_paid", "unavailable_order_paid"):
                continue
            expected = "canceled" if topic == "canceled_order_paid" else "unavailable"
            ev_refs = [order_ev["evidence_ref"]] if order_ev and "evidence_ref" in order_ev else []
            state.setdefault("claim_assessments", []).append({
                "claim_id": claim_id,
                "verdict": "supported" if status == expected else "unsupported",
                "confidence": 0.9 if order_ev else 0.3,
                "evidence_refs": ev_refs,
            })

        self.trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="order-agent",
            target="coordinator",
            decision_code=f"ORDER_{(status or 'unknown').upper()}",
        )
