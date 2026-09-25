from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter


def _parse_iso(val: Any) -> datetime | None:
    """Parse ISO timestamp string or return None if invalid/missing."""
    if not val or not isinstance(val, str):
        return None
    try:
        # Standardize ISO 8601 timestamps
        cleaned = val.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


@dataclass
class CauseRank:
    cause_code: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"cause_code": self.cause_code, "rank": self.rank}


@dataclass
class ResponsibleParty:
    party_type: str
    party_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"party_type": self.party_type, "party_id": self.party_id}


@dataclass
class RootCauseAnalysis:
    ranked_causes: list[dict[str, Any]] = field(default_factory=list)
    responsible_parties: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_causes": self.ranked_causes,
            "responsible_parties": self.responsible_parties,
        }


@dataclass
class ShipmentInvestigationResult:
    shipment_ids: list[str] = field(default_factory=list)
    seller_ids: list[str] = field(default_factory=list)
    root_cause_analysis: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    primary_issue_candidate: str | None = None
    sla_breached_by: str | None = None  # "seller", "logistics", or None
    delay_days_seller: float = 0.0
    delay_days_logistics: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShipmentAgent:
    """Specialist Agent for investigating Shipment logistics, SLA compliance,

    and Root Cause Analysis (Member 3).
    """

    ACTOR_NAME: str = "shipment-agent"

    def __init__(self, gateway: EvidenceGateway, trace: TraceWriter) -> None:
        self.gateway = gateway
        self.trace = trace

    async def investigate(
        self,
        case_id: str,
        claimed_order_id: str | None,
        known_seller_ids: list[str] | None = None,
        case_opened_at: str | None = None,
        order_purchase_at: str | None = None,
    ) -> ShipmentInvestigationResult:
        """Execute shipment and seller investigation for the given case.

        1. Queries get_shipment_summary for tracking timeline and SLA metrics.
        2. Queries get_sellers with order_id for seller profile.
        3. Filters shipping_limits within case window [purchase - 1d, opened_at].
        4. Compares timestamps to accurately distinguish between seller delay
           and logistics carrier delay.
        5. Constructs schema-compliant root_cause_analysis and emits observable
           trace events.
        """
        evidence_refs: list[str] = []
        shipment_ids_set: set[str] = set()
        seller_ids_set: set[str] = set(str(s) for s in (known_seller_ids or []) if s)
        ranked_causes: list[dict[str, Any]] = []
        responsible_parties: list[dict[str, Any]] = []
        primary_issue: str | None = None
        sla_breached_by: str | None = None
        delay_days_seller: float = 0.0
        delay_days_logistics: float = 0.0

        if not claimed_order_id:
            # No order to investigate
            return ShipmentInvestigationResult(
                shipment_ids=[],
                seller_ids=list(seller_ids_set),
                root_cause_analysis=RootCauseAnalysis().to_dict(),
                evidence_refs=[],
                primary_issue_candidate=None,
            )

        # --- 1. Query MCP get_shipment_summary ---
        shipment_data: dict[str, Any] = {}
        try:
            shipment_response = await self.gateway.call(
                "get_shipment_summary",
                case_id=case_id,
                order_id=claimed_order_id,
            )
            ev_ref = shipment_response.get("evidence_ref")
            if ev_ref and ev_ref not in evidence_refs:
                evidence_refs.append(ev_ref)

            self.trace.emit(
                case_id=case_id,
                event_type="tool_result_consumed",
                actor=self.ACTOR_NAME,
                tool_name="get_shipment_summary",
                evidence_refs=[ev_ref] if ev_ref else None,
            )

            raw_data = shipment_response.get("data")
            if isinstance(raw_data, dict):
                shipment_data = raw_data
            elif isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], dict):
                shipment_data = raw_data[0]
        except Exception:
            pass

        # --- 2. Query MCP get_sellers (called once per order using order_id) ---
        try:
            sellers_response = await self.gateway.call(
                "get_sellers",
                case_id=case_id,
                order_id=claimed_order_id,
            )
            s_ev_ref = sellers_response.get("evidence_ref")
            if s_ev_ref and s_ev_ref not in evidence_refs:
                evidence_refs.append(s_ev_ref)

            self.trace.emit(
                case_id=case_id,
                event_type="tool_result_consumed",
                actor=self.ACTOR_NAME,
                tool_name="get_sellers",
                evidence_refs=[s_ev_ref] if s_ev_ref else None,
            )

            s_data = sellers_response.get("data")
            if isinstance(s_data, list):
                for s in s_data:
                    if isinstance(s, dict) and s.get("seller_id"):
                        seller_ids_set.add(str(s["seller_id"]))
            elif isinstance(s_data, dict) and s_data.get("seller_id"):
                seller_ids_set.add(str(s_data["seller_id"]))
        except Exception:
            pass

        # --- 3. Filter shipping_limits by case time window ---
        opened = _parse_iso(case_opened_at)
        purchase = _parse_iso(order_purchase_at)

        raw_limits = shipment_data.get("shipping_limits")
        limits: list[dict[str, Any]] = []
        if isinstance(raw_limits, list):
            for lim in raw_limits:
                if not isinstance(lim, dict):
                    continue
                ts = _parse_iso(lim.get("shipping_limit_at"))
                if not ts:
                    continue
                if opened and ts > opened:
                    continue
                if purchase and ts < purchase - timedelta(days=1):
                    continue
                limits.append(lim)

        for lim in limits:
            if lim.get("seller_id"):
                seller_ids_set.add(str(lim["seller_id"]))
            if lim.get("order_item_id"):
                shipment_ids_set.add(str(lim["order_item_id"]))

        # Fallback IDs from shipment_data root
        raw_shipment_id = (
            shipment_data.get("shipment_id")
            or shipment_data.get("order_item_id")
            or shipment_data.get("package_id")
        )
        if raw_shipment_id:
            shipment_ids_set.add(str(raw_shipment_id))

        if shipment_data.get("seller_id"):
            seller_ids_set.add(str(shipment_data["seller_id"]))
        elif isinstance(shipment_data.get("seller_ids"), list):
            for s in shipment_data["seller_ids"]:
                if s:
                    seller_ids_set.add(str(s))

        # --- 4. Evaluate SLA Timestamps ---
        limit_date = min(
            (
                _parse_iso(lim["shipping_limit_at"])
                for lim in limits
                if lim.get("shipping_limit_at")
            ),
            default=None,
        )
        if limit_date is None:
            limit_date = _parse_iso(
                shipment_data.get("shipping_limit_date")
                or shipment_data.get("seller_shipping_limit_date")
                or shipment_data.get("limit_date")
            )

        carrier = _parse_iso(
            shipment_data.get("delivered_carrier_at")
            or shipment_data.get("order_delivered_carrier_date")
            or shipment_data.get("carrier_handover_date")
            or shipment_data.get("delivered_carrier_date")
        )
        estimated = _parse_iso(
            shipment_data.get("estimated_delivery_at")
            or shipment_data.get("order_estimated_delivery_date")
            or shipment_data.get("estimated_delivery_date")
        )
        arrived = _parse_iso(
            shipment_data.get("delivered_customer_at")
            or shipment_data.get("order_delivered_customer_date")
            or shipment_data.get("delivered_customer_date")
        )

        order_status = str(shipment_data.get("order_status") or "").lower()
        primary_seller_id = (
            sorted(list(seller_ids_set))[0] if seller_ids_set else None
        )

        # Do not assess late delivery for canceled or unavailable orders
        if order_status not in ("canceled", "unavailable") and estimated:
            late = (arrived and arrived > estimated) or (
                not arrived and opened and opened > estimated
            )
            if late:
                seller_late = bool(limit_date and carrier and carrier > limit_date)
                if seller_late:
                    primary_issue = "late_delivery_seller"
                    sla_breached_by = "seller"
                    if limit_date and carrier:
                        delay_days_seller = (carrier - limit_date).total_seconds() / 86400.0
                    ranked_causes.append(
                        CauseRank(cause_code="SELLER_DISPATCH_TIMEOUT", rank=1).to_dict()
                    )
                    responsible_parties.append(
                        ResponsibleParty(party_type="seller", party_id=primary_seller_id).to_dict()
                    )
                    if arrived and arrived > estimated:
                        ranked_causes.append(
                            CauseRank(cause_code="LOGISTICS_DELAY", rank=2).to_dict()
                        )
                        responsible_parties.append(
                            ResponsibleParty(
                                party_type="logistics_provider", party_id=None
                            ).to_dict()
                        )
                else:
                    primary_issue = "late_delivery_logistics"
                    sla_breached_by = "logistics"
                    end_date = arrived or opened
                    if end_date and estimated:
                        delay_days_logistics = (end_date - estimated).total_seconds() / 86400.0
                    ranked_causes.append(
                        CauseRank(cause_code="LOGISTICS_DELAY", rank=1).to_dict()
                    )
                    responsible_parties.append(
                        ResponsibleParty(party_type="logistics_provider", party_id=None).to_dict()
                    )

        # Emit handoff event from shipment-agent to coordinator
        self.trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=self.ACTOR_NAME,
            target="coordinator",
            decision_code="SHIPMENT_INVESTIGATED",
            evidence_refs=evidence_refs if evidence_refs else None,
            attributes={
                "sla_breached_by": sla_breached_by,
                "primary_issue": primary_issue,
            },
        )

        return ShipmentInvestigationResult(
            shipment_ids=sorted(list(shipment_ids_set)),
            seller_ids=sorted(list(seller_ids_set)),
            root_cause_analysis=RootCauseAnalysis(
                ranked_causes=ranked_causes,
                responsible_parties=responsible_parties,
            ).to_dict(),
            evidence_refs=evidence_refs,
            primary_issue_candidate=primary_issue,
            sla_breached_by=sla_breached_by,
            delay_days_seller=round(delay_days_seller, 2),
            delay_days_logistics=round(delay_days_logistics, 2),
        )
