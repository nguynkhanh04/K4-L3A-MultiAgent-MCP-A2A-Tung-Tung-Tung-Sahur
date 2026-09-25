"""Payment & Financial Resolution Specialist Agent.

Chuyên môn: Đối soát Giao dịch, Cổng thanh toán & Tính toán Đền bù Tài chính.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter
from .evidence_rules import WINDOW_SLACK, parse_ts


def _dec(val: Any) -> Decimal:
    """Convert any numeric value to a 2-decimal-place Decimal safely."""
    if val is None:
        return Decimal("0.00")
    if isinstance(val, Decimal):
        return val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if isinstance(val, float):
        return Decimal(f"{val:.10f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _sum_dec(amounts: list[Decimal]) -> Decimal:
    total = Decimal("0.00")
    for a in amounts:
        total += a
    return total


@dataclass
class RefundLine:
    """Một dòng chi tiết đề xuất hoàn tiền."""

    reason_code: str
    amount_brl: Decimal
    entity_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "amount_brl": float(self.amount_brl),
            "entity_id": self.entity_id,
        }


@dataclass
class PaymentInvestigationResult:
    """Kết quả trả về từ PaymentAgent, sẵn sàng để ghép vào L3A output."""

    payment_references: list[str]
    financial_resolution: dict[str, Any]
    detected_issue: str | None
    evidence_refs: list[str]
    captured_total_brl: float
    refunded_total_brl: float
    refundable_total_brl: float
    is_valid_transaction: bool


class PaymentAgent:
    """Specialist Agent – Payment & Financial Resolution (Member 4)."""

    ACTOR = "payment-agent"

    def __init__(self) -> None:
        pass

    async def investigate(
        self,
        *,
        case_id: str,
        order_id: str,
        gateway: EvidenceGateway,
        trace: TraceWriter | None = None,
        expected_order_total: float | int | Decimal | None = None,
        order_status: str | None = None,
        order_purchase_at: str | None = None,
        case_opened_at: str | None = None,
    ) -> PaymentInvestigationResult:
        """Gọi 3 MCP tools, đối soát dòng tiền và trả kết quả tài chính."""
        evidence_refs: list[str] = []

        # ── 1. get_order_payments ──────────────────────────────────────────
        try:
            pmts_ev = await gateway.call(
                "get_order_payments", case_id=case_id, order_id=order_id
            )
            pmts_ref = pmts_ev.get("evidence_ref", "")
            if pmts_ref:
                evidence_refs.append(pmts_ref)
                if trace:
                    trace.emit(
                        case_id=case_id,
                        event_type="tool_result_consumed",
                        actor=self.ACTOR,
                        tool_name="get_order_payments",
                        evidence_refs=[pmts_ref],
                    )
        except Exception:
            pmts_ev = {"data": []}

        # ── 2. get_payment_timeline ────────────────────────────────────────
        try:
            ptl_ev = await gateway.call(
                "get_payment_timeline", case_id=case_id, order_id=order_id
            )
            ptl_ref = ptl_ev.get("evidence_ref", "")
            if ptl_ref:
                evidence_refs.append(ptl_ref)
                if trace:
                    trace.emit(
                        case_id=case_id,
                        event_type="tool_result_consumed",
                        actor=self.ACTOR,
                        tool_name="get_payment_timeline",
                        evidence_refs=[ptl_ref],
                    )
        except Exception:
            ptl_ev = {"data": {"events": []}}

        # ── 3. get_refund_timeline (bọc try/except vì tool lỗi khi không có refund) ──
        try:
            rtl_ev = await gateway.call(
                "get_refund_timeline", case_id=case_id, order_id=order_id
            )
            rtl_ref = rtl_ev.get("evidence_ref", "")
            if rtl_ref:
                evidence_refs.append(rtl_ref)
                if trace:
                    trace.emit(
                        case_id=case_id,
                        event_type="tool_result_consumed",
                        actor=self.ACTOR,
                        tool_name="get_refund_timeline",
                        evidence_refs=[rtl_ref],
                    )
        except Exception:
            rtl_ev = {"data": {"events": []}}

        # ── 4. Chuẩn hóa & lọc dữ liệu trong khoảng thời gian case ─────────
        payments_raw = self._extract_list(pmts_ev.get("data", []), "payments")
        ptl_events = self._extract_list(ptl_ev.get("data", {}), "events")
        if not ptl_events:
            ptl_events = self._extract_list(ptl_ev.get("data", {}), "timeline")

        refund_events = self._extract_list(rtl_ev.get("data", {}), "events")
        if not refund_events:
            refund_events = self._extract_list(rtl_ev.get("data", {}), "refunds")

        start = parse_ts(order_purchase_at)
        end = parse_ts(case_opened_at)

        def in_window(val: Any) -> bool:
            ts = parse_ts(val)
            if ts is None:
                return True
            if end and ts > end:
                return False
            return not (start and ts < start - WINDOW_SLACK)

        ptl_events = [e for e in ptl_events if in_window(e.get("event_at"))]
        refund_events = [e for e in refund_events if in_window(e.get("event_at"))]

        # ── 5. Đối soát ───────────────────────────────────────────────────
        return self._reconcile(
            case_id=case_id,
            order_id=order_id,
            payments=payments_raw,
            payment_timeline=ptl_events,
            refund_timeline=refund_events,
            evidence_refs=evidence_refs,
            expected_order_total=expected_order_total,
            order_status=order_status,
        )

    def _reconcile(
        self,
        *,
        case_id: str,
        order_id: str,
        payments: list[dict[str, Any]],
        payment_timeline: list[dict[str, Any]],
        refund_timeline: list[dict[str, Any]],
        evidence_refs: list[str],
        expected_order_total: float | int | Decimal | None = None,
        order_status: str | None = None,
    ) -> PaymentInvestigationResult:
        """Thuần logic đối soát – không gọi gateway, không có side-effect."""
        payment_references: list[str] = []
        payment_items: list[dict[str, Any]] = []
        captured_total = Decimal("0.00")

        for p in payments:
            ref = (
                p.get("payment_reference")
                or p.get("payment_id")
                or p.get("transaction_id")
            )
            if ref and str(ref) not in payment_references:
                payment_references.append(str(ref))

            amt = _dec(
                p.get("payment_value")
                or p.get("amount_brl")
                or p.get("amount")
                or 0
            )
            captured_total += amt
            payment_items.append({
                "ref": ref,
                "amount": amt,
                "type": str(p.get("payment_type", "")).lower(),
                "raw": p,
            })

        pending_refunds: list[dict[str, Any]] = []
        failed_refunds: list[dict[str, Any]] = []
        completed_refunds: list[dict[str, Any]] = []
        refunded_total = Decimal("0.00")

        for r in refund_timeline:
            r_status = str(r.get("status", "")).lower()
            r_amount = _dec(
                r.get("amount_brl")
                or r.get("amount")
                or r.get("refund_amount")
                or 0
            )
            r_id = r.get("refund_id") or r.get("id") or order_id
            r_time = r.get("event_at") or ""

            if r_status in {"pending", "processing", "in_progress", "submitted"}:
                pending_refunds.append({"id": r_id, "amount": r_amount, "time": r_time})
            elif r_status in {"failed", "error", "rejected", "declined"}:
                failed_refunds.append({"id": r_id, "amount": r_amount, "time": r_time})
            elif (
                r_status in {"completed", "success", "processed", "approved"}
                or r.get("is_completed")
            ):
                completed_refunds.append({"id": r_id, "amount": r_amount, "time": r_time})
                refunded_total += r_amount

        # Phát hiện duplicate từ events hoặc timeline
        duplicate_items: list[dict[str, Any]] = []
        seen_ext_ids: set[str] = set()
        captured_events_amounts: list[Decimal] = []

        for evt in payment_timeline:
            evt_type = str(evt.get("event_type", evt.get("status", ""))).lower()
            evt_amount = _dec(
                evt.get("amount_brl")
                or evt.get("amount")
                or evt.get("payment_value")
                or 0
            )
            if evt_type in {"captured", "approved", "settled"}:
                captured_events_amounts.append(evt_amount)
                ext_id = str(
                    evt.get("external_transaction_id")
                    or evt.get("transaction_id")
                    or ""
                )
                if ext_id and ext_id in seen_ext_ids:
                    ref_id = evt.get("payment_reference") or ext_id
                    duplicate_items.append({"ref": ref_id, "amount": evt_amount})
                if ext_id:
                    seen_ext_ids.add(ext_id)

        # Phát hiện mismatch từ event tường minh
        has_mismatch_event = any(
            evt.get("event_type") == "reconciliation_mismatch"
            for evt in payment_timeline
        )

        expected_total = (
            _dec(expected_order_total)
            if expected_order_total is not None
            else None
        )

        # Heuristic phát hiện duplicate từ captured events
        if not duplicate_items and len(captured_events_amounts) >= 2:
            repeated = len(set(captured_events_amounts)) < len(captured_events_amounts)
            tot = _sum_dec(captured_events_amounts)
            base_total = expected_total or captured_total
            if repeated and base_total > Decimal("0.00") and tot > base_total:
                # Capture trùng tiền và tổng lớn hơn đơn hàng
                dup_amt = captured_events_amounts[0]
                duplicate_items.append({"ref": order_id, "amount": dup_amt})

        # Heuristic bổ sung từ payment_items
        if not duplicate_items:
            sig_map: dict[str, list[dict[str, Any]]] = {}
            for item in payment_items:
                sig = f"{item['type']}|{item['amount']}"
                sig_map.setdefault(sig, []).append(item)

            for _sig_key, items in sig_map.items():
                if len(items) <= 1:
                    continue
                p_type = items[0]["type"]
                amount = items[0]["amount"]
                exp = expected_total
                if p_type == "credit_card" and (exp is None or amount == exp):
                    for extra in items[1:]:
                        duplicate_items.append({"ref": extra["ref"], "amount": extra["amount"]})
                for item in items[1:]:
                    if item["raw"].get("is_duplicate") or item["raw"].get("status") == "duplicate":
                        duplicate_items.append({"ref": item["ref"], "amount": item["amount"]})

        detected_issue: str | None = None
        refund_lines: list[RefundLine] = []

        # Phân loại theo thứ tự ưu tiên:
        # refund_failed / refund_pending > duplicate_charge > payment_mismatch > split
        latest_refund_status = None
        latest_refund_item = None
        if refund_timeline:
            sorted_refunds = sorted(
                refund_timeline, key=lambda e: str(e.get("event_at") or "")
            )
            if sorted_refunds:
                latest_refund_item = sorted_refunds[-1]
                latest_refund_status = str(latest_refund_item.get("status", "")).lower()

        if latest_refund_status in {"failed", "error", "rejected", "declined"} or (
            failed_refunds and not pending_refunds
        ):
            detected_issue = "refund_failed"
            amt_brl = (
                _dec(latest_refund_item.get("amount_brl"))
                if latest_refund_item
                else Decimal("0.00")
            )
            target_fr = failed_refunds or [{"amount": amt_brl, "id": order_id}]
            for fr in target_fr:
                refund_lines.append(
                    RefundLine(
                        reason_code="refund_failed_retry",
                        amount_brl=fr["amount"],
                        entity_id=str(fr["id"]),
                    )
                )

        elif (
            latest_refund_status
            in {"pending", "processing", "in_progress", "submitted"}
            or pending_refunds
        ):
            detected_issue = "refund_pending"
            amt_brl = (
                _dec(latest_refund_item.get("amount_brl"))
                if latest_refund_item
                else Decimal("0.00")
            )
            target_pr = pending_refunds or [{"amount": amt_brl, "id": order_id}]
            for pr in target_pr:
                refund_lines.append(
                    RefundLine(
                        reason_code="refund_pending_expedite",
                        amount_brl=pr["amount"],
                        entity_id=str(pr["id"]),
                    )
                )

        elif duplicate_items:
            detected_issue = "duplicate_charge"
            for dup in duplicate_items:
                refund_lines.append(
                    RefundLine(
                        reason_code="duplicate_charge",
                        amount_brl=dup["amount"],
                        entity_id=str(dup["ref"]) if dup["ref"] else order_id,
                    )
                )

        elif has_mismatch_event:
            detected_issue = "payment_mismatch"
            diff = (captured_total - expected_total) if expected_total else Decimal("0.00")
            if diff > Decimal("0.00"):
                refund_lines.append(
                    RefundLine(
                        reason_code="overcharge_mismatch",
                        amount_brl=diff,
                        entity_id=order_id,
                    )
                )

        elif expected_total is not None and captured_total != expected_total:
            diff = captured_total - expected_total
            if diff > Decimal("0.00"):
                detected_issue = "payment_mismatch"
                refund_lines.append(
                    RefundLine(
                        reason_code="overcharge_mismatch",
                        amount_brl=diff,
                        entity_id=order_id,
                    )
                )

        elif (
            len(payment_items) > 1
            and (expected_total is None or captured_total == expected_total)
            and not has_mismatch_event
        ):
            detected_issue = "valid_split_payment"

        # Đơn bị hủy / không khả dụng
        if order_status in {"canceled", "unavailable"}:
            remaining = max(Decimal("0.00"), captured_total - refunded_total)
            if remaining > Decimal("0.00") and not refund_lines:
                reason = "order_canceled" if order_status == "canceled" else "order_unavailable"
                refund_lines.append(
                    RefundLine(
                        reason_code=reason,
                        amount_brl=remaining,
                        entity_id=order_id,
                    )
                )

        recommended_dec = _sum_dec([line.amount_brl for line in refund_lines])
        serialised_lines = [line.to_dict() for line in refund_lines]

        is_valid = (
            detected_issue in {None, "valid_split_payment"}
            and recommended_dec == Decimal("0.00")
            and not failed_refunds
            and not pending_refunds
        )

        financial_resolution: dict[str, Any] = {
            "currency": "BRL",
            "recommended_refund_brl": float(recommended_dec),
            "refund_lines": serialised_lines,
        }

        refundable = max(Decimal("0.00"), captured_total - refunded_total)

        return PaymentInvestigationResult(
            payment_references=payment_references[:20],
            financial_resolution=financial_resolution,
            detected_issue=detected_issue,
            evidence_refs=evidence_refs,
            captured_total_brl=float(captured_total),
            refunded_total_brl=float(refunded_total),
            refundable_total_brl=float(refundable),
            is_valid_transaction=is_valid,
        )

    @staticmethod
    def _extract_list(data: Any, key: str) -> list[dict[str, Any]]:
        """Chuẩn hóa data từ MCP: có thể là list thẳng hoặc wrapped trong dict."""
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            inner = data.get(key, data)
            if isinstance(inner, list):
                return [d for d in inner if isinstance(d, dict)]
            return [data]
        return []
