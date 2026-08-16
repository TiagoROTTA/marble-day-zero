"""Slack WebClient wrapper. Single responsibility: send approval messages.

Degraded mode: if SLACK_BOT_TOKEN is empty, print the message instead of posting.
"""
from slack_sdk import WebClient

from src.config import settings
from src.slack.blocks import (
    approval_blocks,
    purchase_order_blocks,
    review_queue_blocks,
)

_client = WebClient(token=settings.slack_bot_token) if settings.slack_bot_token else None


def send_approval(thread_id: str, title: str, context: dict) -> str | None:
    """Post an approval message. Returns the message ts (None in degraded mode)."""
    if _client is None:
        print(f"[SLACK DISABLED] would send: {title}")
        print(f"  thread_id={thread_id}")
        for k, v in context.items():
            print(f"  {k}: {v}")
        return None

    resp = _client.chat_postMessage(
        channel=settings.slack_approval_channel,
        blocks=approval_blocks(thread_id, title, context),
        text=title,
    )
    return resp["ts"]


def send_review_queue(
    thread_id: str,
    restaurant_name: str,
    items: list[dict],
    dropped: int,
) -> str | None:
    """Post the low-confidence review card. Returns the ts (None in degraded mode)."""
    # ASCII: this title is print()ed in degraded mode, and a non-ASCII dash
    # raises UnicodeEncodeError on the cp1252 Windows console.
    count = f"{len(items)} item needs" if len(items) == 1 else f"{len(items)} items need"
    title = f"{restaurant_name} - {count} a human eye"

    if _client is None:
        print(f"[SLACK DISABLED] would send: {title}")
        print(f"  thread_id={thread_id}")
        for item in items:
            print(
                f"  [{item.get('kind', 'unknown')}] {item.get('question', '')} "
                f"(confidence={item.get('confidence', 0.0)})"
            )
        if dropped > 0:
            print(f"  {dropped} further items below the review threshold, recorded as gaps")
        return None

    resp = _client.chat_postMessage(
        channel=settings.slack_approval_channel,
        blocks=review_queue_blocks(thread_id, restaurant_name, items, dropped),
        text=title,
    )
    return resp["ts"]


def send_purchase_order(thread_id: str, restaurant_name: str, po: dict) -> str | None:
    """Post the draft opening-order card. Returns the ts (None in degraded mode)."""
    # ASCII, same reason as send_review_queue: this string is print()ed below.
    title = f"Draft opening order - {restaurant_name}"

    if _client is None:
        print(f"[SLACK DISABLED] would send: {title}")
        print(f"  thread_id={thread_id}")
        for vendor, lines in (po.get("vendor_lines") or {}).items():
            print(f"  {vendor}:")
            for line in lines or []:
                print(
                    f"    {line.get('packs', line.get('qty', 0))} x "
                    f"{line.get('pack_unit', '')} {line.get('display_name', '')} "
                    f"- {line.get('line_cost', 0.0)}"
                )
        print(f"  total_cost={po.get('total_cost', 0.0)}")
        excluded = po.get("excluded_skus") or []
        if excluded:
            print(
                f"  withheld={len(excluded)} item(s), "
                f"not_spent={po.get('excluded_cost_total', 0.0)}"
            )
        return None

    resp = _client.chat_postMessage(
        channel=settings.slack_approval_channel,
        blocks=purchase_order_blocks(thread_id, restaurant_name, po),
        text=title,
    )
    return resp["ts"]


# Three-way, because the review card has a third button. A binary
# "approve or else" map would label a skip as a rejection, which is the opposite
# of what happened: skipped items are dropped, not refused. Unknown values fall
# back to the rejection label — the safest thing to claim in public.
_DECISION_LABELS = {
    "approve": "✅ Approved",
    "skip": "⏭️ Flagged items skipped",
    "reject": "❌ Rejected",
}


def confirm_decision(channel: str, message_ts: str, title: str, decision: str) -> None:
    """Replace the approval buttons with a confirmation line."""
    if _client is None:
        print(f"[SLACK DISABLED] would confirm: {decision}")
        return

    label = _DECISION_LABELS.get(decision, _DECISION_LABELS["reject"])
    _client.chat_update(
        channel=channel,
        ts=message_ts,
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": label}},
        ],
        text=f"{title} — {label}",
    )
