"""Block Kit JSON builders. Pure data, no I/O.

Each function returns a list[dict] ready to pass to chat_postMessage(blocks=...).

This module imports nothing from src/nodes, src/tools or src/config: Slack code
does not know about LLMs, nodes or settings. It only knows plain dicts.

Block Kit limits respected here:
  - 50 blocks per message
  - 25 elements per actions block
  - 3000 characters per text object (see _MAX_SECTION_CHARS)
"""

_MAX_SECTION_CHARS = 3000

# Block budgets. A message is capped at 50 blocks by Slack; rendering more
# items than fits does not degrade the card, it makes chat_postMessage fail.
_MAX_QUEUE_ITEMS = 20      # each item costs 2 blocks (section + context)
_MAX_VENDOR_GROUPS = 20    # each vendor group costs 1 block
_MAX_VENDOR_LINES = 10     # lines rendered inside one vendor section
_MAX_EXCLUDED_ITEMS = 6    # withheld items named inside the single withheld section


def _truncate(text: str, limit: int = _MAX_SECTION_CHARS) -> str:
    """Clamp a text object to Slack's per-object character limit.

    Slack rejects the entire chat_postMessage call if any single text object
    exceeds 3000 characters, so one long vendor group would kill the whole card.
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def approval_blocks(thread_id: str, title: str, context: dict) -> list[dict]:
    """Build an approval message with 2 buttons.

    The thread_id is encoded in each button's `value` so the webhook handler
    knows which LangGraph run to resume.
    """
    fields = [
        {"type": "mrkdwn", "text": f"*{k}:*\n{v}"}
        for k, v in list(context.items())[:8]
    ]

    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
        {"type": "section", "fields": fields},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": "approve",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "style": "primary",
                    "value": f"{thread_id}|approve",
                },
                {
                    "type": "button",
                    "action_id": "reject",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "style": "danger",
                    "value": f"{thread_id}|reject",
                },
            ],
        },
    ]


def review_queue_blocks(
    thread_id: str,
    restaurant_name: str,
    items: list[dict],
    dropped: int,
) -> list[dict]:
    """Build the low-confidence review card.

    `items` are review-queue entries:
    {"kind", "ref", "confidence", "question", "detail"}, where `question` states
    what the pipeline concluded and the optional `detail` says why.
    `dropped` is the number of entries the gate left out of the queue.

    Each entry states the conclusion the pipeline reached, not an open question:
    "*White onion* → Onion, yellow jumbo" with the reasoning underneath, so the
    two buttons have something to act on. An open question ("Which SKU is
    'White onion'?") above Approve/Reject asks the reviewer to answer with a
    button that cannot carry an answer.

    Batch approval (one decision over a visible list) rather than per-item
    buttons: Block Kit caps a message at 50 blocks and 25 elements per actions
    block, and a reviewer facing 12 approve/reject pairs will not use it. It
    also keeps the resume payload a single `thread_id|decision` value, which is
    what src/server/app.py's value.split("|", 1) parsing expects.
    """
    # A queue of one is the common case once the floor has done its work, and
    # "1 items" is the first thing a reviewer reads on the card.
    count = "1 item" if len(items) == 1 else f"{len(items)} items"

    blocks: list[dict] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _truncate(f"*{restaurant_name}* — {count} to confirm"),
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Here is what we think each one is. Confirm if that "
                            "is right, reject if it is not.",
                }
            ],
        },
    ]

    for item in items[:_MAX_QUEUE_ITEMS]:
        question = str(item.get("question", ""))
        confidence = item.get("confidence", 0.0)
        kind = str(item.get("kind", "unknown"))
        try:
            pct = f"{float(confidence):.0%}"
        except (TypeError, ValueError):
            pct = "—"
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": _truncate(question)},
            }
        )
        # `detail` is the node's one-clause reason for the proposal above it.
        # Without it "60% confidence" is a number with nothing behind it, and a
        # reviewer cannot confirm what they cannot see the basis for.
        subline = f"{pct} confidence · `{kind}`"
        detail = str(item.get("detail", "") or "").strip()
        if detail:
            subline = f"{subline} · {detail}"
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": _truncate(subline)}],
            }
        )

    overflow = len(items) - _MAX_QUEUE_ITEMS
    if overflow > 0:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": _truncate(
                            f"_…and {overflow} more flagged items not shown in this card._"
                        ),
                    }
                ],
            }
        )

    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": "approve",
                    "text": {"type": "plain_text", "text": "Confirm all"},
                    "style": "primary",
                    "value": f"{thread_id}|approve",
                },
                {
                    "type": "button",
                    "action_id": "reject",
                    "text": {"type": "plain_text", "text": "Reject all"},
                    "style": "danger",
                    "value": f"{thread_id}|reject",
                },
                {
                    "type": "button",
                    "action_id": "skip",
                    "text": {"type": "plain_text", "text": "Skip flagged"},
                    "value": f"{thread_id}|skip",
                },
            ],
        }
    )

    if dropped > 0:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": _truncate(
                            f"_{dropped} further items fell below the review threshold "
                            f"and were recorded as gaps._"
                        ),
                    }
                ],
            }
        )

    return blocks


def purchase_order_blocks(thread_id: str, restaurant_name: str, po: dict) -> list[dict]:
    """Build the draft opening-order card.

    `po` is the draft_po payload: {"vendor_lines": {vendor: [line, ...]},
    "total_cost", "covers_per_week", "assumptions", ...} where each line is
    {"display_name", "packs", "pack_unit", "line_cost", ...}.

    The assumptions are printed next to the total on purpose: a figure shown
    without its basis invites "where did that come from?" at the wrong moment.
    """
    blocks: list[dict] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _truncate(f"*Draft opening order — {restaurant_name}*"),
            },
        }
    ]

    vendor_lines = po.get("vendor_lines") or {}
    for vendor, lines in list(vendor_lines.items())[:_MAX_VENDOR_GROUPS]:
        lines = lines or []
        rendered = [f"*{vendor}*"]
        for line in lines[:_MAX_VENDOR_LINES]:
            qty = line.get("packs", line.get("qty", 0))
            pack_unit = line.get("pack_unit", "")
            display_name = line.get("display_name", line.get("sku_id", ""))
            try:
                cost = f"${float(line.get('line_cost', 0.0)):,.2f}"
            except (TypeError, ValueError):
                cost = "$0.00"
            rendered.append(f"• {qty} × {pack_unit} {display_name} — {cost}")
        remaining = len(lines) - _MAX_VENDOR_LINES
        if remaining > 0:
            rendered.append(f"_…and {remaining} more lines_")
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": _truncate("\n".join(rendered))},
            }
        )

    extra_vendors = len(vendor_lines) - _MAX_VENDOR_GROUPS
    if extra_vendors > 0:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": _truncate(f"_…and {extra_vendors} more vendors_"),
                    }
                ],
            }
        )

    blocks.append({"type": "divider"})

    try:
        total = f"${float(po.get('total_cost', 0.0)):,.2f}"
    except (TypeError, ValueError):
        total = "$0.00"
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": _truncate(f"*Order total: {total}*")},
        }
    )

    # What was deliberately not bought, immediately under the total. This is the
    # claim the product is making — money not spent on a guess — so it sits where
    # the eye already is rather than in the assumptions footnote. Entries are the
    # dicts draft_po writes; plain strings from an older payload still render.
    excluded = po.get("excluded_skus") or []
    if excluded:
        try:
            withheld_total = f"${float(po.get('excluded_cost_total', 0.0)):,.2f}"
        except (TypeError, ValueError):
            withheld_total = "$0.00"

        why: list[str] = []
        for item in excluded:
            reason = str(item.get("reason", "")) if isinstance(item, dict) else ""
            if "confidence_below_floor" in reason and "confidence below review floor" not in why:
                why.append("confidence below review floor")
            if "skipped_in_review" in reason and "skipped in review" not in why:
                why.append("skipped in review")
        if not why:
            why = ["confidence below review floor"]

        rendered = [
            f"*Withheld: {len(excluded)} item(s) ({withheld_total}) — {' / '.join(why)}*"
        ]
        for item in excluded[:_MAX_EXCLUDED_ITEMS]:
            if not isinstance(item, dict):
                rendered.append(f"• {item}")
                continue
            name = item.get("display_name") or item.get("sku_id", "")
            try:
                cost = f"${float(item.get('cost_not_spent', 0.0)):,.2f}"
            except (TypeError, ValueError):
                cost = "$0.00"
            confidence = item.get("confidence")
            try:
                pct = f"{float(confidence):.0%} confidence" if confidence is not None else "—"
            except (TypeError, ValueError):
                pct = "—"
            dishes = item.get("dishes") or []
            where = ", ".join(str(d) for d in dishes[:3])
            if len(dishes) > 3:
                where += f", +{len(dishes) - 3} more"
            rendered.append(f"• {name} — {cost} not spent — {pct} in {where or 'n/a'}")

        more = len(excluded) - _MAX_EXCLUDED_ITEMS
        if more > 0:
            rendered.append(f"_…and {more} more withheld items_")

        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": _truncate("\n".join(rendered))},
            }
        )

    # Days of cover is a per-category policy, so draft_po sends the range it
    # actually applied ("2-30 days cover by category") rather than a number.
    # An older payload carries no label, in which case the card says nothing
    # instead of inventing one — a wrong basis printed under the total is the
    # same class of problem as a wrong total.
    basis = [f"Based on {po.get('covers_per_week', 0)}/wk projected covers"]
    days_cover_label = str(po.get("days_cover_label") or "").strip()
    if days_cover_label:
        basis.append(days_cover_label)
    basis.append("prices from hand-curated catalog")
    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": _truncate(" · ".join(basis))}
            ],
        }
    )

    assumptions = po.get("assumptions") or []
    if assumptions:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": _truncate(
                            " · ".join(str(a) for a in assumptions)
                        ),
                    }
                ],
            }
        )

    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": "approve",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "style": "primary",
                    "value": f"{thread_id}|approve",
                },
                {
                    "type": "button",
                    "action_id": "reject",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "style": "danger",
                    "value": f"{thread_id}|reject",
                },
            ],
        }
    )

    return blocks
