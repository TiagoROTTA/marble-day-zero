"""CLI entrypoint for the Day Zero pipeline.

This script's stdout is the only view anyone gets of a multi-minute run, so the
output format is treated as a deliverable rather than as logging.

Usage:
    uv run python -m scripts.run_dayzero joes-pizza-carmine
    uv run python -m scripts.run_dayzero joes-pizza-carmine --dry-run --no-slack
    uv run python -m scripts.run_dayzero joes-pizza-carmine --resume-status

The run narrates itself: one line per completed stage, carrying the headline
number that stage produced. A pipeline that says what it is doing while it does
it can be followed and debugged; one that prints nothing for ninety seconds and
then dumps JSON cannot.

When the graph reaches a human gate (review_wait / po_wait) the run PAUSES and
this process exits 0. Resumption happens in the other process — the FastAPI
webhook in src/server/app.py, driven by the Slack button click. There is
deliberately no polling loop here: duplicating the resume path would collapse
the two-process design into one and leave the checkpointer with nothing to do.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt

from src.graph import get_compiled_graph
from src.state import initial_dayzero_state
from src.tools.snapshot import list_corpus

# The stage names the pipeline writes into state["stage"], in pipeline order.
# Used only for the width of the narration column and for --dry-run's stop point.
DRY_RUN_STOP_STAGE = "costed"
_STAGE_WIDTH = 19


class _UsageCollector(BaseCallbackHandler):
    """Sum token usage across every LLM response in the run.

    A callback rather than a read of state: the nodes deliberately return domain
    fields only, so usage never lands in AgentState. LangChain propagates the
    callbacks attached to the run config down into each node's llm.invoke(), so
    this sees every call without any node knowing it exists.

    Anything missing is counted as zero — a cost line is a nice-to-have and must
    never be the reason a demo run dies.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_write = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in getattr(response, "generations", []) or []:
            for generation in generations or []:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                if not usage:
                    continue
                self.calls += 1
                self.input_tokens += int(usage.get("input_tokens") or 0)
                self.output_tokens += int(usage.get("output_tokens") or 0)
                details = usage.get("input_token_details") or {}
                self.cache_read += int(details.get("cache_read") or 0)
                self.cache_write += int(details.get("cache_creation") or 0)


def _k(n: int) -> str:
    """184213 -> '184k'. Small numbers stay exact so an empty run reads as 0."""
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def _stage_detail(state: dict[str, Any]) -> str:
    """The headline number the just-completed stage produced.

    Written for a reader unfamiliar with the code: each line should be readable
    on its own, without knowing the field names underneath.
    """
    stage = state.get("stage", "")

    if stage == "ingested":
        r = state.get("restaurant") or {}
        return (
            f"{r.get('name', '?')} — {r.get('cuisine', '?')}, "
            f"{r.get('seats', '?')} seats, menu as {r.get('menu_format', '?')}"
        )

    if stage == "menu_extracted":
        items = state.get("menu_items") or []
        priced = sum(1 for i in items if i.get("price"))
        return f"{len(items)} items ({priced} with a price)"

    if stage == "recipes_decomposed":
        recipes = state.get("recipes") or []
        components = sum(len(r.get("components") or []) for r in recipes)
        return f"{len(recipes)} recipes / {components} ingredient lines"

    if stage == "canonicalized":
        matches = state.get("sku_matches") or []
        alias = sum(1 for m in matches if m.get("method") == "alias" and m.get("sku_id"))
        norm = sum(1 for m in matches if m.get("method") == "normalized" and m.get("sku_id"))
        llm = sum(1 for m in matches if m.get("method") == "llm" and m.get("sku_id"))
        unmatched = sum(1 for m in matches if not m.get("sku_id"))
        return f"{alias} alias / {norm} normalized / {llm} llm / {unmatched} unmatched"

    if stage == "costed":
        plates = state.get("plate_costs") or []
        # A plate with no costable component at all carries coverage 0.0 and no
        # cost; cost_plates keeps it out of its own mean for that reason, and
        # this line has to agree or the narration contradicts the node. Absent
        # flag means costable: an older checkpoint predates the field.
        costed = [p for p in plates if p.get("costable", True)]
        coverage = (
            sum(float(p.get("coverage") or 0.0) for p in costed) / len(costed)
            if costed
            else 0.0
        )
        flagged = len(state.get("review_queue") or [])
        summary = f"{len(plates)} plates"
        if len(costed) != len(plates):
            # ASCII: this is printed to a cp1252 console.
            summary += f" ({len(plates) - len(costed)} not costable)"
        return (
            f"{summary} · mean coverage {coverage:.2f} "
            f"· {flagged} flagged for a human"
        )

    if stage == "reviewed":
        decision = state.get("human_decision") or "auto"
        return f"review queue cleared (human: {decision})"

    if stage == "forecast":
        f = state.get("demand_forecast") or {}
        covers = f.get("covers_per_week")
        if covers is None:
            covers = sum(f.get("covers_per_day") or [])
        return (
            f"{float(covers):,.0f} covers/wk · {len(f.get('item_mix') or {})} items in the mix "
            f"· confidence {float(f.get('confidence') or 0.0):.2f}"
        )

    if stage == "po_drafted":
        po = state.get("purchase_order") or {}
        vendors = po.get("vendor_lines") or {}
        lines = sum(len(v or []) for v in vendors.values())
        summary = (
            f"${float(po.get('total_cost') or 0.0):,.2f} across {len(vendors)} vendors "
            f"· {lines} lines"
        )
        withheld = po.get("excluded_skus") or []
        if withheld:
            summary += (
                f" · {len(withheld)} withheld "
                f"(${float(po.get('excluded_cost_total') or 0.0):,.2f} not spent)"
            )
        return summary

    if stage == "po_approved":
        return f"human said {state.get('human_decision') or '?'}"

    return ""


def _print_stage(state: dict[str, Any], elapsed: float) -> None:
    """One narration line per completed stage, timestamped from the run start."""
    stage = state.get("stage", "")
    detail = _stage_detail(state)
    print(f"[{elapsed:6.1f}s] {stage:<{_STAGE_WIDTH}}· {detail}", flush=True)


def _print_usage(usage: _UsageCollector) -> None:
    """The cost line. Visible after every run, so caching is a fact not a hope."""
    print(
        f"tokens: {_k(usage.input_tokens)} in ({_k(usage.cache_read)} cached) "
        f"/ {_k(usage.output_tokens)} out — {usage.calls} LLM calls",
        flush=True,
    )


def _resume_status(graph: Any, thread_id: str, config: dict) -> int:
    """Print what the checkpoint holds for a thread. Proof the pause survived."""
    snapshot = graph.get_state(config)
    values = snapshot.values or {}

    print(f"═══ resume status · {thread_id} ═══")
    if not values:
        print("no checkpoint found — this thread_id has never run")
        return 1

    waiting_on = list(snapshot.next or ())
    print(f"  stage         : {values.get('stage') or '(nothing completed)'}")
    print(f"  menu items    : {len(values.get('menu_items') or [])}")
    print(f"  plates costed : {len(values.get('plate_costs') or [])}")
    print(f"  review queue  : {len(values.get('review_queue') or [])} pending")
    print(f"  human decision: {values.get('human_decision') or '(none yet)'}")
    if waiting_on:
        print(f"  waiting on    : {', '.join(waiting_on)}")
        print("  status        : ⏸  paused — awaiting approval in Slack")
    else:
        print("  status        : ✅ not waiting — run is finished or never paused")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_dayzero",
        description="Run the Day Zero pipeline for one restaurant snapshot.",
    )
    parser.add_argument("slug", help="restaurant slug under data/restaurants/")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="checkpoint thread id (default: dayzero-<slug>)",
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="force the degraded print-only path instead of posting to Slack",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="dump the final state as JSON to PATH",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"stop after cost_plates (stage={DRY_RUN_STOP_STAGE}); no forecast, no PO, no Slack",
    )
    parser.add_argument(
        "--resume-status",
        action="store_true",
        help="print the checkpoint state for the thread and exit, running nothing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # PowerShell defaults to cp1252 and would crash on the banner glyphs mid-demo.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus = list_corpus()
    if args.slug not in corpus:
        print(f"unknown slug: {args.slug!r}", file=sys.stderr)
        print(f"valid slugs ({len(corpus)}):", file=sys.stderr)
        for slug in corpus:
            print(f"  {slug}", file=sys.stderr)
        return 2

    thread_id = args.thread_id or f"dayzero-{args.slug}"
    usage = _UsageCollector()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [usage]}

    graph = get_compiled_graph()

    if args.resume_status:
        return _resume_status(graph, thread_id, config)

    if args.no_slack or args.dry_run:
        # The WebClient is bound once at import time from settings, so forcing
        # the degraded path means rebinding it before any node sends a card.
        # --dry-run implies it: a dry run must never reach Slack.
        import src.slack.client as slack_client

        slack_client._client = None

    print(f"═══ Day Zero · {args.slug} ═══")
    print(f"thread_id : {thread_id}")
    print(f"mode      : {'dry run (stops at cost_plates)' if args.dry_run else 'full run'}")
    print()

    started = time.monotonic()
    state = initial_dayzero_state(args.slug)
    last_stage = ""
    final: dict[str, Any] = dict(state)
    stopped_early = False
    crashed = ""

    try:
        for values in graph.stream(state, config=config, stream_mode="values"):
            final = values
            stage = values.get("stage", "")
            if stage and stage != last_stage:
                last_stage = stage
                _print_stage(values, time.monotonic() - started)
            if args.dry_run and stage == DRY_RUN_STOP_STAGE:
                stopped_early = True
                break
    except GraphInterrupt:
        # Belt and braces: this LangGraph version ends the stream cleanly on
        # interrupt, but a version that raises must still land on the banner
        # below rather than on a traceback in front of the camera.
        pass
    except Exception as e:
        # Every node already swallows its own failures into retry_count, so an
        # exception reaching here is infrastructure — a dead Slack token, a
        # missing file, a bad model id. Say so in one line and still print the
        # cost of what was spent; a 30-line traceback mid-demo says nothing.
        crashed = f"{type(e).__name__}: {e}"

    print()

    snapshot = graph.get_state(config)
    waiting_on = list(snapshot.next or ())

    if crashed:
        print(f"✖  run failed outside the retry loop (thread_id={thread_id})")
        print(f"   {crashed}")
        if waiting_on:
            print(f"   the checkpoint stopped at: {', '.join(waiting_on)}")
    elif stopped_early:
        print(f"■  dry run complete — stopped after cost_plates (thread_id={thread_id})")
    elif waiting_on:
        print(f"⏸  paused — awaiting approval in Slack (thread_id={thread_id})")
        print(f"   waiting on: {', '.join(waiting_on)}")
        print("   resume happens in the webhook process: uv run uvicorn src.server.app:app")
    elif final.get("human_decision") == "reject":
        print(f"✋ stopped — a human rejected the run (thread_id={thread_id})")
    else:
        print(f"✅ run complete (thread_id={thread_id})")

    if final.get("last_error"):
        print(f"   last_error: {final['last_error']}")

    _print_usage(usage)

    if args.json_out:
        payload = {k: v for k, v in final.items() if k not in ("messages", "__interrupt__")}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        print(f"state written to {args.json_out}")

    return 1 if crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
