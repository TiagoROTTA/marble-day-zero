"""FastAPI server. Exposes the Slack interactivity webhook and a health check.

The webhook flow:
1. Slack POSTs `application/x-www-form-urlencoded` with a `payload` field
   containing URL-encoded JSON.
2. Respond within 3 seconds → resume the LangGraph run in a BackgroundTask.
3. The resume call is graph.invoke(Command(resume={"decision": ...}), ...).
"""
import json
import logging
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from langgraph.types import Command

from src.graph import get_compiled_graph
from src.slack.client import confirm_decision
from src.slack.verify import verify

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="day-zero-agent")
_graph = None


@app.on_event("startup")
def _startup() -> None:
    global _graph
    _graph = get_compiled_graph()
    logger.info("graph compiled and ready")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _resume_graph(thread_id: str, decision: str) -> None:
    """Resume a paused LangGraph run. Called from a BackgroundTask."""
    try:
        _graph.invoke(
            Command(resume={"decision": decision}),
            config={"configurable": {"thread_id": thread_id}},
        )
        logger.info("resumed thread_id=%s decision=%s", thread_id, decision)
    except Exception as e:
        logger.exception("resume failed for thread_id=%s: %s", thread_id, e)


@app.post("/slack/interactions")
async def slack_interactions(request: Request, bg: BackgroundTasks) -> dict:
    body = await request.body()

    if not verify(
        body,
        request.headers.get("x-slack-signature", ""),
        request.headers.get("x-slack-request-timestamp", ""),
    ):
        raise HTTPException(status_code=401, detail="bad signature")

    form = parse_qs(body.decode())
    if "payload" not in form:
        raise HTTPException(status_code=400, detail="missing payload")

    payload = json.loads(form["payload"][0])

    actions = payload.get("actions") or []
    if not actions:
        return {"text": "no action"}

    action = actions[0]
    value = action.get("value", "")

    if "|" not in value:
        raise HTTPException(status_code=400, detail="malformed value")

    thread_id, decision = value.split("|", 1)

    channel = (payload.get("channel") or {}).get("id", "")
    message_ts = (payload.get("message") or {}).get("ts", "")
    # Defensive on purpose: this runs inside the request handler, and the first
    # block is only a section-with-text for the cards we happen to send today. A
    # header block, a divider, or a card with no blocks at all must degrade to a
    # generic title, not raise a KeyError/AttributeError at the reviewer.
    blocks = (payload.get("message") or {}).get("blocks") or []
    first_text = blocks[0].get("text") if blocks and isinstance(blocks[0], dict) else None
    title = first_text.get("text", "Agent review") if isinstance(first_text, dict) else "Agent review"
    if channel and message_ts:
        confirm_decision(channel, message_ts, title, decision)

    bg.add_task(_resume_graph, thread_id, decision)

    return {"text": f"Decision `{decision}` received, processing..."}
