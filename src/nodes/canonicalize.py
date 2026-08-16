"""Ingredient canonicalization: collapse free-form ingredient names onto catalog SKUs.

Everything upstream of this node is parsing and everything downstream is
arithmetic; this is the only genuinely hard join in the pipeline. It runs in
three passes, cheapest first, and records which pass won in the `method` field
so the breakdown is inspectable:

  1. exact alias match   — pure dict lookup, zero cost, confidence 1.0
  2. normalised match    — deterministic normaliser, zero cost, confidence 0.9
  3. LLM match           — ONE batched call for the distinct names that survive

Pass 3 only ever sees names the two free passes could not resolve, and it sees
each distinct name once rather than once per occurrence: a 60-item menu that
mentions mozzarella eleven times pays for it at most once. If the LLM pass is
carrying most of the load, the fix is more aliases in the catalog
(`data/catalog/skus.json`), not a cleverer prompt here.

"Distinct" means distinct under `_norm()`, not distinct as a string, so
"roasted peanuts" and "Roasted peanuts" are one lookup and one review question
instead of two. Only the *work* is deduplicated: every original spelling still
gets its own row in `sku_matches`, because `cost_plates` looks matches up by
`raw_name` and a missing key would silently drop that ingredient from every
plate that uses it. Names `_norm()` does not collapse are left alone — the six
spellings of rice paper on the Madame Vo menu are a catalog gap, not a
text-matching problem, and deciding they are the same product would be exactly
the confident mismatch this node refuses everywhere else.

Why pass 2 reports zero hits on every live run so far, and why it stays anyway:
the hand-authored catalog gives every SKU at least five aliases, plural and
reordered spellings included, so pass 1 already holds every key pass 2 would
compute. On the 135-ingredient Madame Vo run all 90 alias hits were reachable
by their normalised key too, and not one of the 45 names that fell through to
the LLM had a normalised key in the index — the residue is products the catalog
does not carry, and no normaliser invents coverage. 351 of the catalog's 1593
exact keys are hand-written variants of another key: this catalog does by hand
what pass 2 does by algorithm. Redundant against *this* catalog is not redundant
in general; the first customer-supplied catalog with one alias per SKU is where
the safety net earns its keep, and until then it costs one dict lookup.

Every `sku_id` the model returns is validated against the real catalog keys and
a hallucinated id is demoted to None with confidence 0.0 — a wrong match
silently poisons the plate cost, and a wrong number that looks right is far more
damaging than an honest gap.

Failure handling: any exception (missing catalog, validation, network, rate
limit) is caught; retry_count++ ; the error message is fed back to the LLM on
the next attempt via `last_error` and an appended HumanMessage.
"""
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import settings
from src.state import CONF_AUTO_ACCEPT, AgentState

CATALOG_PATH = str(Path(__file__).resolve().parents[2] / "data" / "catalog" / "skus.json")

# Preparation adjectives that describe what the kitchen does to an ingredient,
# not what it buys. "Fresh chopped garlic" and "garlic" are the same purchase.
_STOPWORDS = {
    "fresh", "chopped", "diced", "minced", "shredded", "grated",
    "sliced", "whole", "ground", "large", "small", "organic",
}


class SkuMatch(BaseModel):
    raw_name: str
    sku_id: str | None = Field(
        default=None,
        description="Matching sku_id, or null if nothing in the catalog is a genuine match",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One short clause: why this SKU, or why nothing fits")


class SkuMatchBatch(BaseModel):
    matches: list[SkuMatch]


SYSTEM_PROMPT = (
    "You map free-form restaurant ingredient names onto a fixed purchasing catalog. For each "
    "raw ingredient name you are given, return the sku_id of the catalog entry a chef would "
    "actually buy in order to put that ingredient on the plate.\n"
    "\n"
    "`sku_id: null` IS AN EXPLICITLY GOOD ANSWER, NOT A FAILURE. When the catalog holds no "
    "genuine match — a Vietnamese kitchen's fish sauce when this catalog carries no fish sauce "
    "— return null with confidence 0.0 and say in one clause what is missing. NEVER force a "
    "name onto the nearest wrong SKU. A wrong match silently poisons the plate cost and the "
    "food-cost report downstream, and a wrong number that looks right is far more damaging "
    "than an honest, visible gap that a human can fill in thirty seconds. A catalog gap is "
    "useful information; a confident mismatch is a lie.\n"
    "\n"
    "ONLY EVER RETURN A sku_id THAT APPEARS VERBATIM IN THE CATALOG TABLE BELOW. Do not invent "
    "an id, do not adapt one to look like the ingredient, do not guess at a naming pattern. "
    "Any id not in the table is discarded and the ingredient is left unmatched.\n"
    "\n"
    "CONFIDENCE IS YOUR OWN HONEST SELF-ASSESSMENT. Use 1.0 for an obvious synonym or spelling "
    "variant ('nuoc cham' -> the same product under another name). Around 0.7 for a plausible "
    "but imperfect substitute (a specific regional chile matched to a generic dried chile). "
    "Below 0.6 for a guess you would not defend. Exactly 0.0 whenever sku_id is null. "
    "Anything under 0.85 is routed to a human for confirmation, so an honest 0.7 costs nothing "
    "and an inflated 0.95 costs us a wrong price.\n"
    "\n"
    "Return exactly one entry for every raw name you are given, with raw_name copied verbatim, "
    "and no extras.\n"
    "\n"
    "CATALOG (sku_id | display_name | category):\n"
)

INSTRUCTION = "Match each of these raw ingredient names to the catalog:\n"


def _norm(s: str) -> str:
    """Deterministic normaliser: lowercase, depunctuate, naive singularisation,
    drop preparation adjectives, sort the remaining tokens alphabetically.

    Token sorting is what makes "tomatoes roma" and "roma tomatoes" the same
    key. Applied identically to both sides of the lookup, so it only has to be
    self-consistent, not linguistically correct.

    One function used in one place is not an abstraction: it stays here rather
    than becoming a `src/text/` package.
    """
    depunctuated = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    tokens = []
    for tok in depunctuated.split():
        if tok in _STOPWORDS:
            continue
        # Naive singularisation. The `ies`/`oes` cases exist because the catalog
        # writes "Roma tomato" and menus write "roma tomatoes"; a bare trailing-s
        # strip would leave "tomatoe" and miss.
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 4 and tok.endswith("es") and tok[:-2].endswith(("o", "x", "z", "ch", "sh")):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        if tok:
            tokens.append(tok)
    return " ".join(sorted(tokens))


def _build_alias_index(catalog: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Map every display_name and every alias to its sku_id, twice over.

    Returns (exact_index, normalized_index): the first keyed on the lowercased,
    whitespace-collapsed string, the second on `_norm()` of the same string so
    both sides of the pass-2 lookup are normalised identically. Earlier catalog
    entries win a key collision, which keeps the index deterministic.
    """
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for sku in catalog:
        sku_id = sku["sku_id"]
        names = [sku.get("display_name") or ""] + list(sku.get("aliases") or [])
        for name in names:
            if not name:
                continue
            exact.setdefault(" ".join(name.lower().split()), sku_id)
            key = _norm(name)
            if key:
                normalized.setdefault(key, sku_id)
    return exact, normalized


def _build_llm():
    """Factory isolated to make mocking trivial in tests."""
    return ChatAnthropic(
        model=settings.llm_model_cheap,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
        # Thinking ON. Disabling it here to save tokens carries the same lazy-
        # default risk measured on the decomposer: this node emits a per-match
        # `confidence` that feeds the same review gate (CONF_REVIEW_FLOOR) and
        # the plate-confidence product in `src/nodes/cost_plates.py`, so a pile
        # of undifferentiated 0.50s would quietly empty the human review queue.
        # Deciding whether "Grana Padano" is genuinely the right SKU for a menu
        # ingredient is a judgement, not a transcription -- pass 3 only ever
        # sees the names passes 1 and 2 could not resolve, i.e. the hard ones.
        # Hallucinated sku_ids are still caught structurally: every returned id
        # is validated against the real catalog keys below.
        thinking={"type": "adaptive"},
    ).with_structured_output(SkuMatchBatch)


def canonicalize_node(state: AgentState) -> dict:
    """Resolve every recipe component's raw_name to a catalog sku_id."""
    try:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            catalog = json.load(f)

        exact_index, normalized_index = _build_alias_index(catalog)
        valid_ids = {sku["sku_id"] for sku in catalog}

        # Raw names grouped in first-seen order: a 60-item menu repeats the same
        # mozzarella many times and it only needs resolving once, and spellings
        # that `_norm()` already collapses are the same name for this purpose.
        groups: dict[str, list[str]] = {}
        for recipe in state.get("recipes") or []:
            for component in recipe.get("components") or []:
                raw = (component.get("raw_name") or "").strip()
                if not raw:
                    continue
                # A name made of nothing but stopwords normalises to "": key it
                # on itself so unrelated junk does not pile into one group.
                key = _norm(raw) or " ".join(raw.lower().split())
                spellings = groups.setdefault(key, [])
                if raw not in spellings:
                    spellings.append(raw)

        # The first spelling seen does the work for its group; the others ride
        # along on its answer and still get their own row further down.
        variants = {spellings[0]: spellings for spellings in groups.values()}

        matches: list[dict] = []
        new_items: list[dict] = []
        unresolved: list[str] = []

        # --- Passes 1 and 2: no LLM at all ---
        for raw, spellings in variants.items():
            sku_id = exact_index.get(" ".join(raw.lower().split()))
            if sku_id:
                matches.extend({
                    "raw_name": name,
                    "sku_id": sku_id,
                    "method": "alias",
                    "confidence": 1.0,
                } for name in spellings)
                continue
            sku_id = normalized_index.get(_norm(raw))
            if sku_id:
                matches.extend({
                    "raw_name": name,
                    "sku_id": sku_id,
                    "method": "normalized",
                    "confidence": 0.9,
                } for name in spellings)
                continue
            unresolved.append(raw)

        # --- Pass 3: one call for whatever survived ---
        if unresolved:
            table = "\n".join(
                f"{sku['sku_id']} | {sku.get('display_name', '')} | {sku.get('category', '')}"
                for sku in sorted(catalog, key=lambda s: s["sku_id"])
            )
            # One cache_control block: deterministically ordered, so a rerun on
            # another restaurant reads the catalog back from cache.
            msgs = [
                SystemMessage(content=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT + table,
                    "cache_control": {"type": "ephemeral"},
                }])
            ]
            if state.get("last_error"):
                msgs.append(HumanMessage(
                    f"Previous attempt invalid: {state['last_error']}. "
                    f"Correct it and try again."
                ))
            msgs.append(HumanMessage(
                INSTRUCTION + json.dumps(unresolved, ensure_ascii=False, indent=2)
            ))

            llm = _build_llm()
            result = llm.invoke(msgs)

            returned = {}
            for m in result.matches:
                returned.setdefault(m.raw_name, m)
                returned.setdefault(" ".join(m.raw_name.lower().split()), m)

            for raw in unresolved:
                m = returned.get(raw) or returned.get(" ".join(raw.lower().split()))
                if m is None:
                    sku_id, confidence = None, 0.0
                    reasoning = "the model returned no entry for this name"
                elif m.sku_id is not None and m.sku_id not in valid_ids:
                    # Hallucinated id: demoted, never passed downstream as a price key.
                    sku_id, confidence = None, 0.0
                    reasoning = f"hallucinated sku_id '{m.sku_id}' is not in the catalog"
                else:
                    sku_id, confidence, reasoning = m.sku_id, m.confidence, m.reasoning
                matches.extend({
                    "raw_name": name,
                    "sku_id": sku_id,
                    "method": "llm",
                    "confidence": confidence,
                } for name in variants[raw])
                # One question per group: a human is not asked to adjudicate
                # "roasted peanuts" and "Roasted peanuts" separately.
                if sku_id is None or confidence < CONF_AUTO_ACCEPT:
                    new_items.append({
                        "kind": "sku_match",
                        "ref": raw,
                        "confidence": confidence,
                        "question": f"Which SKU is '{raw}'?",
                        "payload": {"suggested": sku_id, "reasoning": reasoning},
                    })

        n_alias = sum(1 for m in matches if m["method"] == "alias")
        n_norm = sum(1 for m in matches if m["method"] == "normalized")
        n_llm = sum(1 for m in matches if m["method"] == "llm" and m["sku_id"])
        n_unmatched = sum(1 for m in matches if m["sku_id"] is None)
        print(
            f"canonicalize: {n_alias} alias, {n_norm} normalized, "
            f"{n_llm} llm, {n_unmatched} unmatched "
            f"({len(matches)} raw names resolved as {len(variants)} distinct)"
        )

        return {
            "sku_matches": matches,
            "review_queue": state.get("review_queue", []) + new_items,
            "stage": "canonicalized",
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "sku_matches": [],
            "messages": [HumanMessage(f"Canonicalization failed: {e}")],
        }
