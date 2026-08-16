"""Guards on the Day Zero CLI runner. Its stdout is the deliverable, so it is tested.

Nothing here touches the network: the compiled graph is replaced by a fake that
replays a plausible sequence of states. What is under test is the narration, the
stop point of --dry-run and the unknown-slug error, not LangGraph itself.
"""
from types import SimpleNamespace

import pytest

import scripts.run_dayzero as runner

# A plausible run, as graph.stream(stream_mode="values") would yield it: the
# initial state first, then one state per completed node.
_STATES = [
    {"stage": "", "menu_items": []},
    {
        "stage": "ingested",
        "restaurant": {"name": "Joe's Pizza", "cuisine": "pizza", "seats": 20,
                       "menu_format": "html"},
    },
    {
        "stage": "menu_extracted",
        "menu_items": [{"name": "Cheese Slice", "price": 3.75}, {"name": "Water", "price": None}],
    },
    {
        "stage": "recipes_decomposed",
        "recipes": [{"item_name": "Cheese Slice", "components": [{"raw_name": "flour"},
                                                                 {"raw_name": "mozzarella"}]}],
    },
    {
        "stage": "canonicalized",
        "sku_matches": [
            {"raw_name": "flour", "sku_id": "FLR-01", "method": "alias"},
            {"raw_name": "mozzarella", "sku_id": "MOZ-01", "method": "normalized"},
            {"raw_name": "san marzano", "sku_id": "TOM-01", "method": "llm"},
            {"raw_name": "unicorn dust", "sku_id": None, "method": "llm"},
        ],
    },
    {
        "stage": "costed",
        "plate_costs": [{"item_name": "Cheese Slice", "coverage": 0.9},
                        {"item_name": "Sicilian", "coverage": 0.7}],
        "review_queue": [{"kind": "plate_cost", "ref": "Sicilian"}],
    },
    {
        "stage": "forecast",
        "demand_forecast": {"covers_per_week": 1840.0, "item_mix": {"Cheese Slice": 1.0},
                            "confidence": 0.62},
    },
    {
        "stage": "po_drafted",
        "purchase_order": {
            "vendor_lines": {"Baldor": [{"line_cost": 100.0}], "Sysco": [{"line_cost": 82.5}]},
            "total_cost": 4182.5,
        },
    },
]


class _FakeGraph:
    """Replays `states`, records how many were actually consumed."""

    def __init__(self, states, next_nodes=()):
        self._states = states
        self._next = next_nodes
        self.consumed = 0
        self.stream_config = None

    def stream(self, state, config=None, stream_mode=None):
        self.stream_config = config
        for values in self._states:
            self.consumed += 1
            yield values

    def get_state(self, config):
        return SimpleNamespace(values=self._states[-1], next=self._next)


@pytest.fixture
def fake_graph(monkeypatch):
    graph = _FakeGraph(_STATES)
    monkeypatch.setattr(runner, "get_compiled_graph", lambda: graph)
    return graph


# --- the unknown slug ------------------------------------------------------


def test_unknown_slug_exits_non_zero_and_lists_valid_slugs(capsys):
    code = runner.main(["definitely-not-a-restaurant"])
    captured = capsys.readouterr()

    assert code != 0
    assert "definitely-not-a-restaurant" in captured.err
    assert "valid slugs" in captured.err
    # The real corpus index is the source of truth for the suggestion list.
    assert "joes-pizza-carmine" in captured.err


def test_unknown_slug_never_compiles_the_graph(monkeypatch, capsys):
    """Validation happens before any SQLite / LLM machinery is touched."""

    def boom():
        raise AssertionError("get_compiled_graph must not run for an unknown slug")

    monkeypatch.setattr(runner, "get_compiled_graph", boom)
    assert runner.main(["definitely-not-a-restaurant"]) != 0
    capsys.readouterr()


# --- --dry-run stops at costed ---------------------------------------------


def test_dry_run_stops_at_costed(fake_graph, capsys):
    code = runner.main(["joes-pizza-carmine", "--dry-run", "--no-slack"])
    out = capsys.readouterr().out

    assert code == 0
    assert "costed" in out
    assert "2 plates" in out
    # Everything downstream of cost_plates must never have been pulled.
    assert "forecast" not in out
    assert "po_drafted" not in out
    assert fake_graph.consumed == 6, "the stream was drained past cost_plates"
    assert "dry run complete" in out


def test_dry_run_forces_the_degraded_slack_path(fake_graph, monkeypatch, capsys):
    import src.slack.client as slack_client

    monkeypatch.setattr(slack_client, "_client", object())
    runner.main(["joes-pizza-carmine", "--dry-run"])
    capsys.readouterr()

    assert slack_client._client is None


def test_no_slack_forces_the_degraded_slack_path(fake_graph, monkeypatch, capsys):
    import src.slack.client as slack_client

    monkeypatch.setattr(slack_client, "_client", object())
    runner.main(["joes-pizza-carmine", "--no-slack"])
    capsys.readouterr()

    assert slack_client._client is None


# --- the costed line agrees with cost_plates -------------------------------


def test_costed_summary_excludes_uncostable_plates_from_the_mean():
    # cost_plates marks a plate with no costable component at all `costable:
    # False` and keeps it out of its own mean coverage. This line has to agree,
    # or the narration contradicts the node it is narrating.
    state = {
        "stage": "costed",
        "plate_costs": [
            {"item_name": "Cheese Slice", "coverage": 0.9, "costable": True},
            {"item_name": "Sicilian", "coverage": 0.7, "costable": True},
            {"item_name": "Soda", "coverage": 0.0, "costable": False},
        ],
        "review_queue": [],
    }

    summary = runner._stage_detail(state)

    assert "3 plates" in summary
    assert "1 not costable" in summary
    # (0.9 + 0.7) / 2, not / 3.
    assert "mean coverage 0.80" in summary


def test_costed_summary_treats_a_plate_without_the_flag_as_costable():
    # Checkpoints written before the field exists must not read as uncostable.
    state = {
        "stage": "costed",
        "plate_costs": [{"item_name": "Cheese Slice", "coverage": 0.9}],
        "review_queue": [],
    }

    summary = runner._stage_detail(state)

    assert "1 plates" in summary
    assert "not costable" not in summary
    assert "mean coverage 0.90" in summary


# --- the full run and its narration ----------------------------------------


def test_full_run_narrates_every_stage(fake_graph, capsys):
    code = runner.main(["joes-pizza-carmine", "--no-slack"])
    out = capsys.readouterr().out

    assert code == 0
    assert fake_graph.consumed == len(_STATES)
    for stage in ("ingested", "menu_extracted", "recipes_decomposed", "canonicalized",
                  "costed", "forecast", "po_drafted"):
        assert stage in out, f"{stage} never narrated"
    assert "1 alias / 1 normalized / 1 llm / 1 unmatched" in out
    assert "1,840 covers/wk" in out
    assert "$4,182.50 across 2 vendors" in out
    assert "tokens:" in out


def test_thread_id_defaults_to_dayzero_slug(fake_graph, capsys):
    runner.main(["joes-pizza-carmine", "--no-slack"])
    out = capsys.readouterr().out

    assert "dayzero-joes-pizza-carmine" in out
    assert fake_graph.stream_config["configurable"]["thread_id"] == "dayzero-joes-pizza-carmine"


def test_thread_id_override(fake_graph, capsys):
    runner.main(["joes-pizza-carmine", "--no-slack", "--thread-id", "custom-thread"])
    capsys.readouterr()

    assert fake_graph.stream_config["configurable"]["thread_id"] == "custom-thread"


def test_pause_banner_when_the_graph_is_waiting(monkeypatch, capsys):
    graph = _FakeGraph(_STATES[:6], next_nodes=("review_wait",))
    monkeypatch.setattr(runner, "get_compiled_graph", lambda: graph)

    code = runner.main(["joes-pizza-carmine", "--no-slack"])
    out = capsys.readouterr().out

    assert code == 0, "a pause is not an error"
    assert "paused" in out
    assert "review_wait" in out


def test_a_crash_mid_stream_prints_one_line_not_a_traceback(monkeypatch, capsys):
    """A dead Slack token must not put a 30-line traceback on the operator's screen."""

    class _ExplodingGraph(_FakeGraph):
        def stream(self, state, config=None, stream_mode=None):
            yield from _STATES[:6]
            raise RuntimeError("account_inactive")

    monkeypatch.setattr(runner, "get_compiled_graph", lambda: _ExplodingGraph(_STATES[:6]))

    code = runner.main(["joes-pizza-carmine", "--no-slack"])
    out = capsys.readouterr().out

    assert code == 1
    assert "run failed" in out
    assert "RuntimeError: account_inactive" in out
    assert "Traceback" not in out
    assert "tokens:" in out, "the cost of what was already spent is still reported"


def test_json_out_writes_the_final_state(fake_graph, tmp_path, capsys):
    target = tmp_path / "state.json"
    runner.main(["joes-pizza-carmine", "--no-slack", "--json-out", str(target)])
    capsys.readouterr()

    import json

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["stage"] == "po_drafted"
    assert payload["purchase_order"]["total_cost"] == 4182.5


# --- --resume-status --------------------------------------------------------


def test_resume_status_reports_a_waiting_run(monkeypatch, capsys):
    graph = _FakeGraph(_STATES[:6], next_nodes=("review_wait",))
    monkeypatch.setattr(runner, "get_compiled_graph", lambda: graph)

    code = runner.main(["joes-pizza-carmine", "--resume-status"])
    out = capsys.readouterr().out

    assert code == 0
    assert "costed" in out
    assert "review_wait" in out
    assert "paused" in out
    assert graph.consumed == 0, "--resume-status must not run the graph"


def test_resume_status_on_an_unknown_thread_exits_non_zero(monkeypatch, capsys):
    graph = _FakeGraph([{}])
    monkeypatch.setattr(runner, "get_compiled_graph", lambda: graph)

    assert runner.main(["joes-pizza-carmine", "--resume-status"]) != 0
    assert "no checkpoint" in capsys.readouterr().out


# --- the cost line ----------------------------------------------------------


def test_usage_collector_sums_anthropic_usage_metadata(capsys):
    usage = runner._UsageCollector()
    message = SimpleNamespace(usage_metadata={
        "input_tokens": 184_000,
        "output_tokens": 11_000,
        "input_token_details": {"cache_read": 142_000, "cache_creation": 900},
    })
    response = SimpleNamespace(generations=[[SimpleNamespace(message=message)]])

    usage.on_llm_end(response)
    usage.on_llm_end(response)

    assert usage.calls == 2
    assert usage.input_tokens == 368_000
    assert usage.cache_read == 284_000

    runner._print_usage(usage)
    assert "tokens: 368k in (284k cached) / 22k out" in capsys.readouterr().out


def test_usage_collector_survives_a_response_without_usage():
    usage = runner._UsageCollector()
    usage.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=None)]]))
    usage.on_llm_end(SimpleNamespace(generations=None))

    assert usage.calls == 0
    assert usage.input_tokens == 0
