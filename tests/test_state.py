from src.state import initial_state


def test_initial_state_shape():
    s = initial_state("hello")
    assert s["input"] == "hello"
    assert s["retry_count"] == 0
    assert s["extracted_data"] == {}
    assert s["needs_human"] is False
    assert s["human_decision"] == ""
