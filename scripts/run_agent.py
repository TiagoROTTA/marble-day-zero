"""CLI entrypoint for the agent.

Usage:
    uv run python -m scripts.run_agent "Marie bought 3 apples in Paris for 5 EUR"

If the run pauses (interrupt fired), the script exits — the resume happens
later via the Slack button click → FastAPI webhook → graph resume.
"""
import sys
import uuid
from pprint import pprint

from src.graph import get_compiled_graph
from src.state import initial_state


def main() -> None:
    user_input = sys.argv[1] if len(sys.argv) > 1 else "Default test input"

    graph = get_compiled_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== Run starting ===")
    print(f"thread_id: {thread_id}")
    print(f"input: {user_input}\n")

    state = initial_state(user_input)
    result = graph.invoke(state, config=config)

    graph_state = graph.get_state(config)
    paused = bool(graph_state.next)

    print("=== Run finished ===")
    if paused:
        print(">>> Run paused for HITL.")
        print(">>> Approve/Reject via Slack to resume.")
        print(">>> thread_id to resume:", thread_id)
    else:
        print(">>> Run completed without HITL.")

    print("\nFinal state:")
    pprint({k: v for k, v in result.items() if k != "__interrupt__"})


if __name__ == "__main__":
    main()
