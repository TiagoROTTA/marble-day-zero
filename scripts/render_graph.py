"""Print the compiled graph topology as Mermaid.

Generated from the code rather than drawn by hand, so the diagram in the README
cannot drift from the actual wiring. Run: uv run python -m scripts.render_graph
"""
from src.graph import build_graph

if __name__ == "__main__":
    print(build_graph().compile().get_graph().draw_mermaid())
