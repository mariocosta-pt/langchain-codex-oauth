"""v0.7 example (LangGraph + system_prompt_mode drift mitigation).

This file intentionally depends on `langgraph`, but `langgraph` is NOT a runtime
dependency of `langchain-codex-oauth`.

See `examples/README.md` for setup instructions.

Try running with different modes:
  - `python examples/langgraph/system_prompt_drift.py --mode strict`
  - `python examples/langgraph/system_prompt_drift.py --mode default`
  - `python examples/langgraph/system_prompt_drift.py --mode disabled`

In "strict" mode (default for ChatCodexOAuth), system prompts are anchored more
aggressively to reduce instruction drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# LangGraph is an optional dependency (installed in a separate venv).
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Route(TypedDict):
    next: Literal["worker"]
    reason: str


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str | None


ROUTER_SYSTEM = (
    "You are a supervisor/router. Your job is to decide the next node. "
    "You MUST return a JSON object with keys: next, reason. "
    "Set next='worker'."
)

WORKER_SYSTEM = "You are a worker. Reply with a short greeting."


def router_node(state: GraphState, *, mode: str) -> dict:
    from langchain_codex_oauth import ChatCodexOAuth

    router = ChatCodexOAuth(
        model="gpt-5.2-codex", system_prompt_mode=mode
    ).with_structured_output(Route)
    route = router.invoke([SystemMessage(content=ROUTER_SYSTEM), *state["messages"]])
    # Route is a dict-like TypedDict output in the success case.
    return {"route": route.get("next") if isinstance(route, dict) else None}


def worker_node(state: GraphState, *, mode: str) -> dict:
    from langchain_codex_oauth import ChatCodexOAuth

    worker = ChatCodexOAuth(model="gpt-5.2-codex", system_prompt_mode=mode)
    msg = worker.invoke([SystemMessage(content=WORKER_SYSTEM), *state["messages"]])
    return {"messages": [AIMessage(content=msg.content)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["strict", "default", "disabled"],
        default="strict",
        help="ChatCodexOAuth system_prompt_mode",
    )
    args = parser.parse_args()

    mode: str = args.mode

    builder: StateGraph[GraphState] = StateGraph(GraphState)
    builder.add_node("router", lambda state: router_node(state, mode=mode))
    builder.add_node("worker", lambda state: worker_node(state, mode=mode))

    builder.add_edge(START, "router")
    builder.add_edge("router", "worker")
    builder.add_edge("worker", END)

    graph = builder.compile()

    initial: GraphState = {
        "messages": [HumanMessage(content="Say hello.")],
        "route": None,
    }

    out = graph.invoke(initial)
    print("mode:", mode)
    print("route:", out.get("route"))

    final_messages = out.get("messages") or []
    if final_messages:
        print("final:", final_messages[-1].content)


if __name__ == "__main__":
    main()
