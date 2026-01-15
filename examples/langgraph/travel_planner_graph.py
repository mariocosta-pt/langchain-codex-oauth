"""LangGraph real-world-ish example: trip planner agent graph.

Goal
- Exercise system prompts, routing, and tool calling.
- Use `ChatCodexOAuth` (subscription Codex via OAuth) as a drop-in-ish model.

This example intentionally depends on `langgraph`, but `langgraph` is NOT a
runtime dependency of `langchain-codex-oauth`.

See `examples/README.md` for setup instructions.

Run:
  python examples/langgraph/travel_planner_graph.py --city Tokyo --budget 600

Try different system prompt strategies:
  python examples/langgraph/travel_planner_graph.py --mode strict
  python examples/langgraph/travel_planner_graph.py --mode default
  python examples/langgraph/travel_planner_graph.py --mode disabled
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain_codex_oauth import ChatCodexOAuth


def _get_llm(
    *,
    provider: str,
    openai_model: str,
    mode: str,
    temperature: float | None,
    bind_tools: bool,
    tool_choice: str | None,
):
    if provider == "codex":
        model = ChatCodexOAuth(
            model="gpt-5.2-codex",
            system_prompt_mode=mode,  # type: ignore[arg-type]
            temperature=temperature,
        )
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "provider=openai requires `pip install langchain-openai`"
            ) from exc

        model = ChatOpenAI(
            model=openai_model,
            temperature=temperature,
        )
    else:
        raise ValueError("provider must be 'codex' or 'openai'")

    if bind_tools:
        tc = tool_choice
        # LangChain's ChatOpenAI prefers `required`; Codex adapter supports `any`.
        if provider == "openai" and tc == "any":
            tc = "required"
        model = (
            model.bind_tools(TOOLS, tool_choice=tc) if tc else model.bind_tools(TOOLS)
        )

    return model


# Optional dependency (installed in a separate venv).
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# -----------------------------
# Pydantic output models
# -----------------------------


class BudgetLineItem(BaseModel):
    label: str
    cost_usd: float


class BudgetBreakdown(BaseModel):
    total_usd: float
    items: list[BudgetLineItem]


class DayPlan(BaseModel):
    day: int = Field(ge=1)
    morning: str
    afternoon: str
    evening: str


class TripPlan(BaseModel):
    city: str
    constraints: list[str]
    days: list[DayPlan]
    budget: BudgetBreakdown
    notes: list[str]


class RouteDecision(BaseModel):
    next: Literal["planner", "researcher", "budgeter", "writer", "end"]
    reason: str


# -----------------------------
# Tooling (prints are deliberate)
# -----------------------------


_CITY_KB: dict[str, str] = {
    "tokyo": (
        "Tokyo basics (rough, illustrative):\n"
        "- Transit: ~ $6–$12/day (metro + JR short trips)\n"
        "- Food: ~$25–$60/day depending on style\n"
        "- Popular areas: Asakusa, Shinjuku, Akihabara, Ueno\n"
        "- Day trip options: Kamakura, Nikko\n"
    ),
    "lisbon": (
        "Lisbon basics (rough, illustrative):\n"
        "- Transit: ~ $6/day (metro/tram)\n"
        "- Food: ~$20–$55/day\n"
        "- Areas: Baixa, Alfama, Belém\n"
        "- Day trip: Sintra\n"
    ),
}


@tool
def city_info(city: str) -> str:
    """Get basic city info (offline toy knowledge base).

    Use this instead of guessing.
    """

    key = city.strip().lower()
    result = _CITY_KB.get(key) or (
        f"No KB entry for {city!r}. Use general knowledge cautiously."
    )
    print(f"\n[tool] city_info(city={city!r}) -> {result.splitlines()[0]}")
    return result


_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_eval_expr(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))

    raise ValueError("Unsupported expression")


@tool
def calc(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression.

    Supported: + - * / % **, parentheses, ints/floats.
    """

    expr = expression.strip()
    try:
        parsed = ast.parse(expr, mode="eval")
        value = _eval_expr(parsed)
        out = f"{value:.2f}".rstrip("0").rstrip(".")
    except Exception as exc:
        out = f"error: {exc}"

    print(f"\n[tool] calc(expression={expression!r}) -> {out}")
    return out


TOOLS = [city_info, calc]


def _has_recent_tool_results(messages: list[BaseMessage]) -> bool:
    # In a tool loop, ToolNode appends ToolMessage results after an AI tool call.
    recent = messages[-6:] if len(messages) > 6 else messages
    return any(isinstance(m, ToolMessage) for m in recent)


def _post_tools_for_owner(state: PlannerState, owner: str) -> bool:
    # Only treat ToolMessages as "ours" if we were the tool requester.
    if state.get("tool_owner") != owner:
        return False
    return _has_recent_tool_results(state.get("messages") or [])


def _extract_json_object(text: str) -> str | None:
    """Extract a JSON object from a model response.

    Many models return JSON wrapped in markdown fences (```json ... ```). We want
    the graph to evaluate adapter/model behavior, not fail due to formatting.
    """

    raw = text.strip()
    if not raw:
        return None

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            # Drop the opening fence and closing fence.
            raw = "\n".join(lines[1:-1]).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    return raw[start : end + 1]


# -----------------------------
# LangGraph state
# -----------------------------


class PlannerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    city: str
    budget_usd: float
    constraints: list[str]

    plan_outline: str | None
    research_notes: str | None
    budget: dict[str, Any] | None
    final: dict[str, Any] | None

    # routing bookkeeping
    next: str | None
    tool_owner: str | None
    steps: int


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def supervisor_node(
    state: PlannerState,
    *,
    provider: str,
    openai_model: str,
    temperature: float | None,
    mode: str,
    max_steps: int,
) -> dict[str, Any]:
    _print_header("[node] supervisor")

    steps = int(state.get("steps", 0)) + 1
    if steps >= max_steps:
        print(f"max_steps reached ({steps}); ending")
        return {"next": "end", "steps": steps}

    summary = {
        "has_plan_outline": bool(state.get("plan_outline")),
        "has_research_notes": bool(state.get("research_notes")),
        "has_budget": bool(state.get("budget")),
        "has_final": bool(state.get("final")),
        "steps": steps,
    }
    print("state summary:", summary)

    router = _get_llm(
        provider=provider,
        openai_model=openai_model,
        mode=mode,
        temperature=temperature,
        bind_tools=False,
        tool_choice=None,
    ).with_structured_output(RouteDecision)

    system = SystemMessage(
        content=(
            "You are a supervisor for a LangGraph workflow. Decide the next node.\n"
            "Rules:\n"
            "- Choose exactly one of: planner, researcher, budgeter, writer, end\n"
            "- Prefer planner -> researcher -> budgeter -> writer -> end\n"
            "- If a required artifact is missing, pick the node that creates it\n"
            "- Do not chat; only fill the RouteDecision schema"
        )
    )

    human = HumanMessage(
        content=(
            "Given this state summary, choose next node:\n"
            f"{json.dumps(summary)}\n\n"
            "Artifacts: plan_outline, research_notes, budget, final"
        )
    )

    decision = router.invoke([system, human])

    next_node: str | None = None
    reason: str | None = None

    if isinstance(decision, dict):
        next_node = decision.get("next")
        reason = decision.get("reason")
    elif hasattr(decision, "model_dump"):
        dumped = decision.model_dump()
        if isinstance(dumped, dict):
            next_node = dumped.get("next")
            reason = dumped.get("reason")
    else:
        next_node = getattr(decision, "next", None)
        reason = getattr(decision, "reason", None)

    print("router decision:", decision)

    if next_node not in {"planner", "researcher", "budgeter", "writer", "end"}:
        next_node = "planner"

    return {
        "next": next_node,
        "steps": steps,
        "messages": [AIMessage(content=f"ROUTE: {next_node} ({reason})")],
    }


def planner_node(
    state: PlannerState,
    *,
    provider: str,
    openai_model: str,
    temperature: float | None,
    mode: str,
) -> dict[str, Any]:
    _print_header("[node] planner")

    city = state["city"]
    budget_usd = state["budget_usd"]
    constraints = state["constraints"]

    planner = _get_llm(
        provider=provider,
        openai_model=openai_model,
        mode=mode,
        temperature=temperature,
        bind_tools=False,
        tool_choice=None,
    )

    system = SystemMessage(
        content=(
            "You are a travel planner. Produce a 2-day itinerary outline.\n"
            "Constraints:\n"
            "- Be practical and concise\n"
            "- Do not invent prices; leave costs to the budgeter\n"
            "- Use bullet points and headings"
        )
    )

    user = HumanMessage(
        content=(
            f"City: {city}\n"
            f"Budget: ${budget_usd} USD total\n"
            f"Constraints: {constraints}\n\n"
            "Create a 2-day outline (morning/afternoon/evening per day)."
        )
    )

    msg = planner.invoke([system, user])
    print("planner output (first 200 chars):", str(msg.content)[:200])

    return {"plan_outline": str(msg.content), "messages": [msg]}


def researcher_agent_node(
    state: PlannerState,
    *,
    provider: str,
    openai_model: str,
    temperature: float | None,
    mode: str,
) -> dict[str, Any]:
    _print_header("[node] researcher_agent")

    city = state["city"]
    messages = state["messages"]

    # Two-phase tool loop:
    # 1) Force at least one tool call (city_info, ideally calc too)
    # 2) After tool results are present, write research notes without calling tools
    post_tools = _post_tools_for_owner(state, "researcher")

    if not post_tools:
        model = _get_llm(
            provider=provider,
            openai_model=openai_model,
            mode=mode,
            temperature=temperature,
            bind_tools=True,
            tool_choice="any",
        )
        system = SystemMessage(
            content=(
                "You are a research assistant for trip planning.\n"
                "Rules:\n"
                "- You MUST call tools in this step\n"
                "- Call city_info(city) for the target city\n"
                "- Also call calc(...) to compute a 2-day rough food+transit estimate\n"
                "- For calc, ONLY use numbers and operators (no variables)\n"
                "- Do not write the final notes yet"
            )
        )
        user = HumanMessage(
            content=(
                f"City: {city}\n"
                "Pick a plausible food_per_day (25-60) and transit_per_day (6-12), "
                "then call calc with '(food_per_day+transit_per_day)*2' using NUMBERS only. "
                "Example: '(40+10)*2'."
            )
        )
    else:
        model = _get_llm(
            provider=provider,
            openai_model=openai_model,
            mode=mode,
            temperature=temperature,
            bind_tools=True,
            tool_choice=None,
        )
        system = SystemMessage(
            content=(
                "You are a research assistant for trip planning.\n"
                "Rules:\n"
                "- You have tool outputs in context\n"
                "- Do NOT call tools now\n"
                "- Output a short bullet fact list (transit/food/areas/day trips)"
            )
        )
        user = HumanMessage(content=f"Write research notes for {city}.")

    msg = model.invoke([system, user, *messages])
    tool_calls = getattr(msg, "tool_calls", None)
    print("tool_calls:", tool_calls)
    if isinstance(msg.content, str) and msg.content:
        print("researcher content (first 200 chars):", msg.content[:200])

    return {
        "messages": [msg],
        "tool_owner": "researcher" if tool_calls else None,
    }


def budgeter_agent_node(
    state: PlannerState,
    *,
    provider: str,
    openai_model: str,
    temperature: float | None,
    mode: str,
    min_spend_ratio: float,
) -> dict[str, Any]:
    _print_header("[node] budgeter_agent")

    city = state["city"]
    budget_usd = state["budget_usd"]
    constraints = state["constraints"]
    messages = state["messages"]

    # Two-phase tool loop:
    # 1) Force arithmetic tool usage to compute totals
    # 2) After tool results are present, output final JSON without tools
    post_tools = _post_tools_for_owner(state, "budgeter")

    if not post_tools:
        model = _get_llm(
            provider=provider,
            openai_model=openai_model,
            mode=mode,
            temperature=temperature,
            bind_tools=True,
            tool_choice="any",
        )
        system = SystemMessage(
            content=(
                "You are a budgeter.\n"
                "Rules:\n"
                "- You MUST call tools in this step\n"
                "- Call calc(...) to compute a candidate total\n"
                "- For calc, ONLY use numbers and operators (no variables)\n"
                "- Do not output the final JSON yet"
            )
        )
        min_ratio_pct = int(min_spend_ratio * 100)
        user = HumanMessage(
            content=(
                f"Budget cap: ${budget_usd} USD\n"
                f"City: {city}\n"
                f"Call calc with a numeric expression that yields a total between {min_ratio_pct}% "
                "and 100% of the cap. Example: '600*0.95'."
            )
        )
    else:
        model = _get_llm(
            provider=provider,
            openai_model=openai_model,
            mode=mode,
            temperature=temperature,
            bind_tools=True,
            tool_choice=None,
        )
        min_ratio_pct = int(min_spend_ratio * 100)
        system = SystemMessage(
            content=(
                "You are a budgeter. Create a concrete budget breakdown that fits under the total.\n"
                "Rules:\n"
                "- You have tool outputs in context\n"
                "- Do NOT call tools now\n"
                "- Output JSON with keys: total_usd, items (list of {label, cost_usd})\n"
                f"- Total must be close to the cap: at least {min_ratio_pct}% of the cap"
            )
        )
        user = HumanMessage(
            content=(
                f"City: {city}\n"
                f"Budget cap: ${budget_usd} USD\n"
                f"Constraints: {constraints}\n\n"
                "Return only a JSON object (no markdown fences, no ```)."
            )
        )

    msg = model.invoke([system, user, *messages])
    tool_calls = getattr(msg, "tool_calls", None)
    print("tool_calls:", tool_calls)
    if isinstance(msg.content, str) and msg.content:
        print("budgeter content (first 200 chars):", msg.content[:200])

    # If the model already produced JSON, parse it; otherwise we keep the message and
    # let the supervisor decide what to do next.
    parsed_budget: dict[str, Any] | None = None
    if isinstance(msg.content, str):
        extracted = _extract_json_object(msg.content)
        if extracted is not None:
            try:
                parsed_budget = BudgetBreakdown.model_validate_json(
                    extracted
                ).model_dump()
            except Exception as exc:
                print("budget parse failed:", exc)

    # Enforce a "near cap" constraint to push tool+instruction adherence.
    if parsed_budget is not None:
        total = parsed_budget.get("total_usd")
        if isinstance(total, (int, float)):
            if float(total) < float(budget_usd) * float(min_spend_ratio):
                print(
                    "budget too low; forcing retry. total=",
                    total,
                    "min_required=",
                    float(budget_usd) * float(min_spend_ratio),
                )
                parsed_budget = None

    update: dict[str, Any] = {
        "messages": [msg],
        "tool_owner": "budgeter" if tool_calls else None,
    }
    if parsed_budget is not None:
        update["budget"] = parsed_budget
    return update


def writer_node(
    state: PlannerState,
    *,
    provider: str,
    openai_model: str,
    temperature: float | None,
    mode: str,
) -> dict[str, Any]:
    _print_header("[node] writer")

    city = state["city"]
    constraints = state["constraints"]

    writer = _get_llm(
        provider=provider,
        openai_model=openai_model,
        mode=mode,
        temperature=temperature,
        bind_tools=False,
        tool_choice=None,
    ).with_structured_output(TripPlan, include_raw=True)

    system = SystemMessage(
        content=(
            "You are a careful assistant. Produce the final TripPlan output.\n"
            "Rules:\n"
            "- Must conform to the TripPlan schema exactly\n"
            "- Keep day entries concise\n"
            "- Use the provided budget dict if present; do not exceed it"
        )
    )

    user_payload = {
        "city": city,
        "constraints": constraints,
        "plan_outline": state.get("plan_outline"),
        "research_notes": state.get("research_notes"),
        "budget": state.get("budget"),
    }

    user = HumanMessage(
        content=(
            "Use the following context to produce the final TripPlan.\n"
            f"{json.dumps(user_payload, indent=2)}"
        )
    )

    out = writer.invoke([system, user])

    parsed = out.get("parsed") if isinstance(out, dict) else None
    raw = out.get("raw") if isinstance(out, dict) else None

    if parsed is None:
        print("writer did not return parsed output")
        return {"messages": [AIMessage(content=str(out))]}

    final = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    print("final TripPlan parsed OK")

    msg = AIMessage(content=json.dumps(final, indent=2))
    return {"final": final, "messages": [msg, *([raw] if raw else [])]}


def capture_research_notes(state: PlannerState) -> dict[str, Any]:
    """Extract a usable research_notes string from recent AI messages."""

    notes: str | None = None
    for m in reversed(state.get("messages") or []):
        if (
            isinstance(m, AIMessage)
            and isinstance(m.content, str)
            and m.content.strip()
        ):
            # Skip router messages
            if m.content.startswith("ROUTE:"):
                continue
            notes = m.content
            break

    if notes:
        return {"research_notes": notes}
    return {}


def route_from_supervisor(state: PlannerState) -> str:
    return str(state.get("next") or "end")


def route_from_tools(state: PlannerState) -> str:
    """After tool execution, decide where to go.

    We store the requesting node in `tool_owner` when a node produces tool_calls.
    """

    owner = state.get("tool_owner")
    if owner in {"researcher", "budgeter"}:
        return owner

    return "supervisor"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Tokyo")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--mode",
        choices=["strict", "default", "disabled"],
        default="strict",
        help="ChatCodexOAuth system_prompt_mode",
    )
    parser.add_argument(
        "--provider",
        choices=["codex", "openai"],
        default="codex",
        help="Model provider to run this graph with",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Only used when --provider openai",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (shared across providers)",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--min-spend-ratio",
        type=float,
        default=0.85,
        help="Require budget total to be near cap (e.g. 0.85 => >=85% of cap)",
    )
    args = parser.parse_args()

    provider = str(args.provider)
    openai_model = str(args.openai_model)
    temperature = float(args.temperature)

    city = str(args.city)
    budget_usd = float(args.budget)
    mode = str(args.mode)
    max_steps = int(args.max_steps)
    min_spend_ratio = float(args.min_spend_ratio)

    constraints = [
        "2 days",
        f"total budget under ${budget_usd} USD",
        "include 1 cultural activity",
        "include 1 food-focused stop",
        "avoid overly packed schedule",
    ]

    tool_node = ToolNode(TOOLS, handle_tool_errors=True)

    graph: StateGraph[PlannerState] = StateGraph(PlannerState)
    graph.add_node(
        "supervisor",
        lambda state: supervisor_node(
            state,
            provider=provider,
            openai_model=openai_model,
            temperature=temperature,
            mode=mode,
            max_steps=max_steps,
        ),
    )
    graph.add_node(
        "planner",
        lambda state: planner_node(
            state,
            provider=provider,
            openai_model=openai_model,
            temperature=temperature,
            mode=mode,
        ),
    )
    graph.add_node(
        "researcher",
        lambda state: researcher_agent_node(
            state,
            provider=provider,
            openai_model=openai_model,
            temperature=temperature,
            mode=mode,
        ),
    )
    graph.add_node(
        "budgeter",
        lambda state: budgeter_agent_node(
            state,
            provider=provider,
            openai_model=openai_model,
            temperature=temperature,
            mode=mode,
            min_spend_ratio=min_spend_ratio,
        ),
    )
    graph.add_node(
        "writer",
        lambda state: writer_node(
            state,
            provider=provider,
            openai_model=openai_model,
            temperature=temperature,
            mode=mode,
        ),
    )
    graph.add_node("tools", tool_node)
    graph.add_node("capture_research", capture_research_notes)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "researcher": "researcher",
            "budgeter": "budgeter",
            "writer": "writer",
            "end": END,
        },
    )

    graph.add_edge("planner", "supervisor")

    # Researcher tool loop
    graph.add_conditional_edges(
        "researcher",
        tools_condition,
        {
            "tools": "tools",
            END: "capture_research",
        },
    )
    graph.add_edge("capture_research", "supervisor")

    # Budgeter tool loop
    graph.add_conditional_edges(
        "budgeter",
        tools_condition,
        {
            "tools": "tools",
            END: "supervisor",
        },
    )

    graph.add_conditional_edges(
        "tools",
        route_from_tools,
        {
            "researcher": "researcher",
            "budgeter": "budgeter",
            "supervisor": "supervisor",
        },
    )

    graph.add_edge("writer", END)

    app = graph.compile()

    initial: PlannerState = {
        "messages": [HumanMessage(content=f"Plan a trip to {city}.")],
        "city": city,
        "budget_usd": budget_usd,
        "constraints": constraints,
        "plan_outline": None,
        "research_notes": None,
        "budget": None,
        "final": None,
        "next": None,
        "tool_owner": None,
        "steps": 0,
    }

    _print_header("RUN")
    print("provider:", provider)
    print("mode:", mode)
    print("temperature:", temperature)
    if provider == "openai":
        print("openai_model:", openai_model)
    print("city:", city)
    print("budget_usd:", budget_usd)

    out = app.invoke(initial)

    _print_header("RESULT")
    final = out.get("final")
    if isinstance(final, dict):
        print(json.dumps(final, indent=2))
    else:
        print("No final plan produced.")


if __name__ == "__main__":
    main()
