# Using `ChatCodexOAuth` with LangChain, LangGraph, and Deep Agents

This guide explains how to use the Codex models included with a ChatGPT Plus/Pro subscription through `langchain-codex-oauth`, without an `OPENAI_API_KEY`.

## 1. Installation and authentication

```bash
python -m pip install -U langchain-codex-oauth langchain langgraph
langchain-codex-oauth auth login
```

To install directly from the repository:

```bash
python -m pip install -U \
  "langchain-codex-oauth @ git+https://github.com/mariocosta-pt/langchain-codex-oauth.git@main"
```

Optional dependencies:

```bash
python -m pip install -U deepagents langsmith
```

OAuth credentials are stored locally under `~/.langchain-codex-oauth/`. You do not need to set `OPENAI_API_KEY` for `ChatCodexOAuth`.

## 2. Models and reasoning levels

| Model | Recommended use |
| --- | --- |
| `gpt-5.6-luna` | Parsing, classification, and fast or high-volume nodes |
| `gpt-5.6-terra` | Implementation and everyday work with a good quality/cost balance |
| `gpt-5.6-sol` | Difficult planning, review, and the most demanding problems |

Accepted levels:

- `min` or `minimal`
- `low`
- `med` or `medium`
- `high`
- `xhigh`
- `max`

Short names are normalized to `minimal` and `medium`. The level can also be added as a model suffix:

```python
from langchain_codex_oauth import ChatCodexOAuth

fast_model = ChatCodexOAuth(model="gpt-5.6-luna-min")
builder_model = ChatCodexOAuth(model="gpt-5.6-terra-med")
reviewer_model = ChatCodexOAuth(model="gpt-5.6-sol-xhigh")
```

Or configured explicitly:

```python
model = ChatCodexOAuth(
    model="gpt-5.6-sol",
    reasoning={"effort": "high", "summary": "auto"},
    verbosity="medium",
)
```

Reserve `xhigh` and `max` for tasks where tests or evaluations show enough benefit to justify the additional latency. Codex `ultra` is not a normal reasoning level: it is multi-agent orchestration and is not exposed by this chat model.

## 3. Direct use with LangChain

`ChatCodexOAuth` implements `BaseChatModel`, so it works with messages, tools, structured output, and LangChain agents.

### Simple invocation

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_codex_oauth import ChatCodexOAuth

model = ChatCodexOAuth(model="gpt-5.6-terra-med")

response = model.invoke(
    [
        SystemMessage(content="Answer briefly and objectively."),
        HumanMessage(content="Explain what a Singer tap is."),
    ]
)

print(response.content)
```

### LangChain agent with tools

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_codex_oauth import ChatCodexOAuth

@tool
def list_streams() -> list[str]:
    """Return the streams available in the current integration."""
    return ["products", "suppliers", "orders"]

model = ChatCodexOAuth(model="gpt-5.6-terra-med")
agent = create_agent(
    model=model,
    tools=[list_streams],
    system_prompt="Use the available tools before answering.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Which streams are available?"}]}
)
print(result["messages"][-1].content)
```

Tools can be Python functions, `BaseTool` instances, APIs, databases, or tools obtained through MCP. The model can only call tools supplied to the agent.

## 4. Using it with LangGraph

The same model instance can be called inside a `StateGraph` node. For the tap workflow, it can be useful to select a different model for each responsibility:

```python
from langchain_codex_oauth import ChatCodexOAuth

parser = ChatCodexOAuth(model="gpt-5.6-luna-low")
builder = ChatCodexOAuth(model="gpt-5.6-terra-high")
reviewer = ChatCodexOAuth(model="gpt-5.6-sol-xhigh")
```

Minimal example with a conditional decision:

```python
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_codex_oauth import ChatCodexOAuth

builder = ChatCodexOAuth(model="gpt-5.6-terra-high")
reviewer = ChatCodexOAuth(model="gpt-5.6-sol-xhigh")

class State(TypedDict, total=False):
    requirement: str
    implementation: str
    review: str
    approved: bool
    attempts: int


def implement(state: State) -> dict:
    feedback = state.get("review", "")
    response = builder.invoke(
        f"Implement this requirement:\n{state['requirement']}\nFeedback:\n{feedback}"
    )
    return {
        "implementation": str(response.content),
        "attempts": state.get("attempts", 0) + 1,
    }


def review(state: State) -> dict:
    response = reviewer.invoke(
        "Reply only with PASS or FAIL and a short explanation:\n"
        + state["implementation"]
    )
    text = str(response.content)
    return {"review": text, "approved": text.startswith("PASS")}


def route(state: State) -> Literal["retry", "done"]:
    if state.get("approved") or state.get("attempts", 0) >= 3:
        return "done"
    return "retry"


graph = StateGraph(State)
graph.add_node("implement", implement)
graph.add_node("review", review)
graph.add_edge(START, "implement")
graph.add_edge("implement", "review")
graph.add_conditional_edges(
    "review",
    route,
    {"retry": "implement", "done": END},
)
app = graph.compile()

result = app.invoke({"requirement": "Create the products stream."})
```

In a real workflow, the reviewer decision should use structured output, such as a Pydantic model containing `verdict`, `issues`, and `repair_target`, rather than parsing free-form text.

## 5. Using it as a Deep Agent

Yes. `create_deep_agent` accepts a `BaseChatModel` instance, so `ChatCodexOAuth` can be passed directly through `model=`.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from langchain_codex_oauth import ChatCodexOAuth

@tool
def expected_product_fields() -> list[str]:
    """Return the minimum fields expected for a product."""
    return ["id", "sku", "name", "updated_at"]

model = ChatCodexOAuth(model="gpt-5.6-terra-high")

agent = create_deep_agent(
    model=model,
    tools=[expected_product_fields],
    system_prompt=(
        "Implement only the requested stream. Run validations before finishing."
    ),
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    name="tap-builder",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Implement the products stream from the documentation.",
            }
        ]
    },
    config={"configurable": {"thread_id": "tap-products-001"}},
)
```

Deep Agents add capabilities such as planning, a virtual filesystem, subagents, context compaction, and durable execution on top of LangGraph.

### Available tools

Tools can be supplied through `tools=`:

- Python functions decorated with `@tool`.
- LangChain tools (`BaseTool`).
- API and database clients wrapped as tools.
- MCP tools obtained, for example, with `MultiServerMCPClient`.
- Validation, Docker, or Hotglue tools created specifically for the workflow.

Choose the backend separately:

- `StateBackend`: a virtual filesystem stored in agent state.
- `FilesystemBackend`: direct local filesystem access.
- `LocalShellBackend`: local filesystem and command execution.

`FilesystemBackend`, and especially `LocalShellBackend`, provide real machine access. Use them only in an isolated workspace, with minimum permissions and no secrets in accessible files.

## 6. LangSmith: observability, not an agent tool

LangSmith does not need to be passed through `tools=`. It is a tracing, evaluation, and observability layer for LangChain, LangGraph, and Deep Agents.

Installation:

```bash
python -m pip install -U langsmith
```

Typical configuration:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="tap-planner"
```

Once enabled, LangChain and LangGraph executions are traced automatically, including:

- `ChatCodexOAuth` calls.
- Graph node inputs and outputs.
- Tool calls.
- Subagents and their executions.
- Errors, timings, and usage reported by the backend.

Use `@traceable` for functions outside the graph that should also appear in the trace:

```python
from langsmith import traceable

@traceable(run_type="tool", name="validate-product-sample")
def validate_product_sample(sample: dict) -> bool:
    return all(key in sample for key in ("id", "sku", "name"))
```

The generic LangSmith quickstart may request `OPENAI_API_KEY` because it uses the OpenAI client as its example. That is not required for `ChatCodexOAuth`; local OAuth authentication is still used.

### Data considerations

A trace can contain prompts, responses, tool arguments, and data samples. Before enabling LangSmith for a synchronization workflow:

- Do not place OAuth tokens, passwords, or API keys in graph state or messages.
- Use sanitized samples when personal or commercially sensitive data is involved.
- Separate development and production projects.
- Review the retention, region, and access policies of the LangSmith account.

## 7. Recommended tap planner configuration

| Responsibility | Model | Initial reasoning |
| --- | --- | --- |
| Documentation parsing and stream classification | Luna | `low` |
| Global integration planning | Sol | `high` |
| Tap and ETL implementation | Terra | `high` |
| Semantic validation of schemas and samples | Sol | `xhigh` |
| Simple repairs | Terra | `med` |
| Exceptionally difficult problems | Sol | `max` |

Start with lower levels and increase them only when validation fails or evaluations demonstrate an improvement. Routing, retry limits, and validation commands should remain deterministic in LangGraph.

## 8. Official references

- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization)
- [LangSmith observability quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
