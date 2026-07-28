# Usar `ChatCodexOAuth` com LangChain, LangGraph e Deep Agents

Este guia mostra como utilizar os modelos Codex incluídos numa subscrição ChatGPT Plus/Pro através do `langchain-codex-oauth`, sem `OPENAI_API_KEY`.

## 1. Instalação e autenticação

```bash
python -m pip install -U langchain-codex-oauth langchain langgraph
langchain-codex-oauth auth login
```

Para instalar diretamente do repositório:

```bash
python -m pip install -U \
  "langchain-codex-oauth @ git+https://github.com/mariocosta-pt/langchain-codex-oauth.git@main"
```

Dependências opcionais:

```bash
python -m pip install -U deepagents langsmith
```

A autenticação OAuth fica guardada localmente em `~/.langchain-codex-oauth/`. Não é necessário definir `OPENAI_API_KEY` para `ChatCodexOAuth`.

## 2. Modelos e níveis de reasoning

| Modelo | Uso recomendado |
| --- | --- |
| `gpt-5.6-luna` | Parsing, classificação e nós rápidos ou de grande volume |
| `gpt-5.6-terra` | Implementação e trabalho quotidiano com boa relação qualidade/custo |
| `gpt-5.6-sol` | Planeamento difícil, revisão e problemas com maior exigência |

Níveis aceites:

- `min` ou `minimal`
- `low`
- `med` ou `medium`
- `high`
- `xhigh`
- `max`

Os nomes curtos são normalizados para `minimal` e `medium`. O nível também pode ser colocado como sufixo do modelo:

```python
from langchain_codex_oauth import ChatCodexOAuth

fast_model = ChatCodexOAuth(model="gpt-5.6-luna-min")
builder_model = ChatCodexOAuth(model="gpt-5.6-terra-med")
reviewer_model = ChatCodexOAuth(model="gpt-5.6-sol-xhigh")
```

Ou configurado explicitamente:

```python
model = ChatCodexOAuth(
    model="gpt-5.6-sol",
    reasoning={"effort": "high", "summary": "auto"},
    verbosity="medium",
)
```

`xhigh` e `max` devem ser reservados para tarefas onde os testes ou avaliações demonstrem benefício suficiente para justificar maior latência. O modo Codex `ultra` não é um nível normal de reasoning: é orquestração multiagente e não é exposto por este chat model.

## 3. Uso direto com LangChain

`ChatCodexOAuth` implementa `BaseChatModel`, podendo ser usado com mensagens, ferramentas, structured output e agentes LangChain.

### Invocação simples

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_codex_oauth import ChatCodexOAuth

model = ChatCodexOAuth(model="gpt-5.6-terra-med")

response = model.invoke(
    [
        SystemMessage(content="Responde de forma curta e objetiva."),
        HumanMessage(content="Explica o que é uma fila de tarefas."),
    ]
)

print(response.content)
```

### Agente LangChain com ferramentas

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_codex_oauth import ChatCodexOAuth

@tool
def word_count(text: str) -> int:
    """Conta as palavras num texto."""
    return len(text.split())

model = ChatCodexOAuth(model="gpt-5.6-terra-med")
agent = create_agent(
    model=model,
    tools=[word_count],
    system_prompt="Usa as ferramentas disponíveis antes de responder.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Quantas palavras há em 'olá mundo'?"}]}
)
print(result["messages"][-1].content)
```

As ferramentas podem ser funções Python, `BaseTool`, APIs, bases de dados ou ferramentas obtidas por MCP. O modelo apenas pode chamar as ferramentas passadas ao agente.

## 4. Uso com LangGraph

A mesma instância pode ser chamada dentro de um nó `StateGraph`. Num workflow com várias etapas, pode ser útil escolher um modelo diferente por responsabilidade:

```python
from langchain_codex_oauth import ChatCodexOAuth

parser = ChatCodexOAuth(model="gpt-5.6-luna-low")
builder = ChatCodexOAuth(model="gpt-5.6-terra-high")
reviewer = ChatCodexOAuth(model="gpt-5.6-sol-xhigh")
```

Exemplo mínimo com uma decisão condicional:

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
        f"Implementa este requisito:\n{state['requirement']}\nFeedback:\n{feedback}"
    )
    return {
        "implementation": str(response.content),
        "attempts": state.get("attempts", 0) + 1,
    }


def review(state: State) -> dict:
    response = reviewer.invoke(
        "Responde apenas PASS ou FAIL e uma explicação curta:\n"
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

result = app.invoke({"requirement": "Criar um parser para registos CSV."})
```

Num workflow real, a decisão do reviewer deve usar structured output, por exemplo um modelo Pydantic com `verdict`, `issues` e `repair_target`, em vez de interpretar texto livre.

## 5. Uso como Deep Agent

Sim. `create_deep_agent` aceita uma instância de `BaseChatModel`, por isso `ChatCodexOAuth` pode ser passado diretamente em `model=`.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from langchain_codex_oauth import ChatCodexOAuth

@tool
def required_project_files() -> list[str]:
    """Devolve os ficheiros necessários no projeto de exemplo."""
    return ["README.md", "pyproject.toml"]

model = ChatCodexOAuth(model="gpt-5.6-terra-high")

agent = create_deep_agent(
    model=model,
    tools=[required_project_files],
    system_prompt="Implementa apenas a tarefa pedida. Valida-a antes de terminar.",
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    name="project-builder",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Adiciona o comando de terminal descrito na documentação.",
            }
        ]
    },
    config={"configurable": {"thread_id": "project-task-001"}},
)
```

O Deep Agent acrescenta capacidades como planeamento, filesystem virtual, subagentes, compactação de contexto e execução durável sobre LangGraph.

### Ferramentas disponíveis

Podem ser fornecidas através de `tools=`:

- Funções Python decoradas com `@tool`.
- Ferramentas LangChain (`BaseTool`).
- Clientes de APIs e bases de dados encapsulados como ferramentas.
- Ferramentas MCP obtidas, por exemplo, com `MultiServerMCPClient`.
- Ferramentas de validação, containers ou CI criadas especificamente para o workflow.

O backend deve ser escolhido separadamente:

- `StateBackend`: filesystem virtual guardado no estado do agente.
- `FilesystemBackend`: leitura e escrita no disco local.
- `LocalShellBackend`: filesystem e execução de comandos locais.

`FilesystemBackend` e especialmente `LocalShellBackend` dão acesso real à máquina. Devem ser usados apenas num workspace isolado, com permissões mínimas e sem segredos nos ficheiros acessíveis.

## 6. LangSmith: observabilidade, não ferramenta do agente

LangSmith não precisa de ser passado em `tools=`. É uma camada de tracing, avaliação e observabilidade para LangChain, LangGraph e Deep Agents.

Instalação:

```bash
python -m pip install -U langsmith
```

Configuração típica:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="langchain-codex-example"
```

Depois de ativado, as execuções LangChain/LangGraph são registadas automaticamente, incluindo:

- Chamadas ao `ChatCodexOAuth`.
- Entradas e saídas dos nós do graph.
- Chamadas a ferramentas.
- Subagentes e respetivas execuções.
- Erros, tempos e consumo reportado pelo backend.

Para funções externas ao graph que também devam aparecer na trace, pode ser usado `@traceable`:

```python
from langsmith import traceable

@traceable(run_type="tool", name="validate-record")
def validate_record(record: dict) -> bool:
    return all(key in record for key in ("id", "name"))
```

O quickstart genérico do LangSmith pode pedir `OPENAI_API_KEY` porque usa o cliente OpenAI como exemplo. Isso não é necessário para `ChatCodexOAuth`; continua a ser utilizada a autenticação OAuth local.

### Cuidados com dados

Uma trace pode conter prompts, respostas, argumentos de ferramentas e amostras de dados. Antes de ativar LangSmith num workflow de sincronização:

- Não colocar tokens OAuth, passwords ou API keys no estado ou nas mensagens.
- Usar amostras sanitizadas quando existirem dados pessoais ou comerciais sensíveis.
- Separar projetos de desenvolvimento e produção.
- Rever retenção, região e políticas de acesso da conta LangSmith.

## 7. Configuração recomendada para workflows

| Responsabilidade | Modelo | Reasoning inicial |
| --- | --- | --- |
| Parsing de documentação e classificação | Luna | `low` |
| Planeamento global do workflow | Sol | `high` |
| Implementação | Terra | `high` |
| Validação semântica de schemas e resultados | Sol | `xhigh` |
| Reparações simples | Terra | `med` |
| Problemas excecionalmente difíceis | Sol | `max` |

Começar com níveis mais baixos e aumentar apenas quando a validação falha ou quando avaliações demonstrarem melhoria. O routing, os limites de tentativas e os comandos de validação devem permanecer determinísticos no LangGraph.

## 8. Referências oficiais

- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization)
- [LangSmith observability quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
