"""RAG (Chroma) A/B example: embedder vs LLM provider.

This is a LangChain-only example (no LangGraph dependency).

It demonstrates:
- Embedding/indexing with either Ollama (mxbai-embed-large) or OpenAI embeddings.
- Retrieval with Chroma (persisted under examples/output/).
- Answer generation with either ChatCodexOAuth (codex) or ChatOpenAI (openai).

Important rule:
- The embedder used to index documents must match the embedder used for queries.

Prereqs
- For codex LLM: `langchain-codex-oauth auth login`
- For Ollama embeddings: `ollama serve` and `ollama pull mxbai-embed-large`
- For OpenAI embeddings/LLM: set `OPENAI_API_KEY`

See `examples/README.md` for setup instructions.

Run examples:
  python examples/langchain/rag_chroma_ab.py --llm codex --embedder ollama
  python examples/langchain/rag_chroma_ab.py --llm codex --embedder openai
  python examples/langchain/rag_chroma_ab.py --llm openai --embedder openai
  python examples/langchain/rag_chroma_ab.py --llm openai --embedder ollama
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_codex_oauth import ChatCodexOAuth

DATA_DIR = Path(__file__).resolve().parent / "rag_data"
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "output"


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    text: str


def _read_corpus_files() -> list[CorpusFile]:
    files: list[CorpusFile] = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        files.append(CorpusFile(path=path, text=path.read_text(encoding="utf-8")))
    if not files:
        raise RuntimeError(f"No corpus files found under {DATA_DIR}")
    return files


def _fingerprint_corpus(files: list[CorpusFile]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.path.name).encode("utf-8"))
        h.update(b"\0")
        h.update(f.text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    # Simple, dependency-free chunker.
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(cleaned):
            break

        start = max(0, end - overlap)

    return chunks


def _build_documents(files: list[CorpusFile]) -> list[Document]:
    docs: list[Document] = []
    for f in files:
        chunks = _chunk_text(f.text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f.path.name,
                        "chunk": i,
                    },
                )
            )
    return docs


def _get_embeddings(*, embedder: str, ollama_model: str, openai_model: str):
    if embedder == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "embedder=ollama requires `pip install langchain-ollama`"
            ) from exc

        return OllamaEmbeddings(model=ollama_model)

    if embedder == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "embedder=openai requires `pip install langchain-openai`"
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI embeddings (embedder=openai)"
            )

        return OpenAIEmbeddings(model=openai_model)

    raise ValueError("embedder must be 'ollama' or 'openai'")


def _get_llm(*, llm: str, openai_model: str, temperature: float):
    if llm == "codex":
        return ChatCodexOAuth(
            model="gpt-5.2-codex",
            system_prompt_mode="strict",
            temperature=temperature,
        )

    if llm == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llm=openai requires `pip install langchain-openai`"
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for ChatOpenAI")

        return ChatOpenAI(model=openai_model, temperature=temperature)

    raise ValueError("llm must be 'codex' or 'openai'")


def _ensure_indexed(
    *,
    vector_store: Any,
    persist_dir: Path,
    embedder: str,
    embedder_model: str,
    corpus_fingerprint: str,
    documents: list[Document],
) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    marker = persist_dir / ".indexed.json"

    marker_data = {
        "embedder": embedder,
        "embedder_model": embedder_model,
        "corpus_fingerprint": corpus_fingerprint,
        "num_documents": len(documents),
    }

    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            existing = None

        if isinstance(existing, dict) and all(
            existing.get(k) == marker_data.get(k)
            for k in ["embedder", "embedder_model", "corpus_fingerprint"]
        ):
            print(f"Index already present in {persist_dir}")
            return

        print(
            "Index marker exists but does not match current settings; "
            "creating a new collection by re-adding documents."
        )

    print(f"Indexing {len(documents)} chunks into Chroma...")

    # Keep ids deterministic so re-runs are stable.
    ids: list[str] = []
    for d in documents:
        source = d.metadata.get("source")
        chunk = d.metadata.get("chunk")
        ids.append(f"{source}:{chunk}")

    # Best-effort: langchain-chroma supports ids.
    vector_store.add_documents(documents, ids=ids)

    marker.write_text(json.dumps(marker_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote index marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="Can ChatCodexOAuth be used with Ollama embeddings for RAG?",
    )
    parser.add_argument("--k", type=int, default=4)

    parser.add_argument("--llm", choices=["codex", "openai"], default="codex")
    parser.add_argument("--embedder", choices=["ollama", "openai"], default="ollama")

    parser.add_argument("--ollama-embed-model", default="mxbai-embed-large:latest")
    parser.add_argument("--openai-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--openai-chat-model", default="gpt-4o-mini")

    parser.add_argument(
        "--persist-dir",
        default="",
        help="Chroma persistence directory (defaults to examples/output/chroma_<embedder>)",
    )

    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()

    llm_choice = str(args.llm)
    embedder_choice = str(args.embedder)

    corpus_files = _read_corpus_files()
    corpus_fingerprint = _fingerprint_corpus(corpus_files)
    docs = _build_documents(corpus_files)

    if not docs:
        raise RuntimeError("No documents produced from corpus")

    persist_dir = (
        Path(args.persist_dir)
        if args.persist_dir
        else DEFAULT_OUTPUT_DIR / f"chroma_rag_{embedder_choice}"
    )

    embeddings = _get_embeddings(
        embedder=embedder_choice,
        ollama_model=str(args.ollama_embed_model),
        openai_model=str(args.openai_embedding_model),
    )

    # Quick sanity: show embedding dimension.
    probe = embeddings.embed_query("dimension probe")
    print("\n=== CONFIG ===")
    print("llm:", llm_choice)
    print("embedder:", embedder_choice)
    if embedder_choice == "ollama":
        print("ollama_embed_model:", args.ollama_embed_model)
    else:
        print("openai_embedding_model:", args.openai_embedding_model)
    print("embedding_dim:", len(probe))
    print("persist_dir:", persist_dir)

    try:
        from langchain_chroma import Chroma  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "This example requires `pip install langchain-chroma`"
        ) from exc

    vector_store = Chroma(
        collection_name="rag_demo",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    embedder_model = (
        str(args.ollama_embed_model)
        if embedder_choice == "ollama"
        else str(args.openai_embedding_model)
    )

    _ensure_indexed(
        vector_store=vector_store,
        persist_dir=persist_dir,
        embedder=embedder_choice,
        embedder_model=embedder_model,
        corpus_fingerprint=corpus_fingerprint,
        documents=docs,
    )

    question = str(args.question)

    print("\n=== RETRIEVAL ===")
    print("question:", question)

    results = vector_store.similarity_search_with_score(question, k=int(args.k))
    for i, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source")
        chunk = doc.metadata.get("chunk")
        preview = doc.page_content.replace("\n", " ")[:140]
        print(f"{i}. score={score:.4f} source={src} chunk={chunk} :: {preview}")

    context_blocks = []
    for doc, _score in results:
        src = doc.metadata.get("source")
        chunk = doc.metadata.get("chunk")
        context_blocks.append(
            f"[source={src} chunk={chunk}]\n{doc.page_content.strip()}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    llm_model = _get_llm(
        llm=llm_choice,
        openai_model=str(args.openai_chat_model),
        temperature=float(args.temperature),
    )

    system = SystemMessage(
        content=(
            "You are a helpful assistant. Answer using only the provided context. "
            "If the context does not contain the answer, say you don't know."
        )
    )

    user = HumanMessage(
        content=(
            "Use the context to answer the question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )
    )

    print("\n=== ANSWER ===")
    msg = llm_model.invoke([system, user])
    print(msg.content)


if __name__ == "__main__":
    main()
