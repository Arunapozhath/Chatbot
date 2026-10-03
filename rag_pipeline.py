"""
Enhanced RAG pipeline:
  1. Query Router → classify question (rag / sql / hybrid)
  2. ChromaDB retrieval (document knowledge base)
  3. MySQL Text-to-SQL (live database)
  4. DuckDuckGo web search fallback (low confidence)
  5. LLM streaming response
"""
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RESULTS, SYSTEM_PROMPT,
    ENABLE_WEB_SEARCH, WEB_SEARCH_MAX_RESULTS, RAG_CONFIDENCE_THRESHOLD,
    GROQ_API_KEY, GROQ_MODEL,
)

GROQ_MODEL_PREFIX = "groq:"   # frontend sends "groq:llama-3.1-8b-instant"


def _is_groq_model(model: str) -> bool:
    return model.startswith(GROQ_MODEL_PREFIX) or (
        GROQ_API_KEY and model in (
            "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
            "gemma2-9b-it", "mixtral-8x7b-32768",
        )
    )


def _make_llm(model: str, streaming: bool = True):
    """Return a ChatGroq or ChatOllama based on model name."""
    if _is_groq_model(model):
        from langchain_groq import ChatGroq
        groq_model = model.replace(GROQ_MODEL_PREFIX, "") if model.startswith(GROQ_MODEL_PREFIX) else model
        return ChatGroq(
            model=groq_model,
            api_key=GROQ_API_KEY,
            streaming=streaming,
            temperature=0.3,
        )
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        streaming=streaming,
        temperature=0.3,
    )
from query_router import classify
from db_connector import VBizSQLChain, format_sql_result, db_is_available


# ── Web search ─────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"dairy vBiz Milkosoft {query}",
                region="in-en",
                safesearch="moderate",
                max_results=max_results,
            ):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception:
        return []


# ── Main RAG pipeline ──────────────────────────────────────────────────────
class MilkosoftRAG:
    def __init__(self, model: str = OLLAMA_LLM_MODEL):
        self.model_name  = model
        self._embeddings = None
        self._db         = None
        self._llm        = None
        self._sql_chain  = VBizSQLChain()

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL
            )
        return self._embeddings

    @property
    def db(self):
        if self._db is None:
            if not Path(CHROMA_DB_PATH).exists():
                raise RuntimeError(
                    "ChromaDB not found. Run  python ingest.py  first."
                )
            self._db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME,
            )
        return self._db

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _make_llm(self.model_name, streaming=True)
        return self._llm

    def retrieve_docs(self, query: str) -> tuple[list, float]:
        results = self.db.similarity_search_with_relevance_scores(query, k=TOP_K_RESULTS)
        if not results:
            return [], 0.0
        docs   = [doc for doc, _ in results]
        scores = [score for _, score in results]
        return docs, max(scores)

    def build_messages(self, context: str, history: list[dict], question: str) -> list:
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        for turn in history[-8:]:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))
        return messages

    async def astream(
        self,
        question: str,
        history: list[dict],
        user_party_id: Optional[str] = None,
    ):
        """
        Yields dicts:
          {"type": "route",     "data": "rag|sql|hybrid"}
          {"type": "sources",   "data": [...]}
          {"type": "sql_result","data": {"sql":..., "table":...}}
          {"type": "web_used",  "data": [...]}
          {"type": "token",     "data": "<token>"}
          {"type": "done",      "data": "<full_answer>"}
          {"type": "error",     "data": "<message>"}
        """
        try:
            route = classify(question)
            yield {"type": "route", "data": route}

            context_parts = []
            rag_sources   = []
            sql_result    = None

            # ── ChromaDB (rag or hybrid) ───────────────────────────────────
            if route in ("rag", "hybrid"):
                docs, best_score = self.retrieve_docs(question)
                rag_sources = [
                    {
                        "file":    Path(d.metadata.get("source", "")).name,
                        "module":  d.metadata.get("module", ""),
                        "excerpt": d.page_content[:240].strip(),
                        "score":   round(best_score, 3),
                        "type":    "document",
                    }
                    for d in docs
                ]
                if docs:
                    context_parts.append(
                        "=== KNOWLEDGE BASE ===\n" +
                        "\n\n---\n\n".join(d.page_content for d in docs)
                    )

                # web fallback when ChromaDB confidence is low
                if ENABLE_WEB_SEARCH and best_score < RAG_CONFIDENCE_THRESHOLD:
                    web_hits = web_search(question)
                    if web_hits:
                        yield {"type": "web_used", "data": web_hits}
                        context_parts.append(
                            "=== WEB SEARCH RESULTS ===\n" +
                            "\n\n".join(
                                f"[{h['title']}]\n{h['snippet']}" for h in web_hits
                            )
                        )

            yield {"type": "sources", "data": rag_sources}

            # ── MySQL Text-to-SQL (sql or hybrid) ─────────────────────────
            if route in ("sql", "hybrid") and db_is_available():
                sql_result = self._sql_chain.query(question, user_party_id)
                formatted  = format_sql_result(sql_result)
                yield {
                    "type": "sql_result",
                    "data": {
                        "sql":   sql_result.get("sql", ""),
                        "table": formatted,
                        "count": sql_result.get("count", 0),
                        "error": sql_result.get("error"),
                    },
                }
                if "error" not in sql_result:
                    context_parts.append(
                        f"=== DATABASE RESULTS ===\n"
                        f"Query: {sql_result.get('sql','')}\n\n"
                        f"{formatted}"
                    )

            # if everything empty, fall through to pure LLM
            context = "\n\n".join(context_parts) if context_parts else "No specific context found."

            # ── Stream LLM answer ─────────────────────────────────────────
            messages = self.build_messages(context, history, question)
            full = ""
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    full += token
                    yield {"type": "token", "data": token}

            yield {"type": "done", "data": full}

        except RuntimeError as e:
            yield {"type": "error", "data": str(e)}
        except Exception as e:
            yield {"type": "error", "data": f"Unexpected error: {e}"}

    def switch_model(self, model: str):
        self.model_name = model
        self._llm = None
