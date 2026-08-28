"""FastAPI backend — the deployed brain behind the portfolio chatbot.

Replaces the old Gradio app. It is a thin HTTP/SSE shell around the from-scratch
pipeline; almost all of the work lives in the modules it reuses (crawl, rag,
agent, mcp_client) — this file only wires them to the web and to the browser.

On boot it loads the committed crawl snapshot in backend/data/site (refreshed in
CI, not at boot — see .github/workflows/crawl.yml) and spins up the agent; set
CRAWL_SITES to re-enable a live boot crawl. Retrieval is owned by a FastMCP child
process that loads that snapshot and embeds it (fastembed/ONNX) — so this parent
stays light (it only parses page metadata for /pages and /health, never embeds).
A second FastMCP child serves the contact tool. Set MCP_TRANSPORT=inprocess to
skip the subprocesses (this parent then builds the index itself) for offline dev.

Endpoints:
  POST /chat    {question, history?}  → text/event-stream of agent events
  GET  /health                  → {ok, pages, chunks, transport, brain}
  GET  /pages                   → [{title, url, chunks}] for the demo panel

Run:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import json
import os
import sys
import threading
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import crawl  # noqa: E402
from rag import RAG, chunk_markdown, parse_frontmatter  # noqa: E402


# --- config (all env-overridable; defaults suit the Render deploy) ----------
# sites to index. Normally empty (CI commits the snapshot; see crawl.yml), so
# DEFAULT_SITES is only used by the boot-crawl safety net below.
DEFAULT_SITES = ["https://adityajain.me", "https://projects.adityajain.me"]
CRAWL_SITES = os.environ.get(
    "CRAWL_SITES", " ".join(DEFAULT_SITES)).split()
CORPUS_DIR = os.environ.get("CORPUS_DIR", "/tmp/site")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "process").lower()
# where to fall back if a live crawl yields nothing (network down, sitemap gone)
BUNDLED_SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "site")


# --- shared state, populated on startup -------------------------------------
class _State:
    agent = None
    pages: list[dict] = []          # [{title, url, source, chunks}]
    chunks = 0
    brain = "?"
    lock = threading.Lock()         # serialize agent turns (shared stdio child)


STATE = _State()


def _refresh_corpus() -> str:
    """Resolve the corpus to index. By default (CRAWL_SITES empty) we DON'T crawl
    at boot — the CI job (.github/workflows/crawl.yml) crawls on push + weekly and
    commits the snapshot to backend/data/site instead, so startup is fast and
    doesn't depend on the live site.

    Safety net: if that committed snapshot isn't there yet (e.g. the first deploy
    before CI has crawled + committed one), we crawl DEFAULT_SITES live this once
    so the service still boots. CI commits a snapshot shortly after and the next
    deploy is snapshot-based again. Set CRAWL_SITES to force a live boot crawl."""
    if not CRAWL_SITES:
        if os.path.isdir(BUNDLED_SNAPSHOT) and os.listdir(BUNDLED_SNAPSHOT):
            print(f"[boot] no boot crawl (CRAWL_SITES empty); using committed "
                  f"snapshot {BUNDLED_SNAPSHOT}", flush=True)
            return BUNDLED_SNAPSHOT
        print("[boot] CRAWL_SITES empty and no committed snapshot yet; falling "
              "back to a one-time live crawl (CI will commit one shortly)", flush=True)
        sites = DEFAULT_SITES
    else:
        sites = CRAWL_SITES
    try:
        print(f"[boot] crawling {sites} → {CORPUS_DIR}", flush=True)
        pages = crawl.crawl(sites)
    except Exception as exc:
        print(f"[boot] crawl failed ({exc}); using bundled snapshot", flush=True)
        pages = []
    if pages:
        crawl.save_snapshot(pages, CORPUS_DIR)
        print(f"[boot] snapshot: {len(pages)} pages → {CORPUS_DIR}", flush=True)
        return CORPUS_DIR
    if os.path.isdir(BUNDLED_SNAPSHOT) and os.listdir(BUNDLED_SNAPSHOT):
        print(f"[boot] falling back to bundled snapshot {BUNDLED_SNAPSHOT}", flush=True)
        return BUNDLED_SNAPSHOT
    raise RuntimeError("no pages crawled and no bundled snapshot to fall back on")


def _page_index(corpus_dir: str) -> tuple[list[dict], int]:
    """Read the snapshot markdown and build the indexed-pages list WITHOUT
    embedding — just parse frontmatter and count chunks. This is what the light
    FastAPI parent knows about the corpus; the real vectors live in the MCP child."""
    import glob
    pages: list[dict] = []
    total = 0
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md")) +
                       glob.glob(os.path.join(corpus_dir, "*.mdx"))):
        slug = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as fh:
            md = fh.read()
        fm = parse_frontmatter(md)
        n = len(chunk_markdown(md, slug, url=fm.get("url", ""), title=fm.get("title", "")))
        total += n
        pages.append({"title": fm.get("title") or slug.replace("-", " ").title(),
                      "url": fm.get("url", ""), "source": slug, "chunks": n})
    pages.sort(key=lambda p: p["title"].lower())
    return pages, total


app = FastAPI(title="Portfolio Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # public, read-only, no cookies
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    from agent import build_agent
    corpus = _refresh_corpus()
    # the page list/health come from the snapshot on disk (no embedding here)
    STATE.pages, STATE.chunks = _page_index(corpus)

    if MCP_TRANSPORT == "process":
        # retrieval + contact each run as their own FastMCP child; the blog child
        # loads CORPUS_DIR and does the embedding. This parent never embeds.
        # Force (not setdefault) so the child indexes the SAME corpus we resolved
        # — otherwise a crawl-failed → bundled-snapshot fallback would leave the
        # child pointed at an empty /tmp/site and crash on boot.
        os.environ["CORPUS_DIR"] = corpus
        STATE.agent = build_agent(rag=None, use_mcp=True, mcp_transport="process")
    else:
        # single interpreter (fits 512 MB): build the index in-process and use the
        # in-memory MCP clients. Embedder chosen by EMBEDDER env (default gemini —
        # API embeddings, no local model, so no OOM). See rag.make_embedder.
        from rag import make_embedder
        rag = RAG(embedder=make_embedder()).build(corpus)
        STATE.agent = build_agent(rag=rag, use_mcp=True, mcp_transport="inprocess")
    STATE.brain = getattr(STATE.agent, "brain_name", "?")
    print(f"[boot] ready — {len(STATE.pages)} pages, {STATE.chunks} chunks, "
          f"transport={MCP_TRANSPORT}, brain={STATE.brain}", flush=True)


class Turn(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Turn] = []   # prior turns, oldest→newest (for follow-ups)


# how many prior turns to feed back in (keeps the prompt — and cost — bounded)
_MAX_HISTORY = 8


def _with_history(question: str, history: list[Turn]) -> str:
    """Fold prior turns into the question so the (stateless) agent can resolve
    follow-ups like "what about the second one?". The ReAct prompt already ends
    with `Question: {question}`, so we prepend a short transcript here rather
    than change the agent."""
    turns = [t for t in history if t.content.strip()][-_MAX_HISTORY:]
    if not turns:
        return question
    lines = []
    for t in turns:
        who = "User" if t.role == "user" else "Assistant"
        lines.append(f"{who}: {t.content.strip()}")
    convo = "\n".join(lines)
    return (f"Conversation so far:\n{convo}\n\n"
            f"Given that conversation, answer this follow-up. Resolve any "
            f"references to earlier turns.\nFollow-up: {question}")


def _sse(kind: str, **data) -> str:
    return f"data: {json.dumps({'kind': kind, **data})}\n\n"


def _stream(question: str, history: list[Turn] | None = None):
    """Drive the agent and translate each Event into an SSE line. Runs under a
    lock because the agent's MCP tools talk to a single shared stdio child —
    concurrent turns would interleave on that pipe."""
    if STATE.agent is None:
        yield _sse("error", message="agent not ready")
        return
    contextual = _with_history(question, history or [])
    # bounded wait: if another turn is mid-flight (the agent shares one stdio
    # child), fail fast with a clear message rather than hanging the browser.
    if not STATE.lock.acquire(timeout=45):
        yield _sse("error", message="The assistant is busy with another question — "
                   "give it a moment and try again.")
        return
    try:
        try:
            for ev in STATE.agent.run_iter(contextual):
                if ev.kind == "thinking":
                    yield _sse("thinking", brain=ev.data["brain"], prompt=ev.data["prompt"])
                elif ev.kind == "model":
                    yield _sse("model", text=ev.data["text"])
                elif ev.kind == "tool_call":
                    yield _sse("tool_call", tool=ev.data["tool"], input=ev.data["input"])
                elif ev.kind == "observation":
                    yield _sse("observation", tool=ev.data["tool"], output=ev.data["output"])
                elif ev.kind == "final":
                    steps = ev.data["result"].steps
                    used = sorted({s.action for s in steps
                                   if s.action and s.action not in ("__final__",)})
                    yield _sse("final", answer=ev.data["answer"], tools_used=used)
        except Exception as exc:  # never leave the stream hanging on a failure
            low = str(exc).lower()
            if any(s in low for s in ("timed out", "timeout", "deadline")):
                message = ("That took longer than I expected — the model was slow "
                           "to respond just now. Please try asking again in a moment.")
            else:
                message = str(exc)
            yield _sse("error", message=message)
    finally:
        STATE.lock.release()


@app.post("/chat")
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        def _empty():
            yield _sse("final", answer="Ask me something about Aditya.", tools_used=[])
        return StreamingResponse(_empty(), media_type="text/event-stream")
    return StreamingResponse(
        _stream(question, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"ok": STATE.agent is not None, "pages": len(STATE.pages),
            "chunks": STATE.chunks, "transport": MCP_TRANSPORT, "brain": STATE.brain}


@app.get("/pages")
def pages():
    return {"sites": CRAWL_SITES, "total_chunks": STATE.chunks,
            # `source` is the slug the model cites in-line (e.g. [home]); the
            # widget uses it to hyperlink those citations to the real page URL.
            "pages": [{"title": p["title"], "url": p["url"], "source": p["source"],
                       "chunks": p["chunks"]}
                      for p in STATE.pages]}
