"""A real MCP server for site search, built with FastMCP.

This runs as its OWN process and exposes the from-scratch RAG retriever as an
MCP tool. The agent — a separate process — connects over stdio and calls
`search_blog` across the process boundary. The embedding model lives HERE and
nowhere else: this child owns retrieval, so the FastAPI parent stays light.

It indexes CORPUS_DIR — the crawl snapshot the parent writes on boot (see
app.py) — using FastEmbedEmbedder (local ONNX, no API/quota, no torch). Set
CORPUS_DIR to point it at the snapshot; FASTEMBED_MODEL overrides the model.

Run standalone (stdio):  python mcp_server/blog_server.py
The agent normally launches this for you (see mcp_client.py / agent.build_agent).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the backend/ dir, where rag.py lives
# APPEND (not insert) so nothing here shadows the real `mcp` package FastMCP
# imports; our client module is named mcp_client.py precisely to avoid that.
sys.path.append(ROOT)

from fastmcp import FastMCP  # noqa: E402
from rag import RAG, FastEmbedEmbedder, search_blog_text  # noqa: E402

mcp = FastMCP("blog-search")

# Index the crawl snapshot the parent wrote on boot. Semantic embeddings via
# fastembed (ONNX) so "email" finds the "get in touch" section — free, no key.
CORPUS_DIR = os.environ.get("CORPUS_DIR", os.path.join(ROOT, "data", "site"))
_RAG = RAG(embedder=FastEmbedEmbedder()).build(CORPUS_DIR)
print(f"[blog-server] indexed {_RAG.num_chunks} chunks from {CORPUS_DIR}",
      file=sys.stderr, flush=True)


@mcp.tool
def search_blog(query: str, k: int = 3, min_score: float = 0.25) -> str:
    """Search Aditya's website and return the most relevant passages.

    Each result includes its source and a relevance score. If the best match is
    weak, the result says so — the agent uses that to rephrase and search again.
    """
    # formatting lives in rag.search_blog_text so the in-process client matches
    return search_blog_text(_RAG, query, k=k, min_score=min_score)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
