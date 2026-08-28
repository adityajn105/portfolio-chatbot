# Portfolio Chatbot

A retrieval-augmented, agentic Q&A bot for a personal site — built from scratch
(no LangChain/LlamaIndex) and deployed on a **fully-free** stack.

**Live:** [chat.adityajain.me](https://chat.adityajain.me) ·
**Write-up:** [Building the Chatbot on This Site](https://adityajain.me/blogs/building-a-portfolio-chatbot.html) ·
**Code:** [github.com/adityajn105/portfolio-chatbot](https://github.com/adityajn105/portfolio-chatbot)

- **UI** → static, on **GitHub Pages** (`web/`, deployed by `.github/workflows/pages.yml`)
- **Backend** → a small **FastAPI** service on **Render** (free tier) that loads a
  **crawl snapshot** (produced in CI on push + weekly, not at boot), indexes it,
  and answers via a **ReAct agent**
- **Embeddings** → the **Gemini Embedding API** (`gemini-embedding-2`, 768-dim,
  MRL-truncated + re-normalized). No local model, so it fits Render's 512 MB where
  fastembed/torch OOM. Document vectors are **disk-cached** (`backend/data/embcache`,
  committed by CI), so boot is a cache hit — only per-question query embeds hit the
  API at runtime. `fastembed` / `st` / `tfidf` remain available as alternatives.
- **LLM** → **Gemini** (`gemini-3.5-flash`), generation only, with a sticky
  fallback to `gemini-3.5-flash-lite` if the primary hits its quota mid-session
- **Contact** → an **MCP** `send_message` tool emails the owner via **Formspree**
  when a visitor (with explicit consent + an email) wants to get in touch
- **Safety/cost** → a scoped, injection-resistant agent prompt + per-IP and global
  rate limits + input-size caps + an origin allowlist (see [Abuse guards](#abuse-guards))

One backend, two frontends that consume the same `/chat` SSE stream:

- **`web/widget.js`** — an embeddable floating **chat bubble** (Shadow DOM, zero
  deps): the clean *product* you drop into any portfolio. Shows only the answer.
- **`web/index.html` + `demo.js`** — a **behind-the-scenes explainer**: watch the
  ReAct agent think, call the MCP `search_site`/`send_message` tools, observe the
  results, and answer — plus the indexed-pages panel.

## Architecture

```
CI (.github/workflows)                    Render web service (backend/)
  crawl.yml → crawl sites, precompute        FastAPI (single interpreter)
    embeddings, commit backend/data/           │ on boot: load committed snapshot → CORPUS_DIR,
    {site, embcache}                           │          build/load the Gemini-embedded index
  pages.yml → publish web/ to Pages            │ POST /chat drives the ReAct agent
                                               ├─ MCP (in-process): search_site  (owns retrieval)
GitHub Pages (static UI)                       └─ MCP (in-process): send_message (→ Formspree)
  widget.js / demo.js ──POST /chat──►
       ▲   Server-Sent Events
       └───────────────────────────────────
```

Crawling and embedding run in CI (`.github/workflows/crawl.yml`) — on every push
and weekly — and commit both the markdown snapshot (`backend/data/site`) and the
embedding cache (`backend/data/embcache`), so boot is fast, offline, and a cache
hit. The snapshot is **not hand-committed**: CI owns it. If it's ever missing
(e.g. the first deploy before CI has produced one), the app crawls live once at
boot so it still comes up, then reverts to snapshot-based on the next deploy.

**Transport.** In production `MCP_TRANSPORT=inprocess` runs the MCP tools in the
same interpreter — the free tier's 512 MB can't hold three (`process` spawns
parent + `search_site` + `send_message` subprocesses and OOMs). Both transports
speak the same JSON-RPC `tools/call` contract, so the agent and tools are
identical either way; only the transport differs.

## Backend

### Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chat` `{question, history?}` | `text/event-stream`: agent events (`thinking`/`model`/`tool_call`/`observation`/`final`). `history` is prior `{role, content}` turns so follow-ups resolve. |
| GET | `/health` | `{ok, pages, chunks, transport, brain}` |
| GET | `/pages` | `[{title, url, chunks}]` for the demo panel and citation linking |

### Environment variables
| Var | Default (`render.yaml`) | Notes |
|-----|-------------------------|-------|
| `GEMINI_API_KEY` | — | **secret**; set in the Render dashboard, never commit. Generation + embeddings. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | generation; sticky fallback to `-lite` on quota |
| `EMBEDDER` | `gemini` | `gemini` \| `fastembed` \| `st` \| `tfidf` |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2` | used when `EMBEDDER=gemini` |
| `GEMINI_EMBED_DIM` | `768` | MRL-truncated + re-normalized |
| `GEMINI_EMBED_PER_MIN` / `GEMINI_EMBED_COOLDOWN` | `50` / `60` | pace doc embeds under the 100 req/min free cap (50 then wait 60s) |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | unused unless `EMBEDDER=fastembed` |
| `FORMSPREE_ENDPOINT` | `https://formspree.io/f/myeygrly` | must be a real `formspree.io/f/<id>` form, activated. Form ids are public → safe to commit. |
| `CRAWL_SITES` | *(empty)* | empty ⇒ no boot crawl, load committed snapshot (CI-refreshed). Space-separated site roots ⇒ live crawl on boot. |
| `CORPUS_DIR` | `/tmp/site` | crawl → MCP handoff dir |
| `MCP_TRANSPORT` | `inprocess` | `inprocess` = one interpreter (fits 512 MB). `process` spawns 3 and OOMs. (Code default is `process`; `render.yaml` sets `inprocess`.) |
| `ALLOWED_ORIGINS` | `*` | comma-separated in prod, e.g. `https://adityajn105.github.io,https://adityajain.me` |
| `PYTHON_VERSION` | `3.12.6` | |

#### Abuse guards
Public endpoint + a billed API key, so `app.py` runs cheap guards before spending a
token: origin allowlist → input-size caps → rate limiter.

| Var | Default | Notes |
|-----|---------|-------|
| `MAX_QUESTION_CHARS` | `600` | per question |
| `MAX_TURN_CHARS` | `600` | per history turn |
| `RATE_PER_MIN` | `6` | per IP / minute |
| `RATE_PER_DAY` | `40` | per IP / day |
| `GLOBAL_PER_DAY` | `800` | all IPs / day — hard spend cap (`0` disables) |

The agent prompt (`AGENTIC_RAG_PROMPT` in `agent.py`) is scoped to Aditya + the ML
topics the blog covers, declines everything else, treats the visitor's message as a
question (never as instructions), and must `search_site` for facts rather than
guessing.

### Run locally
```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=...            # get one at aistudio.google.com
export FORMSPREE_ENDPOINT=...        # optional, for the contact tool
export CORPUS_DIR=/tmp/site
export MCP_TRANSPORT=inprocess       # one interpreter; no subprocess
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then:
```bash
curl -s localhost:8000/health
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"what is Aditya'\''s email?"}'
```
The app builds the index from the bundled snapshot in `backend/data/site`, reusing
the committed embedding cache in `backend/data/embcache` on a hit.

## Deploy

### 1. Backend → Render
Push this repo to GitHub, then on Render: **New → Blueprint**, connect the repo.
It reads `render.yaml` (root dir `backend`, health check `/health`). Set the
`GEMINI_API_KEY` secret in the dashboard (all other vars are pre-filled). Note the
`https://<service>.onrender.com` URL.

> Free tier sleeps when idle; the first request after a sleep just wakes the
> service (~a minute) — there's no boot crawl. The corpus + embedding cache come
> from `backend/data/{site,embcache}`, which CI (`crawl.yml`) crawls + precomputes
> + commits on push and weekly, so wake-up never depends on the live site.
> (First deploy only: if no snapshot exists yet, the app crawls live once to boot.)

### 2. UI → GitHub Pages
`.github/workflows/pages.yml` publishes `web/` to Pages on push (Settings → Pages →
Source: **GitHub Actions**). Set `API_DEFAULT` in `web/demo.js` to your Render URL
(currently `https://portfolio-chatbot-biow.onrender.com`). The live demo is served
at [chat.adityajain.me](https://chat.adityajain.me).

### 3. Widget → any portfolio
Paste before `</body>` (nothing else needed — the bubble self-injects):
```html
<script
  src="https://chat.adityajain.me/widget.js"
  data-api="https://portfolio-chatbot-biow.onrender.com"
  data-title="Chat with Aditya" defer></script>
```

## Layout
```
backend/
  app.py                   FastAPI: boot-load snapshot, /chat (SSE), /health, /pages,
                           CORS, rate limits + input caps (abuse guards)
  rag.py                   chunk → embed (Gemini API, disk-cached) → store → retrieve → generate
  agent.py                 ReAct loop + GeminiPolicy (fallback/timeout); scoped prompt;
                           wires search_site + send_message
  crawl.py                 stdlib sitemap crawler → markdown snapshot
  contact.py               Formspree send (stdlib only), shared by both transports
  mcp_client.py            stdio + in-process MCP clients; tool builders
  precompute_embeddings.py builds the same index in CI so the cache key matches
  mcp_server/
    blog_server.py         FastMCP: search_site (owns the retrieval index)
    contact_server.py      FastMCP: send_message → Formspree
  data/
    site/                  committed crawl snapshot (CI-owned)
    embcache/              committed embedding cache (CI-owned)
  requirements.txt
web/
  widget.js                embeddable bubble (product)
  index.html / demo.js / styles.css   behind-the-scenes demo (explainer)
.github/workflows/
  crawl.yml                crawl + precompute embeddings + commit (push + weekly)
  pages.yml                publish web/ to GitHub Pages
render.yaml
```
