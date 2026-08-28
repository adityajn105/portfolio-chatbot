# Portfolio Chatbot

A retrieval-augmented Q&A bot for a personal site, built from scratch and
deployed on a **fully-free** stack:

- **UI** → static, on **GitHub Pages** (`jackgriffin105.github.io/portfolio_chatbot/`)
- **Backend** → a small **FastAPI** service on **Render** (free tier) that loads a
  **crawl snapshot** (produced in CI on push + weekly, not at boot), indexes it,
  and answers via **Gemini**
- **Embeddings** → local **fastembed** (ONNX `BAAI/bge-small-en-v1.5`) — free, no
  API/quota, no torch, so it fits Render's 512 MB
- **LLM** → **Gemini** (`gemini-3.5-flash-lite`), generation only
- **Contact** → a second **MCP** server emails the owner via **Formspree** when a
  visitor wants to get in touch

One backend, two frontends that consume the same `/chat` SSE stream:

- **`web/widget.js`** — an embeddable floating **chat bubble** (Shadow DOM, zero
  deps): the clean *product* you drop into any portfolio. Shows only the answer.
- **`web/index.html` + `demo.js`** — a **behind-the-scenes explainer**: watch the
  ReAct agent think, call the MCP `search_blog`/`send_message` tools, observe the
  results, and answer — plus the indexed-pages panel.

## Architecture

```
CI (on push + weekly)                 Render web service (backend/)
  crawl.yml → commit backend/data/site  FastAPI (parent)
                                          │ on boot: load committed snapshot → CORPUS_DIR
Portfolio / Pages (static)                │ POST /chat drives the ReAct agent
  widget.js / demo.js  ──POST /chat──►    ├─ MCP child: blog_server  (search_blog, fastembed)
       ▲   Server-Sent Events             └─ MCP child: contact_server (send_message → Formspree)
       └────────────────────────────────
```

Crawling runs in CI (`.github/workflows/crawl.yml`) — on every push and weekly —
and commits the snapshot to `backend/data/site`, so boot is fast and doesn't
depend on the live site. The snapshot is **not hand-committed**: CI owns it. If
it's ever missing (e.g. the first deploy before CI has produced one), the app
crawls live once at boot so it still comes up, then reverts to snapshot-based on
the next deploy. The parent never embeds: the blog MCP child owns retrieval
(loads `CORPUS_DIR`, runs fastembed); the parent only parses page metadata for
`/pages` and `/health`.

## Backend

### Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chat` `{question, history?}` | `text/event-stream`: agent events (`thinking`/`model`/`tool_call`/`observation`/`final`). `history` is prior `{role, content}` turns so follow-ups resolve. |
| GET | `/health` | `{ok, pages, chunks, transport, brain}` |
| GET | `/pages` | `[{title, url, chunks}]` for the demo panel |

### Environment variables
| Var | Default | Notes |
|-----|---------|-------|
| `GEMINI_API_KEY` | — | **secret**; generation only |
| `FORMSPREE_ENDPOINT` | `https://formspree.io/adityajn105@gmail.com` | the portfolio contact form's endpoint (legacy email-in-URL form; or `formspree.io/f/<id>`) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | any fastembed model |
| `CRAWL_SITES` | *(empty)* | empty ⇒ no boot crawl, load committed snapshot (refreshed by CI). Set to space-separated site roots to crawl live on boot. |
| `CORPUS_DIR` | `/tmp/site` | crawl → MCP handoff dir |
| `ALLOWED_ORIGINS` | `*` | comma-separated in prod |
| `MCP_TRANSPORT` | `process` | `inprocess` for offline dev (no subprocess) |

### Run locally
```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=...            # get one at aistudio.google.com
export FORMSPREE_ENDPOINT=...        # optional, for the contact tool
export CORPUS_DIR=/tmp/site
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then:
```bash
curl -s localhost:8000/health
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"what is Aditya'\''s email?"}'
```
Offline (no live crawl / no subprocess): `MCP_TRANSPORT=inprocess` — the parent
builds the index itself from the bundled snapshot in `backend/data/site`.

## Deploy

### 1. Backend → Render
Push this repo to GitHub, then on Render: **New → Blueprint**, connect the repo.
It reads `render.yaml` (root dir `backend`, health check `/health`). Set the two
secrets — `GEMINI_API_KEY` and `FORMSPREE_ENDPOINT` — in the dashboard. Note the
`https://<service>.onrender.com` URL.

> Free tier sleeps when idle; the first request after a sleep just wakes the
> service (a few seconds) — there's no boot crawl. The corpus comes from the
> snapshot in `backend/data/site`, which CI (`.github/workflows/crawl.yml`)
> crawls + commits on push and weekly, so wake-up never depends on the live site.
> (First deploy only: if no snapshot exists yet, the app crawls live once to boot.)

### 2. UI → GitHub Pages
Copy `web/*` into `jackgriffin105.github.io/portfolio_chatbot/`, set the Render
URL as `data-api` in `index.html`'s widget tag, push, enable Pages. This publishes
the demo app at `/portfolio_chatbot/` and hosts `widget.js`.

### 3. Widget → any portfolio
Paste before `</body>` (nothing else needed — the bubble self-injects):
```html
<script
  src="https://jackgriffin105.github.io/portfolio_chatbot/widget.js"
  data-api="https://<service>.onrender.com"
  data-title="Chat with Aditya" defer></script>
```

## Layout
```
backend/
  app.py              FastAPI: crawl-on-boot, /chat (SSE), /health, /pages, CORS
  rag.py              chunk → embed (fastembed) → store → retrieve → generate (Gemini)
  agent.py            ReAct loop + GeminiPolicy; wires search_blog + send_message
  crawl.py            sitemap crawler → markdown snapshot
  contact.py          Formspree send (stdlib only), shared by both transports
  mcp_client.py       stdio + in-process MCP clients; tool builders
  mcp_server/
    blog_server.py    FastMCP: search_blog (owns the fastembed index)
    contact_server.py FastMCP: send_message → Formspree
  data/site/          bundled crawl snapshot (offline fallback)
  requirements.txt
web/
  widget.js           embeddable bubble (product)
  index.html/demo.js/styles.css   behind-the-scenes demo (explainer)
render.yaml
```
