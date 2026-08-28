"""Phase 2 — a ReAct agent, from scratch, plus agentic RAG.

ReAct = **Reason + Act**. Prompt the model to think out loud and, when it wants
information, emit an *action* instead of guessing. We run the action, paste the
result back as an *observation*, and let it think again. Reason → Act → Observe,
in a loop, until it declares a Final Answer. An "agent" is a `while` loop around
an LLM plus a text protocol for calling tools — this file is that loop.

    Question → Thought → (Action: search_blog) → Observation → Final Answer
                      └── or, if it already knows ──→ Final Answer

Sections:
  1. Tools          — the things an agent can *do* (search_blog, calculator)
  2. ReAct loop     — parse, dispatch, observe; streamed as Events
  3. Policies       — the pluggable "brain" (scripted / OpenAI / local SLM)
  4. Agentic RAG    — assemble an agent that decides whether to even retrieve
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(__file__))

from rag import RAG  # noqa: E402


# ===========================================================================
# 1. Tools — the things an agent can *do*
# ---------------------------------------------------------------------------
# A tool is just a name, a description (the agent reads this to decide when to
# use it), and a function from a string input to a string observation. The
# agent never calls Python directly; it emits `Action: search_blog / Action
# Input: what is PPO?` and we dispatch to the matching tool. Keeping tools this
# dumb is what makes the agent loop simple.
# ===========================================================================

@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]


def _first_sentence(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    for end in (". ", "? ", "! "):
        i = text.find(end)
        if 0 < i < limit:
            return text[: i + 1]
    return text[:limit] + ("…" if len(text) > limit else "")


def make_search_tool(rag: Any, k: int = 3, min_score: float = 0.25) -> Tool:
    """Wrap the Phase-1 RAG retriever as a tool the agent can call.

    Each result shows its relevance score, and when the best match is weak
    (< min_score) the observation says so — that's the signal the agent uses to
    decide whether to rephrase its query and search again."""
    def _search(query: str) -> str:
        hits = rag.query(query, k=k)
        if not hits:
            return "No results found. Try a different query."
        body = " || ".join(
            f"[{h.meta['source']} · relevance {h.score:.2f}] {_first_sentence(h.meta['text'])}"
            for h in hits
        )
        if hits[0].score < min_score:
            body += (f"  [NOTE: top relevance is only {hits[0].score:.2f} — these "
                     "may not answer the question; consider rephrasing the query "
                     "and searching again.]")
        return body
    return Tool(
        name="search_blog",
        description="Search Aditya's website (blog posts, projects, about/contact) "
                    "for a topic. Input: a short search query. Returns the most "
                    "relevant passages, each with a relevance score. If results are "
                    "weak, rephrase and search again.",
        run=_search,
    )


# --- a safe arithmetic evaluator (no eval(); only math on literals) ---------
import ast          # noqa: E402
import operator     # noqa: E402

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def _calc(expr: str) -> str:
    try:
        return str(_eval_node(ast.parse(expr, mode="eval").body))
    except Exception:
        return f"Could not evaluate {expr!r}."


calculator = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression. Input: e.g. '2 * (3 + 4)'.",
    run=_calc,
)


def render_tools(tools: list[Tool]) -> str:
    """The tool menu the agent sees in its prompt."""
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


# ===========================================================================
# 2. The ReAct loop
# ---------------------------------------------------------------------------
# Two prompts: the strict one forces tool use; AGENTIC_RAG_PROMPT lets the model
# answer directly OR retrieve when unsure (the "try the LLM, fall back to RAG"
# behaviour, driven entirely by the prompt).
# ===========================================================================

PROMPT_TEMPLATE = """\
Answer the question using ONLY the tools below. Work in this exact format:

Thought: what you're reasoning about
Action: the tool to use, one of [{tool_names}]
Action Input: the input to the tool
Observation: (the tool's result — this is filled in for you)
... (repeat Thought/Action/Action Input/Observation as needed) ...
Thought: I now know the answer
Final Answer: the answer, citing sources in [brackets]

Tools:
{tools}

Question: {question}
{scratchpad}"""


AGENTIC_RAG_PROMPT = """\
You are answering questions about Aditya. You have your own general knowledge AND
a tool that searches Aditya's website — his blog posts, projects, and about/
contact pages (bio, email, location, experience). Work in this exact format:

Thought: what you're reasoning about
Action: a tool from [{tool_names}]        (optional — omit if you can answer directly)
Action Input: the input to the tool
Observation: (the tool's result — filled in for you)
... (repeat Thought/Action/Action Input/Observation as needed) ...
Thought: I now know the answer
Final Answer: the answer (cite sources in [brackets] when you used the tool)

Rules:
- ANY question about Aditya himself — his email, contact info, background,
  education, location, job/employer, experience, dates, projects, or the
  content/wording/opinions in his writing — MUST be answered from search_blog.
  You do NOT know these facts on your own; you have NO reliable prior knowledge
  about this specific person. Always call search_blog first for these, then
  answer only from the results and cite sources. Never answer a question about
  Aditya directly from memory, even if you feel confident — your guess will be
  about the wrong person.
- Answer directly (no Action) ONLY for general/tutorial concepts that are not
  about Aditya — e.g. "what is PPO?", "how does attention work?". If the
  question mentions Aditya, his site, or "you/your" (the visitor means Aditya),
  it is NOT general knowledge — search first.
- Each search result has a relevance score. If the results are weak (low score
  or they don't actually address the question), DON'T answer from them — instead
  rephrase your query with different keywords and call search_blog again. Use at
  most 2 searches. If after searching you still can't find the specific detail,
  say you couldn't find it — do NOT fall back to a guess from memory.
- Quote specific details (emails, names, dates, numbers) EXACTLY as they appear
  in the search results. Never invent or guess them. If a detail isn't in the
  results, say you couldn't find it. In particular, Aditya's only email is the
  one that appears in the search results — never output any other address.
- Use send_message ONLY when the visitor explicitly wants to contact Aditya or
  leave him a question/message AND has given their own email address. First
  confirm what you'll send, then call it with their message and email. NEVER call
  send_message for an ordinary question you can answer or search for — it emails
  a real person.

Tools:
{tools}

Question: {question}
{scratchpad}"""


@dataclass
class Step:
    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class Result:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped: str = "final_answer"   # or "max_steps"


@dataclass
class Event:
    """A live signal emitted as the agent works, so a UI can narrate it.

    kind is one of:
      "thinking"    — about to call the model; data has {prompt, brain}
      "model"       — model replied;            data has {text}
      "tool_call"   — dispatching a tool;       data has {tool, input}
      "observation" — tool returned;            data has {tool, output}
      "final"       — done;                     data has {answer, result}
    """
    kind: str
    data: dict


# Policy = a function that takes the full prompt and returns the model's next
# chunk of text (up to the next Observation, or the Final Answer).
Policy = Callable[[str], str]


def _parse(text: str) -> Step:
    """Pull the Thought / Action / Action Input (or Final Answer) out of a
    model turn. Lenient on whitespace and casing, like real parsers must be."""
    thought = _grab(text, r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer)\s*:|$)")
    final = _grab(text, r"Final Answer:\s*(.*)")
    if final is not None:
        return Step(thought=thought or "", action="__final__", action_input=final)
    action = _grab(text, r"Action:\s*(.*?)(?=\n|$)")
    action_input = _grab(text, r"Action Input:\s*(.*?)(?=\nObservation:|$)")
    return Step(thought=thought or "", action=(action or "").strip() or None,
                action_input=(action_input or "").strip())


def _grab(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


class ReActAgent:
    def __init__(self, policy: Policy, tools: list[Tool], max_steps: int = 6,
                 prompt_template: str = PROMPT_TEMPLATE,
                 fallback_tool: str | None = None) -> None:
        self.policy = policy
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.prompt_template = prompt_template
        # Safety net: if a turn parses to neither a valid action nor a Final
        # Answer (small models sometimes mangle the format), run this tool with
        # the question instead of looping. Keeps flaky SLMs from hallucinating.
        self.fallback_tool = fallback_tool
        # a human-readable label for the "brain", set by the caller (build_agent)
        self.brain_name = type(policy).__name__

    def run_iter(self, question: str):
        """Drive the loop, yielding an Event at each stage so a UI can show the
        agent's reasoning in real time. The final Event carries the Result.
        `run()` is just this generator with the events thrown away."""
        steps: list[Step] = []
        scratchpad = ""
        for _ in range(self.max_steps):
            prompt = self.prompt_template.format(
                tool_names=", ".join(self.tools),
                tools=render_tools(list(self.tools.values())),
                question=question,
                scratchpad=scratchpad,
            )
            # announce BEFORE the (possibly slow) model call, so "thinking…"
            # shows while the model generates.
            yield Event("thinking", {"prompt": prompt, "brain": self.brain_name})
            raw = self.policy(prompt)
            yield Event("model", {"text": raw})

            step = _parse(raw)
            if step.action == "__final__":
                steps.append(Step(thought=step.thought))
                result = Result(answer=step.action_input or "", steps=steps)
                yield Event("final", {"answer": result.answer, "result": result})
                return

            # no clean action and no final answer → fall back to the safety-net
            # tool (searching for the question) rather than spinning.
            if step.action not in self.tools and self.fallback_tool in self.tools:
                step.action = self.fallback_tool
                step.action_input = step.action_input or question

            # dispatch the tool
            tool = self.tools.get(step.action or "")
            yield Event("tool_call", {"tool": step.action, "input": step.action_input})
            step.observation = (
                tool.run(step.action_input or "") if tool
                else f"Unknown tool {step.action!r}. Available: {', '.join(self.tools)}."
            )
            yield Event("observation", {"tool": step.action, "output": step.observation})
            steps.append(step)
            scratchpad += (
                f"Thought: {step.thought}\n"
                f"Action: {step.action}\n"
                f"Action Input: {step.action_input}\n"
                f"Observation: {step.observation}\n"
            )

        result = Result(answer="(stopped: reached step limit without an answer)",
                        steps=steps, stopped="max_steps")
        yield Event("final", {"answer": result.answer, "result": result})

    def run(self, question: str) -> Result:
        result = Result(answer="")
        for ev in self.run_iter(question):
            if ev.kind == "final":
                result = ev.data["result"]
        return result


# ===========================================================================
# 3. Policies — the pluggable "brain"
# ===========================================================================

class ScriptedPolicy:
    """Deterministic policy for tests/demos: returns canned turns in order.
    No API key, no network — proves the loop, parsing, and dispatch work."""

    def __init__(self, turns: list[str]) -> None:
        self.turns = list(turns)
        self.i = 0

    def __call__(self, prompt: str) -> str:
        turn = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return turn


class OpenAIPolicy:
    """Real policy: an OpenAI chat model that stops before writing its own
    Observation (so we, not the model, supply tool results)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stop=["\nObservation:"],   # hand control back after the action
        )
        return resp.choices[0].message.content or ""


class GeminiPolicy:
    """The deploy brain: a Gemini model driving the ReAct loop. Same contract as
    OpenAIPolicy — it stops before writing its own Observation (via a
    stop_sequence) so *we* supply the tool result. Generation only; embeddings
    are local, so the free tier is spent on reasoning, not on indexing."""

    def __init__(self, model: str | None = None) -> None:
        from google import genai
        from google.genai import types
        self._types = types
        # a SHORT per-attempt timeout (ms): most calls return in ~1s, but the API
        # occasionally stalls for 30s+. We'd rather abandon a stalled attempt fast
        # and retry (the retry almost always connects immediately) than make the
        # visitor wait — and a call with no timeout would hold the /chat lock and
        # block every later request.
        # 20s clears legitimately-slow calls (the tail runs ~7-8s) while still
        # cutting off a true hang; retry recovers a transient stall. (API minimum
        # deadline is 10s, so don't go below that.)
        self.timeout_ms = max(10, int(float(os.environ.get("GEMINI_TIMEOUT", "20")))) * 1000
        self.retries = int(os.environ.get("GEMINI_RETRIES", "2"))
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=self.timeout_ms))
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def __call__(self, prompt: str) -> str:
        cfg = self._types.GenerateContentConfig(
            temperature=0.1,
            stop_sequences=["\nObservation:"],  # hand control back after the action
        )
        last_exc = None
        for attempt in range(self.retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg)
                text = resp.text or ""
                if text.strip():
                    return text
                # empty (e.g. MALFORMED_RESPONSE): retry, else let the loop fall back
            except Exception as exc:            # timeouts, transient 429/503, resets
                last_exc = exc
            if attempt < self.retries - 1:
                time.sleep(0.4 * (attempt + 1))
        if last_exc is not None:                # every attempt errored → surface it
            raise last_exc
        return ""                                # all empty → loop's search fallback


class SLMPolicy:
    """A small *local* instruct model (default Qwen2.5-1.5B-Instruct) driving the
    loop — real reasoning, offline, no API key. Needs `transformers` + `torch`.

    Like OpenAIPolicy, it must stop before writing its own Observation so we
    supply the tool result. Transformers has no server-side stop string, so we
    generate then truncate at the first "\\nObservation:"."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 256) -> None:
        import torch  # lazy: heavy import
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        # CPU by default (works everywhere, incl. the free HF Space); use Apple
        # MPS if present. No device_map, so we don't need the accelerate package.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto").to(self.device)
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        # Wrap our ReAct prompt in the model's chat format.
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        # decode only the newly generated tokens
        gen = out[0][inputs["input_ids"].shape[1]:]
        reply = self.tokenizer.decode(gen, skip_special_tokens=True)
        # emulate the stop string: hand control back after the action
        return reply.split("\nObservation:")[0]


# ===========================================================================
# 4. Agentic RAG — let the model decide whether it even needs to retrieve
# ---------------------------------------------------------------------------
# Phase 1 always retrieved, then answered. That's wasteful for a question the
# model already knows and can't cite for one it doesn't. Here we hand the model
# a search_blog tool and the AGENTIC_RAG_PROMPT: answer directly if confident;
# otherwise search first, then answer. The "brain" is auto-selected:
#   * OPENAI_API_KEY set        → OpenAIPolicy (gpt-4o-mini), most reliable
#   * else if transformers here → SLMPolicy (local Qwen2.5-1.5B), offline & free
#   * else                      → a ScriptedPolicy so the pipeline still runs
# ===========================================================================

# Fallback script used only when there's no OpenAI key and no local model: it
# still exercises the retrieve-then-answer path so the demo never dead-ends.
_FALLBACK_TURNS = [
    "Thought: This asks about the blog's content, so I should search it.\n"
    "Action: search_blog\n"
    "Action Input: {q}",
    "Thought: The passages answer it.\n"
    "Final Answer: Based on the blog: {obs}",
]


def choose_policy(verbose: bool = True):
    """Pick the best available brain for the agent. Returns (policy, label).

    Preference order matches the deploy target: Gemini (free-tier generation) →
    OpenAI → local SLM → scripted fallback."""
    if os.environ.get("GEMINI_API_KEY"):
        model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        label = f"Gemini {model}"
        if verbose:
            print(f"[agentic-rag] brain: {label}")
        return GeminiPolicy(), label
    if os.environ.get("OPENAI_API_KEY"):
        label = "OpenAI gpt-4o-mini"
        if verbose:
            print(f"[agentic-rag] brain: {label}")
        return OpenAIPolicy(), label
    try:
        import transformers  # noqa: F401
        label = "Qwen2.5-1.5B-Instruct (local SLM)"
        if verbose:
            print(f"[agentic-rag] brain: {label} — first run downloads it")
        return SLMPolicy(), label
    except Exception:
        label = "scripted fallback"
        if verbose:
            print(f"[agentic-rag] brain: {label} (no key, no transformers)")
        return None, label  # signal: caller builds a ScriptedAgent instead


def _mcp_command(server_file: str = "blog_server.py") -> list[str]:
    """Command that launches a FastMCP server as a separate process.

    Defaults the interpreter to `sys.executable` — on Render the whole app runs
    in one venv, so the children reuse it (no `.venv-mcp`). `MCP_PYTHON` still
    overrides for the local split-runtime setup (a 3.12 server venv). `server_file`
    picks which server under mcp_server/ to launch (blog vs contact)."""
    here = os.path.dirname(os.path.realpath(__file__))
    py = os.environ.get("MCP_PYTHON") or sys.executable
    server = os.path.join(here, "mcp_server", server_file)
    return [py, server]


def build_agent(rag: RAG, policy=None, max_steps: int = 5, use_mcp: bool = False,
                mcp_transport: str = "inprocess") -> ReActAgent:
    """Assemble the agentic-RAG agent. If policy is None, auto-select one.

    Wires TWO tools: `search_blog` (retrieval) and `send_message` (contact the
    owner by email). use_mcp=True routes both through MCP instead of calling the
    handlers directly. mcp_transport picks how:
      "inprocess" — in-memory MCP clients (no subprocess); deploys anywhere.
      "process"   — real FastMCP server subprocesses over stdio (one per server).
    Either way the agent emits the same `Action: search_blog` / `send_message`."""
    if use_mcp and mcp_transport == "process":
        from mcp_client import (MCPStdioClient, make_mcp_search_tool,
                                make_mcp_contact_tool)
        search_tool = make_mcp_search_tool(
            MCPStdioClient(_mcp_command("blog_server.py")), k=3)
        contact_tool = make_mcp_contact_tool(
            MCPStdioClient(_mcp_command("contact_server.py")))
    elif use_mcp:
        from mcp_client import (InProcessMCPClient, InProcessContactMCPClient,
                                make_mcp_search_tool, make_mcp_contact_tool)
        search_tool = make_mcp_search_tool(InProcessMCPClient(rag), k=3)
        contact_tool = make_mcp_contact_tool(InProcessContactMCPClient())
    else:
        from mcp_client import make_contact_tool
        search_tool = make_search_tool(rag, k=3)
        contact_tool = make_contact_tool()
    tools = [search_tool, contact_tool]
    label = None
    if policy is None:
        policy, label = choose_policy()
    if policy is None:  # scripted fallback: pre-run the search so the script can quote it
        agent = _ScriptedAgent(rag, search_tool)
    else:
        agent = ReActAgent(policy, tools, max_steps=max_steps,
                           prompt_template=AGENTIC_RAG_PROMPT,
                           fallback_tool="search_blog")
    if label:
        agent.brain_name = label
    return agent


class _ScriptedAgent(ReActAgent):
    """No-LLM stand-in that always retrieves then answers, so the offline demo
    still shows the loop end to end. Not real reasoning — a deterministic prop."""

    def __init__(self, rag: RAG, tool) -> None:
        self._rag = rag
        super().__init__(ScriptedPolicy([]), [tool], max_steps=3,
                         prompt_template=AGENTIC_RAG_PROMPT,
                         fallback_tool="search_blog")
        self.brain_name = "scripted fallback"

    def run_iter(self, question: str):
        # set up the canned turns for THIS question, then drive the normal loop
        # (so streaming events work identically to a real brain).
        hits = self._rag.query(question, k=3)
        obs = "; ".join(f"[{h.meta['source']}] {h.meta['heading']}" for h in hits[:2]) \
            or "nothing relevant"
        turns = [t.format(q=question, obs=obs) for t in _FALLBACK_TURNS]
        self.policy = ScriptedPolicy(turns)
        yield from super().run_iter(question)
