---
title: "Using LLMs Efficiently: The Right Model for the Job"
url: https://adityajain.me/blogs/efficient-llm-usage.html
---

# Using LLMs Efficiently: The Right Model for the Job

- LLM
- Cost
- Routing
- Systems

There’s a reflex, once you have an API key to a frontier model, to send everything to it. It’s the
smartest thing in the room — why would you route your autocomplete, your tag-this-ticket job, and your
gnarly multi-step refactor anywhere else? Because most of what you ask an LLM to do is not hard,
and paying frontier prices for easy work is the single most common way teams light money on fire.

Using LLMs efficiently isn’t one clever trick. It’s mostly one decision made well — which model
handles this request — plus a handful of boring levers that shave cost without touching quality. This
post walks through both: how to match a model tier to a task, what the model routers (OpenRouter,
Cursor’s Auto mode) are really doing under the hood, the reuse/context/throughput tricks worth
knowing, and a cautionary tale about optimizing the wrong number.

## Start here: right-size the model to the task

Models come in rough tiers — small/fast, a mid-tier workhorse, and frontier/reasoning
models — and they can differ by 10–50× in price per token. The skill is not “always use the best one.”
It’s knowing that the best one for a job is the cheapest model that clears the quality bar for that
job. A frontier model doing summarization is overqualified; a tiny model doing multi-step math fails
silently and confidently, which is worse than expensive.

Click through the common tasks and the tier I’d reach for first:

Code autocomplete inline suggestions Chat & Q&A general assistant Summarize / rewrite transform text Classify / extract high volume Hard reasoning & math multi-step Agentic coding tools + many turns Long-document analysis big context Creative & marketing copy style over rigor Bulk offline jobs no latency need

Reach for Small / fast

Why It fires on every keystroke and must return in tens of milliseconds. The task is local and low-ambiguity — a small, fast model is not a compromise here, it is the right tool.

Tip Latency and cost per call matter more than raw smarts. Cache aggressively; keep the model close to the user.

Reach for Mid tier

Why Most conversational turns need fluency and broad knowledge, not deep multi-step reasoning. A mid-tier model is the default workhorse — good answers at a fraction of frontier cost.

Tip Escalate to a frontier model only for the turns that actually need it (see routing).

Reach for Small / fast

Why Summarizing, reformatting, and tone changes are shallow transforms — the answer is mostly *in* the input. Small models do this well and cheaply.

Tip Cap the output length. Give one clear example of the format you want.

Reach for Small / fast

Why Narrow, repetitive, high-volume work — routing tickets, tagging, pulling fields from documents. Paying a frontier price per row is money on fire.

Tip A small or fine-tuned model wins; run it through the batch API and use structured output.

Reach for Frontier

Why Long chains where one wrong step derails the whole answer — proofs, tricky analysis, planning. This is exactly what reasoning-tier models are for; a cheap model fails silently and confidently.

Tip Pay for capability here. The expensive answer that is right beats three cheap answers that are wrong.

Reach for Frontier

Why A long tool loop compounds errors — a weak model takes more turns, and every turn re-reads the whole transcript, so a "cheaper" model can cost *more* end to end. Capability buys back turns.

Tip Measure end-to-end task cost, not per-call cost. This is the local-metric trap below.

Reach for Frontier

Why The binding constraint is context window and recall across a lot of text, not raw cleverness. Reach for a model with a large, reliable context — often a frontier or long-context model.

Tip Retrieve the relevant slices first (RAG) instead of stuffing everything — cheaper and often more accurate.

Reach for Mid tier

Why Style, voice, and fluency matter more than provable correctness. Mid-tier models write well and cheaply; frontier models rarely write *enough* better to justify the price.

Tip Spend your budget on iterations and good prompts, not on the biggest model.

Reach for Small / fast

Why Nightly enrichment, backfills, evals — nobody is waiting on the response. Latency is free to trade away, so use the cheapest model that clears your quality bar.

Tip Async batch endpoints are typically ~50% cheaper. If quality wobbles, cascade: cheap first, escalate only the failures.

Pick a task. The tier is a starting point, not a law — green is small/fast, blue is the mid-tier workhorse, red is frontier. Most systems mix all three.

A starting point, not a rulebook. The tiers are relative — “small/mid/frontier”
maps onto whatever model family you’re using. The real move is defaulting low and escalating on
evidence, not defaulting high out of habit.

Two patterns fall out of this table and are worth naming:

- Cascade. Try the cheap model first; run a quick check (does it parse? does it pass a validator?
is the model confident?); only escalate the failures to a bigger model. You pay frontier prices for
the small slice that actually needs them.

- Fine-tune down. For a narrow, high-volume task, fine-tuning or distilling a small model to do
that one thing well often beats prompting a frontier model forever. Higher up-front cost, far lower
marginal cost. (I wrote about the mechanics in Fine-Tuning LLMs.)

## Split one task across tiers

Here’s the move most people miss: a single task isn’t one tier either. A coding change, a research
report, a data pipeline — each has phases with wildly different needs. Planning is a handful of
tokens but decides everything; implementation is the bulk of the tokens but is well-specified once the
plan exists; review is largely mechanical. So don’t pick one model for the whole job — plan with a
frontier model, implement with a mid model, and review with a small one.

Assign a tier to each phase and watch the blended cost against the lazy “all-frontier” default:

A ~2M-token coding task, illustrative blended prices (0.5/0.5 / 0.5/5 / $20 per
million tokens for small/mid/frontier). The frontier model earns its price on the plan; the
implement phase is where the tokens (and the savings) are. Try the “backwards” split to see
how much a cheap planner + frontier implementer wastes.

The principle — spend on the phase that decides the outcome, economize on the phase that has the
volume — shows up all over the place once you look for it:

- Plan → implement. A frontier model writes the spec, the design, the step list; a mid model turns
each well-defined step into code. The hard reasoning happens once, cheaply amortized over the bulk.

- Draft cheap → verify frontier (or the reverse). Generate with a mid model, then have a stronger
model critique rather than rewrite — verification is often shorter (and cheaper) than generation.

- Extract cheap → reason frontier. A small model pulls the structured facts out of messy input;
the frontier model reasons over the compact, clean version. You shrink the expensive model’s context.

- Summarize cheap → answer frontier. A cheap model compresses a long document or chat history into
a tight brief; the frontier model works on the brief, not the raw wall of text.

- Orchestrator → workers. A capable model decomposes and delegates; a swarm of cheap workers each
handles one narrow sub-task in parallel. (More on this shape in
Working with AI Agents.)

- Pay once, run cheap forever. Use a frontier model once to write the prompt, the few-shot
examples, or the golden template — then run a small model against it at scale.

## Model routers: OpenRouter and Cursor’s Auto mode

Doing the right-sizing decision by hand, per request, doesn’t scale. A model router is a layer
that sits in front of many models and picks one for each request — the same idea as the table above,
automated.

- OpenRouter is a unified gateway: one API, one key, hundreds of models
across providers behind a single schema. That alone kills a lot of integration cost and vendor
lock-in — you can swap models with a string change and A/B them on real traffic. On top of that it
does automatic fallbacks (if a provider is down or rate-limits you, the request reroutes) and can
route by price or latency, including an auto option that picks a model for the prompt. The win
is operational: you stop hard-coding a single provider and let a policy choose.

- Cursor’s Auto mode is the same principle aimed at coding. Instead of you
picking a model for every edit, chat, or agent step, the editor selects one for you — leaning on
cheaper/faster models for routine work and stronger ones when the request looks hard, while smoothing
over provider capacity. You feel it as “it’s just fast and usually right,” which is exactly what a
good router should feel like.

Under the hood both are doing the same thing: classify the request, send easy work to a cheap model
and hard work to a capable one. Here’s why that pays — drag the slider to change how hard your
traffic is and watch the three billing strategies diverge:

Illustrative per-query prices (small ≈ 0.0006,mid≈0.0006, mid ≈ 0.0006,mid≈0.006, frontier ≈ $0.03)
over 100k requests/month. Always-cheap is tempting until you see how much traffic it fails;
always-frontier is safe but pays top price for trivial work. Routing captures most of the gap.

Notice what the slider teaches: routing’s payoff depends entirely on your traffic mix. When most
requests are easy, a router saves a fortune. When the work is genuinely hard, it can’t save much — and
that’s the honest answer, not a failure. Don’t route for the sake of routing; route because you
measured that most of your requests are easy.

## The other levers: reuse, context, throughput

Model choice is the biggest knob, but it’s not the only one. A second family of techniques cuts cost
without changing which model you use — and they stack on top of routing. They fall into three groups:
pay for the same tokens only once, send fewer tokens, and shape the request and response.

Prompt caching pay once Semantic response cache pay once Retrieve, don't stuff send less Prune the history send less Compress the prompt send less Batch API shape the I/O Structured output shape the I/O Cap the output shape the I/O Stream the response shape the I/O

Family Reuse

How Cache the static prefix — system prompt, tool definitions, retrieved docs — so repeated calls only pay full price for the *new* tokens.

Saves Cached input is far cheaper and faster. Huge for agents and chat that resend a fixed preamble every turn.

Family Reuse

How Embed or hash the request; if a near-identical one was answered recently, return the stored answer with no model call at all.

Saves 100% on a cache hit. Ideal for FAQs, popular queries, and idempotent lookups.

Family Context

How Use RAG to fetch the handful of relevant chunks instead of pasting the whole knowledge base into every prompt.

Saves Shorter prompts cost less *and* are often more accurate — less irrelevant text for the model to get distracted by.

Family Context

How Summarize or drop old turns instead of resending the entire conversation on every call.

Saves Caps the quadratic blow-up of a long chat — otherwise every turn re-pays for all the turns before it.

Family Context

How Strip boilerplate, dedupe, drop pure noise like line-number prefixes — keep the signal, cut the filler.

Saves A few percent per call, losslessly. But measure end to end — over-compressing backfires (see the local-metric trap below).

Family Throughput

How Submit non-urgent work to an asynchronous batch endpoint instead of firing real-time requests.

Saves Typically around half off the per-token price — the discount you get for not needing an answer *right now*.

Family Throughput

How Constrain responses to a schema (JSON, enum) so the first response parses cleanly instead of triggering a retry.

Saves Kills the retry tax — every malformed answer you *don't* have to regenerate is a call you never pay for.

Family Throughput

How Set max_tokens and stop sequences. Output tokens are usually the pricier half of the bill and the slowest to produce.

Saves A direct cut to your most expensive tokens, and lower latency as a bonus.

Family Throughput

How Stream tokens so the user sees output as it's generated instead of waiting for the whole thing.

Saves No dollars saved — it buys *perceived* speed, so a slightly cheaper/slower model becomes acceptable to users.

Click a lever. Reuse = pay for the same tokens once, Context = send fewer tokens, Throughput = shape the request/response. They stack — and none of them require a bigger model.

None of these require a bigger model — and several (prompt caching, structured
output, batch API) are close to free wins you can turn on today. They compound: a routed, cached,
RAG-trimmed request is cheaper on three axes at once.

The two I’d reach for first, because they’re nearly free:

- Prompt caching. Agents and chat resend a big static preamble — system prompt, tool definitions,
retrieved docs — on every turn. Cache that prefix and you pay full price for it once, then a
fraction on every subsequent call. For a long agent loop this is often the single largest line item.

- Retrieve, don’t stuff. Pasting your whole knowledge base into the prompt is expensive and
hurts accuracy — the model has more irrelevant text to trip over. RAG
fetches just the relevant chunks, so the prompt is shorter and sharper. Cheaper and better is a rare
combination; take it.

## The local metric trap (a cautionary subtopic)

Here’s where efficiency work goes wrong, and it’s subtle enough to deserve its own section. GitHub
wrote up how they made Copilot more cost-efficient without sacrificing task
quality,
and their most instructive finding was a failure: an aggressive tool-output compressor that cut
tokens per call but made whole tasks more expensive.

The mechanism is a trap anyone optimizing an agent can fall into. Compress a tool’s output hard and
each call is smaller — the local metric you’re watching looks great. But if the compression drops
something the agent needed, it has to spend extra recovery turns re-reading the original. And
because an agent re-sends the entire accumulated transcript on every turn, each extra turn is
punishingly expensive. Past a point, the per-call savings are dwarfed by the cost of the turns they
caused. As GitHub put it: you saved tokens locally and spent more globally.

Drag the compression slider and watch the two bars move in opposite directions:

Per-call tokens (top) fall steadily as you compress harder. Total task tokens
(bottom) dip to a sweet spot — trimming obvious noise — then climb past the baseline as dropped
context forces recovery turns. Illustrative model; the shape is the point.

The lessons generalize well beyond compression:

- Optimize the task, not the call. Cost per API call is a local metric. The number that pays your
bill is cost per completed task. They can point in opposite directions — measure the one that
matters, end to end.

- Lossless first. Deleting pure noise (boilerplate, duplicate context, line-number prefixes) is
safe and stacks cleanly. Lossy compression that gambles with what the model needs is where you get
burned — validate that behavior didn’t change before you ship it.

- Re-measure across surfaces. A change that helps offline can hurt in production, and vice versa.
An optimization isn’t done when the per-call number drops; it’s done when the task-level number
drops and stays down. This is the same discipline as evaluating LLM
apps — you can’t improve what you don’t measure honestly.

## Key takeaways

- Default low, escalate on evidence. Most requests are easy. Send them to a small or mid-tier
model and reserve the frontier model for work that provably needs it.

- Split one task across tiers. Plan with a frontier model, implement the well-specified bulk with
a mid model, review with a small one. Spend on the phase that decides the outcome; economize on the
phase that has the volume.

- A router is that decision, automated. OpenRouter (unified gateway, fallbacks, price/latency
routing) and Cursor’s Auto mode (pick-the-model-for-the-edit) both do one thing: match request
difficulty to model capability. Their payoff scales with how much of your traffic is easy.

- Cache the prefix, retrieve don’t stuff, cap the output, batch the non-urgent. These cut cost
without changing the model and stack on top of routing — several are nearly free.

- Structured output kills the retry tax, and streaming buys perceived speed so a slightly cheaper
model becomes acceptable.

- Optimize the completed task, not the individual call. The local metric trap is real: per-call
savings that cause extra turns can make the whole job cost more. Measure end to end.

## Go deeper

- From Query to Next Token: How LLM Inference Gets Fast — why decode is memory-bound, and the serving-side tricks (quantization, speculative decoding, MoE) behind the tiers.

- RAG from Scratch — retrieve-don’t-stuff, built up from first principles.

- Fine-Tuning LLMs — how to fine-tune or distill a small model so it can do a narrow job cheaply.

- Evaluating LLM Apps — how to measure task-level quality so your cost cuts don’t quietly regress it.

- Working with AI Agents — where the transcript-re-read cost and the local-metric trap bite hardest.

- OpenRouter — openrouter.ai · a unified API and router across many model providers.

- GitHub Engineering — How we make AI coding more cost-efficient — the source of the local-metric trap story.

The frontier model isn’t the answer to every prompt — it’s the answer to the prompts that need it.
Efficiency is mostly the discipline of noticing which ones those are, automating that judgment, and
measuring the number that actually pays your bill.
