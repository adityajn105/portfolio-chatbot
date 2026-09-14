---
title: "From Query to Next Token: How LLM Inference Gets Fast"
url: https://adityajain.me/blogs/llm-inference-optimization.html
---

# From Query to Next Token: How LLM Inference Gets Fast

- LLM
- Inference
- Performance
- Systems

You paste a wall of text — a whole document, a long chat history, a million tokens — hit enter, and
the reply starts streaming back in under a second. A 70-billion-parameter model just read all of
that and began composing. How is that even possible?

Here’s the reframing that makes the whole thing click: generating a token is not deep thought — it’s
mostly a memory read. To produce one token, the GPU has to haul the model’s weights out of memory
and multiply them by one vector. The arithmetic is almost incidental; the moving of bytes is the
whole cost. Once you see inference as a bandwidth problem, every optimization in this post falls into
one of four buckets: read fewer bytes, share the read, skip the work, or split it across chips.

Let’s trace a single token from your query to the model’s mouth, then layer on the tricks that take
it from “technically works” to “streams faster than you can read.”

## The journey: query → next token

Strip away the systems tricks and one token comes out of this pipeline:

- Tokenize. Your text is chopped into tokens (subword pieces) and mapped to integer IDs.

- Embed. Each ID becomes a vector; positions are encoded so order matters.

- Forward pass. The vectors flow through every transformer layer — attention mixes information
across tokens, the feed-forward blocks transform each position. (If you want the mechanics,
I built attention from scratch here.)

- Logits → probabilities. The final layer produces one score per vocabulary token; a softmax
turns those into a distribution.

- Sample. Pick a token from that distribution (greedy, temperature, top-p — the sampling
knobs).

- Detokenize & repeat. Append the new token to the sequence and go back to step 3 for the next
one.

Once per prompt Repeats per output token ↺

- 1 Tokenize text → token IDs
- →
- 2 Embed IDs → vectors (+ position)
- →
- 3 Forward pass through every layer bottleneck
- →
- 4 Logits → softmax scores → probabilities
- →
- 5 Sample pick the next token
- →
- 6 Detokenize token → text

↺  append token → back to the forward pass, once per generated token

Steps 1–2 run once on your prompt. Steps 3–6 are the decode
loop — they run again for every single output token, each pass dominated by the forward
pass through every layer. That inner loop is where all the cost (and all the optimization) lives.

That loop runs once per output token. A 500-token answer means 500 trips through the entire
network. Which raises the real question: what does one trip actually cost?

## Prefill vs decode: two very different phases

The single most useful distinction in inference is that step 3 happens in two regimes:

- Prefill — processing your prompt. Every prompt token goes through the network in parallel,
in one big matrix multiply. This phase is compute-bound: the GPU’s math units are the
bottleneck, and they’re busy. This is why a million-token prompt is expensive but not linear-slow —
it’s one enormous parallel pass, and GPUs love parallel.

- Decode — generating the answer, one token at a time. Each new token depends on the previous one,
so it’s inherently sequential. And here’s the twist: generating a single token barely uses the
math units at all. It’s memory-bandwidth-bound — dominated by the time to read the weights out
of HBM.

That asymmetry organizes everything. Prefill wants FLOPs; decode wants bandwidth. Time-to-first-token
is a prefill problem; tokens-per-second after that is a decode problem. Almost every optimization
below targets one phase or the other.

## The memory-bandwidth truth

Why is decode memory-bound? Because to generate one token, the GPU reads every weight in the
model exactly once and does a tiny amount of math with each. There’s no reuse — one token in, one
matrix-vector product per layer, one token out. The time is essentially:
ttoken≈bytes read from memorymemory bandwidtht_{\text{token}} \approx \frac{\text{bytes read from memory}}{\text{memory bandwidth}}ttoken​≈memory bandwidthbytes read from memory​

Plug in real numbers. A 70B model in FP16 is ~140 GB of weights. An H100 moves ~3.35 TB/s. So a
lower bound on per-token latency is 140/3350≈42 ms140 / 3350 \approx 42\text{ ms}140/3350≈42 ms — about 24 tokens/second,
purely from reading the weights. The model isn’t thinking for 42 ms; it’s waiting on the memory
bus. Play with it:

Per-token latency is dominated by bytes ÷ bandwidth. Drop the precision
(FP16 → INT4) and latency falls because there are fewer bytes to read. Crank the batch size
and throughput climbs while per-user latency barely moves — the weight read is shared. Push the
context length toward a million and watch the KV cache overtake the weights entirely. Numbers
are an illustrative bandwidth-bound lower bound (they ignore compute and overhead).

Two things fall out of that widget, and they answer the opening question:

- At batch 1, you pay almost entirely to read the weights. A million-token context doesn’t make
the weight read any bigger — the weights are the weights. That’s why a huge context can still stream
fast: the per-token cost is anchored to model size, not prompt length.

- …but the KV cache scales with context, and at long context it can dwarf the weights. That’s
the real cost of a million tokens — not compute, memory. Which is exactly what the next section
is about.

Every knob from here on is a way to shrink the numerator (read fewer bytes), share it (batching), or
route around it (parallelism).

## The KV cache: don’t recompute the past

The naive decode loop has a catastrophic inefficiency. To generate token 1,000, attention needs the
keys and values (K, V) of all 999 previous tokens. Recompute them from scratch every step and the work
per token grows with position — total work is quadratic, O(n2)O(n^2)O(n2). Generation would grind to a
halt after a few hundred tokens.

The fix is the KV cache: compute each token’s K and V once, store them, and reuse them on
every later step. Now each step only computes K, V for the one new token and reads the rest from
cache. Per-token work drops to O(1)O(1)O(1); total to O(n)O(n)O(n). Toggle it and step through decode:

Cache OFF: every new token re-derives K,V for the whole sequence
(amber) — the per-step cost keeps climbing, so total work is quadratic. Cache ON: only the
newest token is computed (green), the rest are reused — linear work. But watch the cache strip grow:
you traded compute for memory. That growing box, times every layer and every sequence in the
batch, is what makes long context expensive.

The KV cache is non-negotiable — every real system uses it — but it creates the problem the rest of
this section attacks. Its size is roughly:
KV bytes=2×nlayers×dmodel×ntokens×bytes×batch\text{KV bytes} = 2 \times n_{\text{layers}} \times d_{\text{model}} \times n_{\text{tokens}} \times \text{bytes} \times \text{batch}KV bytes=2×nlayers​×dmodel​×ntokens​×bytes×batch

For a 70B model at a million tokens, that’s terabytes — far more than the weights. Two techniques
tame it:

GQA / MQA (Grouped / Multi-Query Attention). Standard attention stores K,V for every attention
head. GQA lets several query heads share one K,V head; MQA takes it to the extreme of a single
shared K,V. The cache shrinks by the sharing factor (often 4–8×) for a negligible quality cost, and
it’s baked in at training time. This is why modern models are almost all GQA.

But if heads share K,V, how do they still specialize? Because only the keys and values are
shared — every head keeps its own query projection. A head’s pattern is softmax(Qₕ·Kᵀ), so
distinct per-head queries interrogate the same keys differently and produce different attention
distributions (and different weighted sums of the shared V). The specialization lives in the
question each head asks, not in the shared shelves — what you give up is head-specific K,V
subspaces, which turned out to be mostly redundant anyway.

PagedAttention (vLLM). Instead of one contiguous KV allocation per sequence — which fragments
memory and forces you to over-reserve for the max length — store the cache in fixed-size pages,
exactly like OS virtual memory. Sequences grow page by page, memory packs tightly, and identical
prefixes (a shared system prompt) can even share pages. It’s the trick that lets a server pack far
more concurrent requests into the same GPU.

Seq A Seq B shared page free reserved / wasted

#### Without paging — one contiguous slab per sequence

Seq A

Seq B

Each sequence grabs one big block sized for the max length up
front. The unused slots sit reserved and idle — internal fragmentation. You run out
of memory long before it's actually full, so fewer requests fit on the GPU.

#### With PagedAttention — fixed-size pages + a block table

Seq A block table

L0 → #3 ⤳ L1 → #7 L2 → #1

Seq B block table

L0 → #3 ⤳ L1 → #5 Physical KV pool (HBM)

#0 · #1 A #2 · #3 A+B #4 · #5 B #6 · #7 A

KV lives in fixed-size pages scattered anywhere in the pool,
indexed by a per-sequence block table — exactly like OS virtual→physical memory.
Allocate only what's used, grow one page at a time, and let an identical prefix
(block #3) be shared by both sequences instead of copied.

Left: the naive scheme reserves a contiguous slab per sequence sized for the
longest it might get — most of it sits idle. Right: PagedAttention keeps a per-sequence
block table mapping logical positions to fixed-size physical pages scattered across the pool,
so nothing is over-reserved and a shared prefix (block #3) is stored once.

## But isn’t attention O(n2)O(n^2)O(n2)?

Fair question — and yes, it is. Every token attends to every other token, so the raw operation is a
genuine n×nn \times nn×n matrix: score every pair with QK⊤QK^\topQK⊤, softmax each row, then multiply by VVV.
A 10,000-token prompt means 100 million attention scores. The KV cache handles decode (each new
token is one query against nnn cached keys — an O(n)O(n)O(n) matrix-vector product, no matrix rebuilt), but
in prefill that full n×nn \times nn×n matrix is real. So why doesn’t it dominate?

Because n2n^2n2 operations is not the same as n2n^2n2 time. That QK⊤QK^\topQK⊤ is a dense matrix
multiply — the single thing a GPU is most brutally good at. Tensor cores grind tens of thousands of
multiply-accumulates in parallel every clock, and prefill is the compute-bound phase where those
math units are exactly the point. A 100-million-entry matmul is a few milliseconds of work. GPUs don’t
fear FLOPs; they fear waiting on memory.

And memory is what the naive version got wrong. Done literally, attention writes that whole
n×nn \times nn×n score matrix out to HBM and reads it back for the softmax — and that traffic grows
quadratically and murders bandwidth long before the arithmetic does. FlashAttention is the fix: it
tiles Q, K, and V into blocks, streams them through the chip’s fast on-chip SRAM, and computes the
softmax online (keeping a running max and sum) so the full matrix is never written to HBM at
all. Identical result, exact attention — but memory traffic drops from O(n2)O(n^2)O(n2) back to O(n)O(n)O(n). That
single kernel is most of why long-context prefill went from painful to routine.

For the truly enormous contexts — that million-token prompt — even a perfectly tiled n2n^2n2 matmul
eventually wins, so long-context models lean on sub-quadratic attention: sliding-window / local
attention (each token sees only a nearby band), sparse patterns, or linear-attention approximations
that trade a little fidelity to get scaling back near-linear.

## Serving many requests: batching amortizes the read

Here’s the lever that makes inference economical. Remember that at batch 1, you spend ~42 ms reading
140 GB of weights to produce one token. But those weights, once loaded, can serve many
sequences at once — the weight read is a fixed cost you can share.

Run 32 requests together and you read the weights once and produce 32 tokens from that read.
Per-token latency for each user barely changes; throughput goes up ~32×. This is why the calculator
above shows throughput climbing with batch size while per-user latency stays flat — until the KV cache
(which is per sequence, so it does scale with batch) starts to dominate the read.

The catch: a static batch stalls. If you wait to assemble a fixed batch, fast requests are held
hostage by slow ones, and finished sequences leave GPUs idle. Continuous (in-flight) batching
fixes this: the scheduler adds and evicts sequences from the running batch every single step, so a
GPU is never idle waiting for a batch to drain and new requests join immediately. Combined with
chunked prefill (interleaving prompt-processing with ongoing decode so a big new prompt doesn’t
freeze everyone else), it’s the backbone of modern serving throughput — and a big reason the chatbot
on this site can feel instant without a dedicated GPU per
user.

## Fewer FLOPs and bytes per token

Batching shares the weight read; the next family of tricks makes each token cost less to begin with.

- Speculative decoding. The clever one. A small, fast draft model guesses the next few tokens;
the big model then verifies all of them in a single parallel pass. How does one pass check k
tokens? Because a causal transformer emits a next-token prediction at every position, not just
the last — so feeding it the whole draft [context, d₁, d₂, d₃] is one prefill-style forward
(weights read once) that hands back the model’s own distribution at each drafted position at the
same time. It accepts the longest matching prefix, resamples the first mismatch from a corrected
distribution, and continues. When the draft is right — which is often, since most tokens are easy —
you get several tokens for roughly the price of one big-model step (one weight read to check k
tokens instead of k reads to make them). And it’s lossless: that correction step makes the
output distribution provably identical to running the big model alone.

- Quantization. Store the weights in 8-bit or 4-bit instead of 16-bit. Fewer bytes per parameter
means fewer bytes to read per token — straight bandwidth savings, plus a smaller memory footprint.
I covered the memory math and the accuracy tradeoff in the fine-tuning
post; the inference takeaway is simply that INT4 weights read in
~a quarter the time of FP16 (try it in the calculator above).

- Knowledge distillation. Train a small student model to imitate a big teacher, then serve
the student. Fewer parameters means fewer bytes and FLOPs per token across the board. You’re
trading a capability ceiling for speed — a genuinely different (smaller) model, not a serving toggle.

- Mixture of Experts (MoE). An architecture-level trick: the model has many “expert”
sub-networks, but a router sends each token to only its top-k (often 2) of them. So a model with
a trillion total parameters might activate only tens of billions per token — you get the
capacity of a huge model at the per-token compute of a small one. The headline number is the
activation ratio = active ÷ total params; modern MoEs run low — GLM-4.x activates only
~3.5–5.9% of its weights per token. The catch is memory: the ratio shrinks compute, not footprint
— all experts must stay resident in HBM. Why not just keep the 2 active ones? Because the
router picks different experts for every token ("the" → experts 17, "mitochondria" →
42), so across a batch nearly all of them get used — and they’re specialists, not redundant
copies, so dropping the rest would delete most of the model’s knowledge.

## When the model doesn’t fit on one GPU

A 70B model in FP16 needs ~140 GB — more than a single 80 GB GPU holds. Bigger models need many GPUs,
and there are two fundamentally different ways to split them:

- Tensor parallelism — split within each layer. Every weight matrix is sharded across GPUs;
each GPU computes its slice of every layer, and they all-reduce to combine results at each step.
This cuts per-token latency (more hands on the same work) but demands a fast interconnect
(NVLink) because the GPUs synchronize on every layer. Kept within a node, as a rule.

- Pipeline parallelism — split across layers. GPU 0 holds layers 1–20, GPU 1 holds 21–40, and
so on; activations flow down the pipeline. Communication is light (only at stage boundaries), so it
scales across nodes — but a naive pipeline has bubbles: while GPU 0 works on the first token, the
others sit idle. Micro-batching (streaming many small batches through the stages) keeps everyone
busy and shrinks the bubble.

They compose. Large deployments stack tensor parallelism within a node, pipeline parallelism
across nodes, and data parallelism across replicas — the “3D parallelism” that trains and serves
frontier models. (Sequence and expert parallelism join the mix for very long contexts and MoE.)

## Putting it together

No single trick makes inference fast — a real serving stack layers most of them, and each one targets
a different axis. Latency (time to a token), throughput (tokens/sec across everyone), and memory
footprint pull against each other; the art is knowing which lever moves which. Click through the
scorecard:

KV Cache the default GQA / MQA shrink the cache PagedAttention vLLM FlashAttention IO-aware kernel Continuous Batching in-flight Speculative Decoding draft + verify Quantization fewer bytes Knowledge Distillation smaller student Tensor Parallelism split each layer Pipeline Parallelism split by layer Mixture of Experts sparse activation

Targets Latency Throughput Memory

How Store each token's K,V once; every later token reuses them instead of recomputing. Turns per-token attention work from O(n) into O(1).

Cost Memory: the cache grows linearly with context × batch and lives in HBM — it *becomes* the bottleneck at long context.

Targets Latency Throughput Memory

How Let multiple query heads share one set of K,V heads, so the cache stores far fewer K,V vectors per token.

Cost A small quality hit vs full multi-head attention — usually negligible, and baked in at train time (not a knob at serving).

Targets Latency Throughput Memory

How Store the KV cache in fixed-size pages like OS virtual memory, so sequences grow without contiguous allocation and pages can be shared.

Cost Implementation complexity — a custom attention kernel and a paged allocator. Nearly eliminates cache fragmentation.

Targets Latency Throughput Memory

How Fuse attention into one kernel that never materializes the full n×n score matrix in HBM — it tiles and keeps work in fast SRAM.

Cost None to accuracy (exact attention). Biggest win in the compute-bound prefill / long sequences.

Targets Latency Throughput Memory

How Add and evict sequences from the running batch every step instead of waiting for the whole batch to finish — the weight read is shared across all of them.

Cost Serving complexity; individual latency can jitter as the batch composition shifts. Massive throughput gain.

Targets Latency Throughput Memory

How A small draft model proposes several tokens; the big model verifies them all in one parallel pass, accepting the longest correct prefix.

Cost Extra compute for the draft + a memory-bound big-model pass per chunk. Output is provably identical to the big model — lossless.

Targets Latency Throughput Memory

How Store weights (and optionally the KV cache) in 8- or 4-bit instead of 16-bit, so there are fewer bytes to haul from HBM per token.

Cost Some accuracy loss that grows as you go lower; needs calibration. See the fine-tuning post for the memory math.

Targets Latency Throughput Memory

How Train a small "student" model to mimic a big "teacher," then serve the student — fewer params means fewer bytes and FLOPs per token.

Cost A capability ceiling below the teacher, plus the up-front training cost. A different model, not a serving toggle.

Targets Latency Throughput Memory

How Shard each weight matrix across GPUs; every GPU does part of every layer and they all-reduce the result. Cuts per-token latency for models too big for one GPU.

Cost Heavy per-layer communication — needs fast interconnect (NVLink). Usually kept within a single node.

Targets Latency Throughput Memory

How Put different layers on different GPUs and stream micro-batches through the stages, so a model far larger than one GPU still fits.

Cost Pipeline "bubbles" — idle stages at the start/end — and added latency. Communication is light (only at stage boundaries).

Targets Latency Throughput Memory

How Route each token to only a couple of expert sub-networks, so a model with huge total params activates only a fraction per token.

Cost Total memory stays large (all experts resident); routing adds complexity and can imbalance load across GPUs.

Click a technique. Targets shows which axis it moves — latency, throughput, or memory. Almost nothing helps all three for free.

Almost nothing improves latency, throughput, and memory for free —
each technique buys one or two and charges you somewhere else. A production stack picks a
complementary set.

So here’s the end-to-end life of your next token in a real system: your prompt is tokenized and runs
through a compute-bound prefill (accelerated by FlashAttention), populating a paged KV
cache shrunk by GQA. Then decode begins: your sequence joins a continuously batched group,
so the quantized weights are read from HBM once and shared across everyone in the batch. A draft
model speculates a few tokens ahead and the big model verifies them in parallel. If the model spans
GPUs, tensor parallelism splits each layer across the chips in the node. The token is sampled,
streamed to you, appended to the cache — and the loop runs again. All of it, per token, in a handful
of milliseconds.

## The techniques in the wild

None of this is theoretical — it’s what today’s frontier models actually run. The striking thing is
how much they’ve converged: nearly every one is a Mixture of Experts, pairs it with a
KV-cache-efficient attention variant, pushes context past 128K, and ships a reasoning (“thinking”)
mode. A quick tour:

Model (org)Techniques it’s known to use

DeepSeek V3 / R1 (open weights)MoE (~671B total, ~37B active); Multi-head Latent Attention (MLA) — their signature KV-cache compression; Multi-Token Prediction; FP8 training; auxiliary-loss-free load balancing. R1 layers large-scale RL for reasoning on top.

Kimi K2 (Moonshot, open weights)Very large MoE (~1T total, ~32B active, hundreds of experts); MLA-style attention; long context; trained with the Muon optimizer.

GLM 4.5 / 4.6 (Zhipu/Z.ai, open weights)MoE (~355B total, ~32B active) at a low activation ratio (~3.5–5.9%); GQA; 128K–200K context; a hybrid reasoning mode.

Grok (xAI)Grok-1 was open-weighted as a MoE (314B total, 2-of-8 experts, ~86B active); later versions are closed, with very long context and a reasoning mode.

ChatGPT / GPT (OpenAI, closed)GPT-4 is widely reported to be a large MoE; the o-series (“reasoning”) models add RL + inference-time chain-of-thought. The open-weight gpt-oss (2025) is a confirmed MoE with GQA and an MXFP4-quantized release.

Claude (Anthropic, closed)Architecture not publicly disclosed. Publicly known: long context (up to ~200K, 1M in beta), an extended-thinking reasoning mode, strong tool use, and Constitutional AI alignment.

A caveat on sources: figures for the open-weight models (DeepSeek, Kimi, GLM, Grok-1, gpt-oss) come
from their published papers and model cards; entries for the closed frontier models (ChatGPT/GPT-4,
Claude, and the latest Grok) reflect what’s publicly reported or officially stated — the exact
architectures aren’t disclosed.

The pattern is unmistakable: MoE for cheap capacity, an efficient-KV attention (MLA or GQA) to
survive long context, and a reasoning mode that spends more decode steps to think. Every one of
those is a bet placed exactly where this post said the costs are.

## Key takeaways

- Decode is memory-bound, not compute-bound. Per token, the GPU mostly reads its weights out of
HBM. Latency ≈ bytes ÷ bandwidth. That single fact explains almost everything else.

- A million-token context streams fast because the weight read is anchored to model size, not prompt
length — but the KV cache scales with context and becomes the real memory cost at long range.

- Prefill wants FLOPs, decode wants bandwidth. Time-to-first-token and tokens-per-second are
different problems with different fixes.

- The KV cache is the pivot: it turns quadratic work linear, then becomes the bottleneck itself —
which is what GQA, PagedAttention, and a quantized cache exist to tame.

- Batching is how inference pays for itself: share one weight read across many sequences.
Continuous batching keeps the GPUs full.

- Everything else reads fewer bytes (quantization, distillation), skips work (speculative decoding,
MoE), or splits it across chips (tensor/pipeline parallelism).

## Go deeper

- How GPT Works, Part 2: Attention from scratch — the K, Q, V that the KV cache stores.

- How GPT Works, Part 4: Training & Generation — the sampling step at the end of every loop.

- Fine-Tuning LLMs — the quantization memory math and the accuracy tradeoff, with a calculator.

- Building the Chatbot on This Site — inference serving in a shipped product.

- Dao et al., FlashAttention — the IO-aware attention kernel.

- Kwon et al., PagedAttention / vLLM — paging the KV cache for high-throughput serving.

- Leviathan et al., Fast Inference via Speculative Decoding — draft-and-verify, losslessly.

- Shoeybi et al., Megatron-LM — tensor parallelism for models too big for one GPU.

A frontier model feels fast not because the math got easier, but because a decade of systems work
went into moving fewer bytes, more times, across more chips. The model is the easy part; the harness
around it is where the speed lives.
