---
title: "Fine-Tuning & Serving LLMs: LoRA, quantization, and vLLM"
url: https://adityajain.me/blogs/fine-tuning-llms.html
---

# Fine-Tuning & Serving LLMs: LoRA, quantization, and vLLM

- LLM
- Fine-Tuning
- LoRA
- Quantization

So far in this series we’ve changed what the model can see — retrieval
gave it documents, agents gave it tools. This post changes the
model itself. When you need it to reliably follow a format, speak in a particular voice, or know
a domain cold, you fine-tune: keep training a pretrained model on your own examples.

The catch is cost. A modern model has billions of parameters; updating all of them needs a
cluster and a small fortune. The techniques that made fine-tuning accessible to everyone —
LoRA and quantization — are clever ways around exactly that cost, and both are simple
enough to understand from the inside. Let’s do that, then serve the result.

## First: do you even need to fine-tune?

Reach for the cheapest tool that works, in this order:

- Prompting — just ask well. Zero training.

- RAG — inject the right knowledge at query time (the first post). Best when the problem is missing facts.

- Fine-tuning — change the model’s behavior. Best when the problem is form, style, or a skill the model won’t reliably follow no matter how you prompt.

Fine-tuning teaches behavior, not facts — a common trap is trying to fine-tune knowledge in
when you should have retrieved it. For “answer in my voice, always with citations,” fine-tuning
earns its keep. For “know what my latest post said,” use RAG.

## LoRA: don’t retrain the matrix, learn a small correction

Fine-tuning updates weight matrices. A single one might be d×dd \times dd×d with d=4096d = 4096d=4096 — that’s
~16.8M numbers, and there are hundreds of them. Updating all of it (full fine-tuning) is the
expensive part.

LoRA (Low-Rank Adaptation) makes a bet that pays off: the change you need to make during
fine-tuning is simple — it has low “rank” — even though the original matrix is huge. So instead of
learning a full d×dd \times dd×d update, freeze the original weight WWW and learn the update as a
product of two skinny matrices, BBB (d×rd \times rd×r) and AAA (r×dr \times dr×d), with the rank rrr tiny
(often 8 or 16):

Weffective=Wfrozen+αr BAW_{\text{effective}} = W_{\text{frozen}} + \frac{\alpha}{r}\, B AWeffective​=Wfrozen​+rα​BA

Only BBB and AAA are trained — 2dr2 d r2dr numbers instead of d2d^2d2. Slide the rank and matrix size
and watch the trainable-parameter count collapse:

The hatched square is W, frozen. Only the two slim matrices
B and A are trained, and their product has the same d×d shape as W so it
slots right in. Rank r is the whole knob: bigger r = more capacity but more parameters.

This isn’t hand-waving — I built it in NumPy to be sure. Freeze a random matrix, initialize the
adapter as a no-op (B=0B = 0B=0), and train only BBB and AAA by gradient descent to match a target
correction. The loss drops to ~10−3210^{-32}10−32: a rank-rrr adapter perfectly recovers a rank-rrr
update, while training a fraction of the parameters.

# the entire LoRA training step, from scratch (src/lora_demo.py)
err = B @ A - target # current adapter output vs the correction we want
gB = err @ A.T # gradient wrt B
gA = B.T @ err # gradient wrt A
B -= lr * gB # W stays frozen; only the adapter moves
A -= lr * gA

The framework version is the same idea on a real model: attach adapters to the attention and MLP
projection matrices and train with Unsloth + TRL. The r and alpha in that config are exactly
the knobs above.

model = FastLanguageModel.get_peft_model(
model, r=16, lora_alpha=16,
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj"],
)

Because only the adapters change, the trained artifact is a few megabytes, not gigabytes —
you can keep dozens of task-specific adapters and swap them over one frozen base model.

## Quantization: store the weights in fewer bits

LoRA cuts what you train. Quantization cuts what you store. A weight kept as a 32-bit
float takes 4 bytes; store it in 8 bits (1 byte) or 4 bits (½ byte) and the model’s memory
footprint shrinks proportionally — at the cost of a little numerical precision. Since inference is
mostly memory-bound, this is often the difference between “fits on your GPU” and “doesn’t.”

Pick a model size and a precision and see what fits:

Memory ≈ parameters × bytes-per-parameter. A 7B model is ~28 GB at
FP32 but only ~3.5 GB at INT4 — which is how a 7B model runs on a laptop. Note this counts
weights only; the KV-cache and activations need more at run time.

QLoRA combines both tricks: load the frozen base model in 4-bit (quantized) and train small
LoRA adapters on top in higher precision. You get low memory and low trainable-parameter count
at once — which is why fine-tuning a 7B model on a single free-tier GPU is now routine. That’s the
load_in_4bit=True in the training script.

## Serving: vLLM

A fine-tuned model that only runs in a notebook isn’t useful. vLLM is the standard high-throughput
server. Two ideas make it fast, and both should feel familiar from the GPT series:

- PagedAttention — it manages the KV-cache like an operating system manages memory (in pages),
so it wastes far less and serves more requests at once.

- Continuous batching — new requests slot into a running batch instead of waiting for it to
finish, keeping the GPU busy.

vLLM can serve the base model and apply your LoRA adapter per request, and exposes an
OpenAI-compatible API — so you point the Phase-1 RAG generator at it and the whole pipeline now
answers in your fine-tuned voice:

vllm serve unsloth/Llama-3.2-1B-Instruct \
--enable-lora --lora-modules blog=outputs/blog-lora

## From scratch → framework

LayerWe built (from scratch)Framework / tool

Low-rank adapterNumPy B@A + gradient stepPEFT / Unsloth LoRA

Training datatemplated Q&A from the blogyour labeled set, or LLM-generated pairs

Training loopmanual gradient descentTRL SFTTrainer

Quantizationround a weight to fewer bitsbitsandbytes (4-bit), GGUF

Serving—vLLM (PagedAttention, batching)

## Key takeaways

- Fine-tune for behavior, retrieve for facts. Try prompting → RAG → fine-tuning, in that order.

- LoRA freezes the big matrix and learns a low-rank correction — 2dr2dr2dr params instead of d2d^2d2,
often ~250× fewer. The adapter is a few MB and swappable.

- Quantization stores weights in fewer bits; memory ≈ params × bytes, so INT4 shrinks a 7B
model from ~28 GB to ~3.5 GB. QLoRA stacks both.

- vLLM serves it efficiently with paged KV-cache and continuous batching, and can apply your
adapter over a shared base model.

## Go deeper

- LoRA paper and QLoRA.

- Unsloth — fast, low-memory LoRA fine-tuning.

- Hugging Face PEFT and TRL.

- vLLM — the serving engine (PagedAttention).

Last in the series: knowing whether any of this actually works — evaluation and
observability — and shipping the whole thing as a live demo.
