---
title: "How GPT Works — Part 5: From Base Model to ChatGPT"
url: https://adityajain.me/blogs/gpt-5-base-to-chatgpt.html
---

# How GPT Works — Part 5: From Base Model to ChatGPT

- GPT
- Deep Learning
- NLP
- RLHF

Over four parts we built a complete picture: tokens become vectors,
attention mixes them, the transformer
stacks that into a next-token predictor, and training teaches
it to predict well. Yet a freshly pretrained model is not ChatGPT. This final part covers the gap —
how a raw predictor becomes an assistant, and how it runs when you actually chat with it.

## Three stages to an assistant

The model that comes out of pretraining is a base model. It has absorbed a staggering amount of
the world’s text, but its only skill is continuation — given some text, guess what comes next. Turning
that into a helpful assistant takes two more stages. Click through all three:

Almost all of the model’s knowledge is acquired in stage one; stages two and three add almost no new
facts. They change behaviour — teaching the model to follow instructions and to prefer helpful,
honest, harmless responses.

### Supervised fine-tuning (SFT), by example

Supervised Fine-Tuning (SFT) — sometimes called instruction tuning — keeps training the same
model on a small, human-curated dataset of (prompt, ideal response) pairs, using the very same
next-token cross-entropy loss from Part 4. Only the data
changes: instead of raw internet text, every example demonstrates the behaviour we want. One
training pair looks like this:

Prompt: "Translate 'good morning' into French."
Response: "Bonjour." ← the ideal answer, written by a human labeler

The model is nudged to make that response high-probability given that prompt. Show it tens of
thousands of diverse demonstrations — question answering, summarization, coding, polite refusals — and
it generalizes into “act like a helpful assistant.” The catch: SFT can only imitate the one gold
answer a labeler wrote. It has no way to express that answer A is better than answer B, and for many
prompts there is no single perfect response (what is the ideal poem?). That gap is what the next stage
fills.

### Reinforcement learning from human feedback (RLHF), by example

Reinforcement Learning from Human Feedback (RLHF) flips the approach: instead of demonstrating
ideal answers, humans rank answers the model already produces, and the model is optimized to
generate more of what people prefer. It runs in two sub-steps:

Reward Model (RM). For a prompt, sample several responses from the SFT model and have a human
rank them. Train a separate reward model to predict a scalar “how much would a human like this?”
score. A single comparison it learns from:

Prompt: "Explain photosynthesis to a 6-year-old."

Response A: "Photosynthesis is the process by which chlorophyll-bearing
autotrophs convert radiant energy into chemical energy..." ← accurate, too complex
Response B: "Plants are like tiny chefs! They take sunlight, water, and
air and cook up their own food." ← right for the audience

Human ranking: B > A

Policy optimization. Fine-tune the language model (the policy) to maximize the reward model’s
score — classically with PPO (Proximal Policy Optimization),
plus a penalty that stops it drifting too far from the SFT model. A popular simpler alternative,
DPO (Direct Preference Optimization), optimizes the preferences directly and skips the separate
reward model.

So the one-line distinction: SFT learns from “here is a good answer” (imitation); RLHF learns from
“this answer is better than that one” (preference). RLHF is the polish that makes responses feel
genuinely helpful — the difference you sense between a raw base model and ChatGPT.

## Base model vs. aligned model

The difference is easiest to feel directly. Here is the same prompt sent to a base model and to an
aligned chat model — toggle between them:

The base model isn’t broken; it’s doing exactly what pretraining rewarded — continuing text. A quiz
question is most often followed by more quiz questions in its training data, so that’s what it
produces. Alignment is what redirects that raw capability toward answering you.

## How it runs: context window and KV-cache

Two practical facts shape every interaction with a GPT.

The context window is the maximum number of tokens the model can attend over at once — its working
memory. Everything must fit: the system prompt, the whole conversation so far, and the response being
generated. Run past it and the earliest tokens fall out of view. Because attention compares every
token with every other, cost grows with the square of context length, which is why longer context
windows are expensive and hard-won.

The KV-cache is why generation isn’t hopelessly slow. Recall from
Part 3 that each token produces key and value vectors. When
generating token by token, the keys and values of all previous tokens don’t change — so the model
computes them once and caches them, and each new token only computes its own query against the
stored keys. Without this, generating the 1,000th token would mean recomputing the first 999 every
time.

## The whole series in one paragraph

A GPT turns text into tokens, each token into a vector, and uses causal self-attention stacked in
transformer blocks to blend those vectors into context-aware representations — producing, at the
final position, a probability distribution over the next token. It’s pretrained on the
internet by minimizing next-token cross-entropy, then aligned with supervised fine-tuning and
RLHF into an assistant. At inference it samples one token at a time — reusing a KV-cache for speed
— appending each token and repeating. That’s it. Everything else is scale and refinement.

## Key takeaways

- A base model only continues text; it is the product of pretraining alone.

- Alignment = supervised fine-tuning (instruction/response pairs) + RLHF (optimizing against learned human preferences).

- Alignment changes behaviour, not knowledge — the facts were learned in pretraining.

- The context window is the model’s finite working memory; attention cost scales quadratically with it.

- The KV-cache stores past keys/values so each new token is cheap to generate.

## Go deeper

- State of GPT — Andrej Karpathy’s talk on the full pretraining → SFT → RLHF pipeline.

- Illustrating RLHF — Hugging Face on reward models and preference optimization.

- Transformer Explainer — a live GPT-2 to tie every concept back to a running model.

That’s the series — from a dot product to ChatGPT. Thanks for reading; jump back to any part below.
