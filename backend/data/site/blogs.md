---
title: "Blog"
url: https://adityajain.me/blogs
---

Writing

# All posts

### Building the Chatbot on This Site: from the series to a live assistant

The capstone: a full, file-by-file walkthrough of the assistant on this site — crawler, from-scratch RAG, a ReAct agent, tools over MCP, SSE streaming, an embeddable Shadow-DOM widget, and the safety-and-cost work tutorials skip. Every step points at the real code on GitHub and the production tool you'd swap in. It's live — go talk to it.

- LLM
- RAG
- Agents
- Production

Sep 3, 2026

### Evaluating & Observing LLM Apps

You can't improve what you can't measure. This post builds evaluation from scratch — a golden set, hit@k and MRR for retrieval, a groundedness check for answers — plus lightweight tracing to see where latency and cost go, then points at how the whole system ships. With interactive metric and trace playgrounds.

- LLM
- Evaluation
- Observability
- RAG

Sep 2, 2026

### Fine-Tuning & Serving LLMs: LoRA, quantization, and vLLM

When prompting isn't enough, you fine-tune. This post builds LoRA's low-rank idea from scratch in NumPy, shows why it trains ~250x fewer parameters, explains quantization and why 4-bit lets big models fit on small GPUs, and covers serving with vLLM. With interactive calculators for rank and memory.

- LLM
- Fine-Tuning
- LoRA
- Quantization

Sep 1, 2026

### How AI Agents Actually Work: ReAct from scratch

An 'agent' is a while-loop around an LLM plus a text protocol for calling tools. This post builds the ReAct pattern — Reason, Act, Observe — by hand with no framework, so you can see there's no magic, then shows what LangGraph and MCP add on top. With an interactive trace you can step through.

- LLM
- Agents
- ReAct
- Tools

Aug 31, 2026

### How GPT Works — Part 5: From Base Model to ChatGPT

The final step: how a raw next-token predictor becomes a helpful assistant. Part 5 covers pretraining, supervised fine-tuning, and RLHF, the difference between a base model and a chat model, plus the context window and KV-cache that govern inference.

- GPT
- Deep Learning
- NLP
- RLHF

Aug 30, 2026

### RAG from Scratch: teaching a model to look things up

Retrieval-Augmented Generation, built by hand with no libraries. Chunk a corpus, turn text into vectors with TF-IDF, retrieve by cosine similarity, and answer grounded in what you found — then see exactly where a purely lexical approach breaks and why semantic embeddings fix it. With interactive playgrounds.

- LLM
- RAG
- NLP
- Retrieval

Aug 30, 2026

### How GPT Works — Part 4: Training & Generation

How a transformer learns and how it writes. Part 4 covers next-token cross-entropy training with an interactive loss-descent demo, then decoding strategies — greedy, temperature, top-k, and top-p sampling — you can reshape live.

- GPT
- Deep Learning
- NLP
- Training

Aug 29, 2026

### How GPT Works — Part 3: The Transformer

How the attention mechanism becomes a working language model. Part 3 covers subword tokenization with live Byte-Pair Encoding, the transformer block (residuals, LayerNorm, MLP), the causal mask that makes a GPT decoder-only, and the full pipeline from text to next-token probabilities.

- GPT
- Deep Learning
- NLP
- Transformers

Aug 28, 2026

### How GPT Works — Part 2: Attention

From the seq2seq bottleneck to the mechanism that replaced recurrence entirely. Part 2 builds attention from the ground up — soft alignment, scaled dot-product self-attention with Q/K/V, multi-head attention, and why a transformer needs positional encoding.

- GPT
- Deep Learning
- NLP
- Attention

Aug 27, 2026

### How GPT Works — Part 1: The Foundations

A visual, hands-on guide to how large language models work. Part 1 covers the only prerequisites you need — vectors, the dot product, matrix multiplication, and softmax — then the one idea the whole model is built on: next-token prediction.

- GPT
- Deep Learning
- NLP

Aug 26, 2026

### Image Captioning with the 'Merge' Architecture

The 'merge' architecture generates image captions differently from the traditional 'inject' approach where image features are fed into the RNN. Here's how it works.

- Computer Vision
- NLP

Jul 13, 2019

### Attention Model for Machine Translation

Attention is one of the most powerful sequence-to-sequence ideas, powering machine translation, image captioning, and more. Here's how the attention model and mechanism work, plus a date-translation demo.

- NLP
- Deep Learning

Jun 10, 2019

### Proximal Policy Optimization (PPO)

PPO became an industry-standard RL algorithm after OpenAI's release. This post explains the clipped surrogate objective and why PPO is stable and effective.

- Reinforcement Learning
- Policy Gradient

Apr 26, 2019

### Policy Gradient and the Actor-Critic Algorithm

Where Deep Q-Learning falls short, policy gradient methods step in. This post builds from the policy gradient up to the Advantage Actor-Critic (A2C) algorithm.

- Reinforcement Learning
- Policy Gradient

Apr 24, 2019

### Deep Q-Learning and Advancements over Deep Q-Networks

Using deep neural networks for Q-learning to build an agent that plays Flappy Bird — plus key improvements to DQN like Double DQN, Dueling Networks, and Prioritized Experience Replay.

- Reinforcement Learning
- Deep Learning

Apr 11, 2019

### Monte Carlo and Temporal Difference Learning

When the full MDP is unknown, an agent must learn from experience. This post covers Monte Carlo and Temporal Difference methods for model-free reinforcement learning.

- Reinforcement Learning

Apr 10, 2019

### Policy Optimization in a Known MDP

When the MDP is fully known, an agent can compute an optimal policy directly. This post walks through policy iteration, value iteration, and related dynamic-programming techniques.

- Reinforcement Learning

Oct 19, 2018

### LSTM — Long Short Term Memory Networks

LSTMs are a special kind of RNN capable of learning long-term dependencies. This post demystifies the cell state, the gates, and how LSTMs remember.

- Deep Learning

Oct 18, 2018

### Word Embeddings in Natural Language Processing

To use words in NLP models we must turn them into numbers. This post explores word embeddings — from one-hot vectors to Word2Vec and GloVe — and why they work.

- NLP

Sep 10, 2018

No posts match this topic.
