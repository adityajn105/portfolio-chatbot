---
title: "How GPT Works — Part 4: Training & Generation"
url: https://adityajain.me/blogs/gpt-4-training-generation.html
---

# How GPT Works — Part 4: Training & Generation

- GPT
- Deep Learning
- NLP
- Training

Part 3 assembled the architecture — text in, a probability over the
next token out. But an untrained transformer’s probabilities are noise. This part covers the two
processes that bracket a working model: training, which teaches it to put probability on the right
token, and generation, which turns those probabilities back into text.

## Training: one loss function, trillions of times

Training a GPT is conceptually the same supervised learning you already know — it just manufactures
its own labels. Take any text; at every position the “correct answer” is simply the token that
actually came next. No human annotation required, which is why the entire internet becomes training
data. Feeding the true previous tokens while predicting the next one is called teacher forcing.

The model is scored with cross-entropy loss: if it assigned probability ppp to the token that
truly came next, the loss for that position is −log⁡p-\log p−logp. Confident and right → tiny loss; confident
and wrong → large loss.

L=−1N∑t=1Nlog⁡P(xt∣x1,…,xt−1)\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \log P(x_t \mid x_1, \ldots, x_{t-1})L=−N1​∑t=1N​logP(xt​∣x1​,…,xt−1​)

The elegant part is the gradient. For softmax + cross-entropy it collapses to predicted minus
target — push the correct token’s probability up, everything else down. Watch one token learn:

Now scale that intuition: the same objective, run over trillions of tokens with billions of
parameters. Nothing about the goal changes — the model just gets relentlessly better at predicting
the next token, and grammar, facts, and reasoning emerge as a side effect (the observation behind
scaling laws: more data + parameters + compute predictably lowers loss).

## Generation: from probabilities back to text

A trained model gives you a probability distribution over the next token. Decoding is the policy
that turns that distribution into an actual choice, and it dramatically changes the output’s character:

- Greedy — always take the single most likely token. Coherent but repetitive and flat.

- Temperature — the knob from Part 1: divide the logits before
softmax to sharpen (low, conservative) or flatten (high, adventurous) the distribution.

- Top-k — sample only from the kkk most likely tokens, discarding the long tail of nonsense.

- Top-p (nucleus) — sample from the smallest set of tokens whose probability sums past ppp, so
the pool grows and shrinks with the model’s confidence.

Reshape a real next-token distribution and watch what survives:

This is exactly the sampler you were playing with in Part 1’s
next-token loop, now with the professional knobs exposed. A more exhaustive alternative,
beam search, keeps several candidate sequences alive at once and is common in translation and
captioning — I walk through it in my Image Captioning post — but
open-ended chat models lean on top-p sampling for variety.

## Key takeaways

- GPT training is self-supervised: the label is just the next token, so raw text is the dataset.

- Cross-entropy loss −log⁡P(true token)-\log P(\text{true token})−logP(true token) rewards confident-correct and punishes confident-wrong.

- Its gradient is simply predicted − target — the whole learning signal.

- Scaling laws: more data, parameters, and compute predictably drive the loss down.

- Decoding converts probabilities to text: greedy, temperature, top-k, and top-p trade off coherence against diversity.

## Go deeper

- Let’s build GPT: from scratch, in code — Karpathy trains a transformer live.

- The Illustrated GPT-2 — includes a clear treatment of sampling.

- How to generate text — Hugging Face’s hands-on guide to greedy, beam, top-k, and top-p decoding.

- My Image Captioning post — beam search and BLEU for sequence generation.

The last part steps back: how a raw next-token predictor becomes the helpful assistant you actually chat with.
