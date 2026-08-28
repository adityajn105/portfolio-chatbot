---
title: "How GPT Works — Part 3: The Transformer"
url: https://adityajain.me/blogs/gpt-3-transformer.html
---

# How GPT Works — Part 3: The Transformer

- GPT
- Deep Learning
- NLP
- Transformers

Part 2 gave us self-attention — a way for every token to gather
context from every other token. But attention is one ingredient, not a meal. This part assembles the
full transformer: how raw text becomes tokens, what wraps around attention inside a block, the one
change that makes a GPT generative, and the end-to-end pipeline from a string to a probability over
the next token.

## First, text has to become numbers

Attention operates on vectors, so before anything else the text must be chopped into tokens and
each mapped to an id. The obvious options both disappoint: splitting on words gives a huge,
brittle vocabulary that can’t handle a typo or a new word, while splitting on characters makes
sequences painfully long and forces the model to relearn spelling from scratch.

Modern GPTs use a middle path — subword tokenization, in practice Byte-Pair Encoding (BPE).
BPE starts from individual characters and greedily merges the most frequent adjacent pair over and
over, so common chunks become single tokens while rare words still decompose into reusable pieces.
Step through the actual algorithm on a tiny corpus:

This is why token counts feel uneven: " the" is one token, but an unusual name might cost five. It
also connects back to Part 1 — each learned token id indexes into an
embedding matrix to become the vector attention consumes. (I cover the embedding side of this
story in more depth in Word Embeddings.)

## The transformer block: what wraps around attention

Self-attention rarely runs alone. It sits inside a transformer block that repeats, identically,
dozens of times. Each block adds three things that make deep stacks actually trainable:

- Residual connections — every sub-layer computes x+sublayer(x)x + \text{sublayer}(x)x+sublayer(x), not just
sublayer(x)\text{sublayer}(x)sublayer(x). The input takes a shortcut around the operation, so gradients flow cleanly
through 96 layers instead of vanishing.

- Layer normalization — re-centers and re-scales activations at each step to keep the numbers in
a stable range.

- A feed-forward MLP — a two-layer network (e.g. 768→3072→768768 \to 3072 \to 768768→3072→768) applied to each token
independently. Attention mixes information between tokens; the MLP then processes it per token.
This is where much of the model’s raw knowledge is stored.

x←x+Attention(Norm(x))x←x+MLP(Norm(x))x \leftarrow x + \text{Attention}(\text{Norm}(x)) \qquad x \leftarrow x + \text{MLP}(\text{Norm}(x))x←x+Attention(Norm(x))x←x+MLP(Norm(x))

## The causal mask: how a GPT becomes a generator

Here is the one modification that turns generic attention into GPT. When predicting token ttt, the
model must not be allowed to see tokens t+1t{+}1t+1 and beyond — otherwise “predict the next word” would
be trivial cheating during training. So GPT applies a causal mask: every position may attend only
to itself and earlier positions.

That single triangular mask is the difference between a decoder-only model like GPT (attends
leftward, generates left to right) and a bidirectional encoder like BERT (attends both ways, built for
understanding rather than generation). Toggle it above to feel the distinction.

## Putting it all together

Stack it up and the whole model is a straight pipeline: tokenize the text, embed the tokens and add
positions, run the causal transformer block NNN times, then project back to vocabulary size to get a
score for every possible next token. Click through each stage:

The remarkable part is how uniform it is — the same block, repeated. GPT-2 small stacks 12 of them;
GPT-3 stacks 96. Scaling a transformer is largely a matter of making that stack taller and wider, an
observation that turns out to matter enormously in Part 4.

## Key takeaways

- Tokenization turns text into integer ids; BPE learns a subword vocabulary by greedily merging frequent pairs.

- A transformer block wraps attention with residual connections, LayerNorm, and a per-token MLP, then repeats.

- Residuals + normalization are what make very deep stacks trainable at all.

- The causal mask restricts each token to the past — the defining trait of a decoder-only GPT.

- The assembled model is a single pipeline: tokens → embeddings + position → N blocks → unembed → logits → softmax.

## Go deeper

- Let’s build the GPT Tokenizer — Andrej Karpathy builds BPE from scratch.

- The Illustrated GPT-2 — Jay Alammar on decoder-only architecture and masked self-attention.

- LLM Visualization — a 3-D walk through every matrix operation in a real GPT.

- My Word Embeddings post — how token ids become meaningful vectors.

Next: how this architecture is actually trained, and how it turns a probability distribution into text.
