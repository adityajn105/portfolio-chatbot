---
title: "How GPT Works — Part 1: The Foundations"
url: https://adityajain.me/blogs/gpt-1-foundations.html
---

# How GPT Works — Part 1: The Foundations

- GPT
- Deep Learning
- NLP

You know what a neural network is. This series takes you the rest of the way — from the
mathematical building blocks, through the attention mechanism that changed everything, to a
working mental model of a modern large language model like GPT. Every idea comes with something
you can poke, drag, and watch move.

## The only math you need

A GPT makes a surprisingly small number of distinct mathematical moves. If the four ideas
below are second nature, skim the playgrounds and jump ahead — everything in the series is
built from exactly these pieces.

### 1 · Everything is a vector

A model can’t do arithmetic on the word “cat”. So the first thing a GPT does is turn every
token into a list of numbers — a vector, called an embedding. GPT-2 uses vectors of
length 768; think of each as a point in a 768-dimensional space where meaning becomes
geometry: related tokens land near each other, and directions encode properties.

embed("cat")=[0.21−0.440.90⋯]∈R768\text{embed}(\texttt{"cat"}) = \begin{bmatrix} 0.21 & -0.44 & 0.90 & \cdots \end{bmatrix} \in \mathbb{R}^{768}embed("cat")=[0.21​−0.44​0.90​⋯​]∈R768

We can’t draw 768 dimensions, so throughout the series we’ll use tiny 2- or 3-number vectors.
The math is identical; only the count changes.

Go deeper on embeddings: my Word Embeddings post walks
through how these vectors are actually learned so that meaning ends up as geometry. For visual
companions, Jay Alammar’s Illustrated Word2Vec
and Google’s Embeddings crash course
are both excellent.

### 2 · The dot product measures similarity

This is the single most important operation in the whole series. The dot product of two
vectors multiplies them element-by-element and adds up the result:

a⋅b=∑iaibi=a1b1+a2b2+⋯a \cdot b = \sum_i a_i b_i = a_1 b_1 + a_2 b_2 + \cdotsa⋅b=∑i​ai​bi​=a1​b1​+a2​b2​+⋯

One number falls out, and it tells you how aligned two vectors are. Point the same way → large
positive. Perpendicular → about zero. Opposite → negative. That is exactly how attention will
decide which tokens are “relevant” to each other, so it’s worth getting a feel for it. Drag the
arrowheads:

Divide the dot product by the two lengths and you get cosine
similarity — the same idea normalized to the range −1…1, so it reports only the angle
between the vectors and ignores how long they are.

### 3 · Matrices multiply many vectors at once

A matrix is a grid of numbers, and matrix multiplication is nothing but dot products done
in bulk. Every output entry is the dot product of a row from the left with a column from the
right:

(XW)ij=∑kXik Wkj(XW)_{ij} = \sum_k X_{ik}\, W_{kj}(XW)ij​=∑k​Xik​Wkj​

Neural-network layers are matrix multiplies. To transform a whole sentence of token vectors,
you stack them into a matrix XXX and multiply by a weight matrix WWW — one operation, every
token transformed at once. When you meet Q=XWQQ = X W_QQ=XWQ​ in Part 2, that is all it is: the sentence,
reshaped by a learned matrix. This bulk-dot-product-on-a-GPU is why transformers train so fast.

### 4 · Softmax turns scores into probabilities

Models produce raw, unbounded scores called logits. To make a decision we need
probabilities: all positive, all summing to 1. Softmax does that — exponentiate each score,
then divide by the total:

softmax(z)i=ezi∑jezj\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}softmax(z)i​=∑j​ezj​ezi​​

It appears twice in every GPT: once inside attention (turning similarity scores into attention
weights) and once at the very end (turning logits into next-token probabilities). The
temperature knob divides the logits before softmax, controlling how peaked or flat the result
is — you’ll meet it again in Part 4.

## The big idea: it’s just next-token prediction

With those pieces in hand, here is the whole model in one sentence. When you send ChatGPT a
message, it does not plan a sentence and write it out. It looks at everything so far and produces
a probability for every possible next token — a token being a chunk of text, roughly a word or
word-piece (we pull tokenizers apart in Part 3). It picks one, appends it, and does the whole
thing again on the slightly longer text. That repeating step is the autoregressive loop, and
it is all a GPT ever does at inference time.

Play with the loop below. You are the model’s sampler: watch the probabilities, then either pick
a token yourself or let it sample.

### Why such a small idea is so powerful

“Predict the next token” sounds too small to explain writing code, translating French, or
summarizing a report. The surprise of the last decade is that it isn’t. To predict the next token
really well across the entire internet, a model is forced to learn grammar, facts, reasoning
patterns, and style — because all of those help it guess better. Capability is a side effect of
relentless prediction.

Formally, the model scores a whole sequence by multiplying its step-by-step guesses (the chain
rule of probability):

P(x1,x2,…,xn)=∏t=1nP(xt∣x1,…,xt−1)P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \ldots, x_{t-1})P(x1​,x2​,…,xn​)=∏t=1n​P(xt​∣x1​,…,xt−1​)

Each factor is one forward pass of the network. Training nudges the model so the tokens that
actually came next in real text get high probability (Part 4). Everything else in this series —
attention, transformer blocks, tokenizers — exists to make that single next-token probability as
accurate as possible.

## Key takeaways

- Tokens become vectors; meaning is encoded as position and direction in space.

- The dot product is a similarity score — the heart of attention.

- Matrix multiplication is just many dot products at once; every NN layer is one.

- Softmax converts raw scores into a probability distribution that sums to 1.

- A GPT maps text so far → a probability over the next token, and generates by sampling one
token and feeding it back in.

## Go deeper

- Transformer Explainer — a live GPT-2 running in your browser.

- LLM Visualization — a 3D walk through every matrix operation.

- 3Blue1Brown: But what is a GPT? — visual intuition for the whole model.
