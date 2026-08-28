---
title: "How GPT Works — Part 2: Attention"
url: https://adityajain.me/blogs/gpt-2-attention.html
---

# How GPT Works — Part 2: Attention

- GPT
- Deep Learning
- NLP
- Attention

In Part 1 we reduced a GPT to one job — turn text so far into
a probability over the next token — and to four pieces of math, the dot product chief among them.
This part is where that dot product earns its keep. Attention is the mechanism that lets every
token look at every other token and decide what matters, and it is the single idea that made modern
language models possible.

## Where attention came from: the seq2seq bottleneck

Before attention, the standard tool for turning one sequence into another — translation, captioning,
summarization — was the encoder–decoder RNN (usually with
LSTM cells to carry information across long spans). An encoder reads the
input one token at a time and compresses everything into a single fixed-length context vector; a
decoder then generates the output from that one vector.

The flaw is right there in the description. Everything the model knows about a fifty-word sentence
has to survive in one vector. The further back a detail sits, the more likely it is crushed. This is
the bottleneck problem, and it is why long-sentence translation was so brittle.

h1,h2,…,hn⏟one per input token  ⟶  c⏟single vector  ⟶  decoder\underbrace{h_1, h_2, \ldots, h_n}_{\text{one per input token}} \;\longrightarrow\; \underbrace{c}_{\text{single vector}} \;\longrightarrow\; \text{decoder}one per input tokenh1​,h2​,…,hn​​​⟶single vectorc​​⟶decoder

## Soft alignment: let the decoder look back

Bahdanau’s 2014 fix was disarmingly simple: don’t throw away the per-token states. Keep all of
them, and at every output step let the decoder compute a weighted average over the inputs — paying
most attention to the words that matter right now. Those weights are the attention distribution.

The heatmap below is a toy English→French alignment. Each output word (a row) looks back over the
whole input and decides where to focus. Hover a row to isolate its attention.

Notice rouge aligns to red even though French flips the word order — something a single
context vector could never express. This is exactly the mechanism behind classic neural machine
translation and image captioning; I walk through a full date-translation model in my
Attention Model for Machine Translation post if you want the
sequence-to-sequence version in depth.

## Self-attention: the transformer’s leap

The 2017 paper Attention Is All You Need took a bolder step: throw away recurrence entirely.
Instead of one sequence attending to another, let a sequence attend to itself. Every token
gathers context directly from every other token, in one parallel operation — no waiting for a
recurrent state to march through the sentence.

Here is the whole mechanism. From each token’s embedding the model produces three vectors via learned
matrices (the WQ,WK,WVW_Q, W_K, W_VWQ​,WK​,WV​ from Part 1):

- Query qqq — what this token is looking for.

- Key kkk — what this token offers to others.

- Value vvv — the information it passes on if attended to.

For one token, dot its query against every key to get raw scores, scale by d\sqrt{d}d​ (so the
numbers don’t explode as dimension grows), softmax into weights, and take the weighted sum of values:

Attention(Q,K,V)=softmax ⁣(QK⊤dk)V\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) VAttention(Q,K,V)=softmax(dk​​QK⊤​)V

That formula is the transformer. Pick a token below — especially it — and watch the scores,
weights, and blended output update:

The softmax is the same one from Part 1, now turning similarity scores into
a blend. The scaling by √d is the only new wrinkle — without it, large dimensions push the
dot products so high that softmax collapses onto a single token and gradients vanish.

Because there is no recurrence, every token’s attention is computed simultaneously as one big
matrix multiply — the “bulk dot products on a GPU” from Part 1. That parallelism is why transformers
train on far more data than RNNs ever could.

## Multi-head attention: several relationships at once

A single attention pattern can only express one kind of relationship at a time. So transformers run
several attentions in parallel — heads — each with its own WQ,WK,WVW_Q, W_K, W_VWQ​,WK​,WV​, then concatenate the
results. Left to learn freely, different heads specialize: one tracks the previous token, another
links verbs to their subjects, another just spreads context broadly.

Switch between heads on the same sentence:

In a real GPT-2 there are 12 heads per layer across 12 layers — 144 distinct attention patterns,
each a learned lens on the sentence.

## Positional encoding: putting order back in

Self-attention has a subtle blind spot. It is a weighted sum over a set of tokens — permute the
input and the output is just as permuted, but otherwise identical. The mechanism literally cannot tell
“dog bites man” from “man bites dog.”

The fix is to add a position-dependent pattern to each token’s embedding before attention ever
runs. The original transformer uses fixed sinusoids of different frequencies; every position gets a
unique fingerprint, and — crucially — nearby positions get similar fingerprints, so the model can
sense distance. Drag the slider to grow the sequence:

Modern GPTs often swap these fixed sinusoids for learned or rotary (RoPE) positional
encodings, but the goal never changes: give an order-blind mechanism a sense of where each token sits.

## Key takeaways

- The seq2seq bottleneck — cramming a whole input into one vector — is the problem attention solves.

- Attention is a learned weighted average: dot-product similarity → softmax → blend of values.

- Self-attention lets a sequence attend to itself, in parallel, with no recurrence — the core of the transformer.

- The scaled dot-product formula softmax(QK⊤/d)V\text{softmax}(QK^\top/\sqrt{d})Vsoftmax(QK⊤/d​)V is the whole mechanism.

- Multi-head attention runs many of these in parallel so different relationships can coexist.

- Positional encoding is added because attention alone is order-blind.

## Go deeper

- The Illustrated Transformer — Jay Alammar’s canonical visual walkthrough of Q/K/V and multi-head attention.

- Attention Is All You Need — the original 2017 paper.

- The Annotated Transformer — the paper reimplemented line-by-line in PyTorch.

- My own Attention Model for Machine Translation and LSTM Networks posts cover the seq2seq lineage attention grew out of.

Next we assemble these blocks into a full transformer and see how raw text becomes tokens.
