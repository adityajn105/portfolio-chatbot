---
title: "LSTM — Long Short Term Memory Networks"
url: https://adityajain.me/blogs/lstm-tutorials.html
---

# LSTM — Long Short Term Memory Networks

- Deep Learning

I think everybody now a days is familiar with working of neural networks and even most of us know how to efficiently apply them to solve problems. As we go by definition, “An artificial neural network is an interconnected group of nodes, akin to the vast network of neurons in a brain. Here, each circular node represents an artificial neuron and an arrow represents a connection from the output of one artificial neuron to the input of another.”. There has been a lot of good material out there to learn about deep learning or say neural network in general. I myself has studied about it through neural networks and deep learning book by Michael Nielsen and I prefer people do the same before moving ahead in this blog.

In this blog I will move a step ahead and try to explain working of RNN and LSTM based on my understanding. This blog is basically a summary of blog by Christopher Olah’s blog titled Understanding LSTM Networks. So for detailed explanation refer to his blog.

## RNN: Recursive Neural Networks

I am assuming people already knew about neural networks, so they will be easily able to understand how previous input to the neural network does not affect the next input at all during predictions.

Think about a human in general, they do not start thinking from scratch every sec. You do not throw everything away and start thinking from scratch again. Traditional Neural Networks can’t do this and it seems a major short coming, For example imagine you want to classify what kind of event is happening at every point in movie. It is unclear how traditional neural network could use its reasoning about previous event.

RNN resolve this issue. They are networks with loops in them, allowing information to persist.

- A - Neural Network

- XtX_tXt​ - Some Input

- hth_tht​ - output a value

RNN look mysterious but it turns out that they aren’t all that different than a neural networks. A RNN can be thought as a multiple copies of same network, each passing a message to successor.

Unrolled RNN

- A - Neural Network

- XtX_tXt​ - Some Input

- hth_tht​ - output a value

Chain like nature reveals RNN are related to sequences and list. There have been incredible success in applying RNN to variety of problems like speech recognition, language modelling, transition, image captioning.

### Problems with RNN

Sometimes we only need to look at recent info to perform present task. But there are also cases where we need more context. Unfortunately as gap grows, RNN become unable to learn. In theory RNN are absolutely capable of handling long term dependencies. Sadly in practice, RNN don’t seem to be able to learn them. This is due to unstable gradient problem (gradient gets smaller and smaller as it propagates back through layers.) This makes learning in early layers extremely slow.

Thankfully, LSTM don’t have this problem.

## LSTM Networks: Long Short Term Memory Networks

LSTM are special kind of RNN capable of learning long-term dependencies. They work on variety of problems and are widely used. LSTM are explicitly designed to avoid long-term dependencies problem. Removing information for long period of time is practically their default behaviour.

In standard RNNs, this repeating module will have a very simple structure, such as single tanh layer.

Repeating module in standard RNN contains a single layer.

LSTM also have this chain like structure, but repeating module has different structure. Instead of having single NN layer, there are 4 layers, interacting in special way.

The repeating module in an LSTM contains four interacting layers.

The key to LSTM is the cell state, horizontal line running through top. LSTM have ability to add and remove information to cell state, carefully regulated by structure called as gates. Gates are way to let information through. Gates are composed to sigmoid neural network and a pointwise multiplication operation.

Sigmoid layer outputs a number between 0 and 1, describing how much component should be let through. 0 means ‘let nothing through’ and 1 means ‘let everything through’.

LSTM have three of these gates and control state. So, lets discuss about them one by one.

### 1. Forget Layer: First step in our LSTM is decide what information to let through and what to throw away from cell state.

The decision is made by sigmoid layer “forget gate” layer. It looks at ht−1h_{t-1}ht−1​ and XtX_tXt​ and output number between 0 and 1 for each number in cell state ( Ct−1C_{t-1}Ct−1​ ).

1 means “completely keep this” while 0 represent “completely get rid of this”.

Example: in language model trying to predict next word based on previous ones. Cell might include information of gender of present subject so correct pronoun can be used. We see a new subject, we want to forget gender of old subject.

### 2. Stage Update Layer: The next step is to decide what new information we’re going to store in cell state. This has 2 parts

First a sigmoid layer called “input gate layer” decide which values will be updated.

Next, a tanh layer creates a vector of new candidate values, C^t\hat{C}_tC^t​. Next we will combine these two to create an update to state.

In our example: we want to add the gender of new subject to the cell state, to replace the one’s we are forgetting.

### 3. Now its time to update the old cell state, Ct−1C_{t-1}Ct−1​ into the new cell state ctc_tct​

First we multiply old cell state Ct−1C_{t-1}Ct−1​ by ftf_tft​, forgetting thing we decide to forget earlier.

Then, we add it∗C^ti_t * \hat{C}_tit​∗C^t​. This is new candidate values scaled by how much we decide to update each state value.

Example: In language model, this is actually when we would actually drop information about the old subject gender and add the new information.

### 4. Finally, we need to decide what we are going to output.

This output will be based on our cell state, but will be filtered version of it. First we run a sigmoid layer which decides what part of the cell state we’re going to output.

Then we put cell state through tanh (to push value between 1 and -1) and multiply it by the output of sigmoid gate, so that we only output that part we decided to.

Example: In language model, since it just saw a subject it might want to output information relevant to a verb, in case that’s what is coming next. For Example it might output whether subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follow next.

## Variants on LSTM

This was a pretty normal LSTM. But not all LSTM are same. Almost every LSTM uses slightly different version. Differences are minor.

All versions are pretty much same, but some worked better than the LSTM on certain task.

## More Resources

- Recurrent Neural Networks tutorial by Denny Britz

- Understanding LSTMs by Colah

- The Unreasonable Effectiveness of Recurrent Neural Networks by Andrej Karpathy

- Some Example for LSTMs and RNNs
