---
title: "Policy Gradient and the Actor-Critic Algorithm"
url: https://adityajain.me/blogs/actor-critic.html
---

# Policy Gradient and the Actor-Critic Algorithm

- Reinforcement Learning
- Policy Gradient

Today most successful reinforcement algorithm A3C, PPO etc belong to the policy gradient family of algorithm often more specifically to the actor-critic family. But before going into mathematics of these algorithm, you must have basic knowledge of Deep Q Learning(below link).

Deep Q Learning

Before going any further, let’s understand some drawbacks of Q Learning.

- The policy implied by Deep Q-Learning is deterministic. This means Q-Learning can’t learn stochastic policies, which can be useful in some environments. It means we also need to create our own exploration strategies.

- There is no way to handle continuous action spaces in Q-Learning. In policy gradient handling continuous action spaces is relatively easy.

- In policy gradient we are calculating the gradient of policy itself. By contrast Q-Learning we are improving value estimates of different actions in a state, which implicitly improves policy. Improving policy directly is more efficient.

## Policy Gradient (REINFORCE Algorithm)

Let’s call πθ(a∣s)\pi \theta(a|s)πθ(a∣s) the probability of taking action a in state s. θ\thetaθ represents parameters of policy (the weights of Neural Network). The goal is to update θ\thetaθ to values that make πθ\pi \thetaπθ the optimal policy. θt\theta_tθt​ represent the values of thetathetatheta in iteration t. We want to find out the update rule that takes use from θt\theta_tθt​ to θt+1\theta_{t+1}θt+1​ to optimize our policy.

For a discrete action space we will use neural network with softmax output unit, so that output can be thought of as a probability of taking each action in state. Clearly, if action a⋆a^\stara⋆ is optimal action, we want πθ(a⋆∣s)\pi\theta(a^\star | s)πθ(a⋆∣s) as close to 1 as possible. For this we can simply perform an gradient ascent to update θ\thetaθ in the following way:

θt+1=θt+α∇πθt(a⋆∣s)\theta_{t+1} = \theta_t + \alpha \nabla \pi_{\theta_t}(a^\star|s)θt+1​=θt​+α∇πθt​​(a⋆∣s)

We can view ∇πθt(a⋆∣s)\nabla \pi \theta_t(a^\star|s)∇πθt​(a⋆∣s) as direction in which we must move θt\theta_tθt​ to increase value of πθ(a⋆∣s)\pi\theta(a^\star | s)πθ(a⋆∣s). Note we are using gradient ascent to increase a value. Thus one way to view this update is that we keep “pushing” towards more of action a⋆a^\stara⋆ in our policy, which is indeed what we want.

Of course, in practice, we won’t know which action is best… After all that’s what we’re trying to figure out in the first place! To get back to the metaphor of “pushing”, if we don’t know which action is optimal, we might “push” on suboptimal actions and our policy will never converge. One solution would be to “push” on actions in a way that is proportional to our guess of the value of these actions. We will call our guess of the value of action a in state s Q̂(s,a). We get the following gradient ascent update, that we can now apply to each action in turn instead of just to the optimal action:

θt+1=θt+αQ^t(s,a)∇πθt(a∣s)\theta_{t+1} = \theta_t + \alpha \hat Q_t(s,a) \nabla \pi_{\theta_t}(a|s)θt+1​=θt​+αQ^​t​(s,a)∇πθt​​(a∣s)

Of course, in practice, our agent is not going to choose actions uniformly at random, which is what we implicitly assumed so far. Rather, we are going to follow the very policy πθ\pi\thetaπθ that we are trying to train! This is called training on-policy. There are two reasons why we might want to train on-policy:

- We accumulate more rewards even as we train, which is something we might value in some contexts.

- It allows us to explore more promising areas of the state space by not exploring purely randomly but rather closer to our current guess of the optimal actions.

This creates a problem with our current training algorithm, however: although we are going to “push” stronger on the actions that have a better value, we are also going to “push” more often on whichever actions happen to have higher values of πθ to begin with (which could happen due to chance or bad initialization)! These actions might end up winning the race to the top in spite of being bad. This means that we need to compensate for the fact that more probable actions are going to be taken more often. How do we do this? Simple: we divide our update by the probability of the action. This way, if an action is 4x more likely to be taken than another, we will have 4x more gradient updates to it but each will be 4x smaller.

θt+1=θt+αQ^t(s,a)∇πθt(a∣s)πθ(a∣s)\theta_{t+1} = \theta_t + \alpha \frac{\hat Q_t(s,a) \nabla \pi_{\theta_t}(a|s)}{\pi_\theta(a|s)}θt+1​=θt​+απθ​(a∣s)Q^​t​(s,a)∇πθt​​(a∣s)​

Now, we can write ∇πθt(a∣s)πθ(a∣s)\frac{\nabla \pi_{\theta_t}(a|s)}{\pi_\theta(a|s)}πθ​(a∣s)∇πθt​​(a∣s)​ as ∇θlogπtheta(s∣a)\nabla_\theta log \pi_theta(s|a)∇θ​logπt​heta(s∣a) and using return vtv_tvt​ as an unbiased sample of Q^t(s,a)\hat Q_t(s,a)Q^​t​(s,a).

θt+1=θt+αvt∇θlogπθ(s∣a)\theta_{t+1} = \theta_t + \alpha v_t \nabla_\theta log \pi_\theta(s|a)θt+1​=θt​+αvt​∇θ​logπθ​(s∣a)

A widely used variation of REINFORCE is to subtract a baseline value from the return vtv_tvt​ to reduce the variance of gradient estimation while keeping the bias unchanged (Remember we always want to do this when possible). As it turns out, REINFORCE will still work perfectly fine if we subtract any function from Q̂(s,a) as long as that function does not depend on the action. This means that using the Â function instead of the Q̂ function is perfectly allowable. For example, a common baseline is to subtract state-value from action-value, and if applied, we would use advantage A(s,a) = Q(s,a) – V(s) in the gradient ascent update.

θt+1=θt+αA^t(s∣a)∇θlogπθ(s∣a)\theta_{t+1} = \theta_t + \alpha \hat A_t(s|a) \nabla_\theta log \pi_\theta(s|a)θt+1​=θt​+αA^t​(s∣a)∇θ​logπθ​(s∣a)

where,

- At(s,a)=Rt−bA_t(s,a) = R_t - bAt​(s,a)=Rt​−b (Return - baseline)

- πθ(s∣a)=policy\pi_\theta(s|a) = policyπθ​(s∣a)=policy

#### Algorithm for Monte Carlo REINFORCE Algorithm:

- Initialize θ\thetaθ arbitrarily, baseline b

- For each episode {s1,a1,r2,....,sT−1,aT−1,rT}\{s_1,a_1,r_2,....,s_{T-1}, a_{T-1}, r_T\}{s1​,a1​,r2​,....,sT−1​,aT−1​,rT​}

- for t= 1 to T-1

- Rt=∑t′=tt′=T−1γt′−trt′R_t = \sum_{t'=t}^{t' = T-1} \gamma^{t'-t} r_t'Rt​=∑t′=tt′=T−1​γt′−trt′​

- A^t(st,at)=Rt−b(st)\hat A_t(s_t,a_t) = R_t - b(s_t)A^t​(st​,at​)=Rt​−b(st​)

- θ←θ+αAt(st,at)∇θlogπθ(st,at)\theta \leftarrow \theta + \alpha A_t(s_t,a_t) \nabla_\theta log \pi_\theta(s_t,a_t)θ←θ+αAt​(st​,at​)∇θ​logπθ​(st​,at​)

- Return θ\thetaθ

#### Code for Monte Carlo Policy Gradient to solve gym Cart-pole environment:

import gym
import numpy as np
import collections

import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PGN(nn.Module):
def __init__(self,input_size,n_actions):
super(PGN,self).__init__()

self.net = nn.Sequential(
nn.Linear(input_size,128),
nn.ReLU(),
nn.Linear(128,n_actions)
)

def forward(self,x):
logits = self.net(x)
return F.softmax(logits)

class MeanBuffer():
def __init__(self,capacity):
self.capacity = capacity
self.deque = collections.deque(maxlen=capacity)
self.sum = 0.0

def add(self,val):
if len(self.deque)==self.capacity:
self.sum -= self.deque[0]
self.deque.append(val)
self.sum+=val

def mean(self):
if not self.deque:
return 0.0
return self.sum/len(self.deque)

TARGET_REWARD = 195
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
ENTROPY_BETA = 0.01
BELLMAN_STEPS = 10
BASELINE_STEPS = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v0')
net = PGN(env.observation_space.shape[0], env.action_space.n).to(device)

agent = ptan.agent.PolicyAgent(net,preprocessor=ptan.agent.float32_preprocessor,device=device)

exp_source = ptan.experience.ExperienceSourceFirstLast(env,agent,gamma=GAMMA,
steps_count=BELLMAN_STEPS)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

total_rewards = []
step_rewards = []
baseline_buf = MeanBuffer(BASELINE_STEPS)
step_idx = 0
done_episodes = 0

batch_states, batch_actions, batch_scales = [], [], []
for step_idx, exp in enumerate(exp_source):
baseline_buf.add(exp.reward)
baseline = baseline_buf.mean()
batch_states.append(exp.state)
batch_actions.append(exp.action)
batch_scales.append(exp.reward - baseline)

episode_rewards = exp_source.pop_total_rewards()
if episode_rewards:
done_episodes += 1
reward = episode_rewards[0]
total_rewards.append(reward)
mean_rewards = float(np.mean(total_rewards[-100:]))
print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
step_idx, reward, mean_rewards, done_episodes))
if mean_rewards > TARGET_REWARD:
print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
break

if len(batch_states) < BATCH_SIZE:
continue

#copy training data to the GPU
states_v = torch.FloatTensor(batch_states).to(device)
batch_actions_t = torch.LongTensor(batch_actions).to(device)
batch_scale_v = torch.FloatTensor(batch_scales).to(device)

#apply gradient descent
optimizer.zero_grad()
#softmax output

prob_v = net(states_v)

#apply logarithm
log_prob_v = torch.log(prob_v)
#scale the log probs according to (reward - baseline)
log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
#take the mean cross-entropy across all batches
loss_policy_v = -log_prob_actions_v.mean()

# subtract the entropy bonus from the loss function
entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
entropy_loss_v = -ENTROPY_BETA * entropy_v
loss_v = loss_policy_v + entropy_loss_v

loss_v.backward()
optimizer.step()

batch_states.clear()
batch_actions.clear()
batch_scales.clear()

## Actor-Critic Algorithm:

Until now we have studied Critic-only methods and Actor-only methods. Critic-only methods that use temporal difference learning have a lower variance in the estimates of expected returns. A straightforward way of deriving a policy in critic-only methods is by selecting greedy actions for which the value function indicates that the expected return is the highest. Actor-only methods (example Policy Gradient) typically work with a parameterized family of policies over which optimization procedures can be used directly. In actor-critic spectrum of continuous actions can be generated, but the optimization methods used (typically called policy gradient methods) suffer from high variance in the estimates of the gradient, leading to slow learning.

Actor-critic methods combine the advantages of actor-only and critic-only methods. While the parameterized actor brings the advantage of computing continuous actions without the need for optimization procedures on a value function, the critic’s merit is that it supplies the actor with low-variance knowledge of the performance. More specifically, the critic’s estimate of the expected return allows for the actor to update with gradients that have lower variance, speeding up the learning process. Actor-critic methods usually have good convergence properties, in contrast to critic-only methods.

In Actor-only policy gradient method we were reducing variance and increasing stability by subtracting the cumulative reward by a baseline. Intuitively, making the cumulative reward smaller by subtracting it with a baseline will make smaller gradients, and thus smaller and more stable updates.

θt+1=θt+αQ^t(s∣a)∇θlogπθ(s∣a)\theta_{t+1} = \theta_t + \alpha \hat Q_t(s|a) \nabla_\theta log \pi_\theta(s|a)θt+1​=θt​+αQ^​t​(s∣a)∇θ​logπθ​(s∣a)

As we know, the Q value can be learned by parameterizing the Q function with a neural network.

This leads us to Actor Critic Methods, where:

- The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).

- The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).

and both the Critic and Actor functions are parameterized with neural networks. In the derivation above, the Critic neural network parameterizes the Q value — so, it is called Q Actor Critic.

### Problem with Policy Gradient

We are in a situation of Monte Carlo, waiting until the end of episode to calculate the reward. We may conclude that if we have a high reward (R(t)), all actions that we took were good, even if some were really bad.

As we can see in this example, even if A3 was a bad action (led to negative rewards), all the actions will be averaged as good because the total reward was important. As a consequence, to have an optimal policy, we need a lot of samples. This produces slow learning, because it takes a lot of time to converge. What if, instead, we can do an update at each time step? The Actor Critic model is a better score function. Instead of waiting until the end of the episode as we do in Monte Carlo REINFORCE, we make an update at each step (TD Learning).

### Advantage Actor-Critic (A2C)

Let’s go back to baseline function which is subtracted from value to reduce variance of gradient estimation, using the V function as the baseline function, we subtract the Q value term with the V value. Intuitively, this means how much better it is to take a specific action compared to the average, general action at the given state. We will call this value the advantage value:

A(st,at)=Q(st,at)−V(st)A(s_t,a_t) = Q(s_t,a_t) - V(s_t)A(st​,at​)=Q(st​,at​)−V(st​)

we can rewrite this using bellman optimality equation as,

A(s,t)=rt+1+γV(st+1)−V(st)A(s,t) = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)A(s,t)=rt+1​+γV(st+1​)−V(st​)

So, this is Advantage Actor-Critic Equation:

θt+1=θt+αA^t(s∣a)∇θlogπθ(s∣a)\theta_{t+1} = \theta_t + \alpha \hat A_t(s|a) \nabla_\theta log \pi_\theta(s|a)θt+1​=θt​+αA^t​(s∣a)∇θ​logπθ​(s∣a)

### Asynchronous Advantage Actor-Critic (A3C)

A3C implements parallel training where multiple workers in parallel environments independently update a global value function—hence “asynchronous.” One key benefit of having asynchronous actors is effective and efficient exploration of the state space.

## More Resources

- An Intuitive Explanation of Policy Gradient.

- RL Course of David Silver - Lecture

- Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto

- Example codes and problems to understand concepts better.
