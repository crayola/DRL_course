# Chapter 1: Introduction to Deep Reinforcement Learning

## Lesson 1.1 Introduction to Reinforcement Learning
- Learner will be able to explain the core concepts of reinforcement learning, including agents, environments, states, actions, and rewards.
- Terms / concepts introduced: agent, environment, state, action, reward.

## 1.2 Traditional reinforcement learning techniques
- Learner will be able to explain the difference between value-based and policy-based methods in reinforcement learning.
- Learner will be able to explain the Q-learning algorithm.
- Terms / concepts introduced: Value function, policy, Q-learning, Bellman equation, on-policy vs off-policy.

## 1.3 Setting up the Deep Reinforcement Learning Environment
- Learner will be able to set up a development environment with PyTorch and OpenAI Gym.
- Tools introduced: PyTorch installation, OpenAI Gym installation, basic environment setup.


# Chapter 2: Deep Q-learning

## 2.1 The DQN algorithm
- Learner will be able to explain the DQN algorithm.
- Learner will be able to describe the architecture and components of Deep Q-Networks (DQN), including how neural networks are used to approximate Q-values, the role of experience replay in breaking correlation between sequences, and the purpose of fixed Q-targets in stabilizing the learning process.
- Terms/concepts introduced: Q-learning, DQN architecture, experience replay, fixed Q-targets, Epsilon greediness.

## 2.2 Implementing DQN on CartPole
- Learner will be able to set up the CartPole environment from OpenAI Gym and understand its dynamics and how they represent reinforcement learning challenges.
- Learner will be able to implement a DQN model in PyTorch to solve the CartPole problem, integrating core components such as neural networks for Q-value approximation, epsilon greediness, experience replay for memory, and fixed Q-targets for stability.
- Learner will be able to evaluate and iterate on the DQN model, using performance metrics to assess learning efficiency and making adjustments to improve the model.
- Tools introduced: implementing DQN in PyTorch

## 2.3 Advanced Topics in Value-Based Learning
- Learner will be able to explain advanced topics in deep Q learning, such as prioritized experience replay and distributional DQN, to understand how they can further improve the performance and efficiency of DQN models.
- Learner will be able to discuss the practical implications of these advanced techniques in real-world applications and their potential impact on model performance.
- Terms / concepts introduced: Prioritized experience replay, distributional DQN, double DQN

## 2.4 Summary and Limitations of Value-Based Methods
- Learner will be able to summarize the key takeaways from value-based methods, highlighting the strengths and where they excel in reinforcement learning problems.
- Learner will be able to articulate the limitations of value-based methods, such as challenges in handling high-dimensional action spaces and the preference for policy-based methods in certain scenarios.
- This lesson provides a bridge to policy-based methods, setting the stage for the next chapter's focus.


# Chapter 3: Policy-Based Methods

## 3.1 Policy Methods: Background & Theory
- Learner will be able to explain the underlying framework of policy-based methods
- Learner will be able to articulate the relative strengths of policy vs value-based methods

## 3.2 Policy gradient methods & Reinforce
- Learner will be able to explain the underlying theory behind policy-gradient methods
- Learner will be able to explain and apply the Reinforce algorithm

## 3.3 Proximal Policy Optimization (PPO) Theory
- Learner will be able to explain the theoretical framework behind PPO and TRPO
- Learner will be able to compare PPO with TRPO and explain the improvements introduced by PPO
- Terms introduced: KL-divergence, clipped surrogate objective function

## 3.4 TRPO and PPO Hands-On with PyTorch
- Learner will be able to apply TRPO and PPO to practical problems using PyTorch, solidifying their understanding through application.
- Tools introduced: torchrl

# Chapter 4: Advanced Topics and Self-Study

## 4.1 Actor-Critic Methods with PyTorch Hands-On
- Learner will be able to explain the actor-critic architecture and how it leverages both the policy and value based frameworks.
- Learner will be able to apply it to solve a reinforcement learning problem in PyTorch.
- Terms introduced: Actor Critic, A2C.

## 4.2 Hyperparameter Optimization with Optuna
- Learner will be able to perform hyperparameter optimization for reinforcement learning models using Optuna.
- Terms introduced: hyperpareter, optuna, trial, study, pruning.

## 4.3 Further Learning and Exploration
- Learner will be able to explain the current landscape of DRL research and industry applications.
- This will include resources and directions for further study, potentially covering topics like multi-agent RL, generalization, advanced actor-critic methods (e.g., A3C, SAC), model-based RL, and exploration techniques.
