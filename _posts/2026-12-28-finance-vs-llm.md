---
title: 'Finance is a Latent LLM'
subtitle: How market simulators connect to LLMs
date: 2026-12-28
permalink: /posts/finance_llms/
category: Ideas
---

As a practitioner in finance, who is also an active researcher in large language models (LLMs), I've found that finance and LLMs are closely intertwined. In a sense, the finance system can be seen as a latent large language models in the following aspects: 


---

#### Environment: Market Simulators vs. Token Simulators

In finance, a market simulator is a synthetic finance world model that generates prices, order flows, volatility, and liquidity and indices to test your strategies and models. 


LLMs simulate **discrete token sequences** to model language and reasoning trajectories.

---

#### Reinforcement Learning

- **Finance**: option pricing, hedging, and insurance → minimize tail risk  
- **LLMs**: RLHF / RLVR → align behavior with user preferences  

RL functions as **risk control** in finance versus **behavior alignment** in LLMs.

---

#### Optimization & Allocation

- **Finance**: portfolio optimization hedges risk across dissimilar (anti-correlated) assets  
- **LLMs / Tech**: recommendation systems allocate similar items to similar users  

Both are constrained allocation problems under uncertainty.

---

#### Scale & Infrastructure

- **HFT**: nanoseconds, latency, hardware dominance  
- **LLMs**: tokens/sec, pipeline parallelism, bandwidth limits  

At scale, systems engineering dominates algorithmic details.

---

#### Alpha–Beta vs. Scaling Laws

Finance uses **alpha** for excess returns and **beta** to model market exposure.

LLMs rely on **scaling laws** to predict final loss and determine when to stop training.

Both guide **capital and compute allocation**.

---

#### State & Control

- **Finance**: hidden latent state inferred from noisy prices  
- **LLMs**: hidden activations inside deep neural networks  

Finance infers the state; LLMs *are* the state.

---

#### Prompt Engineering vs. Technical Analysis

- **Finance (technical analysis)**: conditioning trades via patterns in past prices  
- **LLMs (prompting)**: conditioning behavior via input structure  

Control without retraining.

---

#### Safety

- **Finance**: minimize catastrophic loss via VaR, stress tests, and drawdown limits  
- **LLMs**: minimize harmful outputs by filtering toxicity, hallucination, and misuse  

In both systems, tail risk matters more than average performance.


#### Fundamental analysis v.s. pretraining

#### model training


pretraining + fine-tuning.

<!-- 
continuous Market simulator of continuous prices, index.
discrete token simulator of languages 


RL: option pricing insurace purpose v.s. RLVR in LLM


portfolio optimization similar/ anti-similar for de-risk or leverage? recommend similar items for similar people; v.s. tech de-risk? recommendation? 

can be extremely techie: high freq trading nano seconds v.s. large scale pretraining how to conduct pipeline parallel etc.


alpha beta earn money v.s. scaling law alpha beta predict final loss when to end. what is the optimal training tokens, costs. ...

finance in hidden state; while LLM is the state


Safety: finance min loss; LLM toxic words ...


prompt engineering v.s. techinical analysis -->