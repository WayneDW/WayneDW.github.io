---
title: 'Finance is a Continuous LLMs'
subtitle: How market simulator is connected to LLMs
date: 2025-12-27
permalink: /posts/finance_llms/
category: Others
---

This is a running blog collecting observations on the structural similarities between modern financial systems and large language models.

* Continuous Market Simulators v.s. Discrete Token Simulators

** Finance simulates continuous prices, volatility, and indices

** LLMs simulate discrete token sequences
→ Both are world models rolling forward an uncertain state.

* Reinforcement Learning

Finance: option pricing, hedging, insurance → minimize tail risk

LLMs: RLHF / RLVR → align behavior with preferences
→ RL as risk control vs. behavior alignment

* Optimization & Allocation

Portfolio optimization balances return, risk, and correlation

Recommendation systems allocate similar items to similar users.


* Scale & Infrastructure

HFT: nanoseconds, latency, hardware dominance

LLMs: tokens/sec, pipeline parallelism, bandwidth limits
→ At scale, systems engineering dominates algorithms

* Alpha–Beta vs. Scaling Laws

Finance separates skill from market exposure

LLMs use scaling laws to predict final loss and stop training
→ Both guide capital / compute allocation

* State & Control

Finance: hidden latent state inferred from noisy prices

LLMs: hidden activations are the state

* Prompt engineering ↔ technical analysis: control without retraining

* Safety

Finance: drawdowns, VaR, stress tests

LLMs: toxicity, hallucination, misuse
→ Preventing catastrophic failure matters more than average performance

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