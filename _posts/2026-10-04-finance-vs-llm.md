---
title: 'Finance and LLMs'
subtitle: What They Can Learn from Each Other
date: 2026-10-04
permalink: /posts/finance_llms/
category: Ideas
---

As a finance practitioner, and also an active researcher in LLMs, I’ve found many connections between the two fields, particularly in how they simulate possible future paths and assign value to outcomes.

#### Key Similarity

<style>
.similarity-table {
  display: table !important;
  width: 100%;
  border-collapse: collapse;
  margin: 1.25rem 0 1.75rem;
  font-size: 0.875em;
  line-height: 1.45;
  overflow: visible;
  border: 1px solid #d0d7de;
  border-radius: 8px;
}
.similarity-table th,
.similarity-table td {
  padding: 0.65rem 0.85rem;
  border: 1px solid #d0d7de;
  vertical-align: top;
  text-align: left;
  background: #fff;
}
.similarity-table th:first-child,
.similarity-table td:first-child {
  width: 1%;
  max-width: 7.5rem;
  padding-left: 0.65rem;
  padding-right: 0.65rem;
  font-weight: 600;
  white-space: nowrap;
  background: #f6f8fa;
}
.similarity-table th:not(:first-child),
.similarity-table td:not(:first-child) {
  width: auto;
}
.similarity-table th {
  text-align: center;
  font-weight: 700;
  background: #f6f8fa;
}
.similarity-table tr:nth-child(2n) td:not(:first-child) {
  background: #fafbfc;
}
@media (max-width: 640px) {
  .similarity-table,
  .similarity-table tbody,
  .similarity-table tr,
  .similarity-table th,
  .similarity-table td {
    display: block;
    width: 100% !important;
    white-space: normal;
  }
  .similarity-table th {
    border-bottom: none;
  }
  .similarity-table td {
    border-top: none;
  }
  .similarity-table tr + tr td:first-child {
    border-top: 1px solid #d0d7de;
  }
}
</style>

<table class="similarity-table">
  <thead>
    <tr>
      <th></th>
      <th>Finance</th>
      <th>LLMs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Simulator</td>
      <td>Options: Heston, etc; HFT/ market making: order book/ flow, queue dynamics.</td>
      <td>A pretrained language model generates possible future token sequences.</td>
    </tr>
    <tr>
      <td>Sampling</td>
      <td>Monte Carlo price paths; order, cancellation, and execution events.</td>
      <td>Inference / rollout samples many possible reasoning trajectories.</td>
    </tr>
    <tr>
      <td>Reward</td>
      <td>Option payoff; trading PnL adjusted for risk, costs, and market impact.</td>
      <td>A reward or verifier assigns value to each generated trajectory.</td>
    </tr>
    <tr>
      <td>Scale & Infra</td>
      <td>Options: pricing speed, Greeks, etc / HFT: latency, networking, co-location.</td>
      <td>Inference: throughput, bandwidth, parallelism, serving.</td>
    </tr>
    <tr>
      <td>Resource Allocation</td>
      <td>Expected return, risk, alpha, and correlations guide capital allocation.</td>
      <td>Scaling laws guide compute, data, and model-size allocation.</td>
    </tr>
    <tr>
      <td>Safety</td>
      <td>Control tail risk through stress tests, risk limits, and drawdown constraints.</td>
      <td>Control harmful behavior through evaluations, guardrails, and monitoring.</td>
    </tr>
  </tbody>
</table>

#### Key Difference

<table class="similarity-table">
  <thead>
    <tr>
      <th></th>
      <th>Finance</th>
      <th>LLMs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dimensionality</td>
      <td>Traditionally low- to medium-dimensional; increasingly high-dimensional</td>
      <td>Extremely high-dimensional</td>
    </tr>
    <tr>
      <td>State space</td>
      <td>Continuous in option pricing; discrete / event-driven in HFT</td>
      <td>Discrete token space</td>
    </tr>
    <tr>
      <td>Dynamics</td>
      <td>Stochastic price dynamics or discrete market events</td>
      <td>Discrete autoregressive sampling</td>
    </tr>
    <!-- <tr>
      <td>Methods</td>
      <td>Exploits smoothness with PDEs, Greeks, and numerical integration</td>
      <td>Exploits learned structure with sampling, search, and reinforcement learning</td>
    </tr> -->
    <tr>
      <td>Applications</td>
      <td>Pricing, trading, hedging, and risk management</td>
      <td>Reasoning, coding, and personal AI</td>
    </tr>
    <tr>
      <td>Data Scale</td>
      <td>Limited and noisy financial data; rare regimes and crises are especially scarce</td>
      <td>Massive datasets, but rare, specialized, and high-quality data remain scarce. </td>
    </tr>
  </tbody>
</table>


### What They Can Learn from Each Other

These similarities and differences suggest that the two fields can learn from each other.

Finance can learn from LLMs by moving beyond individually calibrated, low-dimensional models toward high-dimensional, multimodal market simulators learned jointly across assets, signals, and market states.

LLMs, in turn, can learn from finance by placing greater emphasis on safety, tail risk, uncertainty, stress testing, hard constraints, risk controls, and robustness when models fail.

In short, LLMs can help finance model a larger world, while finance can help LLMs handle failure.


#### Acknowledgment

Thanks to ChatGPT for helping refine the ideas and wording in this post.

<!-- 
---

#### Optimization & Allocation

- **Finance**: portfolio optimization hedges risk across dissimilar (anti-correlated) assets  
- **LLMs / Tech**: recommendation systems allocate similar items to similar users  

Both are constrained allocation problems under uncertainty. -->


<!-- 
---

#### State & Control

- **Finance**: hidden latent state inferred from noisy prices  
- **LLMs**: hidden activations inside deep neural networks  

Finance infers the state; LLMs *are* the state. -->
<!-- 
---

#### Fundamental analysis v.s. pretraining

#### model training


pretraining + fine-tuning. -->

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
