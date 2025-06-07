---
title: 'REINFORCE and Stochastic Control'
subtitle: Discrete and continuous-time policy gradients
date: 2025-06-06
permalink: /posts/soc_and_reinforce/
category: Sequential Sampling
---

by ricky's def

$p^*(x)\propto p^{\text{base}}(x)\exp\{(r(x))\}$

use PIS (13)

$$\begin{align*}
KL(Q^{\mu}|Q^0 \mu)=E[\int_0^T \frac{1}{2} \|u\|^2 dt + \log \frac{\mu^0}{\mu}]
\end{align*}$$






[1] Theoretical guarantees for sampling and inference in generative models with latent diffusions.

[2] PATH INTEGRAL SAMPLER: A STOCHASTIC CONTROL APPROACH FOR SAMPLING