---
title: 'Autoregressive Flow'
subtitle: Efficient transport maps
date: 2024-03-12
permalink: /posts/triangular_maps/
category: Flow
---

Normalizing flows are early pioneers in generating modeling by mapping base distributions via a series of invertible transformations. The main idea is to transform the data $x$ through 


https://arxiv.org/pdf/1912.02762.pdf


Normaling flow: change of variable, the key part is Jacobian transformation.


composition

triangular map

understand the loss function:

zs, prior_logprob, log_det = model(x)
logprob = prior_logprob + log_det
loss = -torch.sum(logprob) # NLL

Masked autoencoder (why do we need parity?) RNN is not needed, better efficiency. https://www.youtube.com/watch?v=lNW8T0W-xeE




JAX: Real-NVP --  https://blog.evjang.com/2019/07/nf-jax.html

JAx: Flow: https://github.com/ChrisWaites/jax-flows

https://github.com/karpathy/pytorch-normalizing-flows/tree/master

FYI
