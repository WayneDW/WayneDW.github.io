---
title: 'Inside the Transformer'
subtitle: Ongoing explorations into how Transformers really work (TBC)
date: 2025-08-10
permalink: /posts/transformer_family/
category: Empirics
---


This is an ongoing blog where I explore and improve my understanding of the Transformer family:

I’ll keep sharing important things I discover about Transformers over time.




### Attention

How to reduce KV cache

https://www.spaces.ac.cn/archives/10091/comment-page-1

#### Self-attention


#### Multi-head attention

https://arxiv.org/pdf/2002.07028

https://arxiv.org/pdf/2106.09650

https://medium.com/@hassaanidrees7/exploring-multi-head-attention-why-more-heads-are-better-than-one-006a5823372b

https://medium.com/@nirashanelki/the-secret-of-multi-head-attention-2fdb72208b7f

https://sanjayasubedi.com.np/deeplearning/multihead-attention-from-scratch/

#### Linear Attention

Nyströmformer/ Performer

#### Flash Attention

#### Multi-head Latent Attention 

    DeepSeek

#### Scaling laws

#### Stacking of attentions to achieve better performance via @Shuangfei Zhai's tweets.

### Masks

Casual / Chunk-based Casual/ Bi-directional

### Position embedding

Sinusoidal

limited seq length; independence of PE: the difference between pos 1 and 2 is the same as position 2 and 500? (breaks if it goes beyond wavelength?)

Abs. PE/ Learnable P.E./ ALiBi/

#### RoPE

invariant to shift

1D/ 2D RoPE (ViT) / beta-base encoding

RoPE is a rotary transformation applied to the Query (Q) and Key (K) in Attention.？

the only one that applies to linear attentions so far?

RoPE base number LLaMA 3 choose 

how to do multimodal?

Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation


Rotary Positional Encoding (RoPE has an inductive bias towards left-to-right ordering, Sitan Chen's Train for theWorst, Plan for the Best:
Understanding Token Ordering in Masked Diffusions)


