---
title: 'Generative Transformer'
subtitle: Transformers for image synthesis
date: 2025-08-01
permalink: /posts/generative_transformer/
category: Multimodal
---




an image is flattened into a 1D sequence of tokens, long sequences and quadratic cost; challenges for not only modeling long-term correlation but also renders the slow inference. 



VQVAE 1-2 - Code-book

https://huggingface.co/blog/ariG23498/understand-vq

$$
\begin{align}
\text{Quantize}(E(x)) = e_k \quad \text{where} \quad k = \arg\min_j \| E(x) - e_j \|
\end{align}
$$


VQGAN



<!-- ongoing blog TBD to learn Generative Transformer

VQVAE/ Saining/ MaskGIT/ Muse/ CIP

Masked Autoencoders Are Scalable Vision Learners

causal attention/ bidirectional transformer 

muse -->


