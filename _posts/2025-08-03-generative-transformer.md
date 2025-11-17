---
title: 'Generative Transformer with Enigmatic Orders'
subtitle: Image synthesis via transformers
date: 2025-08-03
permalink: /posts/generative_transformer/
category: Transformer
---


A naïve way to process images with transformers is to flatten them into 1D token sequences and train an autoregressive model. However, this leads to long sequences — especially for high-resolution images — making it hard to capture spatial correlations and causing high computational cost and slow inference. 

To learn the spacial correlations more efficiently, MaskGIT {% cite chang2022maskgit %} leverages the codebook from VQ-VAE {% cite oord2017vqvae %} and adopted a bi-direction transformer decoder with parallel non-autoregressive decoding to speed-up the inference.

<figure style="text-align: center;">
    <img src="/images/maskgit.png" width="600" height="300" />
    <figcaption> Non-causal parallel decoding from MaskGIT. Credit to {% cite chang2022maskgit %} </figcaption>
</figure>

#### Vector Quantized Variational AutoEncoder (VQ-VAE)

VQ-VAE {% cite oord2017vqvae %} {% cite razavi2019vqvae2 %} aims to represent an image as a sequence of discrete token indices. Given an encoder and a learnable codebook of embeddings $\mathrm{\\{e_j\\}}$, the VQ block utilizes an encoder to map each image patch to a feature map to find the index of the nearest codebook vector in the codebook.

$$
\begin{align}
\mathrm{\text{Quantize}(Encoder(x)) = e_k, \quad \text{where} \quad k = \arg\min_j \| Encoder(x) - e_j \|}.\notag
\end{align}
$$

The VQ block can be implementated as follows [Code](https://huggingface.co/blog/ariG23498/understand-vq):
```python
# Initialize the codebook as an embedding matrix
self.embedding = nn.Embedding(num_embeddings, embedding_dim)

# Compute L2 distances between encoded vectors z and each codebook embedding (||a - b||^2)
distances = (
    torch.sum(z ** 2, dim=-1, keepdim=True)                            # ||a||²
    + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)   # ||b||²
    - 2 * torch.matmul(z, self.embedding.weight.t())                   # -2⟨a, b⟩
)

# Find the nearest codebook vector for each encoded input
encoding_indices = torch.argmin(distances, dim=-1)

# Retrieve quantized vectors from the codebook using the nearest indices
z_q = self.embedding(encoding_indices)
```

With a well-optmized VQ block, MaskGIT introduces a novel image reconstruction pipeline that models discrete visual tokens.

<figure style="text-align: center;">
    <img src="/images/VQ-VAE-structure.png" width="600" height="140" />
    <figcaption> MaskGIT model pipeline. Credit to {% cite chang2022maskgit %} </figcaption>
</figure>




<!-- In contrast to CNNs, they contain no inductive bias that prioritizes local interactions.  -->

#### Parallel Decoding

Autoregressive generation is prohibitively slow for high-resolution images. To address this, MaskGIT abandons the autoregressive formulation and instead leverages bidirectional self-attention — similar to BERT {% cite devlin2019bert %} — to enable generation in all directions. In each iteration, the model predicts all masked tokens in parallel and retains only the most confident ones for the next step.

<!-- ###  -->

<!-- ongoing blog TBD to learn Generative Transformer

VQVAE/ Saining/ MaskGIT/ Muse/ CIP

Masked Autoencoders Are Scalable Vision Learners

causal attention/ bidirectional transformer 

muse -->



<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2025_generative_transformer,
  title   ={{Generative Transformer with Enigmatic Orders}},
  author  ={Wei Deng},
  journal ={waynedw.github.io},
  year    ={2025},
  url     ="https://www.weideng.org/posts/generative_transformer/"
}
```
{% endraw %}