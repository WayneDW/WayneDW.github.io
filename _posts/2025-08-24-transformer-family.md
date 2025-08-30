---
title: 'Inside the Transformers'
subtitle: Explorations into how transformers work
date: 2025-08-24
permalink: /posts/transformer_family/
category: Empirics
---


Transformers are the building blocks of modern large language and visions models and have transformed the way we work and create. At their core is the *attention mechanism* {% cite attention_need %}, which captures complex temporal patterns and enables reasoning across sequences.

## Softmax Attention

Attention computes weighted combinations of input tokens so that the model can focus on the most relevant parts of the sequence when producing each output.

$$\begin{align}
\mathrm{\text{Attention}(Q, K, V)=\underbrace{\text{Softmax}\bigg(\frac{Q K^{\intercal}}{\sqrt{d}}\bigg)}_{\text{Attention weight} \ P} V} \notag
\end{align}$$

where $\mathrm{Q\in \mathbb{R}^{n\times d}}$, $\mathrm{K\in \mathbb{R}^{m\times d}}$, $\mathrm{V\in \mathbb{R}^{m\times d}}$, $\mathrm{d}$ is the hidden size. $n$ is the number of query positions and $m$ is number of KV positions. A scaling factor of $\frac{1}{\sqrt{d}}$ is employed to account for the linear variance. 

In a self-attention example: "The VC will not invest money to the startup" with $\mathrm{n=m=L}$, an attention head may focus on the financial relation between VC and money, the syntactic dependency between VC and invest, or other types of relationships. Because the inner product is linear and softmax overly focuses on extreme values, we can only capture a single type of relationship. 

### Masks

Attention masks control what tokens can “see.” A *bidirectional* mask (no mask) lets tokens attend to all others (BERT {% cite devlin2019bert %}), while a *causal* mask enforces left-to-right attention for generation (GPT). A padding mask ignores [PAD] tokens in batching, and an MLM mask randomly hides tokens (BERT’s [MASK]) for prediction from context.


<figure style="text-align: center;">
    <img src="/images/4_masks.png" width="550" height="120" />
    <figcaption> Bidirectional mask  $\ \ \qquad$ causal mask $\ \ \qquad$ padding mask $\ \ \qquad$ random mask </figcaption>
</figure>

```python
def scaled_dot_product(q, k, v, mask=None):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / d**0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn
```



### Multi-head attention 

To integrate knowledge from different representation subspaces, multi-head attention (MHA) proposes an ensemble of locally linear projections of queries, keys and values 

$$\begin{align}
\mathrm{\text{MHA}(X_Q, X_K, X_V)} &= \mathrm{[\text{Head}_1; \text{Head}_2; \ldots, \text{Head}_h] W^O} \notag \\
\mathrm{\text{Head}_i} &=  \mathrm{\text{Attention}(X_Q W_i^Q, X_K W_i^K, X_V W_i^V)}, \notag
\end{align}$$


where $\mathrm{W_i^Q, W_i^K , W_i^V \in \mathbb{R}^{d\times d_{\text{embed}}/h}}$ and  $\mathrm{W^O \in \mathbb{R}^{d_{\text{embed}}\times d}}$ are the model parameters. Increasing the number of heads $\mathrm{h}$ enhances representational diversity, but also introduces a low-rank bottleneck within each head {% cite pmlr-v119-bhojanapalli20a %}. The model architecture is listed below:

<figure style="text-align: center;">
    <img src="/images/multihead_attention.png" width="200" height="250" />
    <figcaption> Multi-head attention {% cite attention_need %} </figcaption>
</figure>

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=bias)
        self.o_proj  = nn.Linear(embed_dim, input_dim, bias=bias)

    def forward(self, x, mask=None):
        B, L, _ = x.size()
        qkv = self.qkv_proj(x) # (B,L,3*E)
        qkv = qkv.view(B, L, self.num_heads, 3*self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1) # each (B,H,L,D)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        out = self.o_proj(values)

        return out, attention
```

<!-- ### Techniques

#### Teacher Forcing -->

#### KV Cache in Inference  

Let $\mathrm{x_{1:t}=(x_1, x_2, ..., x_t)}$ denote the token sequence up to step $\mathrm{t}$. Each transformer layer computes the query/ key/ value vectors $\mathrm{q_i=x_i W^Q, k_i=x_i W^K, v_i=x_i W^V}$ for $\mathrm{i\in\\\{1, ..., t\\\}}$. In the inference of autoregressive transformers, the attention output at position $\mathrm{t}$ is

<!-- $$\begin{equation}
\mathrm{11}. \notag
\end{equation}$$ -->

$$\begin{align}
\mathrm{H_t= \text{Softmax}\bigg(\frac{q_t K_{1:t}^{\intercal}}{\sqrt{d}}\bigg) V_{1:t}}, \notag \\
\mathrm{K_{1:t}} = \begin{bmatrix}
\mathrm{k_1} \\ 
\vdots \\ 
\mathrm{k_t}
\end{bmatrix}, 
\quad 
\mathrm{V_{1:t}} = \begin{bmatrix}
\mathrm{v_1} \\ 
\vdots \\ 
\mathrm{v_t}
\end{bmatrix}.\notag
\end{align}$$

To avoid the quadratic cost of repeatly computing all key value vectors $\mathrm{k_{1:t}}$ and $\mathrm{v_{1:t}}$ at each timestamp $\mathrm{t}$. We only need to compute the key-value pair $\mathrm{k_t, v_t}$ and update the cache as follows

$$\begin{align}
\mathrm{K_{1:t}} = \begin{bmatrix}
\mathrm{K_{1:t-1}} \\ 
\mathrm{k_t}
\end{bmatrix}, 
\quad 
\mathrm{V_{1:t}} = \begin{bmatrix}
\mathrm{V_{1:t-1}} \\ 
\mathrm{v_t}
\end{bmatrix}.\notag
\end{align}$$

### Efficient Attentions

Self-attention requires $\mathrm{O(L^2)}$ time and memory complexity for the length-$\mathrm{L}$ sequence generation. To mitigate the cost, one can exploit structural properties of the attention weight $\mathrm{P}$, such as low-rankness {% cite wang2020linformer %}, sparsity {% cite child2019generating %} {% cite beltagy2020longformer %}, and Kernelization {% cite choromanski2021rethinking %}, among others {% cite tay2022efficient %}.

* **Low-rank**: The query-key inner product acts as a rank-1 approximation to captur one *similarity* pattern. Although softmax relaxes this rank-1 constraint, the attention weight $\mathrm{P}$ remain approximately low-rank in practice (see Theorem 1 in {% cite wang2020linformer %}), yielding $O(L)$ complexity.
* **Sparsity**: Longformer {% cite beltagy2020longformer %} leverages a (dilated) sliding window to capture local dependencies and assigns global attention to pre-specified tokens, enhancing modeling flexibility with $O(L)$ complexity. Reformer {% cite kitaev2020reformer %} employs Locality-Sensitive Hashing (LSH) to group similar items into the same buckets, and each query attends only to tokens within its bucket, resulting $O(L\log L)$ complexity.

<div style="display: flex; justify-content: center; gap: -50px;">
  <figure style="text-align: center;">
    <img src="/images/longformer_d.png" width="120" height="120" />
    <figcaption>Longformer attention.</figcaption>
  </figure>

  <figure style="text-align: center;">
    <img src="/images/reformer_d.png" width="120" height="120" />
    <figcaption>Reformer attention.</figcaption>
  </figure>
</div>
* **Kernelization**: The attention weight $\mathrm{P}$ can be viewed as an exponential kernel $$\mathrm{\exp(x^\intercal y)=\exp(\|x\|_2^2)K_{\text{gaussian}}(x, y) \exp(\|y\|_2^2)}$$ and a prior [random feature blog](https://www.weideng.org/posts/random_fourier_features/) has ever discussed about the Monte Carlo approximations {% cite random_features %}. Building on this idea, Performer {% cite choromanski2021rethinking %} introduces non-negativity random features to avoiding singularities during normalization.

### Autoregressive Transformers v.s. RNNs

Consider a linear relaxation of the exponential linear product {% cite katharopoulos2020transformers %}:

$$\begin{equation}
\mathrm{y_t\propto \sum_{i=1}^t v_i \exp\bigg(\frac{k_i^\intercal q_t}{\sqrt{d}}\bigg) \quad \overset{\text{linearization}}{\Rightarrow} \quad  y_t\propto \sum_{i=1}^t v_i  k_i^\intercal q_t }. \notag
\end{equation}$$

Exploit the *recurrency* property and ignore the normalizing constant {% cite wang2025testtime %}: 

$$\begin{equation}
\mathrm{S_t = S_{t-1} + v_t k_t^\top, \quad y_t = S_t q_t = \sum_{i=1}^t v_i k_i^\intercal q_t,} \notag
\end{equation}$$

where the key–value rank-1 pair $\mathrm{v_t k_t^\top}$ is written into the memory. Although the recurrent update lacks parallelism, this can be mitigated through chunkwise parallelism. One can further increase the expressiveness by replacing the linear inner product  $\mathrm{k^\top q}$ with a feature map $\mathrm{\phi(k)^\top \phi(q)}$:

$$\begin{equation}
\mathrm{S_t = S_{t-1} + v_t \phi(k_t)^\top ,\quad y_t = S_t \phi(q_t), }\notag
\end{equation}$$

where the choices of $\phi$ include $1+\mathrm{ELU}$ {% cite katharopoulos2020transformers %}, random features {% cite choromanski2021rethinking  %}, cosine functions, polynomial expansions, deterministic projections, among others.

#### Gated Linear Attention

Compressing all past pairs equally tends to degrade performance as sequence length grows. To solve this issue, a forgetting gate $\mathrm{G_t\in (0,1)^{d\times d}}$ proposes to learn a $\mathrm{x_t}$-dependent decaying matrix and may result in more hardware-efficient training:

$$\begin{equation}
\mathrm{S_t = G_t \odot S_{t-1} + v_t k_t^\top , \qquad y_t = S_t q_t .}\notag
\end{equation}$$


<figure style="text-align: center;">
    <img src="/images/different_gates.png" width="670" height="150" />
    <figcaption> Different gating formulations {% cite yang2024gated %} </figcaption>
</figure>

<!-- 
#### Flash Attention -->


### Position Embeddings
<!-- 
#### Scaling laws

#### Stacking of attentions to achieve better performance via @Shuangfei Zhai's tweets. -->

#### Absolute Positional Embedding

* Learnable: A common approach is to use nn.Embedding for positional encodings, as in BERT {% cite devlin2019bert %} and GPT-2, which is effective but limited by the training sequence length.

* Sinusoidal: {% cite attention_need %} proposes a deterministic, non-trainable encoding and is in spirit similar to random features {% cite random_features %}. 

<!-- One may increase the base 10000 to support longer sequence length. -->

$$\begin{equation}
\mathrm{PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),  \qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)}.\notag
\end{equation}$$

<!-- limited seq length; independence of PE: the difference between pos 1 and 2 is the same as position 2 and 500? (breaks if it goes beyond wavelength?) -->

#### Relative Position Encodings (RPE)

RPE augments the attention logits with terms that depend on $\mathrm{i-j}$ {% cite shaw2018self dai2019transformer %}:

$$\begin{equation}
\mathrm{e_{ij}=\frac{Q_i^\top K_j}{\sqrt{d}} + Func(i-j, Q_i, K_j)}.\notag
\end{equation}$$

#### Attention with Linear Biases (ALiBi)

To achieve longer extrapolation length beyond the training length, {% cite press2022train %} penalizes distant key-value pairs

$$\begin{equation}
\mathrm{e_{ij} = \frac{Q_i^\top K_j}{\sqrt{d}} + m_h (i - j)}, \notag
\end{equation}$$

where $\mathrm{m_h}$ is a fixed slope constant, e.g. $\mathrm{m_h=-\frac{1}{2^{h/H}}, h\in \\\{1, 2, ...\\\}}$.



#### Rotary Position Embeddings (RoPE)

Building on insights from complex analysis, {% cite su2021roformer %} proposed a point-wise rotation of the Q/K matrices, which has since been widely adopted in state-of-the-art LLM architectures.

$$\begin{equation}
\mathrm{e_{ij}=\frac{(R_i Q_i)^\top R_j K_j}{\sqrt{d}}=\frac{ Q_i\top R_{j-i} K_j}{\sqrt{d}}}.\notag
\end{equation}$$

where $\mathrm{R(\alpha)^\top R(\beta)=R(\beta-\alpha)}$ is the complex inner product and $\mathrm{R(\alpha)}$ is defined below

$$\begin{align}
& \mathrm{z' = e^{im\theta} z  =(\underbrace{\cos(m\theta) + i \sin(m\theta)}_{e^{im\theta}})(\underbrace{q_0 + i q_1}_{z})=} \underbrace{\begin{bmatrix}
\mathrm{\cos(m\theta)} & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}}_{\mathrm{R(m\theta)}}
\begin{bmatrix}
\mathrm{q_0} \\ \mathrm{q_1}
\end{bmatrix}.\notag
\end{align}$$

<!-- where the 2nd row represents the imaginary axis. -->

For higher dimensions, {% cite su2021roformer %} proposes the 2-by-2 block-wise rotation matrix 

$$\begin{equation}
\mathrm{R_m} =
\begin{bmatrix}
\mathrm{\cos(m\theta_0)} & -\mathrm{\sin(m\theta_0)} & 0 & 0 & \cdots   \\
\mathrm{\sin(m\theta_0)} & \;\cos(m\theta_0) & 0 & 0 & \cdots    \\
0 & 0 & \mathrm{\cos(m\theta_1)} & \mathrm{-\sin(m\theta_1)} & \cdots    \\
0 & 0 & \mathrm{\sin(m\theta_1)} & \;\cos(m\theta_1) & \cdots   \\
\vdots & \vdots & \vdots & \vdots & \ddots   \\
\end{bmatrix}
\begin{bmatrix}
\mathrm{q_0} \\ \mathrm{q_1} \\ \mathrm{q_2} \\ \mathrm{q_3} \\ \vdots
\end{bmatrix}. \notag
\end{equation}$$




<!-- 1D/ 2D RoPE (ViT) / beta-base encoding -->

<!-- RoPE is a rotary transformation applied to the Query (Q) and Key (K) in Attention.？ -->

<!-- the only one that applies to linear attentions so far? -->

<!-- RoPE base number LLaMA 3 choose  -->


<!-- #### ALiBi (Attention with Linear Biases) -->

<!-- Abs. PE/ Learnable P.E./ ALiBi/ -->


<!-- how to do multimodal? -->

<!-- Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation -->


<!-- Rotary Positional Encoding (RoPE has an inductive bias towards left-to-right ordering, Sitan Chen's Train for theWorst, Plan for the Best: -->
<!-- Understanding Token Ordering in Masked Diffusions) -->


### The Transformer Family

<!-- 

### Bert

#### RoBert.

#### ALBERT

#### DistilBERT

#### DeBERTa

#### T5

### GPT
 -->
