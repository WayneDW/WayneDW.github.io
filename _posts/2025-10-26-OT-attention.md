---
title: 'Attention = Optimal Transport? Yes.'
subtitle: The backward pass is policy gradient
date: 2025-10-26
permalink: /posts/attention_OT/
category: Transformer
---

The attention module {% cite attention_need %} is the computational engine behind modern LLMs, while *entropic optimal transport* (EOT) studies the efficient way to map one probability distribution to another.  At first glance, the two areas seem unrelated—but inspired by Elon’s work {% cite EOT_attention %}, it is quite mind-blowing to discover that these two mechanisms are, in fact, mathematically equivalent.

## Forward Attention Pass is Optimal Transport

In this post, we show that the optimal coupling $\mathrm{P\in \mathbb{R}^{n\times n}}$ in EOT with cost matrix $\mathrm{C = -QK^{\intercal}\in \mathbb{R}^{n\times n}}$ and entropy regularizer $\mathrm{\varepsilon=\sqrt{d}}$ is identical to the attention score $\mathrm{A\in \mathbb{R}^{n\times n}}$ in Transformers:

$$\begin{align}
\text{Attention:} \ \  \mathrm{A=\text{Softmax}\bigg(\dfrac{QK^{\intercal}}{\varepsilon}\bigg)} \ \  \Longleftrightarrow \ \  \text{EOT:} \ \  \mathrm{L_C(P)= \min_{P^\intercal 1 = 1, P\geq 0}\langle P, C\rangle -  \varepsilon\ H(P)}, \label{EOT_attention}
\end{align}$$

where $\mathrm{Q, K\in \mathbb{R}^{n\times d}}$ and $\mathrm{H(P)=\sum_{i,j=1}^n p_{ij} \log p_{ij}}$ is the discrete entropy. 


$$\textbf{Proof}$$ To handle both the simplex (column-sum) constraints for the RHS, we introduce Lagrange multipliers $$\lambda \in \mathbb{R}^n$$. The Lagrangian is written as

$$
\mathrm{\mathcal{L}(P, \lambda)
= \langle P, C \rangle
+ \sqrt{d}\,\langle P, \log P \rangle
+ \lambda^\top (\mathbf{1} - P^\top \mathbf{1}),}
$$

where the first term represents the transport cost, the second term adds an entropy regularizer to ensure smoothness and positivity. The third term enforces the column-sum simplex constraint.

At the optimum, the derivative of $$\mathcal{L}$$ with respect to $$\mathrm{P_{ij}}$$ must vanish by the KKT condition:

$$
\mathrm{\frac{\partial \mathcal{L}}{\partial P_{ij}}
= C_{ij} + \sqrt{d}(1 + \log P_{ij}) - \lambda_j = 0,}
$$

which leads to the solution:

$$
\mathrm{\log P_{ij} = \frac{\lambda_j - C_{ij}}{\sqrt{d}} - 1 \quad\text{or}\quad  P_{ij} \propto \exp\!\Big(-\frac{C_{ij}}{\sqrt{d}}\Big).}
$$

Since each column of $$\mathrm{P}$$ must satisfy $$\mathrm{\sum_j P_{ij} = 1}$$, we get

$$
\mathrm{1 = \sum_j P_{ij}
= a_i \sum_j \exp\!\Big(-\frac{C_{ij}}{\sqrt{d}}\Big)
\quad \Rightarrow \quad
a_i = \frac{1}{\sum_j \exp(-C_{ij}/\sqrt{d})}.}
$$

Hence, the optimal coupling with the cost matrix $\mathrm{C = -QK^{\intercal}}$ and the entropy regularizer $$\mathrm{\varepsilon=\sqrt{d}}$$ is

$$\begin{align}
\boxed{
\mathrm{P
= \text{Softmax}\bigg(\dfrac{QK^\intercal}{\sqrt{d}}\bigg)\equiv A
}}. \label{solution_OT}
\end{align}$$

Additionally, the Softmax operator in \eqref{EOT_attention} corresponds to the Shannon Entropy in EOT.


### Locality-Biased EOT -- Attention with Linear Biases (AliBi)

{% cite press2022train %} introduced the ALiBi embedding, which imposes a distance-based penalty on attention scores. This is equivalent to augmenting $$\mathrm{H(P)}$$ in \eqref{EOT_attention} with an additional linear penalty.

$$\begin{align}
\mathrm{H'(P)=H(P)+\gamma \sum_{j=1}^n P_{ij} |i-j|}. \notag
\end{align}$$


### PriorSoftmax: A Bayesian Extension

We can further generalize the entropic regularizer to Kullback-Leibler divergence to stay close to the prior $\mathrm{\pi}$. Such extension enables to encode inductive biases or location preference more smoothly. 

$$\begin{align}
\mathrm{L_{C, KL}(P)= \min_{P^\intercal 1 = 1, P\geq 0}\langle P, C\rangle -  \varepsilon\ KL(P, \pi)}. \notag
\end{align}$$

We can also verify that $$\mathrm{L_{C, KL}=L_{C_{KL}}}$$, where $$\mathrm{C_{KL}=C-\varepsilon \log \pi }$$ in the EOT formulation \eqref{EOT_attention}. 








## Backward Attention Gradient is Policy Gradient

Assume $\mathcal{L}\mathrm{(c)}$ is the training loss of the attention module $\mathrm{Attention(c)=\sum_{j=i}^n Softmax(\frac{c}{\varepsilon})_j v_j}$, where $\mathrm{c\in \mathbb{R}^{n}, v\in \mathbb{R}^{d}}$. For convenience, we define the scalar $\mathrm{u_j=Softmax(\frac{c}{\varepsilon})_j}$. 

By checking two cases $\mathrm{k=j}$ and $\mathrm{k\neq j}$ using quotient rule {% cite EOT_attention %}, we can verify that 

$$\begin{align}
\mathrm{\dfrac{\partial u_k}{\partial c_j} = \frac{1}{\varepsilon} u_k(\delta_{kj} - u_j)}\label{partial_grad}.
\end{align}$$


We also denote $\mathrm{g=\dfrac{\partial \mathcal{L}(c)}{\partial u}}$ and a delta function $\mathrm{\delta_{ij}}$ at $\mathrm{i=j}$. Now the back-propagate gradient follows

$$\begin{align}
\mathrm{\dfrac{\partial \mathcal{L}(c)}{\partial c_j}}&= \mathrm{\sum_{k=1}^n  g_k \dfrac{\partial u_k}{\partial c_j}} \notag \\
&=\frac{1}{\varepsilon} \mathrm{\sum_{k=1}^n  g_k  u_k(\delta_{kj} - u_j)} \qquad \text{By Eq.\eqref{partial_grad}}\notag \\
&=\mathrm{\frac{1}{\varepsilon} g_j  u_j - \sum_{k=1}^n  g_k  u_k u_j} \notag \\
&=\mathrm{\frac{u_j}{\varepsilon} \bigg(g_j   - \sum_{k=1}^n  g_k  u_k\bigg)} \notag \\
&=\mathrm{\frac{u_j}{\varepsilon}  \big(g_j   - E_u[g]\big) }. \notag 
\end{align}$$

The term $\mathrm{g_j   - E[g]}$ denotes the advantage function of choosing token $j$, which quantifies the relative gains or losses compared to the mean utility $\mathrm{E_u[g]}$ induced by the current attention policy. This aligns with the REINFORCE policy-gradient formulation that incorporates a baseline. The baseline serves as a reference value to reduce gradient variance and make the training more stable.



<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2025ot_attention,
  title   ={{Attention = Optimal Transport? Yes.}},
  author  ={Wei Deng},
  journal ={waynedw.github.io},
  year    ={2025},
  howpublished = {\url{https://www.weideng.org/posts/attention_OT/}}
}
```
{% endraw %}