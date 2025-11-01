---
title: 'Attention = Optimal Transport? Yes.'
subtitle: And the backward pass is policy gradient
date: 2025-10-26
permalink: /posts/attention_OT/
category: Diffusion
---

The attention module {% cite attention_need %} is the computational engine behind modern LLMs, while *entropic optimal transport* (EOT) studies the optimal way to map one probability distribution to another.  At first glance, the two areas seem unrelated—but inspired by Elon’s work {% cite EOT_attention %}, it is quite remarkable to discover that these two mechanisms are, in fact, mathematically equivalent.


In this post, we show that the optimal coupling $\mathrm{P}$ in EOT with cost matrix $\mathrm{C = -QK^{\intercal}}$ is identical to the attention matrix $\mathrm{A}$ in Transformers:

$$\begin{align}
\text{Attention:} \quad \mathrm{A=\text{Softmax}\bigg(\dfrac{QK^{\intercal}}{\sqrt{d}}\bigg)} \quad \Longleftrightarrow \quad \text{EOT:} \quad \mathrm{L_C(P)= \min_{P^\intercal 1 = 1, P\geq 0}\langle P, C\rangle - \sqrt{d} \ H(P)}, \notag
\end{align}$$

where $\mathrm{Q, K\in \mathbb{R}^{n\times d}, C, P \in \mathbb{R}^{n\times n}}$, $\mathrm{H(P)}$ is the discrete entropy.



$$\textbf{Proof}$$ To handle both the simplex (column-sum) constraints, we introduce Lagrange multipliers $$\lambda \in \mathbb{R}^d$$. The Lagrangian is written as

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
\mathrm{\log P_{ij} = \frac{\lambda_j - C_{ij}}{\sqrt{d}} - 1. \quad\text{or}\quad  P_{ij} \propto \exp\!\Big(-\frac{C_{ij}}{\sqrt{d}}\Big).}
$$

Since each column of $$\mathrm{P}$$ must satisfy $$\mathrm{\sum_j P_{ij} = 1}$$, we get

$$
\mathrm{1 = \sum_j P_{ij}
= a_i \sum_j \exp\!\Big(-\frac{C_{ij}}{\sqrt{d}}\Big)
\quad \Rightarrow \quad
a_i = \frac{1}{\sum_j \exp(-C_{ij}/\sqrt{d})}.}
$$

Hence, the optimal coupling with $$\mathrm{C=-Q K^\intercal}$$ is

$$
\boxed{
\mathrm{P
= \text{Softmax}\bigg(\dfrac{QK^\intercal}{\sqrt{d}}\bigg)\equiv A
}}.
$$



Hence, the *attention matrix* in Transformers can be precisely viewed as the **optimal transport plan** in an entropic OT problem with cost matrix $\mathrm{C = -QK^{\intercal}}$ and the entropy regularizer $$\mathrm{\varepsilon=\sqrt{d}}$$. 