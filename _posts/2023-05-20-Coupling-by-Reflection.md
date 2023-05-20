---
title: 'Coupling by Reflection (II)'
subtitle: A general coupling technique for characterizing a broad range of diffusions.
date: 2023-05-20 
permalink: /posts/coupling_by_reflection/
---

### Limitations of Synchronous Coupling

Consider the diffusion process
\begin{align}
  \mathrm{d}X_t = U(X_t)\mathrm{d}t+\mathrm{d}W_t.\notag
\end{align}

If the drift $U$ is $\kappa$-strongly convex, one can apply the simple synchronous coupling 

$$\begin{align}
  \mathrm{d}X_t = U(X_t)\mathrm{d}t+\mathrm{d}W_t\notag\\
  \mathrm{d}Y_t = U(Y_t)\mathrm{d}t+\mathrm{d}W_t.\notag\\
\end{align}$$

Eliminating the Brownian motion, we obtain a contractivity property

$$\begin{align}
  \|X_t-Y_t\|\leq \|X_0-Y_0\|^2 \exp(-\kappa t).\notag
\end{align}$$

However, we cannot easily obtain the desired contraction when $U$ is not strongly convex. To address this issue, one should consider a more general coupling method. The diffusions may not contract almost surely, but rather in the average sense.

### Reflection Coupling

When the drift term $U$ is zero, we observe that $\\|X_t-Y_t\\|$ remains fixed for any $t$ and synchronous coupling doesn't induce any contraction. Let's explore an alternative coupling where the Brownian motion moves in the opposite direction. We anticipate that, with some probability, the processes will eventually merge.

$$\begin{align}
  \mathrm{d}X_t &= U(X_t)\mathrm{d}t+\mathrm{d}W_t\notag\\
  \mathrm{d}Y_t &= U(Y_t)\mathrm{d}t+R_{X,Y}\mathrm{d}W_t,\notag\\
\end{align}$$

where 

$$\begin{align}
  R_{X,Y}&=(\mathrm{I}d - 2\cdot e_{X, Y}e_{X, Y}^{\intercal})\notag \\
  e_{X, Y}&=\mathbb{I}[X\neq Y]\cdot \frac{X-Y}{\|X-Y\|}\notag\\
\end{align}$$

Add intuition for this ???

### Supermartingales 

We first show $\exp(c\cdot t)f(G_t)$ is a supermartingale, where $G_t=\\|X_t-Y_t\\|$.

Apply Ito's lemma to $f(G_t)$, where $f$ is some concave function to be defined later

$$\begin{align}
  \mathrm{d} f(G_t)=2f'(R_t)\mathrm{d}W_t+\bigg\{-\frac{1}{2}f'(G_t)\cdot \bigg\langle U(X_t)-U(Y_t), \frac{X_t-Y_t}{\|X_t-Y_t\|}\bigg\rangle +2f''(G_t)\bigg\}\cdot \mathrm{d}t.\notag
\end{align}$$

Assume $\langle U(X_t)-U(Y_t), X_t-Y_t\rangle \leq -\kappa(r) \frac{\\|X_t-Y_t\\|^2}{2}$, where $\kappa(r)$ is not necessarily positive

$$\begin{align}
  \bigg\langle U(X_t)-U(Y_t), \frac{X_t-Y_t}{\|X_t-Y_t\|}\bigg\rangle \leq -\frac{1}{2} \cdot G_t \cdot\kappa(G_t). \notag
\end{align}$$

It follows that

$$\begin{align}
  \frac{1}{4} G_t \cdot\kappa(G_t) f'(G_t)+2f''(G_t)+c \cdot f(G_t)\leq 0. \notag
\end{align}$$

It implies that a proper $f$ helps us obtain the desired result

$$\begin{align}
  \mathrm{E}[f(\|X_t-Y_t\|)] \leq f(\|X_0-Y_0\|)\cdot \exp(-c\cdot t).\notag
\end{align}$$


Computational issues?

### How to build such as $f$

TBD




{% cite reflection_coupling_2 %}
{% cite reflection_coupling %}
{% cite durmus_moulines %}
{% cite coupling_hmc %}
{% cite mufa_chen %}

