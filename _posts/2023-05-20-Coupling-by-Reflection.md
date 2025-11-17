---
title: 'Coupling by Reflection (II)'
subtitle: A general coupling technique for characterizing a broad range of diffusions.
date: 2023-05-20 
permalink: /posts/coupling2/
category: Sampling
---

### Limitations of Synchronous Coupling

Given a $\kappa$-strongly convex drift $U$, we can apply the synchronous coupling for the diffusion process

$$\begin{align}
  \mathrm{\mathrm{d}X_t = U(X_t)\mathrm{d}t+\mathrm{d}W_t}\notag\\
  \mathrm{\mathrm{d}Y_t = U(Y_t)\mathrm{d}t+\mathrm{d}W_t}.\notag\\
\end{align}$$

Eliminating the Brownian motion, we obtain a contractivity property

$$\begin{align}
  \mathrm{\|X_t-Y_t\|\leq \|X_0-Y_0\|^2 \exp(-\kappa t)}.\notag
\end{align}$$

However, we cannot easily obtain the desired contraction when $U$ is not strongly convex. To address this issue, one should consider a more general coupling method based on a specic metric instead of the standard Euclidean metric. The diffusions may not contract almost surely, but rather in the average sense.

### Reflection Coupling

Define the coupling time $\mathrm{T_c=\inf \\{ t\geq 0  \| X_t =Y_t \\}}$. By definition, we know that $\mathrm{X_t=Y_t}$ for $\mathrm{t\geq T_c}$ {% cite mufa_chen %} {% cite reflection_coupling %} {% cite reflection_coupling_2 %}  {% cite coupling_hmc %}. When the drift term $U$ is zero, we observe that $\mathrm{\\|X_t-Y_t\\|}$ remains fixed for any $t$ and synchronous coupling doesn't induce any contraction. 

Let's explore an alternative coupling where the Brownian motion moves in the opposite direction. We anticipate with some probability the processes will merge [Why?].

$$\begin{align}
  \mathrm{\mathrm{d}X_t} &\mathrm{= U(X_t)\mathrm{d}t+\mathrm{d}W_t}\notag\\
  \mathrm{\mathrm{d}Y_t} &\mathrm{= U(Y_t)\mathrm{d}t+(\mathrm{I} - 2\cdot e_t e_t^{\intercal})\mathrm{d}W_t},\notag\\
\end{align}$$

where $\mathrm{e_t=\mathbb{I}\_{[X_t\neq Y_t]}\cdot \frac{X_t-Y_t}{\\|X_t-Y_t\\|}}$ and one can identify that $\mathrm{\widetilde W_t=\int_0^t \big[\mathrm{I} - 2\cdot e_s e_s^{\intercal} \big]\mathrm{d} s}$ is also a Brownian motion. In addition, $\mathrm{e_t e_t^{\intercal}}$ is the orthogonal projection onto the unit vector $\mathrm{e_t}$ [\[Hint\]](https://textbooks.math.gatech.edu/ila/projections.html) and you can easily check that $e_t$ is the eigenvector of $\mathrm{\mathrm{I} - 2\cdot e_t e_t^{\intercal}}$ with one eigenvalue $-1$..


### Supermartingales 

We first show $\mathrm{\exp(c\cdot t)f(G_t)}$ is a supermartingale, where $\mathrm{G_t=\\|X_t-Y_t\\|}$.

Apply Ito's lemma to $\mathrm{f(G_t)}$, where $f$ is a concave function to induce a new distance metric $\mathrm{d_f(X, Y)=f(\\|X-Y\\|)}$ {% cite reflection_coupling %}.

$$\begin{align}
  \mathrm{\mathrm{d} f(G_t)=2f'(R_t)\mathrm{d}W_t+\bigg\{f'(G_t)\cdot \bigg\langle U(X_t)-U(Y_t), \frac{X_t-Y_t}{\|X_t-Y_t\|}\bigg\rangle +2f''(G_t)\bigg\} \mathrm{d}t}.\notag
\end{align}$$

Assume $\mathrm{\langle U(X_t)-U(Y_t), X_t-Y_t\rangle \leq -\kappa(r) \frac{\\|X_t-Y_t\\|^2}{2}}$, where $\kappa(r)$ is not necessarily positive

$$\begin{align}
  \mathrm{\bigg\langle U(X_t)-U(Y_t), \frac{X_t-Y_t}{\|X_t-Y_t\|}\bigg\rangle \leq -\frac{1}{2} \cdot G_t \cdot\kappa(G_t)}. \notag
\end{align}$$

Further including the integration factor $\exp(c\cdot t)$, we have

$$\begin{align}
  \mathrm{\dfrac{\mathrm{d} \bigg[\exp(c\cdot t)f(G_t)\bigg]}{\exp(c\cdot t)}\leq 2f'(R_t) \mathrm{d}W_t + \bigg[-\frac{1}{2} G_t \cdot\kappa(G_t) f'(G_t)+2f''(G_t)+c \cdot f(G_t)\bigg]\mathrm{d}t}. \notag
\end{align}$$

In other words, it induces a supermartingale when we have

$$\begin{align}
\mathrm{-\frac{1}{2} G_t \cdot\kappa(G_t) f'(G_t)+2f''(G_t)+c \cdot f(G_t)\leq 0}.\notag
\end{align}$$


It implies that a proper $f$ may help us obtain the desired result

$$\begin{align}
  \mathrm{\mathrm{E}[f(\|X_t-Y_t\|)] \leq f(\|X_0-Y_0\|)\cdot \exp(-c\cdot t)}.\notag
\end{align}$$



### How to build such an $f$

#### A simple case when $c = 0$

We propose to find a $f$ that satisfies 

$$\begin{align}
\mathrm{f''(G_t)\leq \frac{1}{4} G_t \cdot\kappa(G_t) f'(G_t)}.\notag
\end{align}$$

The worst case is given by $\mathrm{f(R)=\int_0^{R} f'(s) \mathrm{d}s}$, where $f'$ is solved by Growall inequality

$$\begin{align}
\mathrm{f'(R)}&\mathrm{=\exp\bigg\{\int_0^R\frac{1}{4} s \cdot\kappa(G_t) \mathrm{d}s\bigg\}}.\notag
\end{align}$$

#### Extention to $c>0$

We aim to obtain the following dimension-independent bound in $R, L\in [0, \infty)$ {% cite reflection_coupling %}.

The general idea is to permit strong convexity outside of a ball with a given radius, within which local non-convexity is allowed.

$\mathrm{-\mathbb{I}\_{[\\|X_t-Y_t\\|< R]} L{\\|X_t-Y_t\\|^2}\leq \langle U(X_t)-U(Y_t), X_t-Y_t\rangle \leq \mathbb{I}\_{[\\|X_t-Y_t\\|\geq R]} K{\\|X_t-Y_t\\|^2}.}$



<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2023_coupling_reflection,
  title   ={{Coupling by Reflection (II)}},
  author  ={Wei Deng},
  journal ={waynedw.github.io},
  year    ={2023},
  howpublished = {\url{https://www.weideng.org/posts/coupling2/}}
}
```
{% endraw %}