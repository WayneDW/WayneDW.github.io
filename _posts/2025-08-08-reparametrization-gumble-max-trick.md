---
title: 'Unfied Reparametrization Tricks'
subtitle: Backpropagation of continuous and discrete random variables
date: 2025-08-08
permalink: /posts/reparametrization_tricks/
category: Empirics
---



$$
\begin{align}
\nabla_\theta \mathbb{E}_{p_\theta(\mathbf{z})}[f_\theta(\mathbf{z})] 
&= \nabla_\theta \left[ \int_{\mathbf{z}} p_\theta(\mathbf{z}) f_\theta(\mathbf{z}) \, d\mathbf{z} \right] \\
&= \int_{\mathbf{z}} \nabla_\theta \left[ p_\theta(\mathbf{z}) f_\theta(\mathbf{z}) \right] \, d\mathbf{z} \\
&= \int_{\mathbf{z}} f_\theta(\mathbf{z}) \nabla_\theta p_\theta(\mathbf{z}) \, d\mathbf{z} 
  + \int_{\mathbf{z}} p_\theta(\mathbf{z}) \nabla_\theta f_\theta(\mathbf{z}) \, d\mathbf{z} \\
&= \underbrace{\int_{\mathbf{z}} f_\theta(\mathbf{z}) \nabla_\theta p_\theta(\mathbf{z}) \, d\mathbf{z}}_{\text{Not necessarily an expectation}}
  + \mathbb{E}_{p_\theta(\mathbf{z})} \left[ \nabla_\theta f_\theta(\mathbf{z}) \right]
\end{align}
$$

The first term of the last equation is not guaranteed to be an expectation. Monte Carlo methods require that we can sample from \(p_\theta(\mathbf{z})\), but not that we can take its gradient. This is not a problem if we have an analytic solution to \(\nabla_\theta p_\theta(\mathbf{z})\), but this is not true in general.

Now that we have a better understanding of the problem, let’s see what happens when we apply the reparameterization trick to our simple example. To be consistent with Kingma, I’ll switch to bold text for vectors and denote the \(i\)-th sample of vector \(\mathbf{v}\) as \(\mathbf{v}^{(i)}\) and \(l \in \{1,\dots,L\}\) to denote the \(l\)-th Monte Carlo sample:

$$
\begin{align}
\boldsymbol{\epsilon} &\sim p(\boldsymbol{\epsilon})
\end{align}
$$

$$
\begin{align}
\mathbf{z} &= g_\theta(\boldsymbol{\epsilon}, \mathbf{x})
\end{align}
$$

$$
\begin{align}
\mathbb{E}_{p_\theta(\mathbf{z})}[f(\mathbf{z}^{(i)})] 
&= \mathbb{E}_{p(\boldsymbol{\epsilon})}\left[ f\left(g_\theta(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right) \right]
\end{align}
$$

$$
\begin{align}
\nabla_\theta \mathbb{E}_{p_\theta(\mathbf{z})}[f(\mathbf{z}^{(i)})] 
&= \nabla_\theta \mathbb{E}_{p(\boldsymbol{\epsilon})}\left[ f\left(g_\theta(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right) \right] \tag{1}
\end{align}
$$

$$
\begin{align}
&= \mathbb{E}_{p(\boldsymbol{\epsilon})}\left[ \nabla_\theta f\left(g_\theta(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right) \right] \tag{2}
\end{align}
$$

$$
\begin{align}
&\approx \frac{1}{L} \sum_{l=1}^L \nabla_\theta f\left(g_\theta(\boldsymbol{\epsilon}^{(l)}, \mathbf{x}^{(i)})\right) \tag{3}
\end{align}
$$