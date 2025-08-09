---
title: 'The Reparametrization and Gumble-Max Tricks'
subtitle: Backpropagation of continuous and discrete random variables
date: 2025-08-08
permalink: /posts/reparametrization_gumble_max_tricks/
category: Empirics
---




$$
\begin{align}
\nabla_\theta \mathbb{E}_{p_\theta(z)}[f_\theta(z)] 
= \nabla_\theta \left[ \int_z p_\theta(z) f_\theta(z) dz \right]
= \int_z \nabla_\theta \left[ p_\theta(z) f_\theta(z) \right] dz
= \int_z f_\theta(z) \nabla_\theta p_\theta(z) dz + \int_z p_\theta(z) \nabla_\theta f_\theta(z) dz
= \underbrace{\int_z f_\theta(z) \nabla_\theta p_\theta(z) dz}_{\text{What about this?}} 
+ \mathbb{E}_{p_\theta(z)} \left[ \nabla_\theta f_\theta(z) \right]
\end{align}
$$

The first term of the last equation is not guaranteed to be an expectation. Monte Carlo methods require that we can sample from \( p_\theta(z) \), but not that we can take its gradient. This is not a problem if we have an analytic solution to \( \nabla_\theta p_\theta(z) \), but this is not true in general.

Now that we have a better understanding of the problem, let’s see what happens when we apply the reparameterization trick to our simple example. To be consistent with Kingma, I’ll switch to bold text for vectors and denote the \( i \)-th sample of vector \(\mathbf{v}\) as \(\mathbf{v}^{(i)}\) and \(l \in L\) to denote the \(l\)-th Monte Carlo sample:

\[
\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})
\]
\[
\mathbf{z} = g_\theta(\boldsymbol{\epsilon}, \mathbf{x})
\]
\[
\mathbb{E}_{p_\theta(z)}[f(\mathbf{z}^{(i)})] = \mathbb{E}_{p(\epsilon)}[f(g_\theta(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))]
\]
\[
\nabla_\theta \mathbb{E}_{p_\theta(z)}[f(\mathbf{z}^{(i)})] 
= \nabla_\theta \mathbb{E}_{p(\epsilon)}[f(g_\theta(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))] \tag{1}
\]
\[
= \mathbb{E}_{p(\epsilon)}[\nabla_\theta f(g_\theta(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))] \tag{2}
\]
\[
\approx \frac{1}{L} \sum_{l=1}^L \nabla_\theta f(g_\theta(\boldsymbol{\epsilon}^{(l)}, \mathbf{x}^{(i)})) \tag{3}
\]