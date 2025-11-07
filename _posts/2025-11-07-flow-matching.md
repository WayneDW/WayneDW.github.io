---
title: 'Flow Matching'
subtitle: Learning vector fields in continuous and discrete Spaces
date: 2025-11-07
permalink: /posts/flow_matching/
category: Diffusion
---


Flow Matching (FM) {% cite flow_match %} has seen impressive success in image and video generation [\[Link\]](https://ai.meta.com/research/movie-gen/). The key idea is to train a neural network to learn the underlying vector (velocity) field that deterministically pushes particles along this transport path.


### Continuous State Spaces

#### Problem setup
Consider a vector field $$u_t : \mathbb{R}^d \to \mathbb{R}^d$$ and define its **flow map** $$\phi_t$$ as the solution of the ODE

$$
\begin{equation}
\frac{d}{dt}\phi_t(x) = u_t(\phi_t(x)), \quad \phi_0(x) = x.
\end{equation}
$$

Let $$p_0$$ be the base distribution (e.g., Gaussian).  $$p_t$$ is defined as the **pushforward** of $$p_0$$ under $$\phi_t$$:

$$
\begin{equation}
p_t = (\phi_t)_\# p_0.
\end{equation}
$$

Equivalently, by the change-of-variables formula [\[Link\]](https://www.cs.ubc.ca/~murphyk/Teaching/Stat406-Spring08/homework/changeOfVariablesHandout.pdf):

$$
\begin{equation}
p_t(x) = p_0(\phi_t^{-1}(x)) \left| \det \nabla \phi_t^{-1}(x) \right|.
\end{equation}
$$

The pair $$(p_t, u_t)$$ satisfies the **continuity equation** (see Theorem 1 {% cite neural_ode %}):

$$
\begin{equation}
\frac{\partial}{\partial t} p_t(x)
= - \nabla_x \cdot \big( u_t(x)\, p_t(x) \big).
\end{equation}
$$

To approximate the true vector field $$u_t(x)$$ via a parameterize $$u_{\theta}(t, x)$$, the ideal regression loss is:

$$
\begin{equation}
\mathcal{L}_{\mathrm{FM}}(\theta)
= \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t}
\big[ \| u_\theta(t,x) - u_t(x) \|^2 \big].\notag
\end{equation}
$$

This is **intractable**, since both $$p_t$$ and $$u_t$$ are unknown.

#### Conditional Flow Matching

To make the loss tractable, we introduce a conditioning variable $$X_1$$ (e.g., the data point) and define a **conditional vector field** $$u_t(x \mid X_1)$$, along with a **conditional** flow map $$\psi_t(\cdot \mid X_1)$$ satisfying

$$
\begin{equation}
\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x \mid X_1)
= 
u_t\big( \psi_t(x \mid X_1) \mid X_1 \big), 
\quad 
\psi_0(x \mid X_1) = x.
\end{equation}
$$

The unconditional velocity field can then be expressed as a conditional expectation:

$$
\begin{equation}
u_t(x)=\int u_t(x \mid x_1) \dfrac{p_t(x\mid x_1) q(x_1)}{p_t(x)}dx_1.
\end{equation}
$$

This means that minimizing the conditional loss will still learn the correct global velocity field. We can now define the CFM loss as an expectation over both the condition $$X_1\sim p_1(\cdot)$$ and the conditional flow trajectories:

$$
\begin{equation}
\mathcal{L}_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{t,\, q(x_1),\, x_0 \sim p_0(\cdot \mid X_1)}
\Big[
\big\|
u_\theta\big(t, \psi_t(x_0 \mid X_1)\big)
-
\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x_0 \mid X_1)
\big\|^2
\Big].
\end{equation}
$$

Intuitively:
- $$\psi_t(x_0 \mid X_1)$$ gives us a sample at time $$t$$;
- $$\frac{\mathrm{d}}{\mathrm{d}t}\psi_t$$ provides the **ground-truth velocity** along this analytic path;
- $$u_\theta$$ tries to predict that velocity.

By regressing $$u_{\theta}$$ to match this conditional velocity, we obtain an unbiased estimator of the original (intractable) FM loss.


### Discrete State Spaces
