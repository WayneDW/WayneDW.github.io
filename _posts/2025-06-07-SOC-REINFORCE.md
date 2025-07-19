---
title: 'Policy Gradients and Stochastic Control'
subtitle: Thoughts on diffusion model alignment
date: 2025-06-07
permalink: /posts/soc_and_reinforce/
category: Sampling
---

## Stochastic Control

Consider a diffusion process described by

$$\begin{align}\label{uncontrolled_process}
    \mathrm{d X_t=b(X_t, t)dt+dW_t, \ \ t\in[0, 1]; \ X_0^u=x_0.}
\end{align}$$

This process induces a probability measure $\mathbb{P}$.

Now introduce a controlled diffusion process governed by

$$\begin{align}\label{controlled_process}
    \mathrm{d X_t^{u}=\big[b(X_t^u, t)+u(X_t^u, t)\big]dt+dW_t, \ \ t\in[0, 1]; \ X_0^u=x_0.}
\end{align}$$

which generates a different path measure $\mathbb{P}^u$ under control $u$.

We also define the cost-to-go function

$$\begin{align}\label{def_J}
    \mathrm{J^u(x, t):=E\bigg[\int_t^1 \frac{\alpha}{2}\|u(X_s^u, s)\|_2^2 ds - r(X_1^u)\bigg|X_t^u = x\bigg]},\notag
\end{align}$$

where $r(\cdot)$ is a terminal reward function.

Define the value function 

$$\begin{align}\label{def_value_func}
    \mathrm{v(x, t):=\inf_{u\in\mathcal{U}} J^u(x, t)}.\notag
\end{align}$$

### Bellman Equation 

The dynamic programming principle yields the Bellman equation {% cite tzen2019theoretical %}:

$$\begin{align}\label{HJB}
    \mathrm{\partial_t v +\nabla v^\intercal b_t+\frac{1}{2} \Delta v =-\inf_{u\in\mathcal{U}} \bigg[\frac{\alpha}{2}\|u(x, t)\|_2^2 + \nabla v^\intercal u_t\bigg]}\notag. 
\end{align}$$


Solving the optimal control yields:

$$\begin{align}
\mathrm{\partial_t v +\nabla v^\intercal b_t -\frac{1}{2\alpha}\|\nabla v\|^2_2 +\frac{1}{2} \Delta v=0}\notag. 
\end{align}$$

<!-- where $\mathrm{v(x, 1)=-r(x)}$. -->


Consider the Cole-Hopf transformation:

$$\begin{align}
\mathrm{\phi(x, t):=\exp\bigg(\frac{-v(x, t)}{\alpha}\bigg) \Leftrightarrow v(x, t)=-\alpha\log \phi(x, t)}\notag. 
\end{align}$$

Applying this change of variable, we can obtain a linear backward Kolmogorov equation 

$$\begin{align}
\mathrm{\partial_t \phi + \nabla \phi^\intercal b_t +\frac{1}{2} \Delta \phi=0}\notag, 
\end{align}$$

where $\mathrm{\phi(x, 1)=\exp\big(\frac{-v(x, 1)}{\alpha}\big)=\exp\big(\frac{r(x)}{\alpha}\big)}$. 

By Feynman-Kac theorem {% cite zhang2022path %}, the value function follows that

$$\begin{align}
\mathrm{v(x, t)=-\alpha\log E\bigg[\exp\bigg(\frac{r(X_1)}{\alpha}\bigg)\bigg|X_t=x\bigg]}\notag. 
\end{align}$$

where $\mathrm{X_t}$ evolves according to the uncontrolled process \eqref{uncontrolled_process}.


### Objective Reformulation

Comparing the path measures $\mathbb{P}^u_{t,x}$ and $\mathbb{P}_{t,x}$, Girsanov theorem leads to the Radon–Nikodym derivative
<!-- \[
\mathrm{KL}\left( \mathbb{P}^u_{t,x} \,\|\, \mathbb{P}_{t,x} \right)
= \mathbb{E}_{\mathbb{P}^u_{t,x}} \left[
    \log \left( \frac{\mathrm{d} \mathbb{P}^u_{t,x}}{\mathrm{d} \mathbb{P}_{t,x}} \right)
\right]
= \mathbb{E}_{\mathbb{P}^u_{t,x}} \left[
    \mathrm{\frac{1}{2\alpha} \int_t^1 \| u(X_s, s) \|^2 \, \mathrm{d}s}
\right]
\] -->

$$\begin{align}
\mathrm{KL}\left( \mathbb{P}^u_{t,x} \,\|\, \mathbb{P}_{t,x} \right)
&= \mathbb{E}_{\mathbb{P}^u_{t,x}} \left[
    \log \left( \frac{\mathrm{d} \mathbb{P}^u_{t,x}}{\mathrm{d} \mathbb{P}_{t,x}} \right)
\right] \notag \\
&= \mathrm{\mathbb{E}_{\mathbb{P}^u_{t,x}} \left[\int_t^1 \| u(X_s, s) \|^2 \, \mathrm{d}s
\right]}\notag
\end{align}$$

With this, the cost-to-go function can be reformulated as {% cite domingo-enrich2024adjoint %}:

$$\begin{align}\label{def_J_2}
    \mathrm{J^u(x, t):=\frac{\alpha}{2}\mathrm{KL}\left( \mathbb{P}^u_{t,x} \,\|\, \mathbb{P}_{t,x} \right) + \mathbb{E}\big[- r(X_1^u)\big|X_t^u = x\big]},
\end{align}$$

where $\mathrm{X^u_t}$ is simulated from the controlled process \eqref{controlled_process}.




## Connections to Alignment in Diffusion Models

Consider a discretization of the diffusion path measure such that $$\mathrm{\mathbb{P}^u_{0,x}\approx \prod_{t=T}^0 p^{\theta}_t}$$ and $$\mathrm{\mathbb{P}_{0,x}\approx \prod_{t=T}^0 p_t}$$. For consistency with the backward time indexing used in diffusion models, we also reverse the time axis.

The minimization of $\mathrm{J^u(x, t)}$ in \eqref{def_J_2} is equivalent to

$$\begin{align}\label{RL_objective}
    \mathrm{argmax_{\{p^{\theta}_t\}_{t=T}^0} E_{\{p^{\theta}_t\}_{t=T}^0}\bigg[r(x_0)-\frac{\alpha}{2} \sum_{t=T}^1 KL(p^{\theta}_{t-1}(x_{t-1}|x_t)\| p_{t-1}(x_{t-1}|x_t))\bigg]}
\end{align}$$

which recovers a standard objective in RL-based fine-tuning for diffusion models {% cite fan2023dpok %}.

Taking the gradient of Eq.\eqref{RL_objective} approximately yields the following

$$\begin{align}
    \mathrm{E_{\{p^{\theta}_t\}_{t=T}^0}\bigg[r(x_0)\sum_{t=T}^1 \nabla \log p^{\theta}_t(x_{t-1}|x_t) -\frac{\alpha}{2} \sum_{t=T}^1 \nabla KL(p^{\theta}_{t-1}(x_{t-1}|x_t)\| p_{t-1}(x_{t-1}|x_t))\bigg]}.\label{grad_RL}\notag
\end{align}$$

The first part of gradient has also been adopted by the classic REINFORCE algorithm, which, however, suffers from the large variance issue. 

#### Variance Reduction with a Value Function Baseline

Motivated by control variate/ actor-critic method, we consider a baseline $$\mathrm{V^{\theta}(x_t):= \mathbb{E}[r(x_0) \\| x_t]}$$ to mimic the advantage function:

$$\begin{align}
    \mathrm{E_{\{p^{\theta}_t\}_{t=T}^0}\bigg[\sum_{t=T}^1 \big(r(x_0)-V^{\theta}(x_t)\big)\nabla \log p^{\theta}_t(x_{t-1}|x_t)\bigg]}.\label{grad_RL_VR}\notag
\end{align}$$

We can easily check that $$
    \mathrm{E_{p^{\theta}_t}\bigg[V^{\theta}(x_t)\nabla \log p^{\theta}_t(x_{t-1}|x_t)\bigg]= V^{\theta}(x_t)\nabla \int p^{\theta}_t(x_{t-1}|x_t)=0}$$. As such, the gradient variance can be reduced significantly given a good approximator of the baseline $$\mathrm{V^{\theta}(x_t)}$$.

#### Importance Sampling and Ratio Clipping

Simulating trajectories can be computationally expensive. To improve sample efficiency, we incorporate importance sampling to reuse previously collected trajectory samples:

$$\begin{align}
    &\quad \mathrm{E_{\{p^{\theta}_t\}_{t=T}^1}\bigg[\sum_{t=T}^1 \big(r(x_0)-V^{\theta}(x_t)\big)\nabla \log p^{\theta}_t(x_{t-1}|x_t)\bigg]}\notag \\
    &=\mathrm{E_{\{p^{\theta_{old}}_t\}_{t=T}^1}\bigg[\sum_{t=T}^1 \big(r(x_0)-V^{\theta}(x_t)\big)\frac{p_t^{\theta}(x_{t-1}|x_t)}{p_t^{\theta_{old}}(x_{t-1}|x_t)}\nabla \log p^{\theta}_t(x_{t-1}|x_t)\bigg]}\notag \\
    &=\mathrm{E_{\{p^{\theta_{old}}_t\}_{t=T}^1}\bigg[\sum_{t=T}^1 \big(r(x_0)-V^{\theta}(x_t)\big)\nabla\frac{p_t^{\theta}(x_{t-1}|x_t)}{p_t^{\theta_{old}}(x_{t-1}|x_t)}\bigg]}.\notag \\
\end{align}$$

Inspired by the trust region approach in TRPO {% cite TRPO %}, we stabilize training by clipping the importance weight within an $$\epsilon$$-bounded interval. This yields the following clipped surrogate objective:

$$\begin{align}
    \mathrm{E_{\{p^{\theta_{old}}_t\}_{t=T}^1}\bigg[\sum_{t=T}^1 \big(r(x_0)-V^{\theta}(x_t)\big)\nabla Clip\bigg(\frac{p_t^{\theta}(x_{t-1}|x_t)}{p_t^{\theta_{old}}(x_{t-1}|x_t)} , 1-\epsilon, 1+\epsilon\bigg)\bigg]}.\notag \\
\end{align}$$

This procedure closely resembles the Proximal Policy Optimization (PPO) algorithm {% cite PPO %}, which has become a standard approach in the alignment of large-scale language models.
