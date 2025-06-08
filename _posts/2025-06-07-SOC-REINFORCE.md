---
title: 'Policy Gradients and Stochastic Control'
subtitle: Thoughts on Diffusion Model Alignment
date: 2025-06-07
permalink: /posts/soc_and_reinforce/
category: Sequential Sampling
---

### Preliminaries

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
    \mathrm{J^u(x, t):=E\bigg[\int_t^1 \frac{1}{2\alpha}\|u(X_s^u, s)\|_2^2 ds - r(X_1^u)\bigg|X_t^u = x\bigg]},
\end{align}$$

where $r(\cdot)$ is a terminal reward function.

Define the value function 

$$\begin{align}\label{def_value_func}
    \mathrm{v(x, t):=\inf_{u\in\mathcal{U}} J^u(x, t)}.
\end{align}$$

### Bellman Equation 

The dynamic programming principle yields the famous Bellman equation {% cite tzen2019theoretical %} {% cite zhang2022path %}:

$$\begin{align}\label{HJB}
    \mathrm{\partial_t v +\nabla v^\intercal b_t+\frac{1}{2} \Delta v =-\inf_{u\in\mathcal{U}} \bigg[\frac{1}{2\alpha}\|u(x, t)\|_2^2 + \nabla v^\intercal u_t\bigg]}. 
\end{align}$$


Solving the optimal control $\mathrm{u^{\star}}$ and plugging the solution back into the Bellman equation \eqref{HJB} yields:

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
\mathrm{\partial_t \phi +\nabla \cdot (b\phi)+\frac{1}{2} \Delta \phi=0}\notag, 
\end{align}$$

where $\mathrm{\phi(x, 1)=\exp\big(\frac{-v(x, 1)}{\alpha}\big)=\exp\big(\frac{r(x)}{\alpha}\big)}$. 

By Feynman-Kac theorem, the value function follows that

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
&= \mathbb{E}_{\mathbb{P}^u_{t,x}} \left[
    \frac{1}{2\alpha} \int_t^1 \| u(X_s, s) \|^2 \, \mathrm{d}s
\right]\notag
\end{align}$$

With this, the cost-to-go function can be reformulated as:

$$\begin{align}\label{def_J_2}
    \mathrm{J^u(x, t):=\mathrm{KL}\left( \mathbb{P}^u_{t,x} \,\|\, \mathbb{P}_{t,x} \right) + \mathbb{E}\big[- r(X_1^u)\big|X_t^u = x\big]}.
\end{align}$$



### Connections to Alignment in Diffusion Models

Consider a discretization of the diffusion path measure such that $$\mathrm{\mathbb{P}^u_{0,x}\approx \prod_{t=T}^0 p^{\theta}_t}$$ and $$\mathrm{\mathbb{P}_{0,x}\approx \prod_{t=T}^0 p_t}$$. For consistency with the backward time indexing used in diffusion models, we also reverse the time axis.

The minimization of $\mathrm{J^u(x, t)}$ in \eqref{def_J_2} is equivalent to

$$\begin{align}
    \mathrm{argmax_{\{p^{\theta}_t\}_{t=T}^0} E_{\{p^{\theta}_t\}_{t=T}^0}\bigg[r(x_0)-\alpha \sum_{t=T}^0 KL(p^{\theta}_{t-1}(\cdot|x_t)\| p_{t-1}(\cdot|x_t))\bigg]}
\end{align}$$

which recovers a standard objective in RL-based fine-tuning for diffusion models {% cite fan2023dpok %}.

By further incorporating importance sampling and clipped gradients, the method closely resembles the PPO algorithm {% cite schulman2017proximal %}, allowing for the reuse of sample trajectories and improving sample efficiency. Removing the KL regularization term from the objective recovers the classic REINFORCE algorithm.

