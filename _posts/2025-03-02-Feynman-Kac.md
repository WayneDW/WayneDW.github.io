---
title: 'Feynman–Kac Formula'
subtitle: A popular tool in finance and stochastic optimal control
date: 2025-03-15
permalink: /posts/feynman_kac/
category: State Space Model
---


Feynman–Kac formula creates an intrinsic connection between PDE and SDE, making it a popular tool in finance, stochastic optimal control (SOC), and mathematical physics. This blog presents a few applications of the Feynman–Kac formula in different areas. 

Assume a stochastic variable $\mathrm{X_t}$ follows a forward SDE (FSDE)

$$
\left\{
\begin{array}{l}
    \mathrm{d X_t = \mu(t, X_t)dt + \Sigma(t, X_t)dW_t} \\
    \ \ \mathrm{X_0=x.}
\end{array}
\right.
$$

Additionally, we have a backward SDE (BSDE) that satisfies a terminal condition

$$
\left\{
\begin{array}{l}
    \mathrm{d Y_t = h(t, X_t, Y_t)dt + \Sigma(t, X_t) dW_t} \\
    \ \ \mathrm{Y_T=g(X_T).}
\end{array}
\right.
$$

Denote $\mathrm{u(t, X_t)\equiv Y_t^{x}}$. The terminal value admits a solution $$\mathrm{u(t, x)\equiv E[Y_t\\|\mathcal{F}_t]}$$ if we back-propagate the conditional expectation. Given the smoothness and linear growth condition, applying Ito's formula:

$$\begin{align}
    &\mathrm{d u=\bigg[u_t +\nabla_x u^\intercal \mu + \frac{1}{2} Tr(u_{xx} \Sigma \Sigma^\intercal)\bigg] dt + \nabla_x u^\intercal \Sigma d W_t}\notag \\
    &\quad\ =\mathrm{h(t, X_t, Y_t)dt + \Sigma(t, X_t) dW_t}. \notag \\
\end{align}$$

**Nonlinear Feynman-Kac** formula {% cite Exarchos2018 %} builds a connection between the solution of PDEs and probabilistic representations of SDEs

$$
\left\{
\begin{array}{l}
    \mathrm{u_t+\nabla_x u^\intercal \mu +\frac{1}{2} Tr(u_{xx} \Sigma \Sigma^\intercal) -h(t, x, u, \Sigma^\intercal \nabla_x u)=0} \\
    \ \ \mathrm{u(T, x)=g(x).}
\end{array}
\right.
$$

 <!-- and price can be derived by applying the Feynman-Kac representation -->

### Feynman-Kac in finance

To protect stocks from unexpected losses, we consider e.g. a European call option with a stike price $\mathrm{K}$ at time $\mathrm{T}$. We denote the stock price by $\mathrm{X_t}$ and the option price at time $t$ with stock price $\mathrm{X_t}$ by $\mathrm{u(t, X_t)}$.

Consider a univariate linear case with $\mathrm{g(x)=(x-K)^+}$, $\mathrm{h\equiv r u}$, where $\mathrm{r_t}$ denotes the risk-free interest rate, the above PDE becomes: 


$$
\left\{
\begin{array}{l}
    \mathrm{u_t+u_x \mu   +\frac{1}{2}\Sigma^2 u_{xx}-r u=0} \\
    \ \ \mathrm{u(T, x)=(x-K)^+.}
\end{array}
\right.
$$



Then $\mathrm{u(t, x)}$ approximates the current price of the future protection, which yields a stochastic solution

$$\begin{align}
    \mathrm{u(t, x)=E\bigg[(X_T-K)^+ exp\bigg\{-\int_t^T r_s ds\bigg\} \bigg]}.\notag \\
\end{align}$$

The proof is an application of Itô's lemma to show the process $$\mathrm{u(t, X_t) exp\{-\int_{t_0}^t r_s ds \}}$$ is a martingale subject to a stopping time {% cite BM_StochasticCalc %}.



To approximate the expectation, we can simulate sufficiently many stock price paths following the forward SDE $\mathrm{X_t}$ and compute the option price in the backward direction. 


### Feynman-Kac in Schrödinger bridge diffusion


Schrödinger bridge diffusion {% cite DSB %} is a transport-optimized diffusion model for the forward-backward SDE (FBSDE) {% cite forward_backward_SDE %} {% cite pardoux1992backward %}

$$
\left\{
\begin{array}{l}
    \mathrm{d X_t = f + g^2 \nabla_x \log \Psi(t, X_t) dt + g dW_t, \ \ \ X_0 \sim p_{\text{data}}} \\
    \mathrm{d X_t = f - g^2 \nabla_x \log \widehat \Psi(t, X_t) dt + g dW_t. \ \ \ X_T \sim p_{\text{prior}}}
\end{array}
\right.
$$

We observe that when $\mathrm{\nabla_x \log \Psi(t, X_t)}=0$, the equation simplifies to the standard diffusion model. The additional forward score function $\mathrm{\nabla_x \log \Psi(t, X_t)}$ is introduced to minimize a stochastic optimal control problem {% cite Chen21 %}, subject to the forward diffusion and marginal constraints. 

Notably, the Hamilton–Jacobi–Bellman (HJB) PDE arises in the derivation of the SOC problems

$$
\begin{align}
\mathrm{\frac{\partial \phi}{\partial t}+\frac{1}{2} g^2\Delta\phi + \langle \nabla \phi, f \rangle=-\frac{1}{2}\|g(t)\nabla\phi(x, t)\|^2_2}, \notag
\end{align}$$

where $\mathrm{\phi=\log \Psi}$. It serves as a continuous-time extension of the Bellman equation, which forms the foundation of reinforcement learning.


Applying the Feynman-Kac formula via Ito's formula to the FBSDE {% cite forward_backward_SDE %}, we obtain the likelihood estimator (instead of the stock price in the first example) for the training of score functions $\mathrm{\nabla_x \log \Psi(t, X_t)}$ and $\mathrm{\nabla_x \log \widehat \Psi(t, X_t)}$

$$\begin{align}
\mathrm{\log p_0(x_0)=E[\log p_T(X_T)]-\int_0^T E\bigg[\frac{1}{2} \|Z_t\|^2 + \frac{1}{2} \|\widehat Z_t\|^2 + \nabla_x \cdot (g \widehat Z_t -f ) + \widehat Z_t^\intercal Z_t\bigg]dt},\notag
\end{align}$$

where $\mathrm{Z_t=g\nabla_x \log \Psi(t, X_t)}$ and $\mathrm{\widehat Z_t=g\nabla_x \log \widehat \Psi(t, X_t)}$.


### Conclusions


The Feynman-Kac representation involves a forward simulation followed by a backward derivation process, conceptually akin to the backpropagation training of deep neural networks. This principle also aligns in spirit with the continuous-time policy gradient {% cite williams1992simple %} and controlled sequential Monte Carlo {% cite heng2020controlled %}, making it a valuable framework for studying more efficient reasonings in language models.
