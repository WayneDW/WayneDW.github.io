---
title: 'Feynman–Kac Formula'
subtitle: A popular tool in finance, stochastic optimal control, and mathematical physics
date: 2025-03-02
permalink: /posts/feynman_kac/
category: Theory
---


Feynman–Kac formula has been widely used in finance, stochastic optimal control, and mathematical physics. This blog presents a few applications of the Feynman–Kac formula in different areas. 

<!-- Most of the knowledge can be found in Wikipedia {% cite feynman_kac_wiki  %} and course slides {% cite feynman_kac_nyu %}; I wrote this blog primarily to refresh my understanding of related techniques and applications. -->

<!-- Assume a stochastic variable follows an SDE 

$$\begin{align}
    \mathrm{d X_t = \mu(X_t, t)dt + \sigma(X_t, t)dW_t},\notag \\
\end{align}$$ -->

### Feynman-Kac Representation

Assume $\mathrm{u(t, x)}$ is smooth and satisfies some growth condition.  $\mathrm{u(t, x)}$ follows a backward PDE which dissipates at a rate of $$\mathrm{r}$$

$$\begin{align}
    \mathrm{\frac{\partial u}{\partial t}+\mu(x, t)\frac{\partial u}{\partial x} +\frac{1}{2}\sigma^2(x, t)\frac{\partial^2 u}{\partial x^2}-r u=0},\notag \\
\end{align}$$

with a terminal condition $\mathrm{u(T, x)=g(x)}$.

Then $\mathrm{u(t, x)}$ yields a stochastic representation

$$\begin{align}
    \mathrm{u(t, x)=E\bigg[g(X_T)exp\bigg\{-\int_t^T r(s, X_s) ds\bigg\} \bigg]}.\notag \\
\end{align}$$

The proof is an application of Itô's lemma to show the process $$\mathrm{u(t, X_t) exp\{-\int_{t_0}^t r(s, X_s) ds \}}$$ is a martingale subject to a stopping time {% cite BM_StochasticCalc %}.






### Applications in finance


<!-- In daily life, we use insurance to manage risks, such as car accidents or unforeseen illnesses. Similarly, in the financial markets, extreme events are inevitable, and financial derivatives serve as a form of "insurance" to hedge these risks and help investors minimize potential losses.  -->

Given the risk-free interest rate $\mathrm{r_t}$, assume a stock price $\mathrm{S_t}$ follows an geometric Brownian motion 

$$\begin{align}
    \mathrm{d \log S_t = \bigg(r_t-\frac{1}{2}\sigma_t^2\bigg)dt + \sigma_t dW_t}. \label{geoBM} \\
\end{align}$$

To protect the stock from potential losses, we can consider a European call option (or others) with a stike price $\mathrm{K}$  at time $\mathrm{T}$. The option price $\mathrm{u(t, x)}$ at time $t$ with stock price $\mathrm{x}$ can be derived by applying the Feynman-Kac representation

$$\begin{align}
    \mathrm{u(t, x)=E\bigg[(S_T-K)^+ exp\bigg\{-\int_t^T r_s ds\bigg\}\bigg| \log S_t = \log x \bigg]}.\notag \\
\end{align}$$

To approximate the expectation, we can simulate sufficiently many sample paths following Eq.\eqref{geoBM} given $\mathrm{S_t=x}$ and evaluate the function at the terminal time $\mathrm{T}$.
<!-- 
where $\mathrm{\frac{\partial u}{\partial \log x}=r_t-\frac{\sigma_t^2}{2}}$, $\mathrm{\frac{\partial^2 u}{\partial (\log x)^2}=\frac{1}{2}\sigma_t^2}$,  is the strike price and   -->

### Applications in Schrödinger bridge




### Applications in SMC



