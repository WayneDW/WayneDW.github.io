---
title: 'The Lyapunov Function Method for Poincare Inequality'
date: 2021-06-01
permalink: /posts/Lyapunov_Poincare/
tags:
  - Lyapunov function
  - Drift condition
  - Poincare Inequality
---


Poincare Inequality or spectral gap inequality is the first important family of functional inequalities that charaterizes the exponential convergence of a random variable towards the equilibrium.

If a Markov Triple satisfies a Poincare equality P(C), we can expect an exponential decay of the variance for any function f

$Var_{\mu}(P_t f)\leq e^{-2t/C} Var_{\mu}(f)$, 

where $P_t$ is a Markov semigroup with the unique invariant distribution $\mu$.

Despite the appealing formulation, identifying the best constant $C>0$ is in general not easy. In this blog, we will show a popular method for determining a crude estimate of such a constant.



Let $\mu(dx)=e^{-U(x)}dx$ be a probability measure on $R^n$, where $U(x)$ is known as the energy function. Let $L=\Delta - \langle\nabla U, \nabla\rangle$ be the $\mu$ symmetric natural operator.


We denote a Lyapunov function by $V$ if $V\geq 1$ and if there exist $\lambda>0, b\geq 0$ and some $R > 0$ such that for all $x$, the following drift condition holds

$LV(x) ≤ -\lambda V(x) + b 1_{B(0, R)}(x)$

### By Theorem 1.4 [1], we show that if there exists a Lyapunov function $V(x)$ satisfying the drift condition, then $\mu $ satisfies a $L^2$ Poincare inequality with constant $C_P=\frac{1}{\lambda}(1+b\kappa_R)$, where $\kappa_R$ is the L2 Poincare constant of $\mu$ restricted to the ball B(0,R).



Given a smooth function $g$, we know that $Var_{\{\mu}}(g)\leq \int (g-c)^2 d\mu$ for all $c$. In what follows, we set $f=g-c$, where $c$ is a constant to be selected later.

Next, we reformulating the drift condition as follows:

$\int f^2 d\mu \leq \int \frac{-LV}{\lambda V} f^2 d\mu + \int f^2 \frac{b}{\lambda V}1_{B(0, R)}d \mu$


Since $L$ is $\mu$-symmetric, by integration by parts, we get

$\int \frac{-LV}{V}f^2d \mu= \int \nabla\left(\frac{f^2}{V}\right) \nabla V d\mu=2\int \frac{f}{V} \nabla f \nabla V d\mu  - \int \frac{f^2}{V^2} \|\nabla V\|^2 d\mu$


$\int \frac{-LV}{V}f^2d \mu = \int \nabla\left(\frac{f^2}{V}) \nabla V d\mu\right=2\int \frac{f}{V} \nabla f \nabla V d\mu - \int \frac{f^2}{V^2} |\nabla V|^2 d\mu$


$\int \frac{-LV}{V}f^2d \mu = \int \nabla\left(\frac{f^2}{V}) \nabla V d\mu\right$


$2\int \frac{f}{V} \nabla f \nabla V d\mu - \int \frac{f^2}{V^2} |\nabla V|^2 d\mu=\int |\nabla f|^2 d\mu - \int |\nabla f - \frac{f}{V} \nabla V|^2 d\mu\leq \int |\nabla f|^2 d\mu$

Coming soon

## References:

[1] D. Bakry, F. Barthe, P. Cattiaux, and A. Guillin. A simple proof of the Poincaré inequality for a large class of probability measures including the log-concave case. Electron. Comm. Probab., 13:60–66, 2008.

[2] D. Bakry, I. Gentil, and M. Ledoux. Analysis and Geometry of Markov Diffusion Operators. Springer, 2014.