---
title: 'The Lyapunov Function Method for Poincare Inequality'
date: 2021-04-08
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



Let $\mu(dx)=e^{-U(x)}dx$ be a probability measure on $R^n$ and $L=\Delta - \langle\nabla U, \nabla,\rangle$ be the $\mu$ symmetric natural operator.


We denote a Lyapunov function by $V$ if $V\geq 1$ and if there exist $\lambda>0, b\geq 0$ and some $R > 0$ such that for all $x$

$LV(x) ≤ -\lambda V(x) + b 1_{B(0, R)}(x)$

The above inequality implies a fast convergence towards the centered Ball of radius $R$ and a certain convergence overall.




Coming soon

## References:

D. Bakry, F. Barthe, P. Cattiaux, and A. Guillin. A simple proof of the Poincaré inequality for a large class of probability measures including the log-concave case. Electron. Comm. Probab., 13:60–66, 2008.

D. Bakry, I. Gentil, and M. Ledoux. Analysis and Geometry of Markov Diffusion Operators. Springer, 2014.
