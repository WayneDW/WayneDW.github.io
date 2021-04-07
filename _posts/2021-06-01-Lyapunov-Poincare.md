---
title: 'The Lyapunov Function Method for Poincare Inequality'
date: 2021-06-01
permalink: /posts/Lyapunov_Poincare/
tags:
  - Langevin diffusion
  - Lyapunov function
  - Drift condition
  - Poincare Inequality
---


Poincare Inequality or spectral gap inequality is the first important family of functional inequalities that charaterizes the exponential convergence of a random variable towards the equilibrium.


## Langevin diffusion

Suppose we are interested in the convergence of the stochastic differential equation

$dx_t = -\nabla U(x_t)dt + \sqrt{2}dW_t$,

where $\nabla U(\cdot)$ denotes the gradient of a energy function $U$ and $(W_t)_{t\geq 0}$ is a Brownian motion. Under some conditions on $U(\cdot)$, $x$ converges to a stationary measure $\mu(x)\propto e^{-U(x)}$.


Define a family of operators $(P_t)_{t\geq 0}$ as follows

$P_t(f(x)) = E[f(x_t)\|x_0=x]$

For a smooth test function $f(x)$, Ito formula implies that

$d f(x_t) = \sqrt{2} \nabla f(x_t) dB_t + Lf(x_t)dt$,

where $L$ is the infinitesimal generator of the symmetric Markov Semigroup $P_t$ 

$Lf=\lim_{t\rightarrow 0} \frac{P_t f -f }{t}=\Delta f - \langle\nabla U, \nabla f\rangle$.

## Poincare Inequality

We say the Gibbs measure $\mu$ satisfies a Poincare equality with a constant $C$ if

$Var_{\mu}(f)\leq C \xi(f)$,

where $\xi$ is the Dirichlet form defined as 

$\xi(f)=\int \Gamma(f)d\mu$.

$\Gamma$ is the Carre du Champ operator satisfying 

$\Gamma(f)=\frac{1}{2}(L(f^2)-2 f L(f) f)$. 

Since $\mu$ is reversible for $P_t$, we have the invariance property $\int L(f)=0$ for almost any f. In other words, for symmetric $\mu$, we have 

$\xi(f)=\int \Gamma(f)d\mu=-\int f L(f) f d\mu$.


## Variance Decay

Now we study the decay of variance

$\Lambda(t)=Var_{\mu}(P_t f)= \int(P_t f)^2d\mu$

By the heat equation and integration by parats, 

$\Lambda_t(t)=2\int P_t f L P_t f d\mu = -2 \xi(P_t f)$

Combining the Poincare equality, we have that

$\Lambda(t)=Var_{\mu}(P_t f)\leq C \xi(P_t f)=-\frac{C}{2}\Lambda_t(t)$

This means that $\Lambda_t(t)\leq -\frac{2}{C} \Lambda(t)$. Including an integration factor $e^{\frac{2t}{C}}$, we have

$\nabla (\Lambda(t) e^{\frac{2t}{C}})=\Lambda_t(t) e^{\frac{2t}{C}} + \Lambda(t) \frac{2}{C} e^{\frac{2t}{C}}\leq 0$

Hence $\Lambda_t(t) e^{\frac{2t}{C}} \leq \Lambda_t(0)$. In other words,

$Var_{\mu}(P_t f)\leq e^{-2t/C} Var_{\mu}(f)$.


## How to identify the Poincare constant

Despite the appealing formulation, identifying the best constant $C>0$ is in general not easy. In this blog, we will show a method for determining a crude estimate of such a constant.

We denote a Lyapunov function by $V$ if $V\geq 1$ and if there exist $\lambda>0, b\geq 0$ and some $R > 0$ such that for all $x$, the following drift condition holds

$LV(x) ≤ -\lambda V(x) + b 1_{B(0, R)}(x)$

### By Theorem 1.4 [1], we show that if there exists a Lyapunov function $V(x)$ satisfying the drift condition, then $\mu $ satisfies a $L^2$ Poincare inequality with constant $C_P=\frac{1}{\lambda}(1+b\kappa_R)$, where $\kappa_R$ is the L2 Poincare constant of $\mu$ restricted to the ball B(0,R).



Given a smooth function $g$, we know that $Var_{\{\mu}}(g)\leq \int (g-c)^2 d\mu$ for all $c$. In what follows, we set $f=g-c$, where $c$ is a constant to be selected later.

Next, we reformulate the drift condition and take an integral for $f^2$ with respect to $\mu$:

$\int f^2 d\mu \leq \int \frac{-LV}{\lambda V} f^2 d\mu + \int f^2 \frac{b}{\lambda V}1_{B(0, R)}d \mu$


#### Control the first term $\int \frac{-LV}{\lambda V} f^2 d\mu$

Since $L$ is $\mu$-symmetric, by integration by parts, we get

$\int \frac{-LV}{V}f^2d \mu= \int \nabla\left(\frac{f^2}{V}\right) \nabla V d\mu$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  =2\int \frac{f}{V} \nabla f \nabla V d\mu  - \int \frac{f^2}{V^2} \|\nabla V\|^2 d\mu$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\int \|\nabla f\|^2 d\mu - \int \|\nabla f - \frac{f}{V} \nabla V\|^2 d\mu$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \leq \int \|\nabla f\|^2 d\mu$

### Control the second term $\int f^2 \frac{b}{\lambda V}1_{B(0, R)} d\mu$

Since $\mu$ satisfies a Poincare inequality on $B(0, R)$ with a constant $\kappa_R$, we have

$\int_{B(0, R)} f^2 d\mu\leq \kappa_R \int_{B(0, R)} \|\nabla f\|^2 d\mu + (1/\mu(B(0, R))) \left(\int_{B(0, R)} fd\mu\right)^2$.

Fix $c=\int_{B(0, R)} gd\mu$. We have
$\int_{B(0, R)} \frac{f^2}{V}d\mu\leq \int_{B(0, R)} f^2 d\mu\leq \kappa_{R}\int_{B(0, R)} \|\nabla f\|^2d\mu$.

Eventually, we have
$Var_{\mu}(f)=\int f^2 d\mu \leq \frac{1}{\lambda} (1+b\kappa_R) \int \|\nabla f\|^2 d\mu$.

In other words, the Poincare inequality has a crude constant $C_p=\frac{1}{\lambda} (1+b \kappa_R)$.

## Construction of the Lypunov function

Suppose we require one tail condition of the energy function $U(x)$, i.e. there exist $\alpha >0$ and $R\geq 0$ such that for $\|x\|\leq R$,

$\langle x, \nabla U(x)\rangle \geq \alpha \|x\|$ (C1)

Now it is sufficient to build a Lyapunov function $V(x)=e^{\gamma \|x\|}$, where $\|x\|=\sqrt{\sum_{i=1}^n x_i^2}$.

Note that $\frac{\partial V(x)}{\partial x_i}= \gamma \frac{x_i}{\|x\|} V(x)$. In the sequel, we have

$\frac{\partial^2 V(x)}{\partial x_i^2}=\frac{\gamma}{\|x\|} V(x)+ \gamma^2 \frac{x_i^2}{\|x\|^2} V(x) - \gamma \frac{x_i^2}{\|x\|^3}V(x)$.

This means that 

$LV(x)=\gamma\left(\frac{n-1}{\|x\|}+\gamma-\frac{x}{\|x\|} \nabla U(x)\right) V(x)$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \leq \gamma\left(\frac{n-1}{\|x\|} + \gamma -\alpha \right) V(x)$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \leq -\gamma(\alpha-\gamma-\frac{n-1}{R}) V(x) + b 1_{B(0, R)}(x)$.

Hence $V(x)$ is a Lyapunov function provided

$\lambda = \gamma(\alpha-\gamma-\frac{n-1}{R})>0$.

This suggests that as long as we choose the radius R large enough and (C1) condition is satisfied, we have the desired result.
 

## References:

[1] D. Bakry, F. Barthe, P. Cattiaux, and A. Guillin. A simple proof of the Poincaré inequality for a large class of probability measures including the log-concave case. Electron. Comm. Probab., 13:60–66, 2008.

[2] D. Bakry, I. Gentil, and M. Ledoux. Analysis and Geometry of Markov Diffusion Operators. Springer, 2014.
