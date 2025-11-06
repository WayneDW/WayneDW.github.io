---
title: 'A Lyapunov Function for Poincaré Inequality'
subtitle: A functional inequality that unifies ODEs, PDEs, SDEs, functional analysis, and Riemannian geometry.
date: 2021-06-01
permalink: /posts/Lyapunov_Poincare/
category: Sampling
keywords: Langevin diffusion, Lyapunov function, Drift condition, Poincaré Inequality, Carré du Champ operator, Dirichlet form
---


Poincaré (spectral gap) inequality {% cite Bakry08 %} is the first important family of functional inequalities that charaterizes the exponential convergence of a random variable towards the equilibrium.


## Langevin diffusion

Suppose we are interested in the convergence of the stochastic differential equation

\begin{equation}\notag
\mathrm{dx_t = -\nabla U(x_t)dt + \sqrt{2}dW_t},
\end{equation}

where $\mathrm{\nabla U(\cdot)}$ denotes the gradient of a energy function $U$ and $\mathrm{(W_t)_{t\geq 0}}$ is a Brownian motion. Under mild growth conditions on $\mathrm{U(\cdot)}$, $\mathrm{x}$ converges to a stationary measure $\mathrm{\mu(x)\propto e^{-U(x)}}$.


Define a family of operators $\mathrm{(P_t)_{t\geq 0}}$ as follows

\begin{equation}\notag
\mathrm{P_t(f(x)) = E[f(x_t)\|$=x],}
\end{equation}
where the expectation is taken over a particular set to denote the conditional density.

For a smooth test function $\mathrm{f(x)}$, Itô's formula implies that

\begin{equation}\notag
\mathrm{d f(x_t) = \sqrt{2} \nabla f(x_t) dB_t + Lf(x_t)dt,}
\end{equation}
where $\mathrm{L}$ is the infinitesimal generator of the symmetric Markov Semigroup $\mathrm{P_t}$ 

\begin{equation}\notag
\mathrm{Lf=\lim_{t\rightarrow 0} \frac{P_t f -f }{t}=\Delta f - \langle\nabla U, \nabla f\rangle,}
\end{equation}
where $\mathrm{\Delta}$ denotes the Laplace operator.

## Poincaré Inequality

We say the Gibbs measure $\mathrm{\mu}$ satisfies a Poincaré equality with a constant $\mathrm{C}$ if

\begin{equation}\notag
\mathrm{Var_{\mu}(f)=\int f^2 d\mu -(\int f d\mu)^2 \leq C \xi(f),}
\end{equation}
where $\mathrm{\xi}$ is the Dirichlet form defined as 

\begin{equation}\notag
\mathrm{\xi(f)=\int \Gamma(f)d\mu.}
\end{equation}

$\mathrm{\Gamma}$ is the Carré du Champ operator satisfying 

\begin{equation}\notag
\mathrm{\Gamma(f)=\frac{1}{2}(L(f^2)-2 f L(f)). }
\end{equation}
Since $\mathrm{\mu}$ is reversible for $\mathrm{P_t}$, we have the invariance property $\mathrm{\int L(f)=0}$ for all f in the Dirichlet domain. In other words, for symmetric $\mu$, we have 

\begin{equation}\notag
\mathrm{\xi(f)=\int \Gamma(f)d\mu=-\int f L(f) d\mu =\int (\nabla f)^2 d\mu.}
\end{equation}
where the last inequality follows by integration by parts such that: $\mathrm{-\int f L(f) d\mu=-\int f\nabla (e^{-U(x)}\nabla f)dx=-\int f d(e^{-U(x)} \nabla f)=f e^{-U(x)} \nabla f\|_{boundary} + \int (\nabla f)^2 d\mu.}$

## Variance Decay

Now we study the decay of variance

\begin{equation}\notag
\mathrm{\Lambda(t)=Var_{\mu}(P_t f)= \int(P_t f)^2d\mu.}
\end{equation}
Reacll $\mathrm{\xi(f)=-\int f L(f) d\mu}$. Taking the derivative

\begin{equation}\notag
\mathrm{\Lambda_t(t)=2\int P_t f L P_t f d\mu = -2 \xi(P_t f).}
\end{equation}

Combining the Poincaré equality, we have that

\begin{equation}\notag
\mathrm{\Lambda(t)=Var_{\mu}(P_t f)\leq C \xi(P_t f)=-\frac{C}{2}\Lambda_t(t)}
\end{equation}
This means that $\mathrm{\Lambda_t(t)\leq -\frac{2}{C} \Lambda(t)}$. Including an integration factor $\mathrm{e^{\frac{2t}{C}}}$, we have

\begin{equation}\notag
\mathrm{\nabla (\Lambda(t) e^{\frac{2t}{C}})=\Lambda_t(t) e^{\frac{2t}{C}} + \Lambda(t) \frac{2}{C} e^{\frac{2t}{C}}\leq 0.}
\end{equation}
Hence $\mathrm{\Lambda(t) e^{\frac{2t}{C}} \leq \Lambda(0)}$. In other words,

\begin{equation}\notag
\mathrm{Var_{\mu}(P_t f)\leq e^{-2t/C} Var_{\mu}(f).}
\end{equation}

## How to identify the Poincaré constant

Despite the appealing formulation, identifying the best constant $\mathrm{C>0}$ is in general not easy. In this blog, we will show a method for determining a crude estimate of such a constant.

We denote a Lyapunov function by $\mathrm{V}$ if $\mathrm{V\geq 1}$ and if there exist $\mathrm{\lambda>0, b\geq 0}$ and some $\mathrm{R > 0}$ such that for all $\mathrm{x}$, the following drift condition holds

\begin{equation}\notag
\mathrm{LV(x) ≤ -\lambda V(x) + b 1_{B(0, R)}(x).}
\end{equation}

#### By Theorem 1.4 [1], we show that if there exists a Lyapunov function $\mathrm{V(x)}$ satisfying the drift condition, then $\mathrm{\mu}$ satisfies a $\mathrm{L^2}$ Poincaré inequality with constant $\mathrm{C_P=\frac{1}{\lambda}(1+b\kappa_R)}$, where $\mathrm{\kappa_R}$ is the L2 Poincaré constant of $\mathrm{\mu}$ restricted to the ball $\mathrm{B(0,R)}$.



Given a smooth function $\mathrm{g}$, we know that $\mathrm{Var_{\{\mu}}(g)\leq \int (g-c)^2 d\mu}$ for all $c$. In what follows, we set $\mathrm{f=g-c}$, where $c$ is a constant to be selected later.

Next, we reformulate the drift condition and take an integral for $\mathrm{f^2}$ with respect to $\mathrm{\mu}$:

\begin{equation}\notag
\mathrm{\int f^2 d\mu \leq \int \frac{-LV}{\lambda V} f^2 d\mu + \int f^2 \frac{b}{\lambda V}1_{B(0, R)}d \mu.}
\end{equation}

### Control the first term $\mathrm{\int \frac{-LV}{\lambda V} f^2 d\mu}$

Since $\mathrm{L}$ is $\mathrm{\mu}$-symmetric, by integration by parts, we get

$\mathrm{\int \frac{-LV}{V}f^2d \mu= \int \nabla\left(\frac{f^2}{V}\right) \nabla V d\mu}$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \mathrm{=2\int \frac{f}{V} \nabla f \nabla V d\mu  - \int \frac{f^2}{V^2} \|\nabla V\|^2 d\mu}$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathrm{=\int \|\nabla f\|^2 d\mu - \int \|\nabla f - \frac{f}{V} \nabla V\|^2 d\mu}$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathrm{\leq \int \|\nabla f\|^2 d\mu}$

### Control the second term $\mathrm{\int f^2 \frac{b}{\lambda V}1_{B(0, R)} d\mu}$

Since $\mathrm{\mu}$ satisfies a Poincaré inequality on $\mathrm{B(0, R)}$ with a constant $\mathrm{\kappa_R}$, we have

\begin{equation}\notag
\mathrm{\int_{B(0, R)} f^2 d\mu\leq \kappa_R \int_{B(0, R)} \|\nabla f\|^2 d\mu + (1/\mu(B(0, R))) \left(\int_{B(0, R)} fd\mu\right)^2.}
\end{equation}

Fix $\mathrm{c=\int_{B(0, R)} gd\mu}$. We have
\begin{equation}\notag
\mathrm{\int_{B(0, R)} \frac{f^2}{V}d\mu\leq \int_{B(0, R)} f^2 d\mu\leq \kappa_{R}\int_{B(0, R)} \|\nabla f\|^2d\mu.}
\end{equation}
Eventually, we have

\begin{equation}\notag
\mathrm{Var_{\mu}(f)=\int f^2 d\mu \leq \frac{1}{\lambda} (1+b\kappa_R) \int \|\nabla f\|^2 d\mu.}
\end{equation}
In other words, the Poincaré inequality has a crude constant $\mathrm{C_p=\frac{1}{\lambda} (1+b \kappa_R)}$.

## Construction of the Lypunov function

Suppose we require one tail condition of the energy function $\mathrm{U(x)}$, i.e. there exist $\mathrm{\alpha >0}$ and $\mathrm{R\geq 0}$ such that

### Assumption $\mathrm{\langle x, \nabla U(x)\rangle \geq \alpha \|x\|$ for all $\|x\|\geq R}$ (C1)

Now it is sufficient to build a Lyapunov function $\mathrm{V(x)=e^{\gamma \|x\|}}$, where $\mathrm{\|x\|=\sqrt{\sum_{i=1}^n x_i^2}}$.

Note that $\mathrm{\frac{\partial V(x)}{\partial x_i}= \gamma \frac{x_i}{\|x\|} V(x)$ and $\frac{\partial^2 V(x)}{\partial x_i^2}=\frac{\gamma}{\|x\|} V(x)+ \gamma^2 \frac{x_i^2}{\|x\|^2} V(x) - \gamma \frac{x_i^2}{\|x\|^3}V(x)}$. 

In the sequel, we have

$\mathrm{LV(x)=\gamma\left(\frac{n-1}{\|x\|}+\gamma-\frac{x}{\|x\|} \nabla U(x)\right) V(x)}$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathrm{\leq \gamma\left(\frac{n-1}{\|x\|} + \gamma -\alpha \right) V(x)}$

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathrm{\leq -\gamma(\alpha-\gamma-\frac{n-1}{R}) V(x) + b 1_{B(0, R)}(x)}$.

Hence $\mathrm{V(x)}$ is a Lyapunov function provided

\begin{equation}\notag
\mathrm{\lambda = \gamma(\alpha-\gamma-\frac{n-1}{R})>0,}
\end{equation}

which suffices to choose $\mathrm{\gamma<\alpha}$, a large $\mathrm{R}$ and assume the (C1) condition.
 
## Discussions

[1] The construction of Lyapunov function implies a tail decay for the distribution $\mathrm{\mu\propto e^{-U(x)}}$ outside the ball $\mathrm{B(0, R)}$.

[2] Obtaining a sharper estimate of Poincaré constant may require isoperimetric inequality.

