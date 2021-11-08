---
title: 'Hamiltonian Monte Caro'
date: 2021-11-07
permalink: /posts/Hamiltonian/
tags:
  - Hamiltonian Monte Caro
---

Hamiltonian Monte Carlo [Nea11] (HMC) is a popular Markov chain Monte Carlo (MCMC) algorithm to simulate from a probability distribution and are generally believed to be faster than the Metropolis Hasting algorithms [MRRT53] and Langevin dynamics. However, the convergence properties are far less understood compared to its empirical successes. In this blog, I will introduce a paper called Optimal Convergence Rate of Hamiltonian Monte Carlo for Strongly Logconcave Distributions [CV19]. 

## Hamiltonian Dynamics

HMC algorithms conserves Hamiltonian and the volume in the phase space and enjoy the reversibility condition. It aims to simulate particles according to the laws of Hamiltonian dynamics. Consider a Hamiltonian function $H(x, v)$ defined as follows

$$H(x, v)=f(x)+\|v\|^2,$$

where $f$ is the potential energy function, $x$ is the position, and $v$ is the velocity variable. In each step, the update of the particles $(x, v)$ follows the system of (ordinary) differential equations as follows

\begin{equation}
\label{Hamiltonian}
\frac{d x}{d t}=\frac{\partial H}{\partial v}=v(t) \ \ \text{and} \ \ \frac{d v}{d t}=-\frac{\partial H}{\partial x}=-\nabla f(x).
\end{equation}

After a time interval $t$, the solutions follow a ``Hamiltonian flow'' $\varphi_t$ that maps $(x,v)$ to $(x_t(x,v), v_t(x, v))$.





## Convergence of Ideal HMC

To prove the convergence of the ideal HMC algorithms, we first assume standard assumptions.

**Strong convexity** We say $f$ is $\mu$-strongly convex if for all $x, y\in R^d$, we have

$$f(y)\geq f(x)+\langle \nabla f(x), y-x\rangle + \frac{\mu}{2} \|y-x\|^2.$$

**Smoothness** We also assume $f$ is $L$-smooth in the sense that

$$f(y)\leq f(x)+\langle \nabla f(x), y-x\rangle + \frac{L}{2} \|y-x\|^2.$$

The convergence analysis hinges on the coupling of two Markov chains such that the distance between the position variables $x_k$ and $y_k$ (the second Markov chain) contracts in each step.

Denote by $x(t)$ and $y(t)$ solutions of HMC (\ref{Hamiltonian}) and denote by $x(0)$ and $y(0)$ the initial positions of two ODEs for HMC. To faciliate the analysis of coupling techniques, we adopt the same initial velocity $v(0)=u(0)$. We first try to prove the contraction bound as follows

<p align="center">
    <img src="/images/HMC_coupling2.png" width="600" />
</p>


**Lemma** Assume the potential function $f$ satisfies the convexity and smoothness assumptions. Then for $0\leq t \leq \frac{1}{2\sqrt{L}}$, we have

$$\|x(t)-y(t)\|^2 \leq (1-\frac{\mu}{4}t^2) \|x(0)-y(0)\|^2.$$


**Proof**
Consider two ODEs for HMCs: 

$$x'(t)=v(t)    \qquad\qquad \quad\text{and}\qquad y'(t)=u(t)$$
$$v'(t)=-\nabla f(x(t))     \quad\qquad\qquad\ \ \  u'(t)=-\nabla f(y(t)),$$

where the initial velocities follow $u(0)=v(0)$. 

Taking the second derivative of $\frac{1}{2}\|x-y\|^2$, we have

$$\frac{d^2}{dt^2}\left(\frac{1}{2} \|x-y\|^2\right)=\frac{d}{dt}\langle v-u, x-y \rangle$$
$$\quad=\langle v'-u', x-y \rangle + \langle v-u, x'-y' \rangle$$
$$\quad=-\rho \|x-y\|^2 + \|v-u\|^2,$$

where $\rho=\rho(t)=\frac{\langle \nabla f(x) - \nabla f(y), x-y \rangle}{\|x-y\|^2}$.

To upper bound $\|v-u\|^2$, recall that $\frac{d}{dt} \|x\|=\frac{d}{dx} \|x\| \cdot \frac{d}{dt} x=\frac{\langle x, \dot{x} \rangle}{\|x\|}$. In what follows, we have

$$\frac{d}{dt}\|v-u\|=\frac{1}{\|v-u\|}\langle v'-u', v-u\rangle =-\frac{\langle \nabla f(x)-\nabla f(y), v-u\rangle}{\|v-u\|}.$$

In particular for the upper bound of $\frac{d}{dt}\|v-u\|$, we have

$\left|\frac{d}{dt}\|v-u\|\right|\leq \|\nabla f(x)-\nabla f(y)\|$
$\leq \sqrt{L \langle \nabla f(x) - \nabla f(y), x-y \rangle}$
$= \sqrt{L\rho \|x-y\|^2}$
$\leq \sqrt{2L\rho \|x_0-y_0\|^2},$

where the first inequality follows by Cauchy–Schwarz inequality, the second inequality follows by the L-smoothness assumption, and the last inequality follows by Claim 7 in [CV19].

Eventually, we can upper bound $\|v-u\|^2$ as follows

$$\|v-u\|^2 \leq  \left(\int_0^t \left|\frac{d}{ds}\|v-u\|\right| ds\right)^2\leq \left(\int_0^t \sqrt{2 L\rho} \|x_0-y_0\| ds\right)^2\leq 2L t \left(\int_0^t \rho ds\right) \|x_0 - y_0\|^2.$$


Define the monotone increasing function

$$P=P(t)=\int_0^t \rho dt,$$
where $P(0)=0$ and $\mu t \leq P(t)\leq L t$ for all $t\geq 0$. Then

$$\|v-u\|^2 \leq 2L t P\|x_0-y_0\|^2.$$

Combining the above upper bounds, we have

$$\frac{d^2}{dt^2} \left(\frac{1}{2}\|x-y\|^2 \right)\leq -\rho \left(\frac{1}{2} \|x_0-y_0\|^2\right)+2Lt P \|x_0-y_0\|^2.$$

Define $\alpha(t)=\frac{1}{2} \|x-y\|^2$, then we have

$$\alpha''(t)\leq -\alpha(0) (\rho(t)-4L t P(t)).$$

Integrating both sides and combining the fact that $\alpha'(0)=0$, we have

$$ \alpha'(t)=\int_0^t \alpha' '(s) ds$$
$$\leq -\int_0^t \alpha(0) (\rho(s)-4L t P(s))ds$$
$$\leq -\alpha(t)\left(P(t) - 4LP(t) \int_0^t s ds\right)$$
$$=-\alpha(0)P(t)(1-2Lt^2).$$

Choose $t \in [0, \frac{1}{2\sqrt{L}}]$, then we can deduce that
$$\alpha'(t)\leq -\alpha(0) \frac{\mu}{2} t.$$

Eventually, we have the desired result such that

$$\alpha(t)=\alpha(0)+\int_0^t \alpha'(s) d s \leq \alpha(0) \left(1-\frac{\mu}{4} t^2\right).$$

## Conclusions

Tuning the number of leapfrog steps is important for controlling the convergence.


## Reference

[Nea11] Radford M Neal. MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, 2:113–162, 2011.

[MRRT53] N. Metropolis, et al. Equation of State Calculations by Fast Computing Machines. Journal of Chemical Physics, 21:1087–1091, 1953.

[MS17] Oren Mangoubi and Aaron Smith. Rapid Mixing of Hamiltonian Monte Carlo on Strongly Log-concave Distributions. arXiv preprint arXiv:1708.07114, 2017.

[Vis21] Nisheeth K. Vishnoi. An Introduction to Hamiltonian Monte Carlo Method for Sampling. arXiv:2108.12107v1. 2021. [Link](https://www.youtube.com/watch?v=efqGwPDnlQY&list=PLJ7WITsfI1LDe6QQ3Uf07AvfxIfvcZ8uI&index=4&t=291s)

[CV19] Zongchen Chen, Santosh S. Vempala. Optimal Convergence Rate of Hamiltonian Monte Carlo for Strongly Logconcave Distributions. Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques. 2019

[Mal20] Alan Maloney. Hamiltonian Monte Carlo For Dummies. [Link](https://www.youtube.com/watch?v=ZGtezhDaSpM&list=PLJ7WITsfI1LDe6QQ3Uf07AvfxIfvcZ8uI&index=7&t=928s). 2020.



