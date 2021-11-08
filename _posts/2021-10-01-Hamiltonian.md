---
title: 'Hamiltonian Monte Caro'
date: 2021-11-07
permalink: /posts/Hamiltonian/
tags:
  - Hamiltonian Monte Caro
---

Hamiltonian Monte Carlo [Nea11] (HMC) is a popular Markov chain Monte Carlo (MCMC) algorithm to simulate from a probability distribution and is believed to be faster than the Metropolis Hasting algorithm [MRRT53] and Langevin dynamics. However, the convergence properties are far less understood compared to its empirical successes. 

In this blog, I will introduce a paper called Optimal Convergence Rate of Hamiltonian Monte Carlo for Strongly Logconcave Distributions [CV19]. 

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

Denote by $x(t)$ and $y(t)$ solutions of HMC (\ref{Hamiltonian}) and denote by $x(0)$ and $y(0)$ the initial positions of two ODEs for HMC. To faciliate the analysis of coupling techniques, we adopt the same initial velocity $v(0)=u(0)$. The convergence study hinges on the contraction bound as follows

<p align="center">
    <img src="/images/HMC_coupling2.png" width="400" />
</p>


**Lemma** Assume the potential function $f$ satisfies the convexity and smoothness assumptions. Then for $0\leq t \leq \frac{1}{2\sqrt{L}}$, we have

$$\|x(t)-y(t)\|^2 \leq (1-\frac{\mu}{4}t^2) \|x(0)-y(0)\|^2.$$


**Proof**
Consider two ODEs for HMCs: 

$$\qquad\qquad\qquad\qquad x'(t)=v(t)    \qquad\qquad \quad\text{and}\qquad y'(t)=u(t)$$
$$\qquad\qquad\qquad\qquad v'(t)=-\nabla f(x(t))     \quad\qquad\qquad\ \ \  u'(t)=-\nabla f(y(t)),$$

where the initial velocities follow $u(0)=v(0)$. 

Taking the second derivative of $$\frac{1}{2}\|x-y\|^2$$, we have

$$\frac{d^2}{dt^2}\left(\frac{1}{2} \|x-y\|^2\right)=\frac{d}{dt}\langle v-u, x-y \rangle\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\qquad\quad=\langle v'-u', x-y \rangle + \langle v-u, x'-y' \rangle\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\qquad\quad=-\rho \|x-y\|^2 + \|v-u\|^2,\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$

where $\rho=\rho(t)=\frac{\langle \nabla f(x) - \nabla f(y), x-y \rangle}{\|x-y\|^2}$.

To upper bound $\|\|v-u\|\|^2$, recall that $\frac{d}{dt} \|\|x\|\|=\frac{d}{dx} \|\|x\|\| \cdot \frac{d}{dt} x=\frac{\langle x, \dot{x} \rangle}{\|\|x\|\|}$. In what follows, we have

$$\frac{d}{dt}\|v-u\|=\frac{1}{\|v-u\|}\langle v'-u', v-u\rangle =-\frac{\langle \nabla f(x)-\nabla f(y), v-u\rangle}{\|v-u\|}.$$

In particular for the upper bound of $\frac{d}{dt}\|\|v-u\|\|$, we have

$$\left|\frac{d}{dt}\|v-u\|\right|\leq \|\nabla f(x)-\nabla f(y)\|\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\quad\leq \sqrt{L \langle \nabla f(x) - \nabla f(y), x-y \rangle}\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\quad= \sqrt{L\rho \|x-y\|^2}\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\quad\leq \sqrt{2L\rho \|x_0-y_0\|^2},\qquad\qquad\qquad\qquad\qquad$$

where the first inequality follows by Cauchy–Schwarz inequality, the second inequality follows by the L-smoothness assumption, and the last inequality follows by Claim 7 in [CV19].

Now, we can upper bound $\|\|v-u\|\|^2$ as follows

$$\|v-u\|^2 \leq  \left(\int_0^t \left|\frac{d}{ds}\|v-u\|\right| ds\right)^2\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\leq \left(\int_0^t \sqrt{2 L\rho} \|x_0-y_0\| ds\right)^2\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\qquad\leq 2L t \left(\int_0^t \rho ds\right) \|x_0 - y_0\|^2.\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$


Define the monotone increasing function

$$P=P(t)=\int_0^t \rho dt,$$
where $P(0)=0$ and $\mu t \leq P(t)\leq L t$ for all $t\geq 0$. Then

$$\|v-u\|^2 \leq 2L t P\|x_0-y_0\|^2.$$

Combining the above upper bounds, we have

$$\frac{d^2}{dt^2} \left(\frac{1}{2}\|x-y\|^2 \right)\leq -\rho \left(\frac{1}{2} \|x_0-y_0\|^2\right)+2Lt P \|x_0-y_0\|^2.$$

Define $\alpha(t)=\frac{1}{2} \|\|x-y\|\|^2$, then we have

$$\alpha''(t)\leq -\alpha(0) (\rho(t)-4L t P(t)).$$

Integrating both sides and combining the fact that $\alpha'(0)=0$, we have

$$ \alpha'(t)=\int_0^t \alpha' '(s) ds\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\ \leq -\int_0^t \alpha(0) (\rho(s)-4L t P(s))ds\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\ \leq -\alpha(t)\left(P(t) - 4LP(t) \int_0^t s ds\right)\qquad\qquad\qquad\qquad\qquad$$
$$\qquad\ =-\alpha(0)P(t)(1-2Lt^2).\qquad\qquad\qquad\qquad\qquad\qquad$$

Choose $t \in [0, \frac{1}{2\sqrt{L}}]$, then we can deduce that
$$\alpha'(t)\leq -\alpha(0) \frac{\mu}{2} t.$$

Eventually, we have the desired result such that

$$\alpha(t)=\alpha(0)+\int_0^t \alpha'(s) d s \leq \alpha(0) \left(1-\frac{\mu}{4} t^2\right).$$


Setting $t=T=1/(c\sqrt{L})$ for some constant $c\geq 2$, we get

$$\|x(T)-y(T)\|^2 \leq \left(1-\frac{1}{4c^2 \kappa}\right)\|x(0)-y(0)\|^2.$$

Eventually, we can achieve the convergence in Wasserstein distance such that 

$$W^2_2(\nu_k, \pi)\leq E\left[\|x_k-y_k\|^2\right]\leq \left(1-\frac{1}{4c^2 \kappa}\right)^k O(1).$$

## Discretized HMC

MV18 proposed to approximate the Hamiltonian trajectory with a second-order Euler integrator such that

$$\hat x_{\eta}(x, v)=x+v \eta - \frac{1}{2} \nabla f(x), \quad \hat v_{\eta}(x, v)=v-\eta \nabla f(x) - \frac{1}{2} \eta^2 \nabla^2 f(x) v.$$

Since Hessian is expensive to compute and store, an approximation is conducted through

$$\nabla^2 f(x) v \approx \frac{\nabla f(\hat x_{\eta}) - \nabla f(x)}{\eta},$$

Now, the numerical integrator follows that

$$\hat x_{\eta}(x, v)=x+v \eta - \frac{1}{2} \nabla f(x), \qquad \hat v_{\eta}(x, v)=v-\frac{1}{2}\eta (\nabla f(x) - \nabla f(\hat x_{\eta})).$$

It can be shown that such a discretized HMC requires O($d^{\frac{1}{4}} \epsilon^{-\frac{1}{2}}$) gradient evaluations under proper regularity assumptions [MV18].



## Conclusions

Properly tuning the number of leapfrog steps is important for maximizing the contraction to accelerate convergence. From the perspective of methodology, the no-U-turn sampler proposes to automatically adjust the number of leapfrog steps and potentially exploits this idea by checking the inner product of postion and velocity [HG14]. 


## Reference

[Nea11] Radford M Neal. MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, 2:113–162, 2011.

[MRRT53] N. Metropolis, et al. Equation of State Calculations by Fast Computing Machines. Journal of Chemical Physics, 21:1087–1091, 1953.

[MS17] Oren Mangoubi and Aaron Smith. Rapid Mixing of Hamiltonian Monte Carlo on Strongly Log-concave Distributions. arXiv preprint arXiv:1708.07114, 2017.

[MV18] Oren Mangoubi and Nisheeth K Vishnoi. Dimensionally tight running time bounds for second-order Hamiltonian Monte Carlo. In NeurIPS. 2018.

[Vis21] Nisheeth K. Vishnoi. An Introduction to Hamiltonian Monte Carlo Method for Sampling. arXiv:2108.12107v1. 2021. [Link](https://www.youtube.com/watch?v=efqGwPDnlQY&list=PLJ7WITsfI1LDe6QQ3Uf07AvfxIfvcZ8uI&index=4&t=291s)

[CV19] Zongchen Chen, Santosh S. Vempala. Optimal Convergence Rate of Hamiltonian Monte Carlo for Strongly Logconcave Distributions. Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques. 2019

[Mal20] Alan Maloney. Hamiltonian Monte Carlo For Dummies. [Link](https://www.youtube.com/watch?v=ZGtezhDaSpM&list=PLJ7WITsfI1LDe6QQ3Uf07AvfxIfvcZ8uI&index=7&t=928s). 2020.

[HG14] Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15 (2014) 1593-1623. 




