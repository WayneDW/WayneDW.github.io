---
title: 'Hutchinson Estimator, Explained'
subtitle: An unbiased Monte Carlo sampler for implicit trace estimation
date: 2024-07-27
permalink: /posts/hutchinson_estimator/
category: Sampling
---


We study the implicit trace estimator $$\mathrm{Tr(A)}$$ for a $d$-dimensional symmetric real matrix $\mathrm{A}$. Such a technique is widely used in neural ODE {% cite neural_ode %} and Schrödinger bridge {% cite forward_backward_SDE %} to learn the divergence operator. 

While the trace computation might seem straightforward, it becomes challenging in high-dimensional settings where $\mathrm{A}$ is not directly accessible. Fortunately, obtaining the matrix-vector product is more feasible. This naturally leads to the use of Hutchinson's trace estimator {% cite Hutchinson89 %}

$$\begin{align}
    \mathrm{Tr(A)=E[z^T A z]},\notag
\end{align}$$

where $\mathrm{z}$ is a standard Gaussian or Rademachar vector.

### Mean Estimator

To this why this holds, we leverage techniques in randomized matrix computations:
$$\begin{align}
    \mathrm{E[z^T A z]=Tr(E[z^T A z])=E[Tr(z^T A z)]\overset{\text{cyc}}{=}E[Tr(A z z^T)]=Tr(A E[z z^T])=Tr(A)},\notag
\end{align}$$

where the cyclical property of trace is used and standard Gaussian or Rademachar has a variance of $\mathrm{I}$. 


### Analysis of Variance

#### (a) $\mathrm{z}$ is a Gaussian Vector  

Note that Gaussian vector with common variance is invariant under orthogonal transformations, i.e. given $\mathrm{z\in \mathrm{N}(0, \sigma^2 I_d)}$ and $\mathrm{Q}$ is a $d$-dim orthogonal matrix, we have $\mathrm{Var[Q z]=Q \sigma^2 I_d Q^\intercal= \sigma^2 I_d=Var[z]}$.


Consider the eigenvalue decomposition $\mathrm{A=Q^\intercal \Lambda Q}$, we have

$$\begin{align}
    \mathrm{Var[z^\intercal A z]=Var[z^\intercal \Lambda z]=Var\bigg[\sum_{i=1}^d \lambda_i z_i^2\bigg]{=}\sum_{i=1}^d \lambda_i^2 Var\big[z_i^2\big]\overset{\mathrm{\sigma^2=1}}{=}\sum_{i=1}^d \lambda_i^2=\|A\|_F},\notag
\end{align}$$


where $$\mathrm{\|\cdot\|_F}$$ denotes the Frobenius norm of a matrix. 

#### (b) $\mathrm{z}$ is a Rademachar Vector ($$\{1, -1\}$$ with equal probability)

Recall that Rademachar Vector $\mathrm{z}$ follows that $\mathrm{E[z^T A z]=Tr(A) = \sum_{i=1}^d z_i^2 A_{ii}}$. It follows that

$$\begin{align}
    \mathrm{z^T A z - E[z^T A z]=\sum_{1\leq i,j\leq d}z_i z_j A_{i,j} - \sum_{i=1}^d z_i^2 A_{ii}=2 \sum_{1\leq i<j\leq d} z_i z_j A_{i,j}},\notag
\end{align}$$


Apply the additivity of varance for independent variables
$$\begin{align}
    \mathrm{Var[z^T A z]=4 \sum_{1\leq i<j\leq d} A_{i,j}^2 Var[z_i z_j]=2 \sum_{1\leq i\neq j\leq d} A_{i,j}^2=2\|A\|_F - \sum_{i=1}^d \lambda_i^2},\notag
\end{align}$$

where the second equality follows since $\mathrm{z_i z_j}$ also follows the Rademachar distribution. The variance seems to be strictly smaller than the Gaussian vector, however, it may not be the case in practice.


The nice computational properties are suggested by {% cite race_est %} to make the analysis much simpler. For a general analysis that applies to any distributions, feel free to check the nice slides over here {% cite utah_quadratic %}.


<!-- included complexity analysis https://arxiv.org/pdf/2012.12895 -->



$$\newline$$
$$\newline$$
$$\newline$$
### Citation

```
@article{deng2024_hutchinson,
  title   ={{Hutchinson Estimator, Explained}},
  author  ={Wei Deng},
  journal ={waynedw.github.io},
  year    ={2024},
  url     ="https://www.weideng.org/posts/hutchinson_estimator/"
}
```