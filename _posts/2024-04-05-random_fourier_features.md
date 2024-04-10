---
title: 'Random Fourier Features'
subtitle: An Monte Carlo sampler for RBF kernels
date: 2024-04-02
permalink: /posts/random_fourier_features/
category: Regression
---

### Linear Regression - Weight-space {% cite GP_ML %}

Bayesian linear regression  is a linear model with Gaussian noise

$$\begin{align}
\mathrm{f(x)=x^\intercal w, \quad y=f(x)+\varepsilon},\notag
\end{align}$$

where $\mathrm{\varepsilon\sim N(0, \sigma^2)}$. Given a Gaussian prior following $\mathrm{w \sim N(0, \Sigma)}$, the posterior follows

$\begin{align}
\mathrm{p(w|X,y)\sim N\bigg(A^{-1}X Y, A^{-1}\bigg)}, \label{linear_posterior}
\end{align}$

where $\mathrm{X, Y}$ are the training data and label; $\mathrm{A=X X^\intercal + \sigma^2\Sigma^{-1}}$. 



**Projection of Features** Linear model is known to suffer from limited modeling capacity. To tackle this issue, we may project the features using proper basis functions, such as  $\mathrm{\phi(x)=(1, x, x^2, x^3, \cdots)}$. In other words, we map a nonlinear model into higher-dimensional linear space.

Now we study the map function $\phi(x)$ in an $N$ dimensional feature space and consider the model

$$\begin{align}
\mathrm{f(x)=\phi(x)^\intercal w, \quad y=f(x)+\varepsilon},\notag
\end{align}$$

The posterior distribution in Eq.\eqref{linear_posterior} is mapped to a higher feature space such that

$\begin{align}
\mathrm{p(\bar w|X,y)\sim N\bigg(\sigma^{-2} A^{-1}\Phi y, A^{-1} \bigg),}\label{post_v1}
\end{align}$

where $$\mathrm{\Phi\equiv \Phi(X)}=\begin{bmatrix} \mathrm{\phi_1(x_1)} & \cdots & \mathrm{\phi_N(x_1)} \\ 
                                     \cdots & \cdots & \cdots \\  
                                     \mathrm{\phi_1(x_n)} & \cdots & \mathrm{\phi_N(x_n)} \end{bmatrix}$$ is a $\mathrm{N\times n}$ matrix and $\mathrm{A=\sigma_n^{-2} \Phi \Phi^\intercal + \Sigma^{-1}}$. Note that  $\mathrm{A}$ is a $\mathrm{N}$-dimensional matrix and inverting the matrix is too expensive if $\mathrm{N}$ is large or infinite. 

Define $\mathrm{K=\Phi^\intercal \Sigma \Phi}$ to be a $n$-dimensional matrix. By the definition of $\mathrm{A}$, we know that 

$$\begin{align}
&\mathrm{\underbrace{A\Sigma \Phi}_{:=Q}=\sigma_n^{-2} \Phi \Phi^\intercal \Sigma \Phi + \Sigma^{-1} \Sigma \Phi=\sigma_n^{-2} \Phi(K+\sigma^2 I)} \notag \\
&\mathrm{\Sigma \Phi (K+\sigma^2 I)^{-1}=A^{-1} \underbrace{A\Sigma \Phi}_{:=Q}  (K+\sigma^2 I)^{-1}=\sigma_n^{-2} A^{-1}\Phi}. \notag
\end{align}$$

Now we can deal with a square matrix of dimension $n$ instead of $N$ and the posterior is reduced to 

$\begin{align}
\mathrm{p(\bar w|X,y)\sim N\bigg(\Sigma \Phi (K+\sigma^2 I)^{-1} y, A^{-1} \bigg)}.\label{post_v2}
\end{align}$


We can expect a significant computational savings when $n\ll N$.



**Covariance/ Kernel** Define $\mathrm{k(x, x')=\phi(x)^\intercal \Sigma \phi(x')}$ and $\mathrm{\psi(x)=\Sigma^{1/2} \phi(x)}$, we can write $\mathrm{k(x, x')}$ as an inner product $\mathrm{k(x, x')=\psi(x)^\intercal \psi(x')}$, which is known to be a kernel. The most popular one is the radial basis function (RBF) kernel,

$$\begin{align}
\mathrm{k(x, x')=\exp\bigg\{\frac{|x-x'|^2}{2}\bigg\}}.
\end{align}$$

which has an infinite number of basis functions and is widely used in nonlinear support vector machines (SVMs). Now we can write $$\begin{align}\mathrm{K}
=\begin{bmatrix} \mathrm{k(x_1, x_1)} & \cdots & \mathrm{k(x_1, x_n)} \\ 
                \cdots & \cdots & \cdots \\  
                \mathrm{k(x_n, x_1)} & \cdots & \mathrm{k(x_n, x_n)} \end{bmatrix}\end{align}$$

However, for large datasets, computing and inverting the covariance matrix scales poorly w.r.t. $n$.



### Random Fourier Features


To tackle this issue, random Fourier features {% cite random_features %} propose an unbiased Monte Carlo approximation to approximate the RBF kernel. 

Define $$\mathrm{f(x):= \exp\{i w^\intercal x\}}$$. Given a Gaussian vector $\mathrm{w\sim N(0, \frac{1}{\sigma^2} I)}$, we have that

Use a more general variance.

$$\begin{align}
\mathrm{E[f(x) f(y)^*]} &= \mathrm{E\big[\exp\big(iw^\intercal (x-y) \big)\big]} \notag \\
               &= \mathrm{(2\pi )^{-D/2}\int \exp(-\frac{1}{2} w^\intercal w)\exp(i w^\intercal \underbrace{(x-y)}_{:=\delta}) dw} \notag \\
               &= \mathrm{(2\pi )^{-D/2}\int \exp(-\frac{1}{2}(w^\intercal w - 2i w^\intercal \delta - \delta^\intercal \delta) - \frac{1}{2}\delta^\intercal \delta ) dw} \notag \\
               &= \mathrm{(2\pi )^{-D/2}\exp\bigg(-\frac{1}{2}\delta^\intercal \delta\bigg)\int \exp(-\frac{1}{2}(w-i\delta)^\intercal (w-i\delta) ) dw} \notag\\
               &= \mathrm{\exp\bigg(-\frac{1}{2}(x-y)^\intercal (x-y)\bigg):=k(x, y),}\notag
\end{align}$$

where $*$ denotes the conjugate transpose.

#### Empirical approximations

Define $$\mathrm{\tilde z^\intercal=\begin{bmatrix} \cos(w^\intercal x) \\ \sin(w^\intercal x) \end{bmatrix}}$$, where $\mathrm{w\sim N(0, \frac{1}{\sigma^2})}$. In practice, we want to approximate 

$$\begin{align} 
\mathrm{\exp\big(iw^\intercal (x-y) \big)} &\mathrm{=\cos(w^\intercal (x-y) ) - i \sin(w^\intercal (x-y))} \notag \\
                                  &\mathrm{\approx \cos(w^\intercal (x-y) )} \notag \\
                                  &\mathrm{=\cos(w^\intercal x)\cos(w^\intercal y) + \sin(w^\intercal x) \sin(w^\intercal y)} \notag \\
                                  &=\begin{bmatrix} \mathrm{\cos(w^\intercal x)} \\ \mathrm{\sin(w^\intercal x)} \end{bmatrix}^\intercal \begin{bmatrix} \mathrm{\cos(w^\intercal y)} \notag\\ \mathrm{\sin(w^\intercal y)} \end{bmatrix} \notag \\
                                  &\mathrm{=\tilde z^\intercal \tilde z},\notag
\end{align}$$


This implies that given sufficiently many samples of $$\mathrm{\{w_i\}_{i=1}^R}$$, we use approximate the kernel 

$$\begin{align} 
\mathrm{k(x, y) \approx E[\cos(w^\intercal (x-y) )] \approx \frac{1}{R} \sum_{i=1}^R \tilde z_i^\intercal \tilde z_i}. \notag  
\end{align}$$

#### Tuning

The key is to tune the variance parameters {% cite num_embed %}.


#### Applications 


transformer. time embedding. 





Acknowledge to Greg's blog. 



