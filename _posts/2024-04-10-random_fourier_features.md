---
title: 'Random Fourier Features'
subtitle: An Monte Carlo sampler for radial basis function kernels and positional embedding
date: 2024-04-10
permalink: /posts/random_fourier_features/
category: Sampling
---

### A Weight-space View of Linear Regression 

Bayesian linear regression {% cite GP_ML %} is a linear model with Gaussian noise

$$\begin{align}
\mathrm{f(x)=x^\intercal w, \quad y=f(x)+\varepsilon},\notag
\end{align}$$

where $\mathrm{\varepsilon\sim N(0, \sigma^2)}$. Given a Gaussian prior following $\mathrm{w \sim N(0, \Sigma)}$, the posterior follows

$\begin{align}
\mathrm{p(w|X,y)\sim N\bigg(\sigma^{-2} A^{-1}X Y, A^{-1}\bigg)}, \label{linear_posterior}
\end{align}$

where $$\mathrm{X}=\begin{bmatrix} \mathrm{x_1} \\ \mathrm{x_2} \\ \cdots \\ \mathrm{x_n} \end{bmatrix}$$  and  $$\mathrm{Y}=\begin{bmatrix} \mathrm{y_1} \\ \mathrm{y_2} \\ \cdots \\ \mathrm{y_n} \end{bmatrix}$$ are the training data and label; $\mathrm{A=\sigma^{-2} X X^\intercal + \Sigma^{-1}}$. 



**Projection of Features** Linear model is known to suffer from limited modeling capacity. To tackle this issue, we may project the features using proper basis functions, such as  $\mathrm{\phi(x)=(1, x, x^2, x^3, \cdots)}$. In other words, we map a nonlinear model into higher-dimensional linear space.

Now we study the map function $\phi(x)$ in an $N$ dimensional feature space and consider the model

$$\begin{align}
\mathrm{f(x)=\phi(x)^\intercal w, \quad y=f(x)+\varepsilon},\notag
\end{align}$$

The posterior distribution in Eq.\eqref{linear_posterior} is mapped to a higher feature space such that

$\begin{align}
\mathrm{p(\bar w|X,y)\sim N\bigg(\sigma^{-2} A^{-1}\Phi y, A^{-1} \bigg),}\label{post_v1}
\end{align}$

where $$\mathrm{\Phi\equiv \Phi(X)}=\begin{bmatrix} \mathrm{\phi_1(x_1)} & \cdots & \mathrm{\phi_1(x_n)} \\ 
                                     \cdots & \cdots & \cdots \\  
                                     \mathrm{\phi_N(x_1)} & \cdots & \mathrm{\phi_N(x_n)} \end{bmatrix}$$ is a $\mathrm{N\times n}$ matrix and $\mathrm{A=\sigma_n^{-2} \Phi \Phi^\intercal + \Sigma^{-1}}$. 
                                     
The distribution of the predictions given a test set $\mathrm{X}_{*}$ follows that

$$\begin{align}
\mathrm{f}_{\star}|\mathrm{X}_{\star}\mathrm{, X,y\sim N\bigg(\sigma^{-2}\Phi_{\star}^\intercal A^{-1}\Phi y, \Phi_{\star}^\intercal A^{-1}\Phi_{\star} \bigg),}\label{dist_y}
\end{align}$$

where $\mathrm{\Phi_{\star}=\Phi(X_{\star})}$. Note that  $\mathrm{A}$ is a $\mathrm{N}$-dimensional matrix and inverting the matrix is too expensive if $\mathrm{N}$ is large or infinite.  Define $\mathrm{K=\Phi^\intercal \Sigma \Phi}$ to be a $n$-dimensional matrix. By the definition of $\mathrm{A}$, we know that 

$$\begin{align}
&\mathrm{\underbrace{A\Sigma \Phi}_{:=Q}=\sigma_n^{-2} \Phi \Phi^\intercal \Sigma \Phi + \Sigma^{-1} \Sigma \Phi=\sigma_n^{-2} \Phi(K+\sigma^2 I)} \notag \\
&\mathrm{\Sigma \Phi (K+\sigma^2 I)^{-1}=A^{-1} \underbrace{A\Sigma \Phi}_{:=Q}  (K+\sigma^2 I)^{-1}=\sigma_n^{-2} A^{-1}\Phi}. \notag
\end{align}$$

Now the posterior and the predictive distribution \eqref{dist_y} is reduced to 

$\begin{align}
\mathrm{p(\bar w|X,y)\sim N\bigg(\Sigma \Phi (K+\sigma^2 I)^{-1} y, A^{-1} \bigg)}.\label{post_v2}
\end{align}$

$$\begin{align}
\mathrm{f}_{\star}|\mathrm{X}_{\star}\mathrm{, X,y\sim N\bigg(\Phi_{\star}^\intercal\Sigma \Phi (K+\sigma^2 I)^{-1} y, \Phi_{\star}^\intercal \Sigma \Phi_{\star} - \Phi_{\star}^\intercal \Sigma \Phi (K+\sigma^2 I)^{-1} \Phi^\intercal \Sigma \Phi_{\star} \bigg),}\label{dist_y_v2}
\end{align}$$

where the covariance term follows by invoking the matrix inversion lemma. Note that $$\mathrm{\Phi_{\star}^\intercal \Sigma \Phi_{\star}}$$, $$\mathrm{\Phi_{\star}^\intercal \Sigma \Phi}$$, $$\mathrm{\Phi^\intercal \Sigma \Phi}$$ are all matrices of dimension $n$ instead of $N$ and we expect a significant savings when $n\ll N$.



**Covariance/ Kernel** Define $\mathrm{k(x, x')=\phi(x)^\intercal \Sigma \phi(x')}$ and $\mathrm{\psi(x)=\Sigma^{1/2} \phi(x)}$, we can write $\mathrm{k(x, x')}$ as an inner product $\mathrm{k(x, x')=\psi(x)^\intercal \psi(x')}$, which is known to be a kernel or covariance. The most popular one is the radial basis function (RBF) kernel:

$$\begin{align}
\mathrm{k_\sigma(x, x')=\exp\bigg\{\frac{|x-x'|^2}{2 \sigma^2}\bigg\}}.
\end{align}$$

which has an infinite number of basis functions and is widely used in nonlinear support vector machines (SVMs). Now we can write $$\begin{align}\mathrm{K}_\sigma
=\begin{bmatrix} \mathrm{k_\sigma(x_1, x_1)} & \cdots & \mathrm{k_\sigma(x_1, x_n)} \\ 
                \cdots & \cdots & \cdots \\  
                \mathrm{k_\sigma(x_n, x_1)} & \cdots & \mathrm{k_\sigma(x_n, x_n)} \end{bmatrix}\end{align}.$$

However, for large datasets, computing and inverting the kernel/ covariance matrix still scales poorly w.r.t. $n$.

$$\textcolor{blue}{\text{Can we do better?}}$$


### Random Fourier Features


To tackle this issue, random Fourier features {% cite random_features %} propose a Monte Carlo approximation to approximate the radial basis function (RBF) kernel. Such transformations have widely been used in the sinusoidal/ positional embeddings of [attentions](https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/modules/sinusoidal_positional_embedding.py#L15) {% cite attention_need %}, and also used to demonstrate DNN are faster than trees {% cite num_embed %} in tabular data.

Define $$\mathrm{f(x):= \exp\{i w^\intercal x\}}$$. Given a Gaussian vector $\mathrm{w\sim N(0, c I)}$, we have that

$$\begin{align}
\mathrm{E[f(x) f(y)^*]} &= \mathrm{E\big[\exp\big(iw^\intercal (x-y) \big)\big]} \notag \\
               &= \mathrm{(2\pi \sqrt{c})^{-D/2}\int \exp\bigg(-\frac{1}{2c} w^\intercal w\bigg)\exp(i w^\intercal \underbrace{(x-y)}_{:=\delta}) dw} \notag \\
               &= \mathrm{(2\pi \sqrt{c})^{-D/2}\int \exp\bigg(-\frac{1}{2c}\big(w^\intercal w - 2i c w^\intercal \delta - c^2\delta^\intercal \delta\big) - \frac{c}{2}\delta^\intercal \delta \bigg) dw} \notag \\
               &= \mathrm{(2\pi \sqrt{c})^{-D/2}\exp\bigg(-\frac{c}{2}\delta^\intercal \delta\bigg)\int \exp\bigg(-\frac{1}{2c}\|w-i c\delta\|^2_2 \bigg) dw} \notag\\
               &= \mathrm{\exp\bigg(-\frac{c}{2}(x-y)^\intercal (x-y)\bigg):=k_{c^{-1/2}}(x, y),}\notag
\end{align}$$

where $*$ denotes the conjugate transpose.

#### Empirical approximations

Define $$\mathrm{\tilde z^\intercal=\begin{bmatrix} \mathrm{\cos(w^\intercal x)} \\ \mathrm{\sin(w^\intercal x)} \end{bmatrix}}$$, where $\mathrm{w\sim N(0, c)}$. In practice, we want to approximate 

$$\begin{align} 
\mathrm{\exp\big(iw^\intercal (x-y) \big)} &\mathrm{=\cos(w^\intercal (x-y) ) - i \sin(w^\intercal (x-y))} \notag \\
                                  &\mathrm{\approx \cos(w^\intercal (x-y) )} \notag \\
                                  &\mathrm{=\cos(w^\intercal x)\cos(w^\intercal y) + \sin(w^\intercal x) \sin(w^\intercal y)} \notag \\
                                  &=\begin{bmatrix} \mathrm{\cos(w^\intercal x)} \\ \mathrm{\sin(w^\intercal x)} \end{bmatrix}^\intercal \begin{bmatrix} \mathrm{\cos(w^\intercal y)} \notag\\ \mathrm{\sin(w^\intercal y)} \end{bmatrix} \notag \\
                                  &\mathrm{=\tilde z^\intercal \tilde z},\notag
\end{align}$$


This implies that given sufficiently many samples of $$\mathrm{\{w_i\}_{i=1}^R}$$, we use random samples to approximate the kernel 

$$\begin{align} 
\mathrm{k_{c^{-1/2}}(x, y) \approx E[\cos(w^\intercal (x-y) )] \approx \frac{1}{R} \sum_{i=1}^R \tilde z_i^\intercal \tilde z_i}. \notag  
\end{align}$$

An alternative of numerical feature based on $\mathrm{\sqrt{2}cos(wx+b)}$, where $$b\sim \text{Uniform}(0, 2\pi)$$, is studied in [Greg's blog](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/#sutherland2015error). The binning approach also has shown reasonable improvement on the performance and is studied in {% cite num_embed %}.

