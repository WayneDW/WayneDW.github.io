---
title: 'Girsanov and MLE'
subtitle: An application of Girsanov theorem in parameter estimation.
date: 2023-03-20
permalink: /posts/Girsanov_MLE/
category: Diffusion
---

### Maximum likelihood estimation

Given $N$ independent observations, the likelihood function given $\theta=(\mu, \sigma)$ follows that

$$\begin{align}
\mathcal{L}(\{x_i\}_{i=1}^N|\theta)=\prod_{i=1}^N f(x_i|\theta),\notag
\end{align}$$

which models the density of the random variable $X$. The maximum likelihood estimator (MLE) is given by

$$\begin{align}
\widehat\theta=\text{argmax} \mathcal{L}(\mathbf{x}|\theta),\notag
\end{align}$$

where $\mathbf{x}=\\{x_i\\}_{i=1}^N$. When $X$ is a Gaussian random variable that follows $X\sim \mathcal{N}(\mu, \sigma^2)$. The likelihood function is expressed as

$$\begin{align}\label{MLE}
\mathcal{L}(\{x_i\}_{i=1}^N|\theta)=\bigg(\frac{1}{\sqrt{2\pi \sigma^2}}\bigg)^{N/2} \exp\bigg(-\frac{\sum_{i=1}^N (x_i-\mu)^2}{2\sigma^2}\bigg).
\end{align}$$

Taking the gradient w.r.t. $\mu$ and $\sigma^2$, we have

$$\begin{align}
\widehat \mu=\frac{1}{N}\sum_{i=1}^N x_i, \quad \widehat\sigma^2=\frac{1}{N} \sum_{i=1}^N (x_i-\widehat \mu)^2.\notag
\end{align}$$


### Girsanov theorem

Assume we have a diffusion process

$$\begin{align}
\mathrm{d}X_t = b(X_t;\theta)\mathrm{d}t+\sigma\mathrm{d}W_t.\notag
\end{align}$$

We observe the whole path of the process $X\_t$ from time $[0, T]$. Denote by $\mathbb{P}\_X$ the law of the process on the path space, which is absolutely continuous w.r.t. the Wiener measure. The density of $\mathbb{P}\_X$ w.r.t. the Wiener measure is determined by the Radon-Nikodym derivative 

$$\begin{align}\label{girsanov}
\frac{\mathrm{d}\mathbb{P}_X}{\mathrm{d}\mathbb{P}_W}=\exp\bigg(\frac{1}{\sigma}\int_0^T b(X_s; \theta)\mathrm{d}W_s-\int_0^T \frac{b^2(X_s;\theta)}{2\sigma^2}\mathrm{d}s \bigg).
\end{align}$$




We can observe a close connection between Eq.\eqref{MLE} and a discrete variant of Eq.\eqref{girsanov} 

Remark: While the Girsanov theorem is commonly employed, it is prone to mistakes that can be easily made. Have I made any such errors?


### Applications

Given a stationary Ornstein-Uhlenbeck process, how do you estimate the parameters using MLE {% cite Grigorios_14 %}..

$$\begin{align}
\mathrm{d}X_t = -\alpha X_t \mathrm{d}t+\sigma\mathrm{d}W_t.\notag
\end{align}$$

Hint: 1) write the likelihood; 2) take the gradient.

