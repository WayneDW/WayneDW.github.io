---
title: 'Gaussian Process and Kalman Filter'
date: 2023-12-01
permalink: /posts/filtering_gaussian_process/
tags:
  - Online linear regression
  - Kalman Filter
  - State-space models
  - Gaussian Process
---

ongoing

### Recursive linear regression

Suppose we are interested in solving a linear regression problem

\begin{align}\notag
Y_n=X_n^\intercal \beta + \epsilon,
\end{align}

where $\epsilon$ is a Gaussian noise with mean $0$. Together with a prior $\mathbb{N}(0, \frac{1}{2}\lambda I_n)$, we have

\begin{align}\label{ori_linear_reg}
\widehat \beta_n=\big(X_n^\intercal X_n + \lambda I_n\big)^{-1} X_n^\intercal Y_n.
\end{align}

It is worth noting that X and Y often comes as a streaming set of data. As such, it is expensive to solve \eqref{ori_linear_reg} everytime new data comes in. Instead, we consider an recursive algorithm

Given an existing posterior conditioned on measurements $1,2,\cdots, k-1$.

\begin{align}\notag
p(\theta|y_{1:k-1})=\mathbb{N}(m_{k-1}, P_{k-1}).
\end{align}

Now when a new measurement comes, the likelihood follows
\begin{align}\notag
p(y_k|\theta)=\mathbb{N}(H_k \theta, \sigma^2).
\end{align}

Using Bayes rule, we have

$$
\begin{align}\notag
p(\theta|y_{1:k})&\propto p(y_k|\theta) p(\theta|y_{1:k-1})\\\notag
                 &\propto \mathbb{N}(\theta|m_k, P_k),\\\notag
\end{align}$$

where the parameters follow

$$
\begin{align}\notag
m_k&=\bigg[P_{k-1}^{-1} +\frac{1}{\sigma^2} H_k^\intercal H_k\bigg]^{-1}\bigg[\frac{1}{\sigma^2} H_k^\intercal y_k + P_{k-1}^{-1} m_{k-1}\bigg],\\
P_k&=\bigg[P_{k-1}^{-1} +\frac{1}{\sigma^2} H_k^\intercal H_k\bigg]^{-1}.\notag\\
\end{align}$$

\begin{align}\notag
\end{align}

Recall that the matrix inversion formula follows that 

$$\begin{align}\notag
(A+BD)^{-1}=A^{-1} - A^{-1} B(I+DA^{-1}B)^{-1}DA^{-1}.
\end{align}$$

The covariance update follows that

$$\begin{align}\notag
P_k=P_{k-1}-P_{k-1}H_k^\intercal \big[H_k P_{k-1} H_k^\intercal +\sigma^2 I\big]^{-1} H_k P_{k-1}.
\end{align}$$

Now including auxiliary variables $S_k$ and $K_k$, the updates follow that

$$\begin{align}
S_k &= H_k P_{k-1} H_k^\intercal +\sigma^2 I \notag\\
K_k &= P_{k-1} H_k^\intercal S_k^{-1} \notag\\
m_k &= m_{k-1} + K_k \bigg[y_k-H_k m_{k-1}\bigg] \notag\\
P_k &= P_{k-1}-K_k S_k K_k^\intercal \notag\\
\end{align}$$

where the update resembles the update equestions of the Kalman filter.
### Kalman Filter


### Gaussian Processes
An Introduction to Gaussian Processes for the Kalman Filter
Expert

KALMAN FILTERING AND SMOOTHING SOLUTIONS TO TEMPORAL GAUSSIAN
PROCESS REGRESSION MODELS. 

TBD


{% cite bayes_filtering %}
{% cite GP_ML %}

{% cite GP_KF %}
