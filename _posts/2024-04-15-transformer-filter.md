---
title: 'Transformer Filter'
subtitle: Can a Transformer represent a Kalman filter?
date: 2024-04-15
permalink: /posts/transformer_filter/
category: State Space Model
---

Transformers have achieved unprecedented successes in large language models. Transformers utilize non-recurrent structures and have achieved great efficiency in modeling long sequences. However, it is still not clear if transformers can model state-space models or Kalman filters with arbitrary accuracy.

We follow {% cite Transformer_filter %} and study the basic properties of transformers in approximating the linear state-space models and Kalman filters. We show

$$\begin{align}
\text{Softmax self-attention}\supseteq \text{Gaussian kernel smooother} \approx \text{Kalman filter}.\notag
\end{align}$$

### Preliminaries


#### Kalman filter 

Kalman Filter {% cite bayes_filtering %} or state space model describes the evolution of a partially observed linear system over time in terms of its unobservable states and observable outputs

$$\begin{align}
\mathrm{x}_n&=\mathrm{A} \mathrm{x}_{n-1} + \mathrm{w}_{n-1}, \ \ \mathrm{w}_{n-1}\sim \mathrm{N}(0, \mathrm{Q})\notag\\
\mathrm{y}_n&=\mathrm{H}\mathrm{x}_n + \mathrm{r}_n, \ \ \mathrm{r}_n \sim \mathrm{N}(0, \mathrm{R}) \notag.
\end{align}$$


Assume $$\mathrm{\widehat x}_n^{\star}$$ is the optimal estimate from Kalman filter. Kalman filter follows that

$$\begin{align}
    \mathrm{\widehat x}_n^{\star}=\mathrm{(A-KH)} \mathrm{\widehat x}_{n-1}^{\star} + \mathrm{K y}_{n-1}=\begin{bmatrix}  \mathrm{A - KH} &  \mathrm{K} \end{bmatrix}  \begin{bmatrix} \mathrm{\widehat x}_{n-1}^{\star} \\ \mathrm{y}_{n-1} \end{bmatrix}. \label{kf_setting}
\end{align}$$

where $\mathrm{K}$ is the Kalman gain and more details are presented in [EnKF blog](https://www.weideng.org/posts/ensemble_kalman_filter/).

#### Transformers and Single-head Self-Attention


We study transformers with only one self-attention head. The multi-layer perceptron (MLP) is simplified to an identify function to facilitate understanding. The self-attention head focuses on the most relevant tokens using the input tokens $\mathrm{q_0, q_1, \cdots, q_N}$ and a query token $\mathrm{q}$ 

$$\begin{align}
\mathrm{F(q_0, \cdots, q_N; q)=\sum_{i=0}^N \frac{e^{q^\intercal D q_j}}{\sum_{j=0}^N e^{q^\intercal D q_j}} M q_i,} \notag
\end{align}$$

where $\mathrm{D}$ and $\mathrm{M}$ are the attention parameters. We drop all the unobserved tokens given sequential tokens. We call it transformer filter  {% cite Transformer_filter %} and the variant follows

$$\begin{align}
\mathrm{F(q_{n-m+1}, \cdots, q_n; q)=\sum_{i=n-m+1}^n \frac{e^{q^\intercal D q_j}}{\sum_{j=n-m+1}^n e^{q^\intercal D q_j}} M q_i,} \label{transformer_filter} 
\end{align}$$

In our context, the token $\mathrm{q}_i$ is a map of the $\mathrm{i}$-th state estimate and observation. Motivated by Eq.\eqref{kf_setting}, the transformer filter can be written as

$$\begin{align}
    \mathrm{\widehat x}_n=\mathrm{F(q_{n-m+1}, \cdots, q_n; q_n)}, \notag
\end{align}$$

where $\mathrm{m=1}$ recovers the Kalman filter formulation. 



### Part I: $\text{Softmax self-attention}\supseteq\text{Gaussian kernel smooother}$

Gaussian kernel smooother (GKS) provides a weighted average of neighboring data

$$\begin{align}
\mathrm{F(q_0, \cdots, q_N; q)=\sum_{i=0}^N \frac{e^{-(z - z_i)^\intercal P (z - z_i)}}{\sum_{j=0}^N e^{-(z - z_j)^\intercal P (z - z_j)}} M z_i,} \label{gks}
\end{align}$$

where a closer data point $z_i$ is assigned with a higher weight. 

By the relation $$\mathrm{u^\intercal P v=\sum_{i,j=1}^n  u_i P_{i,j} v_j}$$, we have that

$$\begin{align}
\mathrm{(u-v)^\intercal P (u-v)=\sum_{i,j=1}^n \bigg(u_i P_{i,j} u_j - u_i P_{i,j} v_j -  v_i P_{i,j} u_j + v_i P_{i,j}v_j \bigg)}. \notag
\end{align}$$

Define basis functions $$\mathrm{\phi(u)=(1, \underbrace{u_1, u_2, \cdots, u_n}_{n \text{ items}}, \underbrace{u_1 u_1, u_1 u_2, \cdots, u_{n-1} u_n, u_n u_n}_{n^2 \text{ items}})}$$ and define 

$$\begin{align}
&\mathrm{\mathrm{A}_{i,j}=0} \text{ if } \mathrm{j\neq 1} \text{ or } \mathrm{i=\{1, 2, \cdots, n+1\}} \text{ otherwise } \mathrm{\mathrm{A_{i,j}=P_{i,j}}} \notag \\
& \mathrm{\mathrm{B}_{i,j}=0} \text{ if } \mathrm{i,j=\{1, n+2, n+3, \cdots, n^2+n+1\}} \text{ otherwise } \mathrm{\mathrm{B_{i,j}=P_{i,j}}} \notag \\
& \mathrm{\mathrm{C}_{i,j}=0} \text{ if } \mathrm{i\neq 1} \text{ or } \mathrm{j=\{1, 2, \cdots, n+1\}} \text{ otherwise } \mathrm{\mathrm{C_{i,j}=P_{i,j}}}. \notag \\
\end{align}$$

We can easily see that

$$\begin{align}
\mathrm{\sum_{i,j=1}^n \bigg(u_i P_{i,j} u_j - u_i P_{i,j} v_j -  v_i P_{i,j} u_j + v_i P_{i,j}v_j \bigg)=\phi(u) (A-2B+C) \phi(v)}, \notag
\end{align}$$

This implies that the transformer filter \eqref{transformer_filter} can be represented as Gaussian kernel smooother \eqref{gks} given the basis function $\phi(\cdot)$ and matrix $\mathrm{D=A-2B+C}$. 


### Part II: $\text{Gaussian kernel smooother}\approx \text{Kalman filter}$


#### Is a Transformer filter $\varepsilon$-close to the Kalman Filter in terms of state estimates?

We consider a transformer filter that takes the past $\mathrm{m}$ state estimates and observations as input.

$$\begin{align}
    \begin{bmatrix} \mathrm{\widehat x}_{n-m} \\ \mathrm{y}_{n-m} \end{bmatrix}, \cdots, \begin{bmatrix} \mathrm{\widehat x}_{n-2} \\ \mathrm{y}_{n-2} \end{bmatrix}, \begin{bmatrix} \mathrm{\widehat x}_{n-1} \\ \mathrm{y}_{n-1} \end{bmatrix} \rightarrow \mathrm{\widehat x_{n}=\sum_{i=n-m+1}^n \alpha_{i, n} \widetilde x_i},  \label{update_trans_filter}
\end{align}$$

where 

$$\begin{align}
     \mathrm{\alpha_{i, n}=\dfrac{\exp(-\beta\|\widetilde x_i - \widetilde x_n\|_2^2)}{\sum_{j=n-m+1}^n \exp(-\beta\|\widetilde x_j - \widetilde x_n\|_2^2)}, \quad \widetilde x_n }=  \begin{bmatrix}  \mathrm{A - KH} &  \mathrm{K} \end{bmatrix}  \begin{bmatrix} \mathrm{\widehat x}_{n-1} \\ \mathrm{y}_{n-1} \end{bmatrix}.\notag
\end{align}$$


By definition in Eq.\eqref{update_trans_filter}, we have 

$$\begin{align}
    \mathrm{\|\widehat x_{n}-\widetilde x_n\|_2} &=\mathrm{\bigg\|\sum_{i=n-m+1}^n \alpha_{i, n} (\widetilde x_i-\widetilde x_n)}\bigg\|_2 \notag \\
    &\leq \mathrm{\sum_{i=n-m+1}^n \alpha_{i, n} \|\widetilde x_i-\widetilde x_n\|_2} \notag \\
    &< \mathrm{\sum_{i=n-m+1}^n \exp\big(-\beta \|\widetilde x_i - \widetilde x_n\|_2^2 \big) \|\widetilde x_i-\widetilde x_n\|_2} \notag \\
    &\leq \mathrm{\max_{\gamma\geq 0} e^{-\beta \gamma^2} \gamma m} \notag \\
    &=\mathrm{e^{-1/2}(2\beta)^{-1/2}m},\label{upper_bound_1}
\end{align}$$

where the second inequality follows since $$\mathrm{\sum_{j=n-m+1}^n \exp(-\beta\|\widetilde x_j - \widetilde x_n\|_2^2)}>1$$; the third follows since  $$\mathrm{\frac{d e^{-\beta \gamma^2} \gamma}{d \gamma}\propto (1-2\beta \gamma^2)}$$ and we can easily check that $\gamma=(2\beta)^{-1/2}$ achieves the optimality of $$\mathrm{e^{-\beta \gamma^2} \gamma}$$ by inspecting the monotonicity.


Consider the eigenvalue decomposition of $$\mathrm{A-KH=Q \Sigma Q^{-1}}$$ and assume the maximum eigenvalue of $$\mathrm{\Sigma}$$ is less than 1, namely $$\mathrm{\lambda_{\max}(\Sigma)}<1$$. We have that  $$\mathrm{\|(A-KH)^i\|_2< \lambda_{\max}^i \|Q\|_2 \|Q^{-1}\|_2} $$. Now we upper bound the approximation error of state estimates from a transformer filter and Eq.\eqref{kf_setting} as follows 

$$\begin{align}
    \|\mathrm{\widehat x}_t -  \mathrm{\widehat x}_t^{\star}\|_2 &\leq \|\mathrm{\widehat x}_t -  \mathrm{\widetilde x}_t\|_2 + \|\mathrm{\widetilde x}_t -  \mathrm{\widehat x}_t^{\star}\|_2 \notag \\
    & \leq \|\mathrm{\widehat x}_t -  \mathrm{\widetilde x}_t\|_2 + \|\mathrm{(A-KH)}(\mathrm{\widehat x}_{t-1} -  \mathrm{\widehat x}_{t-1}^{\star})\|_2 \notag \\
    & \leq \|\mathrm{\widehat x}_t -  \mathrm{\widetilde x}_t\|_2 + \|\mathrm{(A-KH)}(\mathrm{\widehat x}_{t-1} -  \mathrm{\widetilde x}_{t-1})\|_2 + \|\mathrm{(A-KH)^2}(\mathrm{\widehat x}_{t-2} -  \mathrm{\widehat x}_{t-2}^{\star})\|_2 \notag \\
    & \leq \cdots \notag \\
    &\leq \|\mathrm{\widehat x}_t -  \mathrm{\widetilde x}_t\|_2 \sum_{i=0}^n \mathrm{\|\mathrm{(A-KH)^i}\|_2}\notag\\
    &\leq \mathrm{e^{-1/2}(2\beta)^{-1/2}m} \frac{\mathrm{\|Q\|_2 \|Q^{-1}}\|_2}{1-\lambda_{\max}}\notag.
\end{align}$$

where the last inequality follows by Eq.\eqref{upper_bound_1}. We observe that by setting a high enough temperature 

$$\begin{align}
    \mathrm{\beta\geq \frac{m^2 \|Q\|_2^2 \|Q^{-1}\|^2_2}{2e\varepsilon^2 (1-\lambda_{\max})^2 }},\notag
\end{align}$$

the state estimates from transformer filters are $$\mathrm{\varepsilon}$$-close to the estimates from Kalman filter.



#### Conclusion and Future Works

* Positional embedding is not needed in transformers to approximate Kalman filter.
* The predictions are not affected given permuted historical state and observation estimates (TBD).
* What is the motivation of the multi-head mechanism in attention layers?

