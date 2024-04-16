---
title: 'Transformer Filter'
subtitle: Can a Transformer Represent a Kalman Filter?
date: 2024-04-15
permalink: /posts/transformer_filter/
category: Filter
---

Transformers have achieved unprecedented successes in large language models. Transformers utilize autoregressive structures without employing recurrent units and have achieved great efficiency in dealing with long sequences and parallelism. However, it is still not clear if transformers can model standard state-space models or Kalman filters with arbitrary accuracy.

We follow {% cite Transformer_filter %} and study the basic properties of transformers in approximating the linear state-space models and Kalman filters.

$$\begin{align}
\text{Softmax self-attention=Gaussian kernel smooother} \approx \text{Kalman filter}.\notag
\end{align}$$

### Preliminaries


#### Kalman filter 

Kalman Filter {% cite bayes_filtering %} or state space model describes the evolution of a partially observed linear system over time in terms of its unobservable states and observable outputs

$$\begin{align}
\mathrm{x}_n&=\mathrm{A} \mathrm{x}_{n-1} + \mathrm{w}_{n-1}, \ \ \mathrm{w}_{n-1}\sim \mathrm{N}(0, \mathrm{Q})\notag\\
\mathrm{y}_n&=\mathrm{H}\mathrm{x}_n + \mathrm{r}_n, \ \ \mathrm{r}_n \sim \mathrm{N}(0, \mathrm{R}) \notag.
\end{align}$$


Assume $$\mathrm{\widehat x}_n^{\star}$$ is the optimal estimate from Kalman filter, see details in  [blog](https://www.weideng.org/posts/ensemble_kalman_filter/). We have 

$$\begin{align}
    \mathrm{\widehat x}_n^{\star}=\mathrm{(A-KH)} \mathrm{\widehat x}_{n-1}^{\star} + \mathrm{K y}_{n-1}=\begin{bmatrix}  \mathrm{A - KH} &  \mathrm{K} \end{bmatrix}  \begin{bmatrix} \mathrm{\widehat x}_{n-1}^{\star} \\ \mathrm{y}_{n-1} \end{bmatrix}. \label{kf_setting}
\end{align}$$

#### Transformers and Single-head Self-Attention


We focus on transformers with only one self-attention head. The multi-layer perceptron (MLP) is simplified to an identify function to bridge the connection between transformer and Kalman filter. 

The self-attention head make predictions (of what?) using the input tokens $\mathrm{q_0, q_1, \cdots, q_N}$ and a query token $\mathrm{q}$ 

$$\begin{align}
\mathrm{F(q_0, \cdots, q_N; q)=\sum_{i=0}^N \frac{e^{q^\intercal A q_j}}{\sum_{j=0}^N e^{q^\intercal A q_j}} M q_i,} \notag
\end{align}$$

where $\mathrm{A}$ and $\mathrm{M}$ are the attention parameters. In sequential settings, we index the token by time and drop all the tokens we have not yet observed. We call it transformer filter and the underlying variant follows

$$\begin{align}
\mathrm{F(q_{n-m+1}, \cdots, q_n; q)=\sum_{i=n-m+1}^n \frac{e^{q^\intercal A q_j}}{\sum_{j=n-m+1}^n e^{q^\intercal A q_j}} M q_i,} \notag
\end{align}$$

In our context, the token $\mathrm{q}_i$ is a nonlinear function of the $i$-th state estimate and observation. Motivated by Eq.\eqref{kf_setting}, the transformer filter can be written as

$$\begin{align}
    \mathrm{\widehat x}_n=\mathrm{F(q_{n-m+1}, \cdots, q_n; q_n)}. 
\end{align}$$

where $\mathrm{m=1}$ recovers the Kalman filter formulation. 



### Part I: $\text{Softmax self-attention=Gaussian kernel smooother}$

Gaussian kernel smooother (GKS) provides a weighted average of neighboring data

$$\begin{align}
\mathrm{F(q_0, \cdots, q_N; q)=\sum_{i=0}^N \frac{e^{-(z - z_i)^\intercal P (z - z_i)}}{\sum_{j=0}^N e^{-(z - z_j)^\intercal P (z - z_j)}} W z_i,} \notag
\end{align}$$

where a closer data point $z_i$ is assigned with a higher weight. 

By the relation $$\mathrm{u^\intercal P v=\sum_{i,j=1}^n  u_i P_{i,j} v_j}$$, we have that

$$\begin{align}
\mathrm{(u-v)^\intercal P (u-v)=\sum_{i,j=1}^n \bigg(u_i P_{i,j} u_j - u_i P_{i,j} v_j -  v_i P_{i,j} u_j + v_i P_{i,j}v_j \bigg)}, \notag
\end{align}$$

Define basis functions $$\mathrm{\phi(u)=(1, \underbrace{u_1, u_2, \cdots, u_n}_{n \text{ items}}, \underbrace{u_1 u_1, u_1 u_2, \cdots, u_{n-1} u_n, u_n u_n}_{n^2 \text{ items}})}$$

We also define 

$$\begin{align}
&\mathrm{\mathrm{A}_{i,j}=0} \text{ if } \mathrm{j\neq 1} \text{ or } \mathrm{i=\{1, 2, \cdots, n+1\}} \text{ otherwise } \mathrm{\mathrm{A_{i,j}=P_{i,j}}} \notag \\
& \mathrm{\mathrm{B}_{i,j}=0} \text{ if } \mathrm{i,j=\{1, n+2, n+3, \cdots, n^2+n+1\}} \text{ otherwise } \mathrm{\mathrm{B_{i,j}=P_{i,j}}}.\notag \\
& \mathrm{\mathrm{C}_{i,j}=0} \text{ if } \mathrm{i\neq 1} \text{ or } \mathrm{j=\{1, 2, \cdots, n+1\}} \text{ otherwise } \mathrm{\mathrm{C_{i,j}=P_{i,j}}} \notag \\
\end{align}$$

We can easily see that

$$\begin{align}
\mathrm{\sum_{i,j=1}^n \bigg(u_i P_{i,j} u_j - u_i P_{i,j} v_j -  v_i P_{i,j} u_j + v_i P_{i,j}v_j \bigg)=\phi(u) (A-2B+C) \phi(v)}, \notag
\end{align}$$



We can see that the self-attention layer can be represented as Gaussian kernel smooother given appropriate basic function $\phi(\cdot)$ and matrix $\mathrm{D=A-2B+C}$.





$$\begin{align}
\mathrm{F(q_0, \cdots, q_N; q)=\sum_{i=0}^N \frac{e^{q^\intercal D q_j}}{\sum_{j=0}^N e^{q^\intercal D q_j}} M q_i,} \notag
\end{align}$$


### Part II: $\text{Gaussian kernel smooother}\approx \text{Kalman filter}$

"Transformers use Dot-Product Attention to focus on the most relevant tokens."

#### Is a Transformer filter $\varepsilon$-close to the Kalman Filter in terms of state estimates?

We consider a transformer filter whose self-attention blocks takes as input embeddings of the past $m$ state estimates and observations

$$\begin{align}
    \begin{bmatrix} \mathrm{\widehat x}_{n-m} \\ \mathrm{y}_{n-m} \end{bmatrix}, \cdots, \begin{bmatrix} \mathrm{\widehat x}_{n-2} \\ \mathrm{y}_{n-2} \end{bmatrix}, \begin{bmatrix} \mathrm{\widehat x}_{n-1} \\ \mathrm{y}_{n-1} \end{bmatrix} \rightarrow \mathrm{\widehat x_{n}=\sum_{i=n-m+1}^n \alpha_{i, n} \widetilde x_i}, 
\end{align}$$

where 

$$\begin{align}
     \mathrm{\alpha_{i, n}=\dfrac{\exp(-\beta\|\widetilde x_i - \widetilde x_n\|_2^2)}{\sum_{j=n-m+1}^n \exp(-\beta\|\widetilde x_j - \widetilde x_n\|_2^2)}, \quad \widetilde x_n }=  \begin{bmatrix}  \mathrm{A - KH} &  \mathrm{K} \end{bmatrix}  \begin{bmatrix} \mathrm{\widehat x}_{n-1} \\ \mathrm{y}_{n-1} \end{bmatrix}.
\end{align}$$


By definition, we have 

$$\begin{align}
    \mathrm{\|\widehat x_{n}-\widetilde x_n\|_2} &=\mathrm{\bigg\|\sum_{i=n-m+1}^n \alpha_{i, n} (\widetilde x_i-\widetilde x_n)}\bigg\|_2 \notag \\
    &\leq \mathrm{\sum_{i=n-m+1}^n \alpha_{i, n} \|\widetilde x_i-\widetilde x_n\|_2} \notag \\
    &< \mathrm{\sum_{i=n-m+1}^n \exp\big(-\beta \|\widetilde x_i - \widetilde x_n\|_2^2 \big) \|\widetilde x_i-\widetilde x_n\|_2} \notag \\
    &\leq \mathrm{m \max_{\gamma\geq 0} \exp(-\beta \gamma^2) \gamma} \notag \\
    &\leq \mathrm{\varepsilon_1},\notag
\end{align}$$

where $\mathrm{\beta \geq \frac{m^2}{2e \varepsilon_1}}$.



We observe that as we increase the temperature $\beta$, the dependence on the historical estimates $$\mathrm{\widehat x}_{n-2}, \mathrm{\widehat x}_{n-3}\cdots, \mathrm{\widehat x}_{n-m}$$ decays exponentially fast if $$\mathrm{\lambda_{\max}(A-KH)}<1$$, where $\mathrm{\lambda_{\max}(\cdot)}$ denotes the maximum eigenvalue of a matrix. 

#### Conclusion

* Positional embedding is not needed in transformers to approximate Kalman filter.
* The predictions are not affected given permuted historical state and observation estimates. 

