---
title: 'Ensemble Kalman Filter'
subtitle: A general template for time series predictions
date: 2024-03-16
permalink: /posts/ensemble_kalman_filter/
category: Filter
---


### Recursive least squares

Consider linear regression

$\begin{align}
\mathrm{y}=\mathrm{x} \beta + \varepsilon.\label{OLS}
\end{align}$

where $\mathrm{y}, \varepsilon\in \mathrm{R}$, $\mathrm{x}\in\mathrm{R}^{1\times d}$, and $\beta\in \mathrm{R}^{d\times 1}$.

Given $n$ observations $(\mathrm{x}_1, \mathrm{y}_1), (\mathrm{x}_2, \mathrm{y}_2), \cdots, (\mathrm{x}_n, \mathrm{y}_n)$, the solution of Eq.\eqref{OLS} follows that

$$\begin{align}
\widehat\beta_n = (\mathrm{X}_n^\intercal \mathrm{X}_n)^{-1} \mathrm{X}_n^\intercal \mathrm{Y}_n, \label{solution_n}  
\end{align}$$

where $\mathrm{X}_n=(\mathrm{x}_1; \mathrm{x}_2; \cdots; \mathrm{x}_n)$ is a $n\times d$ matrix and $\mathrm{Y}_n=(\mathrm{y}_1, \mathrm{y}_2, \cdots, \mathrm{y}_n) \in \mathrm{R}^n$.

Consider online learning when we have a new $\mathrm{x}_{n+1}$, the solution can be updated as follows

$$\begin{align}
\widehat\beta_{n+1} &= (\mathrm{X}_{n+1}^\intercal \mathrm{X}_{n+1})^{-1} \mathrm{X}_{n+1}^\intercal \mathrm{Y}_{n+1}\notag\\
&=(\mathrm{P}_{n}^{-1} + \mathrm{x}_{n+1}^\intercal \mathrm{x}_{n+1})^{-1} (\mathrm{X}_{n}^\intercal \mathrm{Y}_{n} + \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1})\notag \\
&=\underbrace{(\mathrm{P}_{n}^{-1} + \mathrm{x}_{n+1}^\intercal \mathrm{x}_{n+1})^{-1}}_{\mathrm{P}_{n+1}} \big(\mathrm{P}_{n}^{-1}\widehat\beta_n +  \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} \big),\label{decomposition}
\end{align}$$

where $$\mathrm{P}_n=(\mathrm{X}_{n}^\intercal \mathrm{X}_{n})^{-1}$$, the second equality holds by the block matrix multiplication, and the last equality is followed by Eq.\eqref{solution_n}. 

Applying the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity), 
<!-- $$\begin{align}
(\mathrm{A}+\mathrm{U}\mathrm{C}\mathrm{V})^{-1} = \mathrm{A}^{-1} - \mathrm{A}^{-1} \mathrm{U} (\mathrm{C}^{-1}+\mathrm{V} \mathrm{A}^{-1}\mathrm{U})^{-1} \mathrm{V} \mathrm{A}^{-1},\notag
\end{align}$$  -->
the item $\mathrm{P}_{n+1}$ in Eq.\eqref{decomposition} can be simplified

$$\begin{align}
\mathrm{P}_{n+1}&=\mathrm{P}_n - \mathrm{P}_n \mathrm{x}_{n+1}^\intercal [\underbrace{1 + \mathrm{x}_{n+1} \mathrm{P}_n \mathrm{x}_n^\intercal}_{\mathrm{S}_{n+1} \text{, which is a scalar.}}]^{-1} \mathrm{x}_{n+1} \mathrm{P}_n\notag\\
&=\mathrm{P}_n - \mathrm{K}_{n+1} \mathrm{S}_n \mathrm{K}_{n+1}^\intercal,\label{P_solution} \\
\text{where}\ \  \mathrm{K}_{n+1}&=\mathrm{P}_n \mathrm{x}_{n+1}^\intercal \mathrm{S}_{n+1}^{-1}. \label{def_K}
\end{align}$$

Combining Eq.\eqref{decomposition}, Eq.\eqref{P_solution}, and Eq.\eqref{def_K}, we have 

$$\begin{align}
\widehat\beta_{n+1} &= (\mathrm{P}_n - \mathrm{K}_{n+1} \mathrm{S}_n \mathrm{K}_{n+1}^\intercal) \big(\mathrm{P}_{n}^{-1}\widehat\beta_n +  \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} \big) \notag \\
&=\widehat\beta_n + \mathrm{P}_n \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} - \mathrm{K}_{n+1} \mathrm{S}_n \mathrm{K}_{n+1}^\intercal \mathrm{P}_{n}^{-1}\widehat\beta_n - \mathrm{K}_{n+1} \mathrm{S}_n \mathrm{K}_{n+1}^\intercal \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} \notag \\
&=\widehat\beta_n + \mathrm{P}_n \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} - \mathrm{K}_{n+1} \mathrm{x}_{n+1}\widehat\beta_n - \mathrm{K}_{n+1} \mathrm{S}_n \mathrm{K}_{n+1}^\intercal \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} \notag \\
&=\widehat\beta_n + \mathrm{P}_n \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} - \mathrm{K}_{n+1} \mathrm{x}_{n+1}\widehat\beta_n - \mathrm{K}_{n+1} \mathrm{x}_{n+1} \mathrm{P}_{n} \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} \notag \\
&\overset{\eqref{def_K}}{=}\widehat\beta_n + \mathrm{K}_{n+1} \mathrm{S}_{n+1} \mathrm{y}_{n+1} - \mathrm{K}_{n+1} \mathrm{x}_{n+1}\widehat\beta_n - \mathrm{K}_{n+1} \mathrm{x}_{n+1} \mathrm{P}_{n} \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1}  \notag \\
&=\widehat\beta_n + \mathrm{K}_{n+1} (\mathrm{y}_{n+1} - \mathrm{x}_{n+1}\widehat\beta_n), \notag \\
\end{align}$$

where the last equality follows by $$\mathrm{S}_{n+1}=1 + \mathrm{x}_{n+1} \mathrm{P}_n \mathrm{x}_n^\intercal$$ in Eq.\eqref{P_solution}.

To summarize, the update scheme for recursive least squares follow that

$$\begin{align}
\mathrm{K}_{n+1}&=\mathrm{P}_n \mathrm{x}_{n+1}^\intercal \mathrm{S}_{n+1}^{-1}\notag \\
\mathrm{S}_{n+1}&=1 + \mathrm{x}_{n+1} \mathrm{P}_n \mathrm{x}_n^\intercal \notag \\
\widehat\beta_{n+1}&=\widehat\beta_n + \mathrm{K}_{n+1} (\mathrm{y}_{n+1} - \mathrm{x}_{n+1}\widehat\beta_n),\notag\\
\mathrm{P}_{n+1}&=\mathrm{P}_n - \mathrm{K}_{n+1} \mathrm{S}_n \mathrm{K}_{n+1}^\intercal. \notag
\end{align}$$


### Kalman Filter



Kalman Filter {% cite bayes_filtering %} or state space model are powerful tools in finance for estimating and predicting the state of a system based on incomplete and noisy measurements. It describes the evolution of a system over time in terms of its unobservable states and observable outputs.
<p align="center">
    <img src="/images/state-space.png" width="250" />
</p>
The dynamics and the measurements follow a linear Gaussian model

$$\begin{align}

\mathrm{x}_n&=\mathrm{A}_{n-1} \mathrm{x}_{n-1} + \mathrm{w}_{n-1}, \notag\\
\mathrm{y}_n&=\mathrm{H}_n\mathrm{x}_n + \mathrm{r}_n \label{linear_ss}.
\end{align}$$



where $$\mathrm{x}_n\in\mathrm{R}^d$$ is the latent state and $$\mathrm{y}_n\in\mathrm{R}^p$$ is the measurement; $$\mathrm{w}_{n-1}\sim \mathrm{N}(0, \mathrm{Q}_{n-1})$$ and $$\mathrm{r}_n \sim \mathrm{N}(0, \mathrm{R}_n)$$; the prior $$\mathrm{x}_0\sim \mathrm{N}(\mathrm{m}_0, \mathrm{P}_0)$$. $$\mathrm{A}_{n-1}$$ is the transition matrix and $\mathrm{H}_n$ is the measurement model. Both matrices are assumed to be known or can be estimated through MLE.  


The probabilistic formulation is

$$\begin{align}
\mathrm{P}(\mathrm{x}_n|\mathrm{x}_{n-1})&=\mathrm{N}(\mathrm{x}_n|\mathrm{A}_{n-1} \mathrm{x}_{n-1}, \mathrm{Q}_{n-1}), \notag \\
\mathrm{P}(\mathrm{y}_n|\mathrm{x}_n)&=\mathrm{N}(\mathrm{y}_n|\mathrm{H}_n\mathrm{x}_n, \mathrm{R}_n) \notag.
\end{align}$$

**Theorem** The Bayesian filtering equations \eqref{linear_ss} can be evaluated in a closed-form Gaussian distribution:

$$\begin{align}
\mathrm{P(x_n|y_{1:n-1})}&=\mathrm{N(x_n|m_n^-, P_n^-)},\notag \\
\mathrm{P(x_n|y_{1:n})}&=\mathrm{N(x_n|m_n, P_n)},\notag \\
\mathrm{P(y_n|y_{1:n-1})}&=\mathrm{N(y_n|H_n m_n, S_n)},\notag
\end{align}$$

The prediction step follows

$$\begin{align}
\mathrm{m_n^-} &= \mathrm{A_{n-1} m_{n-1}},\notag\\
\mathrm{P_n^-} &= \mathrm{A_{n-1} P_{n-1} A_{n-1}^\intercal + Q_{n-1}}.\notag
\end{align}$$

The update step follows

$$\begin{align}
\mathrm{S_n} &= \mathrm{H_n P_n^- H_n^\intercal + R_n}\notag,\\
\mathrm{K_n} &= \mathrm{P_n^- H_n^\intercal S_n^{-1}}, \notag\\
\mathrm{m_n} &= \mathrm{m_n^- + K_n (y_n - H_n m_n-)},\notag\\
\mathrm{P_n} &= \mathrm{P_n^- - K_n S_n K_n^\intercal} \notag.
\end{align}$$


$$\begin{bmatrix}a & b\\c & d\end{bmatrix}$$

**Proof** 

By Lemma A.1, the joint distribution of $\mathrm{x_n, x_{n-1}}$ given $\mathrm{y_{1:n-1}}$ is

$$\begin{align}
&\mathrm{\quad\ P(x_{n-1}, x_n|y_{1:n-1})}\notag\\
&=\mathrm{P( x_n|x_{n-1}) P(x_{n-1}|y_{1:n-1})}\notag\\
&=\mathrm{N(x_n; A_{n-1}x_{n-1}, Q_{n-1}) N(x_{n-1}; m_{n-1}, P_{n-1})}\notag\\
&=\mathrm{N((x_{n-1}; x_n)|m', P')},\notag
\end{align}$$

where 

$$\begin{align}
\mathrm{m'}&=\mathrm{(m_{n-1}, A_{n-1} m_{n-1})} \notag \\
\mathrm{P'}&=\mathrm{(P_{n-1}, P_{n-1} A_{n-1}^\intercal; A_{n-1}P_{n-1}, A_{n-1} P_{n-1} A_{n-1}^\intercal+ Q_{n-1})}.\notag
\end{align}$$

#### Appendix


**Lemma A.1** The joint distribution of $\mathrm{x, y}$ and the marginal $\mathrm{y}$ follows 

$$\begin{align}
(\mathrm{x}; \mathrm{y})&\sim \mathrm{N}\bigg(\begin{bmatrix}\mathrm{m} \\ \mathrm{Hm+u}\end{bmatrix}, \begin{bmatrix}\mathrm{P} & \mathrm{PH^\intercal} \\ \mathrm{HP} & \mathrm{HPH^\intercal +R}\end{bmatrix}\bigg) \notag\\
\mathrm{y}&\sim \mathrm{N}(\mathrm{H}\mathrm{M}+\mathrm{u}, \mathrm{H}\mathrm{P}\mathrm{H}^\intercal + \mathrm{R}).\notag
\end{align}$$

given that 

$$\begin{align}
\mathrm{x}  &\sim \mathrm{N}(\mathrm{m}, \mathrm{P}) \notag\\
\mathrm{y|x} &\sim \mathrm{N}(\mathrm{H}\mathrm{x}+\mathrm{u}, \mathrm{R}).\notag
\end{align}$$

**Lemma A.2** The conditional distribution of $\mathrm{x}$ given $\mathrm{y}$ follows that

$$\begin{align}
\mathrm{x|y} &\sim \mathrm{N(a + CB^{-1}(y-b), A-CB^{-1}C^\intercal)}.\notag
\end{align}$$

given that

$$\begin{align}
\mathrm{(x; y) \sim N\bigg(\begin{bmatrix}\mathrm{a} \\ \mathrm{b} \end{bmatrix}, \begin{bmatrix}\mathrm{A} & \mathrm{C}\\ \mathrm{C^\intercal} & \mathrm{B}\end{bmatrix}}\bigg)\notag
\end{align}$$

### Ensemble Kalman Filter

#### Appendix
### Code 

