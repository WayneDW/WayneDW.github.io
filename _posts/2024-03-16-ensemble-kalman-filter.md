---
title: 'Ensemble Kalman Filter'
subtitle: A general template for time series predictions
date: 2024-03-16
permalink: /posts/ensemble_kalman_filter/
category: Filter
---

This study is mainly adapted from Andrew's lectures {% cite kf_lecture %} on Kalman filter.


$\begin{align}
.        \notag
\end{align}$

### Recursive least squares

Consider linear regression

$\begin{align}
\mathrm{y}=\mathrm{x} \beta + \varepsilon.\label{OLS}
\end{align}$

where $\mathrm{y}, \varepsilon\in \mathrm{R}$, $\mathrm{x}\in\mathrm{R}^{1\times d}$, and $\beta\in \mathrm{R}^{d\times 1}$.

Given $n$ observations $(\mathrm{x}_1, \mathrm{y}_1), (\mathrm{x}_2, \mathrm{y}_2), \cdots, (\mathrm{x}_n, \mathrm{y}_n)$, the solution of Eq.\eqref{OLS} follows that

$$\begin{align}
\widehat\beta_n = (\mathrm{X}_n^\intercal \mathrm{X}_n)^{-1} \mathrm{X}_n^\intercal \mathrm{Y}_n:=\mathrm{N}_n^{-1} \mathrm{V}_n,\label{solution_n}
\end{align}$$

where $\mathrm{X}_n=(\mathrm{x}_1; \mathrm{x}_2; \cdots; \mathrm{x}_n)$ is a $n\times d$ matrix and $\mathrm{Y}_n=(\mathrm{y}_1, \mathrm{y}_2, \cdots, \mathrm{y}_n) \in \mathrm{R}^n$, $$\mathrm{N}_n=\mathrm{X}_{n}^\intercal \mathrm{X}_{n}$$ and $$\mathrm{V}_n=\mathrm{X}_{n}^\intercal \mathrm{Y}_n$$.

Consider online learning when we have a new $\mathrm{x}_{n+1}$, the solution can be updated as follows

$$\begin{align}
\widehat\beta_{n+1} &= (\mathrm{X}_{n+1}^\intercal \mathrm{X}_{n+1})^{-1} \mathrm{X}_{n+1}^\intercal \mathrm{Y}_{n+1}\notag\\
&=\mathrm{N}_{n+1}^{-1} \mathrm{V}_{n+1}\notag.\\
&=\underbrace{(\mathrm{N}_{n} + \mathrm{x}_{n+1}^\intercal \mathrm{x}_{n+1})^{-1}}_{\text{I}: \ \ \mathrm{N}_{n+1}^{-1}} \underbrace{(\mathrm{V}_{n} + \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1})}_{\text{II}:\ \  \mathrm{V_{n+1}}}.\label{decomposition}
\end{align}$$

where the last equality holds by the block matrix multiplication.

Applying the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) associated with  an invertible square matrix $\mathrm{A}$

$$\begin{align}
(\mathrm{A}+\mathrm{U}\mathrm{C}\mathrm{V})^{-1} = \mathrm{A}^{-1} - \mathrm{A}^{-1} \mathrm{U} (\mathrm{C}^{-1}+\mathrm{V} \mathrm{A}^{-1}\mathrm{U})^{-1} \mathrm{V} \mathrm{A}^{-1},\notag
\end{align}$$ 

the item $\text{I}$ in Eq.\eqref{decomposition} can be simplified as follows

$$\begin{align}
\text{I}:=\mathrm{N}_{n+1}^{-1}&=\mathrm{N}_n^{-1} - \mathrm{N}_n^{-1} \mathrm{x}_{n+1}^\intercal (\underbrace{1 + \mathrm{x}_{n+1} \mathrm{N}_n^{-1} \mathrm{x}_n^\intercal}_{\mathrm{S}_{n+1}})^{-1} \mathrm{x}_{n+1} \mathrm{N}_n^{-1}\notag\\
&=\mathrm{N}_n^{-1} - \mathrm{K}_{n+1} \mathrm{x}_{n+1} \mathrm{N}_n^{-1},\notag \\
\mathrm{K}_{n+1}&=\mathrm{N}_n^{-1} \mathrm{x}_{n+1}^\intercal \mathrm{S}_{n+1}^{-1}.
\end{align}$$

where $\mathrm{S}_{n+1}$ is a scalar.

Combining the solution in Eq.\eqref{solution_n}, the item $\text{II}$ in Eq.\eqref{decomposition} follows that

$$\begin{align}
\text{II}&=\mathrm{N}_{n+1}\widehat\beta_{n+1}\notag\\
        &=\mathrm{V}_n + \mathrm{x}_{n+1}^\intercal \mathrm{y}_{n+1} \notag \\
        &=\mathrm{V}_n + \mathrm{x}_{n+1}^\intercal \mathrm{x}_{n+1}\widehat\beta_n + \mathrm{x}_{n+1}^\intercal \epsilon_{n+1} \notag \\
        &=\underbrace{(\mathrm{N}_n + \mathrm{x}_{n+1}^\intercal \mathrm{x}_{n+1})}_{\mathrm{N}_{n+1}}\widehat\beta_n + \mathrm{x}_{n+1}^\intercal \epsilon_{n+1}, \notag \\
\end{align}$$ 

where the third equality follows by Eq.\eqref{OLS}.


Dividing $\mathrm{N}_{n+1}$ on both sides of the above equation, we have

$$\begin{align}
\widehat\beta_{n+1}&=\widehat\beta_n + \mathrm{N}_{n+1}^{-1}\mathrm{x}_{n+1}^\intercal \epsilon_{n+1} \notag \\
                   &=\widehat\beta_n + \mathrm{K}_{n+1} \epsilon_{n+1}, \notag \\
\end{align}$$ 





### Kalman Filter


### Ensemble Kalman Filter


### Code 

