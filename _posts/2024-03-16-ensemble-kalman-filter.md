---
title: 'Ensemble Kalman Filter'
subtitle: A general template for time series predictions
date: 2024-03-16
permalink: /posts/ensemble_kalman_filter/
category: Filter
---

This study is mainly based on Andrew's lectures {% cite kf_lecture %} on Kalman filter.


$\begin{align}
.        \notag
\end{align}$

### Recursive least squares

Consider linear regression

$\begin{align}
\mathrm{Y}=\mathrm{X}^\intercal \beta + \varepsilon.\label{OLS}
\end{align}$

where $\mathrm{Y}, \varepsilon\in \mathrm{R}$, $\mathrm{X}\in\mathrm{R}^d$.

Given $n$ observations $(\mathrm{x}_1, \mathrm{y}_1), (\mathrm{x}_2, \mathrm{y}_2), \cdots, (\mathrm{x}_n, \mathrm{y}_n)$, the solution of Eq.\eqref{OLS} follows that

$\begin{align}
\widehat\beta = (\mathcal{X}_n^\intercal \mathcal{X}_n)^{-1} \mathcal{X}_n^\intercal \mathcal{Y}_n,
\end{align}$

where $\mathcal{X}_n==(\mathrm{x}^\intercal_1, \mathrm{x}^\intercal_2, \cdots, \mathrm{x}^\intercal_n)$ is a $n\times d$ matrix and $\mathcal{Y}_n=(\mathrm{y}_1, \mathrm{y}_2, \cdots, \mathrm{y}_n) \in \mathrm{R}^n$.

Consider the streaming setting when every new observation is obtained sequentially. leads to an updated value of $\beta$


### Kalman Filter


### Ensemble Kalman Filter


### Code 

