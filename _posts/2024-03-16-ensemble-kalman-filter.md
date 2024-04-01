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

Kalman Filter {% cite bayes_filtering %} or state space model is the go-to framework for Bayesian filtering problem. The dynamics and the measurements follow a linear Gaussian model

$$\begin{align}
\mathrm{x}_k&=\mathrm{A}_{k-1} \mathrm{x}_{k-1} + \mathrm{w}_{k-1}, \notag \\
\mathrm{y}_k&=\mathrm{H}_k\mathrm{x}_k + \mathrm{r}_k \notag.
\end{align}$$

where $$\mathrm{x}_k\in\mathrm{R}^n$$ is the latent state and $$\mathrm{y}_k\in\mathrm{R}^m$$ is the measurement; $$\mathrm{w}_{k-1}\sim \mathrm{N}(0, \mathrm{Q}_{k-1})$$ and $$\mathrm{r}_k \sim \mathrm{N}(0, \mathrm{R}_k)$$; the prior $$\mathrm{x}_0\sim \mathrm{N}(\mathrm{m}_0, \mathrm{P}_0)$$. $$\mathrm{A}_{k-1}$$ is the transition matrix and $\mathrm{H}_k$ is the measurement model. Both matrices are assumed to be known. For example, we can be obtained through MLE estimation.  

### Ensemble Kalman Filter


### Code 

