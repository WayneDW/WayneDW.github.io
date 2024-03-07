---
title: 'Implicit Ridge Regularization'
date: 2023-03-06
permalink: /posts/implicit_ridge_regularization/
---


We discuss the ordinary least squares (OLS) linear regression

$$\begin{align}
  y=Ax.\notag\\
\end{align}$$

where $X\in \mathbb{R}^{n\times p}$ and a standard solution is $X=(A^\intercal A)^{-1} Ay$

It is known that $X^\intercal X$ is often poor or ill conditioned. To solve this issue and 


Ridge regularization is a popular technique in 

Linear regression

large n > small d
small n < large d

Minimum-norm estimator 

https://arxiv.org/pdf/1805.10939.pdf

why the pseudo-inverse (check this slide: https://ds12.github.io/talks/lectures/lecture4.html#slide16, page ) linear regression corresponds to the solution with minimum norm.


singular value decomposition and/or the Moore-Penrose pseudoinverse.


TBD
