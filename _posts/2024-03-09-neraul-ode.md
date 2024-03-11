---
title: 'Comments on Rectified Flow'
subtitle: Does a straighter flow always yield more efficient transport and accelerate the simulation?
date: 2024-03-09
permalink: /posts/flow_match_stochastic_interpolant/
category: Diffusion Model
---


Diffusion models have demonstrated interesting applications, such as text-to-image generation (stable diffusion, DALLE), text-to-video generation (Sora), and text-to-audio generation. However, diffusion models often require tens or hundreds of steps in generation and is quite slow.

### Rectified Flow

To propose more efficient transport, rectified flow (RecFlow)  {% cite rec_flow %} proposes to straighten the flow to enable fewer discretization steps. For a given coupling $(\mathrm{X}_0, \mathrm{X}_1)\sim \pi_0\otimes \pi_1$, RecFlow proposes to solve the following probability-flow ODE 

$$\begin{align}
    \mathrm{d} \mathrm{Z}_t = \nu_t^{\star} (\mathrm{Z}_t) \mathrm{d} t, \label{ODE}
\end{align}$$
where $t\in[0, 1]$, and $\mathrm{Z}_0\sim \mathrm{X}_0$. $\nu_t^{\star}$ is the velocity field and is the solution of the objective

$$\begin{align}
    \inf_{\nu} \int_0^1 \mathbb{E}\bigg[ \big\|\mathrm{X}_1 - \mathrm{X}_0 - \nu_t(\mathrm{X}_t, t) \big\|_2^2 \bigg] \mathrm{d} t.\label{obj}
\end{align}$$

where $\mathrm{X}_t = t \mathrm{X}_0 + (1-t) \mathrm{X}_1$. The solution of Eq.\eqref{obj} is a straight line interpolation between $\mathrm{X}_0$ and $\mathrm{X}_1$ such that


$$\begin{align}
    \nu_t^{\star}(z)=\mathbb{E}\bigg[ \mathrm{X}_1 - \mathrm{X}_0 | \mathrm{X}_t = z\bigg].\label{velocity_field}
\end{align}$$

To obtain a valid coupling $(\mathrm{Z}_0, \mathrm{Z}_1)$, we can draw $\mathrm{Z}_1$ by simulating RecFlow \eqref{ODE} with $\mathrm{Z}_0\sim \pi_0$. 

#### Lower Transport Costs

Define a transport cost function $c(\cdot) = \\|\cdot\\|_2^2$. We show the rectified coupling $(\mathrm{Z}_0, \mathrm{Z}_1)$ yields a lower transport cost in that $\mathbb{E}\big[c(\mathrm{Z}_1 - \mathrm{Z}_0) \big]\leq \mathbb{E}\big[c(\mathrm{X}_1 - \mathrm{X}_0) \big]$ for any convex cost function $c$. 


$$\begin{align}\notag
    \mathbb{E}\big[c(\mathrm{Z}_1 - \mathrm{Z}_0) \big] &= \mathbb{E}\bigg[c\bigg(\int_0^1 \nu_t^{\star}(\mathrm{Z}_t)\mathrm{d} t \bigg) \bigg] \\ \notag
    & \leq \mathbb{E}\bigg[\int_0^1 c(\nu_t^{\star}(\mathrm{Z}_t)) \mathrm{d} t \bigg] \\ \notag
    & = \mathbb{E}\bigg[\int_0^1 c(\nu_t^{\star}(\mathrm{X}_t)) \mathrm{d} t \bigg] \\ \notag
    & = \mathbb{E}\bigg[\int_0^1 c(\mathbb{E}\big[(\mathrm{X}_1 - \mathrm{X}_0 \big| \mathrm{X}_t)\big]) \mathrm{d} t \bigg] \\ \notag
    & \leq \mathbb{E}\bigg[\int_0^1 \mathbb{E}\big[c(\mathrm{X}_1 - \mathrm{X}_0) \big| \mathrm{X}_t\big] \mathrm{d} t \bigg] \\ \notag
    & = \int_0^1 \mathbb{E}\big[c(\mathrm{X}_1 - \mathrm{X}_0)\big]\mathrm{d} t \\ \notag
    & = \mathbb{E}\big[c(\mathrm{X}_1 - \mathrm{X}_0) \big].
\end{align}$$

where the first equality follows by Eq.\eqref{ODE} and the two inequalities follow by Jensen’s inequality. The second equality holds since Law($\mathrm{X}_t$)=Law($\mathrm{Z}_t$). The nice property also motivates to iterate this procedure to keep reducing the transport cost.


Nevertheless, straight interpolation is a necessary condition for optimal transport when the transport cost is the Euclidean distance {% cite Optimal_transport %}, but it is not sufficient because the couplings $(\mathrm{Z}_0, \mathrm{Z}_1)$ are also subject to optimize. 


#### More general cost functions

The quadratic transport cost function corresponds to the Brownian motion prior. For more general convex cost function $c$, the vector field {% cite rec_flow_general %} follows that

$$\begin{align}
   \nu_t(\mathrm{Z})=\nabla c^{\star} (\nabla f_t (\mathrm{Z})),\notag
\end{align}$$

where $c^{\star}$ is the conjugate function of $c$, $f$ is the solution of the Bregman divergence

$$\begin{align}
    \inf_f \int_0^1 \mathbb{E}\bigg[c^{\star}(\nabla f(\mathrm{X}_t)) - (\mathrm{X}_1 - \mathrm{X}_0)^\intercal \nabla f(\mathrm{X}_t) + c(\mathrm{X}_1 - \mathrm{X}_0) \bigg] \mathrm{d} t.\notag
\end{align}$$

The new path $\{\mathrm{Z}_t: t\in[0, 1]\}$ can be simulated via $\mathrm{d} \mathrm{Z}_t=\nabla c^{\star}(\nabla f_t(\mathrm{Z}_t))\mathrm{d}t$ with $\mathrm{Z}_0=\mathrm{X}_0$.



#### Comments:

**Straight flow with optimized coupling**: Similar to flow matching {% cite flow_match %}, RecFlow has a fairly user-friendly training loss via a quadratic cost and is appealing for straight-path simulation. The iterative rectify procedure further optimizes the coupling and makes the transport more efficient.

**Formulation**: The extension to general transport cost function looks a bit more complex than the Schrödinger bridge formulation, where the latter {% cite forward_backward_SDE %} is really elegant for optimizing general (convex/ non-convex) transport cost functions.

**Flow or diffusion**: Similar to SGD, RecFlow utilizes the randomness in mini-batch simulation and appears to be scalable for real-world datasets. This brings us a question if the manually injected noise via diffusion models are really needed. For example, stochastic interpolant {% cite stochastic_interpolant %} does unify flow and diffusion, but the training loss becomes implicit again, which may affect the scalability. In my opinion, diffusion model should still outperform flow models in training due to the ease of annealing. 

**Straight v.s. non-straight flows**: I am not fully convinced if displacement interpolantion is always the goal for general non-linear transport due to energy barriers. Empirically, we often observe a decent model with 12-16 NFEs, but a much worse model when we further decrease NFE and we have to resort to hacky tricks and distillation. Such empirical findings also match my personal conjecture.
