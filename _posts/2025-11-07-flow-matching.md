---
title: 'Flow Matching'
subtitle: Learning vector fields in continuous and discrete Spaces
date: 2025-11-07
permalink: /posts/flow_matching/
category: Diffusion
---

Flow Matching (FM) {% cite flow_match %} has seen impressive success in image and video generation [\[Link\]](https://ai.meta.com/research/movie-gen/). The key idea is to train a neural network to learn the underlying vector (velocity) field that deterministically pushes particles along this transport path.


## Continuous State Spaces

### Problem setup
Consider a vector field $$\mathrm{u_t : \mathbb{R}^d \to \mathbb{R}^d}$$ and define its **flow map** $$\mathrm{\phi_t}$$ as the solution of the ODE

$$
\begin{equation}
\mathrm{\frac{d}{dt}\phi_t(x) = u_t(\phi_t(x)), \quad \phi_0(x) = x}.\notag
\end{equation}
$$

Let $$\mathrm{p_0}$$ and $$\mathrm{p_1}$$ be the prior and data distributions, respectively.  $$\mathrm{p_t:=[\phi_t]_\# p_0}$$ is defined as the **pushforward** of $$\mathrm{p_0}$$ under $$\mathrm{\phi_t}$$. Equivalently, by the change-of-variables formula [\[Link\]](https://www.cs.ubc.ca/~murphyk/Teaching/Stat406-Spring08/homework/changeOfVariablesHandout.pdf): 

$$
\begin{equation}
\mathrm{p_t(x) = p_0(\phi_t^{-1}(x)) \mid \det \nabla \phi_t^{-1}(x) \mid.}
\end{equation}
$$

The probability $$\mathrm{p_t}$$ satisfies the **continuity equation** (see Theorem 1 {% cite neural_ode %}):

$$
\begin{equation}
\mathrm{\frac{\partial}{\partial t} p_t(x)
= - \nabla_x \cdot \big( u_t(x)\, p_t(x) \big).}\notag
\end{equation}
$$

To approximate the true vector field $$\mathrm{u_t(x)}$$ via a parameterize $$\mathrm{u_{\theta}(t, x)}$$, the ideal regression loss is:

$$
\begin{equation}
\mathrm{\mathcal{L}_{\mathrm{FM}}(\theta)= \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t}\big[ \| u_\theta(t,x) - u_t(x) \|^2 \big]}.\notag
\end{equation}
$$

This is, however, **intractable**, since both $$\mathrm{p_t}$$ and $$\mathrm{u_t}$$ are unknown.

### Conditional Flow Matching (CFM)

To make the loss more tractable, we introduce a conditioning variable $$\mathrm{x_1\sim p_1}$$ and define a **conditional flow map** $$\mathrm{\psi_t(\cdot \mid x_1)}$$ along with a **conditional vector field** $$\mathrm{u_t(x \mid x_1)}$$  satisfying

$$
\begin{align}
\mathrm{p_t(\cdot\mid x_1)} &\mathrm{= [\psi_t]_\# p_0(\cdot|x_1)=(\psi_t)_\# p_0.}\label{map_psi} \\
\mathrm{\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x \mid x_1)} &\mathrm{= u_t\big( \psi_t(x \mid x_1) \mid x_1 \big),
\quad 
\psi_0(x \mid x_1) = x.}\label{flow_eqn}\\
\end{align}
$$

The unconditional velocity field can then be expressed as a conditional expectation:

$$
\begin{equation}
\mathrm{u_t(x)=\int u_t(x \mid x_1) \dfrac{p_t(x\mid x_1) q(x_1)}{p_t(x)}dx_1}.\notag
\end{equation}
$$

Recall that $$\mathrm{\psi_t}$$ pushes the prior distribution from $\mathrm{p_0}$ to $\mathrm{p_t(\cdot\mid x_1)}$ in \eqref{map_psi}. We can re-define the CFM loss:

$$
\begin{align}
\mathrm{\mathcal{L}_{\mathrm{CFM}}(\theta)}&\ = \ \ \ \mathrm{\mathbb{E}_{t,\, x_1\sim p_1,\ x \sim p_t(\cdot|x_1)}\big[ \| u_\theta(t,x) - u_t(x\mid x_1) \|^2 \big]} \notag\\
                                  &\mathrm{\overset{\text{Eq.}\eqref{map_psi}}{=}\mathbb{E}_{t,\, x_1\sim p_1,\, x_0 \sim p_0}\Big[\big\|u_\theta\big(t, \psi_t(x_0 \mid x_1)\big)-
u_t\big( \psi_t(x \mid x_1) \mid x_1 \big)\big\|^2\Big]}\notag \\
                                  &\mathrm{\overset{\text{Eq.}\eqref{flow_eqn}}{=}\mathbb{E}_{t,\, x_1\sim p_1,\, x_0 \sim p_0}\Big[\big\|u_\theta\big(t, \psi_t(x_0 \mid x_1)\big)-
\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x_0 \mid x_1)\big\|^2\Big].} \notag \\
\end{align}
$$


By regressing $$\mathrm{u_{\theta}}$$ to match the conditional vector field, we obtain an unbiased estimator of the FM loss.

### Special Flow Maps

Consider a map $$\mathrm{\psi_t(x\mid x_1)} \mathrm{=\sigma_t(x_1) x + \mu_t(x_1)}$$. Taking time gradient and combining Eq.\eqref{flow_eqn}, we have 

$$
\begin{align}
\mathrm{\dfrac{d }{dt}\psi_t(x\mid x_1)} &\mathrm{=\sigma_t'(x_1) x + \mu_t'(x_1)=u_t\big( \psi_t(x \mid x_1) \mid x_1 \big)}. \notag \\
\end{align}
$$

Replacing $\mathrm{\psi_t(x \mid x_1)}$ with $\mathrm{x}$, s.t. $$\mathrm{\psi_t(x \mid x_1):=\dfrac{x-\mu_t(x_1)}{\sigma_t(x_1)}}$$, we have

$$
\begin{align}
\mathrm{u_t\big(x \mid x_1 \big)} &\mathrm{=\sigma_t'(x_1) \bigg(\dfrac{x-\mu_t(x_1)}{\sigma_t(x_1)}\bigg) + \mu_t'(x_1)}. \label{vector_field_formula} \\
\end{align}
$$

**Connections to Diffusion Models**: For VE-SDE {% cite song2021scorebased %}, we have $$\mathrm{p_t(x)=N(x\mid x_1, \sigma_{1-t}^2I)},$$ the conditional vector field follows $$\mathrm{u_t(x\mid x_1)=-\frac{\sigma_{1-t}'}{\sigma_{1-t}}(x-x_1)}$$ via Eq.\eqref{vector_field_formula}. For VP-SDE, the conditional probability follows
$$\begin{equation}
\mathrm{p_t(x \mid x_1)
= N\!\left(
x \,\middle|\, \alpha_{1-t} x_1,\,
(1 - \alpha_{1-t}^2) I
\right)}\notag
\end{equation}$$, where $$\mathrm{\alpha_t = e^{-\tfrac{1}{2}\int_0^t \beta(s)\,ds}}$$. The vector field can be derived in the same way.


**Connections to Optimal Transport (OT)**: Consider the OT displacement map: 

$$\begin{equation}\mathrm{p_t=[(1-t)id+t\psi]_* p_0}.\end{equation}$$

More specfically, define $\psi_t(x\mid x_1)=(1-t)x+tx_1$, the conditional vector field follows $$\mathrm{u_t(x\mid x_1)=\dfrac{x_1-x}{1-t}}$$. The simplified CFM loss function follows that

$$
\begin{align}
\mathrm{\mathcal{L}_{\mathrm{CFM-OT}}(\theta)=\mathbb{E}_{t,\, x_1\sim p_1,\, x_0 \sim p_0}\Big[\big\|u_\theta\big(t, \psi_t(x_0 \mid x_1)\big)-
(x_1-x_0)\big\|^2\Big]}. \notag \\
\end{align}
$$

#### Network Parameterization

In diffusion models, noise prediction {% cite song2021scorebased %} learns to predict the added noise $$\mathrm{u_t(x \mid x_0)=p_t(x\mid x_0)}$$, while data prediction {% cite karras2022elucidating %} learns to recover the clean data $$\mathrm{u_t(x \mid x_1)=p_t(x\mid x_1)}$$ directly. They’re mathematically equivalent, but the denoiser view often makes training more stable and easier to control.


## Discrete State Spaces
