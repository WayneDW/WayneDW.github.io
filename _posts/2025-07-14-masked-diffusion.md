---
title: 'Discrete Diffusion Models'
subtitle: Diffusion language models for images, languages, and general state spaces
date: 2025-07-14
permalink: /posts/Discrete Diffusion Models/
category: Diffusion Model
---


Diffusion models have gained notable attention in image generation for continuous state spaces. However, their application to discrete state spaces, such as texts, remains limited. To tackle this issue, 
Discrete Denoising Diffusion Probabilistic Models (D3PM) {% cite austin2021structured %} propose to model $\mathrm{d}$-dim tokens as follows:

$$\begin{align}
    \mathrm{\mathbf{q}(x_t|x_{t-1})=\text{Cat}(x_t; \mathbf{q}=x_{t-1} \mathbf{T}_t)},\label{D3PM}
\end{align}$$

where $\mathrm{x_t}$ is a $\mathrm{(m+1)}$-dim one-hot row vector, augmented with an additional mask state $\mathrm{m}$; $$\mathrm{[\mathbf{T}_t]_{i,j}}$$ denotes the transition probability $$\mathrm{q(x_t=j\\|x_{t-1}=i)}$$. 

Given an absorbing matrix s.t. $\mathbf{T}_t=(1-\beta_i) \mathbf{I} + \beta_i \mathbf{e_m\cdot 1^\intercal}$, iterating the transitions {% cite shi2024simplified %}

$$\begin{align}
    &\mathrm{\mathbf{q}(x_t|x_0)=\text{Cat}(x_t; \mathbf{q}=x_{t-1} \overline{\mathbf{T}}_t), \quad \overline{\mathbf{T}}_t=\mathbf{T}_1 \mathbf{T}_2 \dots \mathbf{T}_t=\alpha_t \mathbf{I} + (1-\alpha_t) \mathbf{e_m\cdot 1^\intercal}},\notag\\
    &\mathrm{\mathbf{q}(x_{t-1}|x_t, x_0)=\dfrac{\mathbf{q}(x_t|x_{t-1}) \mathbf{q}(x_{t-1}|x_0)}{\mathbf{q}(x_{t}|x_0)}=\text{Cat}(x_t; \mathbf{q}=\dfrac{x_t \mathbf{T}_t^\intercal \odot x_0 \overline{\mathbf{T}}_{t-1}}{x_0 \overline{\mathbf{T}}_t x_t^\intercal}) },\notag
\end{align}$$

where $\alpha_t=\Pi_{i=1}^t (1-\beta_i)$. Extend the reverse transition {% cite shi2024simplified %} via Bayes rule again

$$\begin{align}
    \mathrm{\mathbf{q}(x_s \mid x_t, x_0)} 
&\mathrm{= \frac{\mathbf{q}(x_t \mid x_s) \mathbf{q}(x_s \mid x_0)}{\mathbf{q}(x_t \mid x_0)}
= \Bigg\{
\begin{array}{ll}
\mathrm{\frac{\alpha_s - \alpha_t}{1 - \alpha_t} \, x_s^\top x_0} & \mathrm{x_s \ne m,\, x_t = m} \label{reverse_transition} \\
\mathrm{\frac{1 - \alpha_s}{1 - \alpha_t}} & \mathrm{x_s = m,\, x_t = m} \\
\mathrm{x_s^\top x_t} & \mathrm{x_t \ne m}
\end{array}}=\mathrm{Cat(x_s; \bar{R}^{x_0}(t, s)^\top x_t)}\\
\end{align}$$

where $\mathrm{\bar{R}^{x_0}(t, s) = \mathbf{I} + \frac{\alpha_s - \alpha_t}{1 - \alpha_t} \mathbf{e_m} (x_0 - \mathbf{e_m})^\top}$, implying $$\mathrm{p(x_s=x_0\\|x_t=m)=\frac{\alpha_s - \alpha_t}{1 - \alpha_t}}$$. 


Next, we approximate $\mathrm{x_0}$ via a soft estimate $\mathrm{\mu_\theta(x_t, t)}$, defined as:

$$
\begin{align}
\mathrm{\mu_\theta(x_t, t) = 
\begin{cases}
\mathrm{softmax}(f_\theta(x_t, t)) & \text{if } x_t = m, \\
\mathrm{x_t} & \text{otherwise}.
\end{cases}}\label{param}
\end{align}
$$

We parametrize $\mathrm{\mathbf{q}(x_s \mid x_t, \mu_\theta(x_t, t)) \approx \mathbf{q}(x_s \mid x_t, x_0)}$ and approximate the loss via Eq.\eqref{reverse_transition} and \eqref{param}

$$\begin{equation}
\begin{aligned}
\mathrm{\mathrm{KL}(\mathbf{q}(x_s \mid x_t, x_0) \parallel \mathbf{q}(x_s \mid x_t, \mu_\theta(x_t, t)))}
&= 
\begin{cases}
\mathrm{\sum_{x_s=0}^m \mathbf{q}(x_s \mid x_t, x_0) \log \frac{\mathbf{q}(x_s \mid x_t, x_0)}{\mathbf{q}(x_s \mid x_t, \mu_\theta(x_t, t))}} & \mathrm{x_t = m} \\
0 & \mathrm{x_t \ne m}
\end{cases} \\
&= \mathrm{\sum_{k \ne m} \frac{\alpha_s - \alpha_t}{1 - \alpha_t} x_0^\top e_k \mathbf{1}_{x_t = m}\log \frac{x_0^\top e_k}{\mu_\theta(x_t, t)^\top e_k}} \\
&= \mathrm{- \frac{\alpha_s - \alpha_t}{1 - \alpha_t} x_0^\top \mathbf{1}_{x_t = m} \log \mu_\theta(x_t, t)},
\end{aligned}
\notag
\end{equation}$$

which forms a cross-entropy loss between the predicted logits and the clean target.

Integrating from $\delta_0\rightarrow 0$ to 1, we have the lower bound of the log
marginal likelihood (ELBO) 

$$
\begin{align}
\mathrm{ELBO= \int_{\delta_0}^1 \frac{\alpha_t'}{1 - \alpha_t} \, \mathbb{E}_{\mathbf{q}(x_t \mid x_0)} \left[ \mathbf{1}_{x_t = m} \cdot x_0^\top \log \mu_\theta(x_t, t) \right] \, dt.}\label{shi_loss}
\end{align}
$$

where $\mathrm{\alpha_t'}$ is the time derivative of $\mathrm{\alpha_t}$.



#### Continuous-Time Markov Chains (CTMC)

For the continuous-time limit, Discrete Diffusion Models {% cite SEDD %} describe the evolution of the probability mass $\mathrm{\mathbf{p}_t} \in \mathbb{R}^{m+1}$:

$$\begin{align}
    \mathrm{\frac{d \mathbf{p}_t}{dt}=\mathbf{Q}_t \mathbf{p}_t, \ \ \mathbf{p}_0\sim \mathbf{p}_{\text{data}},\ \ \mathbf{p}_T\approx \mathbf{p}_{\text{base}}},\notag
\end{align}$$


$$\begin{align}\mathrm{\mathbf{Q}_t = \lim_{\Delta t\rightarrow 0}\frac{\mathbf{T}_t-I}{\Delta t} \in \mathbb{R}^{(m+1) \times (m+1)}}\notag\end{align}$$ 

is a rate matrix (generator) that governs the frequency and destination of state jumps or transitions {% cite gat2024discrete_flow_matching %}, with each column summing to zero. 


For scalability, a common choice is $\mathrm{\mathbf{Q}_t = \sigma_t \mathbf{Q}^{\text{absorb}}}$, where the design of $\sigma_t$ is detailed in {% cite ou2025absorbingDiscrete %}, and the absorbing-type matrix $\mathrm{\mathbf{Q}^{\text{absorb}}}$ has demonstrated superior empirical performance compared to uniform alternatives {% cite austin2021structured %}.

$$\begin{align}
\mathrm{\mathbf{Q}^{\text{absorb}} = }
\begin{bmatrix}
-1 & 0 & \cdots & 0 & 0 \\
0 & -1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \notag \\
0 & 0 & \cdots & -1 & 0 \\
1 & 1 & \cdots & 1 & 0
\end{bmatrix}=\mathbf{e_m\cdot 1^\intercal - I},
\end{align}$$

where $\mathrm{\mathbf{e_m}}$ is the one-hot encoding of the [MASK] token at index $\mathrm{m}$ under zero-based indexing and $\mathrm{\mathbf{1}}$ is a vector of all 1's. The bottom-right **0** corresponds to the absorbing [MASK] token, which remains unchanged once reached. The diagonal entries (–1) define the rates at which non-mask tokens independently transition to [MASK].

In practice, we can implement the process via Euler steps, which approximates Eq.\eqref{D3PM} as follows

$$\begin{align}
    \mathrm{\mathbf{p}(x_{t+\Delta t}=y|x_t=x)=\mathbf{1}_{x=y}+\mathbf{Q}_t(y, x)\Delta t+o(\Delta t)}\notag
\end{align}$$

Invoking the Bayes rule in Lemma A {% cite shi2024simplified %}, the reverse-time process follows that

$$\begin{align}
    &\mathrm{\frac{d\mathbf{p}_{T-t}}{dt}=\mathbf{R}_{T-t} \mathbf{p}_{T-t}},\notag \\
\end{align}$$

where $\mathrm{\mathbf{R}_t(y, x)=\frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\mathbf{Q}_t(x, y)}$,  $$\mathrm{\mathbf{R}_t (x, x)=-\sum_{y \neq x} \mathbf{R}_t(y, x)}$$. $\frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}$ is referred to as the concrete score {% cite meng2022concrete %}, which generalizes the continuous score  $\mathrm{\nabla \log p_t}$ in discrete space. The score can be further simplfied by utilizing the absorbing structures {% cite shi2024simplified %}.


**Lemma A** The reverse-time transition rate matrix satisfy $$\mathrm{\mathbf{R}_t(y, x) = \mathbf{Q}_t(x, y) \frac{\mathbf{p}_{t}(y)}{\mathbf{p}_{t}(x)} \quad \text{for } x \neq y.}$$

**Proof** Reversing the time from time $\mathrm{t+\Delta t}$ to $\mathrm{t}$, Bayes' rule yields for $\mathrm{x \neq y}$:

$$\begin{align}
\mathrm{\mathbf{p}(x_t = y \mid x_{t+\Delta t} = x)}
&\mathrm{= \frac{\mathbf{p}_t(y) \mathbf{p}(x_{t+\Delta t} = x \mid x_t = y)}{\mathbf{p}_{t+\Delta t}(x)}} \notag \\
&\mathrm{= \frac{\mathbf{p}_t(y) \big(\mathbf{1}_{x=y} + \mathbf{Q}_t(x, y) \Delta t + o(\Delta t)\big)}{\mathbf{p}_{t+\Delta t}(x)}} \notag \\
&\mathrm{\approx  \mathbf{1}_{x=y} + \frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)} \mathbf{Q}_t(x, y) \Delta t} \quad \text{ for small $\mathrm{\Delta t}$}.\notag \\
\end{align}$$

#### On the Dimensionality of Sequences

For a length-$n$ sequence $\mathrm{\mathbf{x}=x^1 x^2 \cdots x^n}$, modeling transitions in $\mathrm{\mathbb{R}^{(m+1)^n \times (m+1)^n}}$ is computationally infeasible. To mitigate this, we assume position-wise independence {% cite SEDD %} and employ a shared transition matrix $\mathrm{\mathbf{Q}_t \in \mathbb{R}^{(m+1) \times (m+1)}}$.

Setting zero values in $\mathrm{\mathbf{Q}_t}$ for all sequences with a Hamming distance larger than 1, it inspires to model the concrete score network between sequences $\mathrm{\widehat{\mathbf{x}}_t=x_t^1 \cdots \widehat{x}_t^i \cdots x_t^n}$ and $\mathrm{\mathbf{x}_t=x_t^1 \cdots x_t^i \cdots x_t^n}$ as $$\mathrm{\mathbf{s}_\theta(\cdot, t) : \{0, 1, 2, \ldots, m\}^n \rightarrow \mathbb{R}^{n \times (m+1)}}$$:

$$\begin{align}
    \mathrm{\mathbf{s}_\theta(\mathbf{x}_t, t)_{\widehat{\mathbf{x}}_t} = \mathbf{s}_\theta(x_t^1 \cdots x_t^i  \cdots x_t^n, t)[i, \widehat{x}_t^i]\approx \dfrac{\mathbf{p}_t(x_t^1 \cdots \widehat{x}_t^i  \cdots x_t^n)}{\mathbf{p}_t(x_t^1 \cdots x_t^i  \cdots x_t^n)}}. \notag
\end{align}$$


Given the approximation of the concrete score, the reverse-time transition rate matrix follows

$$\begin{align}
\mathrm{\mathbf{R}_t(x_t^1 \cdots x_t^i  \cdots x_t^n, x_t^1 \cdots \widehat{x}_t^i  \cdots x_t^n)}&=\mathrm{\mathbf{Q}_t(\widehat{x}_t^i, x_t^i)\frac{\mathbf{p}_t(x_t^1 \cdots \widehat{x}_t^i  \cdots x_t^n)}{\mathbf{p}_t(x_t^1 \cdots x_t^i  \cdots x_t^n)}} \notag \\
&\approx \mathrm{\mathbf{Q}_t(\widehat{x}_t^i, x_t^i) \mathbf{s}_\theta(x_t^1 \cdots x_t^i  \cdots x_t^n, t)[i, \widehat{x}_t^i]}\notag.
\end{align}$$


Moreover, we observe that $\mathrm{\mathbf{Q}_t(\widehat{x}_t^i, x_t^i)=\sigma_t \mathbf{Q}^{absorb}(\widehat{x}_t^i, x_t^i)=0}$ for any unmasked states $x_t^i\notin [\mathbf{M}]$ and $\mathrm{\widehat{x}_t^i \neq x_t^i}$. As such, the concrete score parameterization can be significantly simplified by focusing only on $\mathrm{x_t^i= [\mathbf{M}]}$ and $\mathrm{\widehat{x}_t^i \neq [\mathbf{M}]}$. The reparameterized score can be interpreted as a **time-independent** conditional probability on clean data {% cite ou2025absorbingDiscrete %} and the network parametrization is also optimized:


<figure style="text-align: center;">
  <img src="/images/RADD_net.png" width="300" />
  <figcaption style="font-size: 0.75em; margin-top: 0.1em;">
    RADD Network {% cite ou2025absorbingDiscrete %}. The time conditions are removed from the Diffusion Transformer (DiT) {% cite SEDD %}
  </figcaption>
</figure>


### Training Loss

A straightforward way to optimize the concrete score function is to use L2 loss {% cite meng2022concrete %}:

$$\begin{align}
\mathrm{\frac{1}{2} \mathbb{E}_{\mathbf{x}_t \sim \mathbf{p}_t} \left[ \sum_{\mathbf{x}_t \ne {\mathbf{y}}_t} \left( \mathbf{s}_\theta(\mathbf{x}_t, t)_{\mathbf{y}_t}  - \frac{\mathbf{p}_t({\mathbf{y}}_t)}{\mathbf{p}_t({\mathbf{x}}_t)} \right)^2 \right]}.\notag
\end{align}$$

However, such a loss does not guarantee positivity of the concrete score and neglects the underlying probabilistic structure. To resolve this issue, {% cite SEDD %} proposes the Bregman divergence $\mathrm{D_F}(\mathbf{s}_\theta, \frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)})$ with $\mathrm{F(x)=\mathrm{-\log(x)}}$ as follows

$$\begin{align}
\mathrm{D_F\bigg(\mathbf{s}_\theta, \frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\bigg)}&=\mathrm{-\log\mathbf{s}_\theta+\log\bigg(\frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\bigg)+\frac{\mathbf{p}_t(x)}{\mathbf{p}_t(y)}\mathbf{s}_\theta-1} \notag \\
&=\mathrm{\mathbb{E}_{\mathbf{x}_t\sim \mathbf{p}_t}\bigg[\frac{1}{\mathbf{p}_t(y)} \bigg(\mathbf{s}_\theta - \frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\log\mathbf{s}_\theta+K\bigg(\frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\bigg)\bigg)\bigg]},\notag \\
&\approx \mathrm{\mathbb{E}_{\mathbf{x}_t\sim \mathbf{p}_t}\bigg[\sum_{y\neq x} w_{xy} \bigg(\mathbf{s}_\theta - \frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\log\mathbf{s}_\theta+K\bigg(\frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\bigg)\bigg)\bigg]},\label{absorb_loss} \\
\end{align}$$

where $\mathrm{K(a)=a(\log a -1)}$ is a normalizing constant such that $\mathrm{D_F\big(\mathbf{s}_\theta, \frac{\mathbf{p}_t(y)}{\mathbf{p}_t(x)}\big)}\geq 0$ and $$\mathrm{w_{xy} \geq 0}$$. 

The above loss \eqref{absorb_loss} was proposed by SEDD {% cite SEDD %}, which substantially improves the training of discrete diffusion models—achieving GPT-2-level performance but at significantly higher computational cost. To further improve scalability, {% cite ou2025absorbingDiscrete %} draws insightful connections between \eqref{absorb_loss} and the any-order autoregressive loss in \eqref{AO_loss} {% cite pmlr-v32-uria14 %}{% cite hoogeboom2022autoregressive_diffusion %}{% cite shih2022anyorder %}.

$$\begin{align}
\mathrm{\mathcal{L}(\theta) \triangleq - \mathbb{E}_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^L \mathbf{1}[x_t^i = \text{M}] \log p_\theta(x_0^i \mid x_t) \right]} \label{AO_loss}.
\end{align}$$

This simplified loss \eqref{AO_loss} is in a spirit similar to the loss \eqref{shi_loss} (ask Kevin ) and forms the core training objective in {% cite LLaDA %}, enabling scalability comparable to that of large-scale language models such as LLaMA3 and other multimodal applications {% cite rojas2025diffuse %}. 


<!-- ### Discrete Flows



<!-- PDE 


ODE
Backward derivation


Discrete flow model

2.2. Kolmogorov equation -->

<!-- ### Appendix --> 







