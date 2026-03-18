---
title: 'Power Sampling in LLMs'
subtitle: How a training/verifier/data-free sampler outperforms GRPO
date: 2026-03-15
permalink: /posts/power_samplers/
category: Sampling
---




GRPO {% cite shao2024deepseekmath %} has become a dominant RL method for improving reasoning in LLMs. However, it requires expensive training, hyperparameter tuning, and building ad-hoc reward functions, which often reduces sample diversity and generalization ability. A natural question is: Can we unlock the same reasoning performance without costly RL training?


#### Sub-Optimal Sampling

Our goal is to sample high-likelihood trajectories $$\mathrm{p(x_{0:T})}$$. Since we don't have look-ahead oracles, the likelihood is often approximated by **autoregressive property**:

$$\mathrm{p(x_{0:T})=\Pi_{t=0}^T p(x_t\mid x_{<t})}.$$

Despite the simplicity, decoding locally may be only sub-optimal. Consider a simple example of a length-2 sentence with the token vocabulary of $$\{a, b\}$$. The sequence probabilies become

$$
\begin{equation}
\mathrm{p(aa)=0, \ \ p(ab)=0.45, \ \ p(ba)=0.25, \ \ p(bb)=0.3},\notag
\end{equation}
$$

where we know that $$\mathrm{p(x_0=a)=0.45, p(x_0=b)=0.55}.$$ In this example, decoding the token $$\mathrm{b}$$ looks like the optimal solution, which is actually unreasonable since $$\mathrm{p(ab)}$$ has the highest probability.


### Power Sampling

To sample $$\mathrm{p(ab)}$$, {% cite karan2026reasoning %} proposed a simple recipe by sampling the power distribution $$\mathrm{p^{\alpha}}$$:

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 24px;">
  <figure style="text-align: center; margin: 0;">
    <img src="/images/sharpening_distribution_power_sample.png" width="424" height="287" />
    <figcaption> A baby example for sharpened distribution {% cite karan2026reasoning %}.</figcaption>
  </figure>
</div>

Take the above example, if we choose $$\mathrm{\alpha=2.0}$$, we have

$$
\begin{equation}
\mathrm{p_{\text{power}}(x_0=a)\propto 0.45^2=0.2025 > p_{\text{power}}(x_0=b)\propto 0.25^2 + 0.3^2=0.1525}.\notag
\end{equation}
$$




The sub-optimal decoding issue is easily solved by sampling from a sharpening distribution $$\mathrm{p^2}$$. By contrast, the well-known low-temperature sampling simulates $$\mathrm{p(x_t\mid x_{<t})^{\alpha}}$$ and fails in this case

$$
\begin{equation}
\mathrm{p_{\text{low-temperature}}(x_0=a)\propto (0+0.45)^2=0.2025 < p_{\text{power}}(x_0=b)\propto (0.25 + 0.3)^2=0.3025}.\notag
\end{equation}
$$

The above example shows that $$\mathrm{p^{\alpha}}$$ motivates future high-likelihood trajectories, while low-temperature sampling only focuses on the current token.


#### Metropolis-Hastings


For a general length, we first set a block size $$\mathrm{B}$$, e.g. 192, to save computations and denote $$\mathrm{\pi_k(x_{0:kB})\propto p(x_{0:kB})^\alpha}$$. To sample from $$\mathrm{\pi_{k+1}}$$, {% cite karan2026reasoning %} proposed to first initialize a base sample $$\mathrm{x^{(0)}\rightarrow x}$$: 

$$\mathrm{x_t^{(0)}\sim p_{\text{proposal}}(x_t\mid x_{<t})} \quad \text{for} \ \ \mathrm{kB+1\leq t \leq (k+1)B}$$

where $$\mathrm{p_{\text{proposal}}}$$ is a temperature-$$\mathrm{1/\alpha}$$ sampler from the base model.

Next for a random $$\mathrm{m\in \{1, 2, \cdots, (k+1)B\}}$$, we accept the new sample $$\mathrm{x'}$$ 

$$\mathrm{x_t'\sim p_{\text{proposal}}(x_t\mid x_{<t})} \quad \text{for} \ \ \mathrm{m\leq t \leq (k+1)B}$$



with a probability $$\mathrm{1\wedge A(x', x)}$$ via the Metropolis-Hastings algorithm and repeat $$\mathrm{N_{\text{MCMC}}}$$ times

### Reasoning Performance on Math500

Empirically, power sampling significantly improves over the base model and achieves single-shot reasoning performance comparable to GRPO, without any post-training. Moreover, because it preserves sampling diversity, it delivers stronger Pass@k performance across multiple samples.

$$
\mathrm{A(x', x) \leftarrow \min \left\{ 1,\,
\frac{\pi_k(x')}{\pi_k(x)} \cdot
\frac{p_{\mathrm{proposal}}(x \mid x')}{p_{\mathrm{proposal}}(x' \mid x)}
\right\}.}
$$


<div style="display: flex; justify-content: center; align-items: flex-start; gap: 24px;">
  <figure style="text-align: center; margin: 0;">
    <img src="/images/power_sampling_multi_shot.png" width="477" height="302" />
    <figcaption> Multi-shot (Pass@k) reasoning on MATH500 {% cite karan2026reasoning %}.</figcaption>
  </figure>
</div>


The following shows the impact of $\mathrm{\alpha, N_{\text{MCMC}}}$ where $\mathrm{\alpha, N_{\text{MCMC}>1}}$ significantly improves the performance.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 24px;">
  <figure style="text-align: center; margin: 0;">
    <img src="/images/power_sampling_hp.png" width="626" height="237" />
    <figcaption> Impact of $\alpha$ and $N_{\text{MCMC}}$ {% cite karan2026reasoning %}.</figcaption>
  </figure>
</div>



<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2026_power,
  title   ={{Power Sampling in LLMs}},
  author  ={Wei Deng},
  journal ={waynedw.github.io},
  year    ={2026},
  howpublished = {\url{https://www.weideng.org/posts/power_samplers/}}
}
```
{% endraw %}
