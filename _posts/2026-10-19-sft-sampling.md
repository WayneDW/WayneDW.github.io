---
title: 'Finetuning with Sampling'
subtitle: How MCMC turns off-policy expert demonstrations into on-policy training data
date: 2026-03-15
permalink: /posts/sft-sampling/
category: Sampling
---


Reinforcement learning (RL) has shown stronger generalization and less catastrophic forgetting than supervised finetuning (SFT) in LLM post-training. A common explanation is that RL learns from on-policy trajectories, while SFT imitates off-policy expert demonstrations that may be far from the base model distribution.

However, RL struggles when the base model cannot discover correct solutions through repeated sampling. In contrast, SFT can directly learn from expert demonstrations containing knowledge unavailable to the base model.

A natural question is: Can we preserve the knowledge in off-policy expert data while making it more on-policy?

{% cite karan2026finetuning %} propose Projection Sampling, an MCMC algorithm that transforms expert demonstrations into trajectories closer to the base model distribution before applying standard SFT.

Information Projection

Consider a base model $p(x)$ and an expert trajectory $\tau$. Let $\mathcal C$ denote the set of trajectories that preserve the information in $\tau$, such as mathematically correct solutions or responses containing equivalent factual knowledge.

Our goal is to find a training distribution that preserves expert information while remaining as close as possible to the base model:

\arg\min_{\pi:,\operatorname{supp}(\pi)\subseteq\mathcal C}
D_{\mathrm{KL}}(\pi|p).
$$

Interestingly, this optimization problem has a simple solution:

\frac{p(x)\mathbf 1[x\in\mathcal C]}
{\sum_{x'}p(x')\mathbf 1[x'\in\mathcal C]}.
}
$$

In other words, the optimal training distribution is simply the base model distribution conditioned on preserving the expert information.

To see this, denote $Z=p(\mathcal C)$. For any distribution $\pi$ supported on $\mathcal C$,

D_{\mathrm{KL}}(\pi|p_{\mathcal C})-\log Z.
$$

The minimum is therefore achieved at $\pi=p_{\mathcal C}$.

This gives us a principled target: sample correct trajectories that the base model would naturally generate.

Unfortunately, directly sampling from $p_{\mathcal C}$ by rejection sampling can be prohibitively expensive when $p(\mathcal C)$ is small. This motivates MCMC.

Metropolis–Hastings

Starting from an expert trajectory $x^{(0)}=\tau\in\mathcal C$, we construct a Markov chain targeting $p_{\mathcal C}$.

Given a current trajectory $x$, propose a new trajectory

$$
x'\sim\kappa_{\mathcal C}(\cdot\mid x,\tau),
$$

where $\kappa_{\mathcal C}$ is an information-preserving proposal distribution guided by the expert trajectory.

The Metropolis–Hastings acceptance probability is

1\wedge
\frac{p(x')}{p(x)}
\frac{\kappa_{\mathcal C}(x\mid x',\tau)}
{\kappa_{\mathcal C}(x'\mid x,\tau)}.
}
$$

Here, both trajectories are assumed to belong to $\mathcal C$.

The key distinction is that the proposal can use privileged expert information, while the target distribution remains defined by the original base model.

In practice, the authors implement the proposal through suffix resampling. Given a trajectory $x_{1}$, they randomly select an index $m$, preserve the prefix $x_{<m}$, and regenerate the remaining tokens using the base model conditioned on the expert demonstration:

$$
x'{m}
\sim
\kappa{\mathcal C}
(\cdot\mid x_{<m},\tau).
$$

Why include the expert trajectory $\tau$ in-context? Without it, the base model might rarely generate a correct completion for difficult problems. The expert information guides the proposal toward correct solutions, while MH corrects for the proposal distribution.

To reduce the cost of long-sequence sampling, the algorithm progressively extends the trajectory in blocks of size $B$, performing $N_{\mathrm{MCMC}}$ updates at each stage. This follows the blockwise sampling strategy used in Power Sampling.

Once the sampled trajectories are generated, they are used for ordinary SFT. No modification to the training loss is required.

Why Does Sampling Help?

A particularly interesting theoretical result is that the distribution of MCMC samples gets progressively closer to the base model.

Let $\pi_k$ denote the trajectory distribution after $k$ MH steps. Under an exact MH transition kernel preserving $p_{\mathcal C}$, we have

$$
\boxed{
D_{\mathrm{KL}}(\pi_{k+1}|p)
\leq
D_{\mathrm{KL}}(\pi_k|p).
}
$$

The proof follows directly from the data processing inequality. If $\Psi$ denotes the MH transition kernel, then

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_{k+1}|p_{\mathcal C})
&=
D_{\mathrm{KL}}(\Psi\pi_k|\Psi p_{\mathcal C})\
&\leq
D_{\mathrm{KL}}(\pi_k|p_{\mathcal C}).
\end{aligned}
$$

Since all distributions remain supported on $\mathcal C$, the KL identity derived earlier gives the desired result.

Thus, sampling progressively moves off-policy expert demonstrations toward the base model distribution without changing their intended information content.

There is, however, an important distinction between theory and implementation.

Information preservation is not the same as irreducibility. The former requires trajectories to remain in $\mathcal C$, whereas the latter requires every positive-probability state in $\mathcal C$ to be reachable through the MH chain.

Moreover, placing an expert solution in-context does not mathematically guarantee correctness. The theoretical convergence argument assumes a valid, information-preserving MH kernel; the actual LLM implementation is an approximation.

Reasoning Performance

The authors evaluate Projection Sampling on chemistry, mathematical reasoning, and medical question answering.

For Qwen2.5-3B on the challenging MATH Levels 3–5 benchmark, they report:

Method

MATH (3–5)

MATH500

Base model

31.5%

24.5%

SFT

24.3%

16.8%

GRPO

45.7%

31.3%

UFT

47.0%

29.7%

Sampling SFT

49.5%

58.2%

Sampling SFT + RL

54.5%

65.2%

Interestingly, standard SFT performs worse than the base model, whereas Sampling SFT outperforms both RL baselines. Initializing RL from the Sampling SFT checkpoint further improves performance.

The chemistry experiments also demonstrate substantially less forgetting. On Qwen2.5-7B-Instruct, standard SFT improves chemistry accuracy from 34.3% to 61.8%, but reduces MMLU accuracy from 68.7% to 58.6%. Sampling SFT improves chemistry accuracy to 66.0% while retaining 69.2% MMLU accuracy.

The paper further shows that sampling increases the likelihood of expert trajectories under the base model, and that increasing MCMC steps reduces the distributional gap while improving downstream accuracy.

Discussion

The central message is that the performance gap between SFT and RL may not be entirely explained by their optimization objectives. The distribution of training trajectories also matters.

Instead of designing a more sophisticated training loss, Projection Sampling adapts the data distribution to the learner:

$$
\boxed{
\text{Expert Data}
;\xrightarrow{\mathrm{MCMC}};
\text{Model-Aligned Data}
;\xrightarrow{\mathrm{SFT}};
\text{Finetuned Model}.
}
$$

Unlike inference-time power sampling, this sampling cost is paid once during dataset construction. However, the additional computational cost can still be significant, and the quality of the approximation depends on the proposal distribution, correctness preservation, and MCMC mixing.

An interesting open question is whether expert-conditioned suffix proposals can efficiently explore the entire set of semantically equivalent trajectories. Even if MH satisfies detailed balance, poor reachability or slow mixing can prevent practical sampling from approaching the desired projection.

Nevertheless, the results suggest a promising perspective: sampling can be used not only to improve inference, but also to transform training data into a distribution that is easier for a model to learn.



<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2026_power,
  title   ={{Finetuning with Sampling}},
  author  ={Wei Deng},
  journal ={A Sampler's Blog},
  year    ={2026},
  howpublished = {\url{https://www.weideng.org/posts/power_samplers/}}
}
```
{% endraw %}
