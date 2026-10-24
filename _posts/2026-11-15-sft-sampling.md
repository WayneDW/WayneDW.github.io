---
title: 'Finetuning with Sampling'
subtitle: MCMC avoids catastrophic forgetting in SFT
date: 2026-11-15
permalink: /posts/sft-sampling/
category: Sampling
---


Supervised Fine-Tuning (SFT) equips LLMs with new knowledge and reasoning capabilities. However, off-policy expert demonstrations may have low likelihood under the base model, requiring substantial distributional shifts that can cause catastrophic forgetting. While techniques, such as experience replay, LoRA, and regularization help mitigate this issue, principled solutions remain elusive.


### From Off-Policy Demonstrations to On-Policy Data via MCMC {% cite karan2026finetuning %}


Let $$\mathrm{p(x)}$$ denote the base model distribution over trajectories for a given question, and let $$\mathrm{C}$$ denote the set of correct trajectories that preserve the expert's information. Ideally, we want to sample trajectories that satisfy the correctness constraints while remaining as close as possible to the base model:

$$\mathrm{\arg\min_{\pi:\,supp(\pi)\subseteq C}\, D_{KL}(\pi|p)}.$$

Interestingly, the optimal distribution has a simple closed-form solution:

$$
\mathrm{p^*(x)=\frac{p(x)\,\mathbf{1}[x\in C]}{\sum_{x'}p(x')\,\mathbf{1}[x'\in C]}.}
$$


In other words, the ideal training distribution is the base model distribution conditioned on generating correct, information-preserving solutions $$\mathrm{p^*(x)=p_{base}(x\mid x\in C)}$$.


Consider a simple example where the base model assigns probabilities

$$\mathrm{p(a)=0.02,\quad p(b)=0.06,\quad p(c)=0.12,\quad p(d)=0.80,}$$

where $$\mathrm{a,b,c}$$ are correct trajectories in $$\mathrm{C}$$ and $$\mathrm{d}$$ is incorrect.

Conditioning on the set of correct trajectories $$\mathrm{C}$$ gives

$$\mathrm{p_C(a)=0.1,\quad p_C(b)=0.3,\quad p_C(c)=0.6.}$$

Instead of directly imitating an arbitrary expert trajectory, the projected distribution favors correct trajectories that are already more natural for the base model.

### Sampling Expert-Guided Trajectories

Directly sampling correct trajectories can be extremely inefficient when the model struggles with difficult problems. To address this issue, the authors introduce Projection Sampling, which uses expert demonstrations to guide the generation of new training trajectories.

The solusion is fairly simple. Given a current trajectory $$\mathrm{x}$$, an expert solution $$\mathrm{\tau}$$, and a base model $$\mathrm{p}$$.

**Algorithm (Projection Sampling step).**
1. Sample a cut index $$\mathrm{m}$$ uniformly from the positions of $$\mathrm{x}$$.
2. Preserve the prefix $$\mathrm{x_{<m}}$$ and discard the suffix $$\mathrm{x_{\ge m}}$$.
3. Form a proposal prompt that includes the question, the expert solution $$\mathrm{\tau}$$, and the prefix $$\mathrm{x_{<m}}$$.
4. Sample a continuation from the base model in its own reasoning style:
$$\mathrm{x'_t \sim p_{proposal}(x_t \mid x'_{<t}, \tau),\qquad m \le t \le T.}$$
5. Accept the candidate $$\mathrm{x'}$$ with probability
$$\mathrm{A(x',x)=\min\left\{1,\;\frac{p(x')}{p(x)}\frac{\kappa_C(x\mid x',\tau)}{\kappa_C(x'\mid x,\tau)}\right\},}$$
and otherwise keep $$\mathrm{x}$$; then repeat.

The third step is particularly important. Without expert guidance, the base model may struggle to complete a difficult reasoning problem **correctly**. By providing the expert solution as in-context information, the model can use knowledge it may not otherwise possess, while generating the continuation in its own words.

After several sampling iterations, the generated trajectories become more compatible with the base model while retaining the information contained in the expert demonstrations.

The resulting trajectories are then used for standard SFT, without changing the training objective.


### Reasoning Performance

Empirically, Projection Sampling improves over standard SFT. On chemistry (Qwen2.5-7B-Instruct), Sampling SFT reaches 66.0% vs. 61.8% for vanilla SFT, while retaining MMLU (69.2% vs. 58.6%). On harder math (Qwen2.5-3B), the gap is larger:

<style>
.perf-tables {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  margin: 0.6rem 0 0.8rem;
}
.perf-tables > div { flex: 1; min-width: 0; }
.perf-tables .caption {
  font-size: 0.85em;
  font-weight: 600;
  margin: 0 0 0.35rem;
  text-align: center;
}
.perf-tables table {
  display: table !important;
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82em;
  line-height: 1.35;
  margin: 0 !important;
}
.perf-tables th,
.perf-tables td {
  padding: 0.28rem 0.45rem !important;
  border: 1px solid #d0d7de;
}
.perf-tables th { background: #f6f8fa; text-align: center; }
.perf-tables td:not(:first-child) { text-align: right; }
@media (max-width: 640px) {
  .perf-tables { flex-direction: column; }
}
</style>

<div class="perf-tables">
  <div>
    <div class="caption">Chemistry &amp; MMLU</div>
    <table>
      <thead>
        <tr><th>Method</th><th>Chemistry</th><th>MMLU</th></tr>
      </thead>
      <tbody>
        <tr><td>Base Model</td><td>34.3%</td><td>68.7%</td></tr>
        <tr><td>SFT</td><td>61.8%</td><td>58.6%</td></tr>
        <tr><td>OPSD</td><td>61.8%</td><td>65.1%</td></tr>
        <tr><td><strong>Sampling SFT</strong></td><td><strong>66.0%</strong></td><td><strong>69.2%</strong></td></tr>
      </tbody>
    </table>
  </div>
  <div>
    <div class="caption">Mathematical Reasoning</div>
    <table>
      <thead>
        <tr><th>Method</th><th>Math Acc.</th></tr>
      </thead>
      <tbody>
        <tr><td>Base Model</td><td>31.5%</td></tr>
        <tr><td>SFT</td><td>24.3%</td></tr>
        <tr><td>GRPO</td><td>45.7%</td></tr>
        <tr><td>UFT</td><td>47.0%</td></tr>
        <tr><td><strong>Sampling SFT</strong></td><td><strong>49.5%</strong></td></tr>
      </tbody>
    </table>
  </div>
</div>

Vanilla SFT underperforms the base model on math, while Sampling SFT beats GRPO and UFT; combining it with RL further reaches 54.5%.

### Discussion

The main takeaway is that the effectiveness of SFT depends not only on the correctness of expert demonstrations, but also on how compatible those demonstrations are with the model's existing distribution.

Rather than designing a more complicated training objective, Projection Sampling improves the distribution of training data before applying standard SFT.


One limitation is that expert-guided generation does not strictly guarantee correctness, and the theoretical distributional guarantees rely on idealized sampling assumptions.

Nevertheless, the paper suggests a promising direction: instead of only asking how to train LLMs more effectively, we should also ask how to sample better data for training.

<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2026_sft_sampling,
  title   ={{Finetuning with Sampling}},
  author  ={Wei Deng},
  journal ={A Sampler's Blog},
  year    ={2026},
}
```
{% endraw %}
