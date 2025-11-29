---
title: 'Generative Adversarial Distillation'
subtitle: On-policy distillation for personalized or fast experts
date: 2025-11-27
permalink: /posts/adversarial_distillation/
category: Transformer
---

Generative Adversarial Networks (GANs) {% cite goodfellow2014generative %} were once among the best generative models. Their adversarial training enables the generator (G) to produce high-fidelity samples that even a strong discriminator (D) cannot distinguish from real data. Despite this success, however, the generation of **discrete sequences** remains poorly understood due to two challenges:

- 1. The loss gradient of D from G samples is not well defined on discrete data;
- 2. 


### Sequence Generative Adversarial Nets (SeqGAN)




SeqGAN handles this issue by formulating the discrete sequence generator as a stochastic policy in RL. 

<!-- On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

One-step Diffusion with Distribution Matching Distillation 
https://arxiv.org/pdf/2311.18828 -->

<!-- SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient -->

### On-policy Distillation

<br><br><br>

### Citation

{% raw %}
```bibtex
@misc{deng2025distill,
  title   ={{Adversarial Distillation}},
  author  ={Wei Deng},
  journal ={waynedw.github.io},
  year    ={2025},
  howpublished = {\url{https://weideng.org/posts/adversarial_distillation}}
}
```
{% endraw %}
