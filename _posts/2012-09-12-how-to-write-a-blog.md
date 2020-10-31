---
layout: post
title:  "Welcome to Jekyll!"
---

# Welcome

**Hello world**, this is my first Jekyll blog post.

I hope you like it!


... which is shown in the screenshot below:
![My helpful screenshot](/assets/profile4.png)


... you can [get the PDF](/assets/profile4.png) directly. v1



<ul>
  {% for post in site.posts %}
    <li>
      <a href="{https://github.com/WayneDW/Contour-Stochastic-Gradient-Langevin-Dynamics/blob/master/figures/CSGLD.gif }">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
