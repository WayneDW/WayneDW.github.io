---
layout: post
title:  "Welcome to Jekyll!"
---

# Welcome

**Hello world**, this is my first Jekyll blog post.

I hope you like it!


... which is shown in the screenshot below:
![My helpful screenshot](/assets/screenshot.jpg)


... you can [get the PDF](/assets/mydoc.pdf) directly.



<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ https://github.com/WayneDW/Contour-Stochastic-Gradient-Langevin-Dynamics/blob/master/figures/CSGLD.gif }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
