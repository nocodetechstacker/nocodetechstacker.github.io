---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Welcome to Nocode Tech Stacker
permalink: /
---

## {{ page.title }}

이 블로그는 노코드 기술과 AI를 활용한 개발 경험을 공유하는 공간입니다.

<ul class="post-list">
{% for post in site.posts %}
  <li class="post-item">
    <h3>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h3>
    <div class="post-meta">{{ post.date | date: "%Y년 %m월 %d일" }}</div>
    {% if post.description %}
    <div class="post-description">{{ post.description }}</div>
    {% endif %}
    {% if post.categories %}
    <div class="post-categories">
      {% for category in post.categories %}
      <span class="post-category">{{ category }}</span>
      {% endfor %}
    </div>
    {% endif %}
  </li>
{% endfor %}
</ul>
