---
layout: page
title: 카테고리
permalink: /categories/
---

{% for category in site.category_list %}
## [{{ category[1].name }}](/categories/{{ category[0] }}/index.html)
{{ category[1].description }}

{% assign posts = site.posts | where: "categories", category[0] %}
{% for post in posts limit:3 %}
[{{ post.title }}]({{ post.url | relative_url }}) - {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}
{% if posts.size > 3 %}
[더보기...](/categories/{{ category[0] }}/index.html)
{% endif %}

{% endfor %} 